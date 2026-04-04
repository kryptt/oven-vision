use std::f64::consts::PI;

use opencv::core::{CV_16S, Mat, Rect, Scalar, Size, Vec4i, Vector};
use opencv::imgproc;
use opencv::prelude::*;

use super::stage::{Line, PipelineState, StageDescriptor, StageOutcome, VerticalPair};
use super::util::draw_line;
use super::{DebugImage, ImageOutput, Stage};
use crate::annotate::encode_jpeg;

/// Minimum horizontal separation (pixels) between left and right boundaries.
const MIN_SEPARATION: f64 = 150.0;

/// How far into the image to search for the left boundary (fraction of width).
const LEFT_SEARCH_FRAC: f64 = 0.40;

/// How far from the right edge to start searching for the right boundary.
const RIGHT_SEARCH_FRAC: f64 = 0.60;

/// Vertical band of the image to use for column profiling (fraction of height).
const BAND_TOP_FRAC: f64 = 0.15;
const BAND_BOTTOM_FRAC: f64 = 0.85;

/// Minimum absolute gradient strength for a peak to be accepted.
const MIN_GRADIENT_STRENGTH: f64 = 5.0;

/// Multi-scale pyramid: original, half, quarter resolution.
const SCALES: &[f64] = &[1.0, 0.5, 0.25];

/// Number of top peaks to extract per side at each scale.
const PEAKS_PER_SCALE: usize = 2;

/// Non-maximum suppression radius in original-resolution pixels.
const NMS_RADIUS: f64 = 30.0;

/// Neighborhood suppression radius at each scale's pixel coordinates.
const SUPPRESS_RADIUS_SCALE: usize = 40;

/// A gradient peak candidate with metadata for cross-scale merging.
#[derive(Debug, Clone)]
struct PeakCandidate {
    /// X-coordinate in original resolution.
    x_orig: f64,
    /// Raw gradient strength (negative for left, positive for right).
    strength: f64,
    /// Strength normalized by the max at this scale (0..1).
    norm_strength: f64,
    /// Which scale produced this candidate.
    scale: f64,
}

/// Stage S3: Detect the left and right boundaries of the stove panel using
/// multi-scale Sobel-X gradient pyramid with cross-scale NMS.
///
/// The left boundary is cream cabinet -> dark stove (negative horizontal gradient).
/// The right boundary is dark stove -> white wall (positive horizontal gradient).
///
/// **Variant C: Multi-Scale Gradient Pyramid**
///
/// Builds a 3-level image pyramid (1x, 0.5x, 0.25x), extracts the top-2
/// gradient peaks per side at each scale, merges them via NMS at 30px in
/// original coordinates, then forms the Cartesian product of left x right
/// candidates sorted by combined normalized strength. Each iteration selects
/// the Nth pair.
pub struct FindVerticals;

impl FindVerticals {
    pub fn new() -> Self {
        Self
    }
}

pub(crate) const DESCRIPTOR: StageDescriptor = StageDescriptor {
    name: "FindVerticals",
    label: "S3:FindVerticals",
    fallback: Some("FindLines"),
};

impl Stage for FindVerticals {
    fn descriptor(&self) -> StageDescriptor {
        DESCRIPTOR
    }

    fn run(
        &self,
        state: &mut PipelineState,
        src: &Mat,
        _dst: &mut Mat,
        _raw: &Mat,
        iteration: u32,
    ) -> Result<(StageOutcome, ImageOutput), opencv::Error> {
        if state.crop.is_none() {
            return Ok((
                StageOutcome::Exhausted("no crop region from Stage 1".into()),
                ImageOutput::Passthrough,
            ));
        }

        let w = src.cols();
        let h = src.rows();
        if w < 50 || h < 50 {
            return Ok((
                StageOutcome::Exhausted("source image too small".into()),
                ImageOutput::Passthrough,
            ));
        }

        // Step 1: CLAHE + median blur with fixed parameters.
        let smooth = super::util::enhance_gray(src, 7)?;

        // Step 2-4: Build multi-scale pyramid, extract peaks at each scale.
        let mut left_candidates: Vec<PeakCandidate> = Vec::new();
        let mut right_candidates: Vec<PeakCandidate> = Vec::new();

        for &scale in SCALES {
            let scaled = if (scale - 1.0).abs() < f64::EPSILON {
                smooth.try_clone()?
            } else {
                let mut resized = Mat::default();
                let new_size = Size::new(
                    (w as f64 * scale) as i32,
                    (h as f64 * scale) as i32,
                );
                imgproc::resize(&smooth, &mut resized, new_size, 0.0, 0.0, imgproc::INTER_AREA)?;
                resized
            };

            let sw = scaled.cols();
            let sh = scaled.rows();
            if sw < 20 || sh < 20 {
                continue;
            }

            // Sobel-X at this scale.
            let mut sobel_x = Mat::default();
            imgproc::sobel(
                &scaled,
                &mut sobel_x,
                CV_16S,
                1,
                0,
                3,
                1.0,
                0.0,
                opencv::core::BORDER_DEFAULT,
            )?;

            // Vertical band 15%-85% at this scale's dimensions.
            let y_start = (BAND_TOP_FRAC * sh as f64) as i32;
            let y_end = (BAND_BOTTOM_FRAC * sh as f64) as i32;
            if y_end <= y_start + 5 {
                continue;
            }

            let roi = Rect::new(0, y_start, sw, y_end - y_start);
            let sobel_roi = Mat::roi(&sobel_x, roi)?;

            // Column-wise mean gradient profile.
            let roi_h = sobel_roi.rows();
            let mut profile = vec![0.0f64; sw as usize];
            for x in 0..sw {
                let mut sum = 0.0f64;
                for y in 0..roi_h {
                    sum += *sobel_roi.at_2d::<i16>(y, x)? as f64;
                }
                profile[x as usize] = sum / roi_h as f64;
            }

            // LEFT peaks: local minima in [0, 40%) below -MIN_GRADIENT_STRENGTH.
            let left_end = (sw as f64 * LEFT_SEARCH_FRAC) as usize;
            let left_peaks = extract_top_peaks(
                &profile[..left_end],
                PEAKS_PER_SCALE,
                SUPPRESS_RADIUS_SCALE,
                true, // looking for minima (negative gradients)
            );

            // RIGHT peaks: local maxima in [60%, width) above +MIN_GRADIENT_STRENGTH.
            let right_start = (sw as f64 * RIGHT_SEARCH_FRAC) as usize;
            let right_peaks = extract_top_peaks(
                &profile[right_start..],
                PEAKS_PER_SCALE,
                SUPPRESS_RADIUS_SCALE,
                false, // looking for maxima (positive gradients)
            );

            // Find max strength at this scale for normalization.
            let left_max = left_peaks
                .iter()
                .map(|(_, v)| v.abs())
                .fold(0.0f64, f64::max);
            let right_max = right_peaks
                .iter()
                .map(|(_, v)| v.abs())
                .fold(0.0f64, f64::max);

            // Step 5: Map coordinates back to original resolution.
            for (x, val) in left_peaks {
                let norm = if left_max > 0.0 { val.abs() / left_max } else { 0.0 };
                left_candidates.push(PeakCandidate {
                    x_orig: x as f64 / scale,
                    strength: val,
                    norm_strength: norm,
                    scale,
                });
            }
            for (x, val) in right_peaks {
                let x_global = right_start + x;
                let norm = if right_max > 0.0 { val.abs() / right_max } else { 0.0 };
                right_candidates.push(PeakCandidate {
                    x_orig: x_global as f64 / scale,
                    strength: val,
                    norm_strength: norm,
                    scale,
                });
            }
        }

        // Step 6: NMS across scales at 30px in original-resolution coordinates.
        let left_nms = nms_peaks(left_candidates, NMS_RADIUS);
        let right_nms = nms_peaks(right_candidates, NMS_RADIUS);

        if left_nms.is_empty() || right_nms.is_empty() {
            return Ok((
                StageOutcome::Exhausted("no gradient peaks found at any scale".into()),
                ImageOutput::Passthrough,
            ));
        }

        // Step 7: Cartesian product of left x right, filtered by MIN_SEPARATION,
        // sorted by combined normalized strength (descending).
        let mut pairs: Vec<(PeakCandidate, PeakCandidate)> = Vec::new();
        for l in &left_nms {
            for r in &right_nms {
                let sep = r.x_orig - l.x_orig;
                if sep >= MIN_SEPARATION {
                    pairs.push((l.clone(), r.clone()));
                }
            }
        }
        pairs.sort_by(|(la, ra), (lb, rb)| {
            let score_a = la.norm_strength + ra.norm_strength;
            let score_b = lb.norm_strength + rb.norm_strength;
            score_b.total_cmp(&score_a) // descending
        });

        if pairs.is_empty() {
            return Ok((
                StageOutcome::Exhausted("no valid left/right pair meets separation".into()),
                ImageOutput::Passthrough,
            ));
        }

        // Step 8: iteration selects the Nth pair.
        let idx = iteration as usize;
        if idx >= pairs.len() {
            return Ok((
                StageOutcome::Exhausted(format!(
                    "all {} candidate pairs exhausted",
                    pairs.len()
                )),
                ImageOutput::Passthrough,
            ));
        }
        let (ref left_peak, ref right_peak) = pairs[idx];
        let left_x = left_peak.x_orig.round() as i32;
        let right_x = right_peak.x_orig.round() as i32;

        eprintln!(
            "  [S3] iter={iteration} pair {}/{}: L={}(s={},ns={:.2}) R={}(s={},ns={:.2})",
            idx + 1,
            pairs.len(),
            left_x,
            left_peak.scale,
            left_peak.norm_strength,
            right_x,
            right_peak.scale,
            right_peak.norm_strength,
        );

        // Expected perpendicular angle from S2's horizontal lines.
        let expected_perp = state
            .lines
            .as_ref()
            .map(|lp| lp.avg_theta - PI / 2.0)
            .unwrap_or(0.0);

        // Step 9: Sobel-X at original resolution for slope refinement.
        let mut sobel_x_full = Mat::default();
        imgproc::sobel(
            &smooth,
            &mut sobel_x_full,
            CV_16S,
            1,
            0,
            3,
            1.0,
            0.0,
            opencv::core::BORDER_DEFAULT,
        )?;

        let left_line =
            refine_boundary_slope(&sobel_x_full, left_x, w, h, true, expected_perp)?;
        let right_line =
            refine_boundary_slope(&sobel_x_full, right_x, w, h, false, expected_perp)?;

        state.verticals = Some(VerticalPair {
            left: left_line,
            right: right_line,
        });

        Ok((StageOutcome::Success, ImageOutput::Passthrough))
    }

    fn max_retries(&self) -> u32 {
        15
    }

    fn debug_image(
        &self,
        state: &PipelineState,
        working: &Mat,
        _raw: &Mat,
    ) -> Result<Option<DebugImage>, opencv::Error> {
        let Some(verticals) = &state.verticals else {
            return Ok(None);
        };

        let mut canvas = working.try_clone()?;

        draw_line(
            &mut canvas,
            &verticals.left,
            Scalar::new(255.0, 0.0, 255.0, 0.0), // magenta
        )?;
        draw_line(
            &mut canvas,
            &verticals.right,
            Scalar::new(255.0, 255.0, 0.0, 0.0), // cyan
        )?;

        let jpeg = encode_jpeg(&canvas, 80)?;
        let label = format!(
            "S3:FindVerticals_L{:.0}_R{:.0}",
            verticals.left.x1, verticals.right.x1
        );
        Ok(Some((label, jpeg)))
    }
}

/// Extract top-N peaks from a 1D profile using greedy suppression.
///
/// If `is_min` is true, searches for the strongest minima (most negative values
/// below `-MIN_GRADIENT_STRENGTH`). Otherwise, searches for the strongest maxima
/// (most positive values above `+MIN_GRADIENT_STRENGTH`).
///
/// Returns `(local_index, value)` pairs.
fn extract_top_peaks(
    profile: &[f64],
    n: usize,
    suppress_radius: usize,
    is_min: bool,
) -> Vec<(usize, f64)> {
    let mut peaks: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut suppressed = vec![false; profile.len()];

    for _ in 0..n {
        let best = profile
            .iter()
            .enumerate()
            .filter(|(i, _)| !suppressed[*i])
            .filter(|&(_, &v)| {
                if is_min {
                    v < -MIN_GRADIENT_STRENGTH
                } else {
                    v > MIN_GRADIENT_STRENGTH
                }
            })
            .min_by(|(_, a), (_, b)| {
                if is_min {
                    a.total_cmp(b) // most negative first
                } else {
                    b.total_cmp(a) // most positive first
                }
            });

        let Some((idx, &val)) = best else { break };

        peaks.push((idx, val));

        // Suppress neighborhood around this peak.
        let lo = idx.saturating_sub(suppress_radius);
        let hi = (idx + suppress_radius + 1).min(profile.len());
        for s in &mut suppressed[lo..hi] {
            *s = true;
        }
    }

    peaks
}

/// Non-maximum suppression: merge peak candidates within `radius` pixels
/// in original-resolution coordinates, keeping the strongest.
fn nms_peaks(mut candidates: Vec<PeakCandidate>, radius: f64) -> Vec<PeakCandidate> {
    // Sort by absolute strength descending so strongest peaks survive.
    candidates.sort_by(|a, b| b.strength.abs().total_cmp(&a.strength.abs()));

    let mut kept: Vec<PeakCandidate> = Vec::new();
    for c in candidates {
        let dominated = kept
            .iter()
            .any(|k| (k.x_orig - c.x_orig).abs() < radius);
        if !dominated {
            kept.push(c);
        }
    }
    kept
}

/// Refine the slope of a detected boundary using HoughLinesP on a narrow
/// vertical strip around the detected X column.
///
/// `expected_perp` is the expected angle (radians) of the vertical boundary
/// relative to true vertical, derived from the horizontal lines in S2.
/// Segments are scored by proximity to this angle.
///
/// If HoughLinesP finds a segment, returns its endpoints as a Line.
/// Otherwise, returns a vertical line at the detected X.
fn refine_boundary_slope(
    sobel_x: &Mat,
    center_x: i32,
    img_w: i32,
    img_h: i32,
    is_negative: bool,
    expected_perp: f64,
) -> Result<Line, opencv::Error> {
    let strip_half = 25;
    let x0 = (center_x - strip_half).max(0);
    let x1 = (center_x + strip_half).min(img_w);
    let strip_w = x1 - x0;

    if strip_w < 10 {
        // Fallback: vertical line
        return Ok(Line {
            x1: center_x as f64,
            y1: 0.0,
            x2: center_x as f64,
            y2: img_h as f64,
        });
    }

    let strip_roi = Rect::new(x0, 0, strip_w, img_h);
    let strip = Mat::roi(sobel_x, strip_roi)?;

    // Convert to absolute 8-bit
    let mut abs_strip = Mat::default();
    opencv::core::convert_scale_abs(&strip, &mut abs_strip, 1.0, 0.0)?;

    // Threshold to keep only strong gradients
    let mut thresh = Mat::default();
    imgproc::threshold(&abs_strip, &mut thresh, 30.0, 255.0, imgproc::THRESH_BINARY)?;

    // HoughLinesP on the narrow strip
    let mut lines = Vector::<Vec4i>::new();
    imgproc::hough_lines_p(
        &thresh,
        &mut lines,
        1.0,
        PI / 180.0,
        20,   // threshold (votes)
        30.0, // min line length
        15.0, // max line gap
    )?;

    // Filter to negative-slope segments only (x decreases as y increases,
    // matching the known camera angle where vertical edges tilt left going down).
    // Ensure the segment goes top-to-bottom by normalizing so y1 < y2.
    let negative_slope: Vec<Vec4i> = lines
        .iter()
        .map(|l| {
            if l[1] <= l[3] {
                l // already top-to-bottom
            } else {
                Vec4i::from([l[2], l[3], l[0], l[1]]) // flip
            }
        })
        .filter(|l| {
            if is_negative {
                l[2] <= l[0] // x2 <= x1 = negative slope (left-leaning)
            } else {
                l[2] >= l[0] // x2 >= x1 = positive slope (right-leaning)
            }
        })
        .collect();

    if negative_slope.is_empty() {
        return Ok(Line {
            x1: center_x as f64,
            y1: 0.0,
            x2: center_x as f64,
            y2: img_h as f64,
        });
    }

    // Score segments by length and proximity to the expected perpendicular angle.
    // Perpendicular tolerance: 20 degrees max deviation.
    const PERP_TOLERANCE: f64 = 20.0 * PI / 180.0;
    let max_len: f64 = negative_slope
        .iter()
        .map(|l| {
            let dx = (l[2] - l[0]) as f64;
            let dy = (l[3] - l[1]) as f64;
            (dx * dx + dy * dy).sqrt()
        })
        .fold(1.0f64, f64::max);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_line = negative_slope[0];
    for l in &negative_slope {
        let dx = (l[2] - l[0]) as f64;
        let dy = (l[3] - l[1]) as f64;
        let len = (dx * dx + dy * dy).sqrt();
        // Angle from vertical (atan2 gives angle from horizontal, subtract PI/2)
        let seg_angle = dx.atan2(dy); // radians from vertical
        let angle_dev = (seg_angle - expected_perp).abs();
        let angle_score = 1.0 - (angle_dev / PERP_TOLERANCE).min(1.0);
        let len_score = len / max_len;
        // Weight: 60% angle proximity, 40% length
        let score = 0.6 * angle_score + 0.4 * len_score;
        if score > best_score {
            best_score = score;
            best_line = *l;
        }
    }

    // Convert strip-local coordinates back to image coordinates
    Ok(Line {
        x1: (best_line[0] + x0) as f64,
        y1: best_line[1] as f64,
        x2: (best_line[2] + x0) as f64,
        y2: best_line[3] as f64,
    })
}
