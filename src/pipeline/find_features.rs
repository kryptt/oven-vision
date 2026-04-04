use std::sync::Arc;

use opencv::core::{BORDER_CONSTANT, Mat, Point, Point2f, Scalar, Size};
use opencv::imgproc;
use opencv::prelude::*;
use rayon::prelude::*;

use super::stage::{CircleFeature, DetectedFeatures, PipelineState, StageDescriptor, StageOutcome};
use super::util::{Templates, enhance_gray};
use super::{DebugImage, ImageOutput, Stage};
use crate::annotate::encode_jpeg;

/// Expected knob count (excluding the clock).
const EXPECTED_KNOBS: usize = 10;

/// Template matching threshold (TM_CCOEFF_NORMED). Matches below this are ignored.
const MATCH_THRESHOLD: f64 = 0.35;

/// Minimum distance between two accepted match positions (pixels in warped image).
/// Prevents double-detections of the same knob.
const NMS_MIN_DIST: f64 = 15.0;

/// Scale factors to try. The phone template knob ring is ~100px diameter;
/// the warped knob is ~20-50px. Scale range 0.15-0.50.
const SCALE_FACTORS: [f64; 8] = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.15];

/// Rotation step in degrees for knob template matching.
const ROTATION_STEP_DEG: f64 = 10.0;

/// A pre-computed edge template at a specific scale and rotation.
struct PrecomputedEdgeTemplate {
    edge: Mat,
    angle: f64,
    scale: f64,
}

/// Rayon thread count for parallel template matching (sweet spot on 16-core).
const RAYON_THREADS: usize = 16;

/// Stage 8: Detect features using multi-scale, multi-rotation template matching.
///
/// Uses reference photos of the actual knob and clock to find their positions
/// in the knob area (right of clock). Each knob match also yields the handle
/// angle directly (the rotation that scored highest).
///
/// Coordinates are translated back using stored offsets from FindClock and
/// ExtractBand.
///
/// Pre-computes all 8 scales × 36 rotations = 288 edge templates at
/// construction to avoid redundant resize+rotate+canny per iteration.
pub struct FindFeatures {
    /// Shared CLAHE-enhanced templates.
    templates: Arc<Templates>,
    /// Pre-computed knob edge templates: Canny applied to the CLAHE template,
    /// then rotated with INTER_NEAREST to preserve clean binary edges.
    /// Indexed by scale_idx * n_rotations + rot_idx.
    knob_edge_templates: Vec<PrecomputedEdgeTemplate>,
    /// Dedicated rayon pool for parallel template matching.
    pool: rayon::ThreadPool,
}

impl FindFeatures {
    pub fn new(templates: Arc<Templates>) -> Self {
        // Pre-compute all edge templates: scale → Canny → rotate with INTER_NEAREST
        let n_rotations = (360.0 / ROTATION_STEP_DEG) as i32;
        let mut knob_edge_templates = Vec::new();

        if !templates.knob.empty() {
            for &scale in &SCALE_FACTORS {
                if let Ok(scaled) = resize_template(&templates.knob, scale) {
                    if scaled.cols() < 5 || scaled.rows() < 5 {
                        continue;
                    }
                    // Apply Canny FIRST (on the clean scaled template)
                    let mut edge_scaled = Mat::default();
                    if imgproc::canny(&scaled, &mut edge_scaled, 50.0, 150.0, 3, false).is_err() {
                        continue;
                    }
                    // Then rotate the binary edge map with INTER_NEAREST
                    for rot_idx in 0..n_rotations {
                        let angle = rot_idx as f64 * ROTATION_STEP_DEG;
                        if let Ok(rotated_edge) = rotate_edge_template(&edge_scaled, angle) {
                            if !rotated_edge.empty() {
                                knob_edge_templates.push(PrecomputedEdgeTemplate {
                                    edge: rotated_edge,
                                    angle,
                                    scale,
                                });
                            }
                        }
                    }
                }
            }
            tracing::info!(count = knob_edge_templates.len(), "pre-computed knob edge templates");
        }

        // Disable OpenCV internal threading — rayon owns parallelism.
        let _ = opencv::core::set_num_threads(1);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(RAYON_THREADS)
            .build()
            .expect("failed to build rayon thread pool");

        Self {
            templates,
            knob_edge_templates,
            pool,
        }
    }
}

pub(crate) const DESCRIPTOR: StageDescriptor = StageDescriptor {
    name: "FindFeatures",
    label: "S8:FindFeatures",
    fallback: Some("FindClock"),
};

impl Stage for FindFeatures {
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
        if self.templates.knob.empty() || self.templates.clock.empty() {
            return Ok((
                StageOutcome::Exhausted("template images not loaded".into()),
                ImageOutput::Passthrough,
            ));
        }

        let Some(search) = &state.knob_search else {
            return Ok((
                StageOutcome::Exhausted("missing knob search area".into()),
                ImageOutput::Passthrough,
            ));
        };

        // src is the knob area (right of clock) from FindClock
        let x_offset = search.x_min; // offset to translate back to band coords
        let y_offset = search.y_min; // offset to translate back to warped-image coords

        let blurred = enhance_gray(src, 3)?;

        // Edge detection on the working image
        let mut edge_img = Mat::default();
        imgproc::canny(&blurred, &mut edge_img, 50.0, 150.0, 3, false)?;

        // Pick scale based on iteration (cycle through scales, then repeat with lower threshold)
        let scale_idx = (iteration as usize) % SCALE_FACTORS.len();
        let scale = SCALE_FACTORS[scale_idx];
        let threshold_adj = (iteration as usize / SCALE_FACTORS.len()) as f64 * 0.05;
        let threshold = (MATCH_THRESHOLD - threshold_adj).max(0.15);

        // --- Find knobs: use pre-computed edge templates for this scale ---
        let templates_for_scale: Vec<&PrecomputedEdgeTemplate> = self
            .knob_edge_templates
            .iter()
            .filter(|t| (t.scale - scale).abs() < 1e-6)
            .collect();

        if templates_for_scale.is_empty() {
            return Ok((
                StageOutcome::Retry(format!("no pre-computed templates for scale {scale:.2}")),
                ImageOutput::Passthrough,
            ));
        }

        // Parallel template matching across rotations using rayon.
        let edge_ref = &edge_img;
        let mut all_knob_matches: Vec<TemplateMatch> = self.pool.install(|| {
            templates_for_scale
                .par_iter()
                .filter_map(|pct| {
                    if pct.edge.cols() >= edge_ref.cols() || pct.edge.rows() >= edge_ref.rows() {
                        return None;
                    }

                    let mut result = Mat::default();
                    imgproc::match_template(
                        edge_ref,
                        &pct.edge,
                        &mut result,
                        imgproc::TM_CCOEFF_NORMED,
                        &Mat::default(),
                    )
                    .ok()?;

                    find_peaks(&result, threshold, pct.edge.cols(), pct.edge.rows(), pct.angle)
                        .ok()
                })
                .flatten()
                .collect()
        });

        // Compute knob radius from first matching template
        let knob_r = templates_for_scale
            .first()
            .map(|t| t.edge.cols() as f64 / 2.0)
            .unwrap_or(10.0);

        if all_knob_matches.len() < EXPECTED_KNOBS {
            return Ok((
                StageOutcome::Retry(format!(
                    "found {} knob matches (need {}), scale={scale:.2}, threshold={threshold:.2}",
                    all_knob_matches.len(),
                    EXPECTED_KNOBS
                )),
                ImageOutput::Passthrough,
            ));
        }

        // NMS: keep best non-overlapping matches
        let knob_results = nms(&mut all_knob_matches, NMS_MIN_DIST);

        if knob_results.len() < EXPECTED_KNOBS {
            return Ok((
                StageOutcome::Retry(format!(
                    "only {} knob matches after NMS (need {}), scale={scale:.2}",
                    knob_results.len(),
                    EXPECTED_KNOBS
                )),
                ImageOutput::Passthrough,
            ));
        }

        // --- Find clock: scale-only edge-based matching (no rotation) ---
        let scaled_clock = resize_template(&self.templates.clock, scale)?;
        let mut clock_match: Option<TemplateMatch> = None;

        if scaled_clock.cols() >= 5
            && scaled_clock.rows() >= 5
            && scaled_clock.cols() < edge_img.cols()
            && scaled_clock.rows() < edge_img.rows()
        {
            let mut edge_clock = Mat::default();
            imgproc::canny(&scaled_clock, &mut edge_clock, 50.0, 150.0, 3, false)?;

            let mut result = Mat::default();
            imgproc::match_template(
                &edge_img,
                &edge_clock,
                &mut result,
                imgproc::TM_CCOEFF_NORMED,
                &Mat::default(),
            )?;

            let peaks = find_peaks(
                &result,
                threshold,
                scaled_clock.cols(),
                scaled_clock.rows(),
                0.0,
            )?;
            if !peaks.is_empty() {
                clock_match = Some(peaks[0].clone());
            }
        }

        // --- Select the best 10+1 pattern from matches ---
        let clock_r = scaled_clock.cols().max(scaled_clock.rows()) as f64 / 2.0;

        // Sort knob matches by score descending
        let mut sorted_knobs = knob_results;
        sorted_knobs.sort_by(|a, b| {
            b.score.total_cmp(&a.score)
        });

        // Select top matches that are Y-aligned (best 10 by Y-consistency)
        let selected = select_y_aligned_knobs(&sorted_knobs, EXPECTED_KNOBS, knob_r);

        let Some(knobs) = selected else {
            return Ok((
                StageOutcome::Retry(format!(
                    "could not find {EXPECTED_KNOBS} Y-aligned knob matches, scale={scale:.2}"
                )),
                ImageOutput::Passthrough,
            ));
        };

        // Translate coordinates back to warped-image space
        let knob_median_y = {
            let mut ys: Vec<f64> = knobs.iter().map(|k| k.center_y + y_offset).collect();
            ys.sort_by(|a, b| a.total_cmp(b));
            ys[ys.len() / 2]
        };
        let leftmost_knob_x = knobs
            .iter()
            .map(|k| k.center_x + x_offset)
            .fold(f64::INFINITY, f64::min);

        let clock = if let Some(cm) = clock_match {
            let cm_x = cm.x + x_offset;
            let cm_y = cm.y + y_offset;
            // Validate clock is left of knobs and near the same Y
            if cm_x < leftmost_knob_x && (cm_y - knob_median_y).abs() < knob_r * 4.0 {
                CircleFeature {
                    center_x: cm_x,
                    center_y: cm_y,
                    radius: clock_r,
                }
            } else {
                // Clock match in wrong position -- synthesize from knob row
                tracing::warn!(
                    cm_x, cm_y = cm.y + y_offset, leftmost_knob_x,
                    "clock match in wrong position, synthesizing from knob row"
                );
                CircleFeature {
                    center_x: leftmost_knob_x - knob_r * 4.0,
                    center_y: knob_median_y,
                    radius: clock_r,
                }
            }
        } else {
            // No clock match -- estimate position from knob row
            tracing::warn!("no clock match, synthesizing position from knob row");
            CircleFeature {
                center_x: leftmost_knob_x - knob_r * 4.0,
                center_y: knob_median_y,
                radius: clock_r,
            }
        };

        // Extract off-angles from the best-matching rotation per knob
        let off_angles: Vec<f64> = knobs.iter().map(|k| k.angle).collect();

        state.features = Some(DetectedFeatures {
            clock,
            knobs: knobs
                .iter()
                .map(|k| CircleFeature {
                    center_x: k.center_x + x_offset,
                    center_y: k.center_y + y_offset,
                    radius: knob_r,
                })
                .collect(),
            off_angles,
        });

        Ok((StageOutcome::Success, ImageOutput::Passthrough))
    }

    fn max_retries(&self) -> u32 {
        10
    }

    fn debug_image(
        &self,
        state: &PipelineState,
        working: &Mat,
        _raw: &Mat,
    ) -> Result<Option<DebugImage>, opencv::Error> {
        let Some(features) = &state.features else {
            return Ok(None);
        };

        let search = state.knob_search.as_ref();
        let x_offset = search.map(|s| s.x_min).unwrap_or(0.0);
        let y_offset = search.map(|s| s.y_min).unwrap_or(0.0);

        // working is the knob area; draw features translated to local coords
        let mut canvas = working.try_clone()?;

        // Draw knobs in green with angle indicators in yellow
        let green = Scalar::new(0.0, 255.0, 0.0, 0.0);
        let yellow = Scalar::new(0.0, 255.0, 255.0, 0.0);

        for (i, knob) in features.knobs.iter().enumerate() {
            // Translate from warped-image coords back to local knob-area coords
            let cx = (knob.center_x - x_offset) as i32;
            let cy = (knob.center_y - y_offset) as i32;
            let r = knob.radius as i32;

            imgproc::circle(
                &mut canvas,
                Point::new(cx, cy),
                r,
                green,
                2,
                imgproc::LINE_8,
                0,
            )?;

            // Draw angle indicator
            if let Some(&angle) = features.off_angles.get(i) {
                let rad = angle.to_radians();
                let end_x = cx + (r as f64 * rad.cos()) as i32;
                let end_y = cy + (r as f64 * rad.sin()) as i32;
                imgproc::line(
                    &mut canvas,
                    Point::new(cx, cy),
                    Point::new(end_x, end_y),
                    yellow,
                    2,
                    imgproc::LINE_8,
                    0,
                )?;
            }
        }

        let n_knobs = features.knobs.len();
        let label = format!(
            "S8:FindFeatures_{}k_clk{:.0}",
            n_knobs, features.clock.center_x
        );
        let jpeg = encode_jpeg(&canvas, 80)?;
        Ok(Some((label, jpeg)))
    }
}

/// A single template match result.
#[derive(Clone)]
struct TemplateMatch {
    x: f64,
    y: f64,
    score: f64,
    angle: f64,
}

/// Knob match with center and angle.
struct KnobMatch {
    center_x: f64,
    center_y: f64,
    angle: f64,
}

/// Resize a grayscale template to a given scale factor.
fn resize_template(template: &Mat, scale: f64) -> Result<Mat, opencv::Error> {
    let mut resized = Mat::default();
    let new_w = (template.cols() as f64 * scale) as i32;
    let new_h = (template.rows() as f64 * scale) as i32;
    if new_w < 3 || new_h < 3 {
        return Ok(Mat::default());
    }
    imgproc::resize(
        template,
        &mut resized,
        Size::new(new_w, new_h),
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;
    Ok(resized)
}

/// Rotate a binary edge template by the given angle using INTER_NEAREST
/// to preserve clean binary edges (no interpolation blur).
fn rotate_edge_template(template: &Mat, angle_deg: f64) -> Result<Mat, opencv::Error> {
    let cx = template.cols() as f64 / 2.0;
    let cy = template.rows() as f64 / 2.0;
    let rot_mat =
        imgproc::get_rotation_matrix_2d(Point2f::new(cx as f32, cy as f32), -angle_deg, 1.0)?;

    let mut rotated = Mat::default();
    imgproc::warp_affine(
        template,
        &mut rotated,
        &rot_mat,
        Size::new(template.cols(), template.rows()),
        imgproc::INTER_NEAREST,
        BORDER_CONSTANT,
        Scalar::default(),
    )?;
    Ok(rotated)
}

/// Find peaks in a matchTemplate result map above the threshold.
/// Returns matches sorted by score descending.
/// Uses row-pointer access to avoid per-pixel bounds checks.
fn find_peaks(
    result: &Mat,
    threshold: f64,
    templ_w: i32,
    templ_h: i32,
    angle: f64,
) -> Result<Vec<TemplateMatch>, opencv::Error> {
    let rows = result.rows();
    let cols = result.cols() as usize;
    let half_w = templ_w as f64 / 2.0;
    let half_h = templ_h as f64 / 2.0;
    let threshold_f32 = threshold as f32;

    let mut matches = Vec::new();

    for y in 0..rows {
        let row = result.at_row::<f32>(y)?;
        for (x, &val) in row.iter().enumerate().take(cols) {
            if val >= threshold_f32 {
                matches.push(TemplateMatch {
                    x: x as f64 + half_w,
                    y: y as f64 + half_h,
                    score: val as f64,
                    angle,
                });
            }
        }
    }

    matches.sort_by(|a, b| {
        b.score.total_cmp(&a.score)
    });
    Ok(matches)
}

/// Non-maximum suppression: keep the highest-scoring matches, removing
/// any that are within `min_dist` of a higher-scoring match.
fn nms(matches: &mut Vec<TemplateMatch>, min_dist: f64) -> Vec<TemplateMatch> {
    matches.sort_by(|a, b| {
        b.score.total_cmp(&a.score)
    });

    let mut kept: Vec<TemplateMatch> = Vec::new();
    let min_dist_sq = min_dist * min_dist;

    for m in matches.iter() {
        let dominated = kept.iter().any(|k| {
            let dx = m.x - k.x;
            let dy = m.y - k.y;
            dx * dx + dy * dy < min_dist_sq
        });
        if !dominated {
            kept.push(m.clone());
        }
    }

    kept
}

/// From NMS'd knob matches, select the best Y-aligned subset of N.
fn select_y_aligned_knobs(
    matches: &[TemplateMatch],
    n: usize,
    knob_r: f64,
) -> Option<Vec<KnobMatch>> {
    if matches.len() < n {
        return None;
    }

    // Try each match's Y as a reference, collect the N closest
    let mut best: Option<(Vec<KnobMatch>, f64)> = None;

    let unique_ys: Vec<f64> = {
        let mut ys: Vec<f64> = matches.iter().map(|m| m.y).collect();
        ys.sort_by(|a, b| a.total_cmp(b));
        ys.dedup_by(|a, b| (*a - *b).abs() < 3.0);
        ys
    };

    for &ref_y in &unique_ys {
        let mut by_y: Vec<&TemplateMatch> = matches
            .iter()
            .filter(|m| (m.y - ref_y).abs() < knob_r * 3.0)
            .collect();

        if by_y.len() < n {
            continue;
        }

        // Sort by X
        by_y.sort_by(|a, b| a.x.total_cmp(&b.x));

        // Take the best N by score among those that are Y-close
        let mut by_score: Vec<&TemplateMatch> = by_y.clone();
        by_score.sort_by(|a, b| {
            b.score.total_cmp(&a.score)
        });
        by_score.truncate(n);

        // Re-sort by X
        by_score.sort_by(|a, b| a.x.total_cmp(&b.x));

        // Check monotonic X with min gap
        let mut ok = true;
        for i in 1..by_score.len() {
            if by_score[i].x - by_score[i - 1].x < 5.0 {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }

        // Score: Y-variance (lower is better)
        let mean_y: f64 = by_score.iter().map(|m| m.y).sum::<f64>() / n as f64;
        let y_var: f64 = by_score.iter().map(|m| (m.y - mean_y).powi(2)).sum::<f64>() / n as f64;
        let y_score = 1.0 / (1.0 + y_var);

        let is_better = best.as_ref().is_none_or(|(_, s)| y_score > *s);
        if is_better {
            let knobs = by_score
                .iter()
                .map(|m| KnobMatch {
                    center_x: m.x,
                    center_y: m.y,
                    angle: m.angle,
                })
                .collect();
            best = Some((knobs, y_score));
        }
    }

    best.map(|(k, _)| k)
}
