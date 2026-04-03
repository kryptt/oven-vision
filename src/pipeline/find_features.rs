use opencv::core::{
    BORDER_CONSTANT, Mat, MatTraitConst, Point, Point2f, Rect, Scalar, Size, Vector,
};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;

use super::perspective::transform_to_mat;
use super::stage::{CircleFeature, DetectedFeatures, PipelineState, StageId, StageOutcome};
use super::{DebugImage, Stage};
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

/// Path to template images (bundled in container at build time).
const KNOB_TEMPLATE_PATH: &str = "/templates/knob.jpg";
const CLOCK_TEMPLATE_PATH: &str = "/templates/clock.jpg";

/// Stage 4: Detect features using multi-scale, multi-rotation template matching.
///
/// Uses reference photos of the actual knob and clock to find their positions
/// in the perspective-corrected image. Each knob match also yields the handle
/// angle directly (the rotation that scored highest).
pub struct FindFeatures {
    knob_template: Mat,
    clock_template: Mat,
}

impl FindFeatures {
    pub fn new() -> Self {
        let knob_template = imgcodecs::imread(KNOB_TEMPLATE_PATH, imgcodecs::IMREAD_GRAYSCALE)
            .unwrap_or_else(|e| {
                tracing::warn!(%e, path = KNOB_TEMPLATE_PATH, "failed to load knob template, using empty");
                Mat::default()
            });
        let clock_template = imgcodecs::imread(CLOCK_TEMPLATE_PATH, imgcodecs::IMREAD_GRAYSCALE)
            .unwrap_or_else(|e| {
                tracing::warn!(%e, path = CLOCK_TEMPLATE_PATH, "failed to load clock template, using empty");
                Mat::default()
            });
        Self {
            knob_template,
            clock_template,
        }
    }
}

impl Stage for FindFeatures {
    fn id(&self) -> StageId {
        StageId::FindFeatures
    }

    fn run(
        &self,
        state: &mut PipelineState,
        frame: &Mat,
        iteration: u32,
    ) -> Result<StageOutcome, opencv::Error> {
        let (Some(crop), Some(persp)) = (&state.crop, &state.perspective) else {
            return Ok(StageOutcome::Exhausted(
                "missing crop or perspective from previous stages".into(),
            ));
        };

        if self.knob_template.empty() || self.clock_template.empty() {
            return Ok(StageOutcome::Exhausted("template images not loaded".into()));
        }

        // Warp the cropped frame
        let roi_rect = Rect::new(
            crop.x as i32,
            crop.y as i32,
            crop.width as i32,
            crop.height as i32,
        );
        let cropped = Mat::roi(frame, roi_rect)?;
        let mat = transform_to_mat(&persp.matrix)?;

        let mut warped = Mat::default();
        imgproc::warp_perspective_def(
            &cropped,
            &mut warped,
            &mat,
            Size::new(persp.output_width as i32, persp.output_height as i32),
        )?;

        // Compute the Y-band where knobs should be.
        // The warped image height is top_pad + inter_line + bot_pad.
        // The trim lines are at Y=top_pad and Y=top_pad+inter_line.
        // From the Boretti reference, knobs are at ~15-35% of image height
        // (between the bottom trim and partway to the oven doors).
        let img_h = persp.output_height as f64;
        let knob_y_min = img_h * 0.10;
        let knob_y_max = img_h * 0.45;

        // Convert to grayscale
        let mut gray = Mat::default();
        imgproc::cvt_color_def(&warped, &mut gray, imgproc::COLOR_BGR2GRAY)?;

        // Enhance contrast
        let mut enhanced = Mat::default();
        let mut clahe = imgproc::create_clahe(2.0, Size::new(8, 8))?;
        clahe.apply(&gray, &mut enhanced)?;

        // Pick scale based on iteration (cycle through scales, then repeat with lower threshold)
        let scale_idx = (iteration as usize) % SCALE_FACTORS.len();
        let scale = SCALE_FACTORS[scale_idx];
        let threshold_adj = (iteration as usize / SCALE_FACTORS.len()) as f64 * 0.05;
        let threshold = (MATCH_THRESHOLD - threshold_adj).max(0.15);

        // --- Find knobs: multi-rotation template matching ---
        let scaled_knob = resize_template(&self.knob_template, scale)?;
        if scaled_knob.cols() < 5 || scaled_knob.rows() < 5 {
            return Ok(StageOutcome::Retry(format!(
                "knob template too small at scale {scale:.2}"
            )));
        }

        // Enhance template contrast to match
        let mut knob_enhanced = Mat::default();
        clahe.apply(&scaled_knob, &mut knob_enhanced)?;

        let mut all_knob_matches: Vec<TemplateMatch> = Vec::new();

        let n_rotations = (360.0 / ROTATION_STEP_DEG) as i32;
        for rot_idx in 0..n_rotations {
            let angle = rot_idx as f64 * ROTATION_STEP_DEG;
            let rotated = rotate_template(&knob_enhanced, angle)?;
            if rotated.empty() {
                continue;
            }

            // Skip if template is larger than image
            if rotated.cols() >= enhanced.cols() || rotated.rows() >= enhanced.rows() {
                continue;
            }

            let mut result = Mat::default();
            imgproc::match_template(
                &enhanced,
                &rotated,
                &mut result,
                imgproc::TM_CCOEFF_NORMED,
                &Mat::default(),
            )?;

            // Find peaks above threshold
            let matches = find_peaks(&result, threshold, rotated.cols(), rotated.rows(), angle)?;
            all_knob_matches.extend(matches);
        }

        // Filter to Y-band where knobs should be
        all_knob_matches.retain(|m| m.y >= knob_y_min && m.y <= knob_y_max);

        if all_knob_matches.len() < EXPECTED_KNOBS {
            return Ok(StageOutcome::Retry(format!(
                "found {} knob matches in Y-band [{knob_y_min:.0}..{knob_y_max:.0}] (need {}), scale={scale:.2}, threshold={threshold:.2}",
                all_knob_matches.len(),
                EXPECTED_KNOBS
            )));
        }

        // NMS: keep best non-overlapping matches
        let knob_results = nms(&mut all_knob_matches, NMS_MIN_DIST);

        if knob_results.len() < EXPECTED_KNOBS {
            return Ok(StageOutcome::Retry(format!(
                "only {} knob matches after NMS (need {}), scale={scale:.2}",
                knob_results.len(),
                EXPECTED_KNOBS
            )));
        }

        // --- Find clock: scale-only matching (no rotation) ---
        let scaled_clock = resize_template(&self.clock_template, scale)?;
        let mut clock_match: Option<TemplateMatch> = None;

        if scaled_clock.cols() >= 5
            && scaled_clock.rows() >= 5
            && scaled_clock.cols() < enhanced.cols()
            && scaled_clock.rows() < enhanced.rows()
        {
            let mut clock_enhanced = Mat::default();
            clahe.apply(&scaled_clock, &mut clock_enhanced)?;

            let mut result = Mat::default();
            imgproc::match_template(
                &enhanced,
                &clock_enhanced,
                &mut result,
                imgproc::TM_CCOEFF_NORMED,
                &Mat::default(),
            )?;

            let mut peaks = find_peaks(
                &result,
                threshold,
                scaled_clock.cols(),
                scaled_clock.rows(),
                0.0,
            )?;
            // Clock should also be in the knob Y-band
            peaks.retain(|m| m.y >= knob_y_min && m.y <= knob_y_max);
            if !peaks.is_empty() {
                clock_match = Some(peaks[0].clone());
            }
        }

        // --- Select the best 10+1 pattern from matches ---
        let knob_r = scaled_knob.cols() as f64 / 2.0;
        let clock_r = scaled_clock.cols().max(scaled_clock.rows()) as f64 / 2.0;

        // Sort knob matches by score descending
        let mut sorted_knobs = knob_results;
        sorted_knobs.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select top matches that are Y-aligned (best 10 by Y-consistency)
        let selected = select_y_aligned_knobs(&sorted_knobs, EXPECTED_KNOBS, knob_r);

        let Some(knobs) = selected else {
            return Ok(StageOutcome::Retry(format!(
                "could not find {EXPECTED_KNOBS} Y-aligned knob matches, scale={scale:.2}"
            )));
        };

        // Determine clock
        let knob_median_y = {
            let mut ys: Vec<f64> = knobs.iter().map(|k| k.center_y).collect();
            ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            ys[ys.len() / 2]
        };
        let leftmost_knob_x = knobs
            .iter()
            .map(|k| k.center_x)
            .fold(f64::INFINITY, f64::min);

        let clock = if let Some(cm) = clock_match {
            // Validate clock is left of knobs and near the same Y
            if cm.x < leftmost_knob_x && (cm.y - knob_median_y).abs() < knob_r * 4.0 {
                CircleFeature {
                    center_x: cm.x,
                    center_y: cm.y,
                    radius: clock_r,
                }
            } else {
                // Clock match in wrong position — synthesize from knob row
                CircleFeature {
                    center_x: leftmost_knob_x - knob_r * 4.0,
                    center_y: knob_median_y,
                    radius: clock_r,
                }
            }
        } else {
            // No clock match — estimate position from knob row
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
                    center_x: k.center_x,
                    center_y: k.center_y,
                    radius: knob_r,
                })
                .collect(),
            off_angles,
        });

        Ok(StageOutcome::Success)
    }

    fn max_retries(&self) -> u32 {
        30
    }

    fn debug_image(
        &self,
        state: &PipelineState,
        frame: &Mat,
    ) -> Result<Option<DebugImage>, opencv::Error> {
        let (Some(crop), Some(persp), Some(features)) =
            (&state.crop, &state.perspective, &state.features)
        else {
            return Ok(None);
        };

        let roi_rect = Rect::new(
            crop.x as i32,
            crop.y as i32,
            crop.width as i32,
            crop.height as i32,
        );
        let cropped = Mat::roi(frame, roi_rect)?;
        let mat = transform_to_mat(&persp.matrix)?;

        let mut canvas = Mat::default();
        imgproc::warp_perspective_def(
            &cropped,
            &mut canvas,
            &mat,
            Size::new(persp.output_width as i32, persp.output_height as i32),
        )?;

        // Draw clock in cyan
        let cyan = Scalar::new(255.0, 255.0, 0.0, 0.0);
        imgproc::circle(
            &mut canvas,
            Point::new(
                features.clock.center_x as i32,
                features.clock.center_y as i32,
            ),
            features.clock.radius as i32,
            cyan,
            2,
            imgproc::LINE_8,
            0,
        )?;

        // Draw knobs in green with angle indicators in yellow
        let green = Scalar::new(0.0, 255.0, 0.0, 0.0);
        let yellow = Scalar::new(0.0, 255.0, 255.0, 0.0);

        for (i, knob) in features.knobs.iter().enumerate() {
            let cx = knob.center_x as i32;
            let cy = knob.center_y as i32;
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
            "S4:FindFeatures_{}k_clk{:.0}",
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

/// Rotate a grayscale template by the given angle in degrees.
fn rotate_template(template: &Mat, angle_deg: f64) -> Result<Mat, opencv::Error> {
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
        imgproc::INTER_LINEAR,
        BORDER_CONSTANT,
        Scalar::default(),
    )?;
    Ok(rotated)
}

/// Find peaks in a matchTemplate result map above the threshold.
/// Returns matches sorted by score descending.
fn find_peaks(
    result: &Mat,
    threshold: f64,
    templ_w: i32,
    templ_h: i32,
    angle: f64,
) -> Result<Vec<TemplateMatch>, opencv::Error> {
    let rows = result.rows();
    let cols = result.cols();
    let half_w = templ_w as f64 / 2.0;
    let half_h = templ_h as f64 / 2.0;

    let mut matches = Vec::new();

    for y in 0..rows {
        for x in 0..cols {
            let val = *result.at_2d::<f32>(y, x)? as f64;
            if val >= threshold {
                matches.push(TemplateMatch {
                    x: x as f64 + half_w,
                    y: y as f64 + half_h,
                    score: val,
                    angle,
                });
            }
        }
    }

    matches.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(matches)
}

/// Non-maximum suppression: keep the highest-scoring matches, removing
/// any that are within `min_dist` of a higher-scoring match.
fn nms(matches: &mut Vec<TemplateMatch>, min_dist: f64) -> Vec<TemplateMatch> {
    matches.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
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
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
        by_y.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

        // Take the best N by score among those that are Y-close
        let mut by_score: Vec<&TemplateMatch> = by_y.clone();
        by_score.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        by_score.truncate(n);

        // Re-sort by X
        by_score.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

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
