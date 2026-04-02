use opencv::core::{Mat, Point, Rect, Scalar, Size, Vec3f, Vector};
use opencv::imgproc;
use opencv::prelude::*;

use super::perspective::transform_to_mat;
use super::stage::{CircleFeature, DetectedFeatures, PipelineState, StageId, StageOutcome};
use super::{DebugImage, Stage};
use crate::annotate::encode_jpeg;
use crate::detect::radial_edge_scan;

/// Expected knob count (excluding the clock).
const EXPECTED_KNOBS: usize = 10;

/// Minimum radius threshold — circles smaller than this are noise.
const MIN_RADIUS: i32 = 5;

/// Maximum radius for any feature (knobs and clock).
const MAX_RADIUS: i32 = 50;

/// Raise the max-circles cap: we over-detect on purpose and filter by model.
const MAX_CIRCLES: usize = 60;

/// --- Proportion constraints from the Boretti Toscana reference image ---

/// The clock radius is ~1.5× the median knob radius. Accept 1.2-2.5×.
const CLOCK_RADIUS_RATIO_MIN: f64 = 1.2;
const CLOCK_RADIUS_RATIO_MAX: f64 = 2.5;

/// The clock center Y must be within this many knob-radii of the knob row
/// median Y (the clock is in the same row as the knobs).
const CLOCK_Y_TOLERANCE_RADII: f64 = 3.0;

/// Maximum coefficient of variation of inter-knob X spacing. The reference
/// image shows very even spacing (CV < 0.15). Reject sets with CV > this.
const MAX_SPACING_CV: f64 = 0.5;

/// Maximum coefficient of variation of knob radii. The reference shows
/// uniform knob sizes (CV < 0.1). Reject sets with CV > this.
const MAX_RADIUS_CV: f64 = 0.4;

/// Stage 4: Detect circular features (clock + 10 knobs) in the
/// perspective-corrected image.
///
/// Strategy: over-detect with loose HoughCircles, then use model-based
/// consensus to find the best 10+1 subset matching the geometric pattern
/// (10 Y-aligned, similarly-sized knobs + 1 larger clock to their left).
pub struct FindFeatures;

impl FindFeatures {
    pub fn new() -> Self {
        Self
    }

    fn params_for_iteration(iteration: u32) -> (f64, f64, f64) {
        let param1 = 100.0;
        // Start loose, get even looser — we want many candidates
        let param2 = (40.0 - iteration as f64 * 1.5).max(8.0);
        let min_dist = (25.0 - iteration as f64 * 0.5).max(8.0);
        (param1, param2, min_dist)
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

        // Grayscale + CLAHE + blur
        let mut gray = Mat::default();
        imgproc::cvt_color_def(&warped, &mut gray, imgproc::COLOR_BGR2GRAY)?;

        let mut enhanced = Mat::default();
        let mut clahe = imgproc::create_clahe(2.0, Size::new(8, 8))?;
        clahe.apply(&gray, &mut enhanced)?;

        let mut blurred = Mat::default();
        imgproc::median_blur(&enhanced, &mut blurred, 5)?;

        // Over-detect: loose HoughCircles params
        let (param1, param2, min_dist) = Self::params_for_iteration(iteration);

        let mut circles = Vector::<Vec3f>::new();
        imgproc::hough_circles(
            &blurred,
            &mut circles,
            imgproc::HOUGH_GRADIENT,
            1.0,
            min_dist,
            param1,
            param2,
            MIN_RADIUS,
            MAX_RADIUS,
        )?;

        let count = circles.len();
        if count < EXPECTED_KNOBS + 1 {
            return Ok(StageOutcome::Retry(format!(
                "found {count} circles (need {}+), param2={param2:.0}",
                EXPECTED_KNOBS + 1
            )));
        }
        if count > MAX_CIRCLES {
            return Ok(StageOutcome::Retry(format!(
                "found {count} circles (max {MAX_CIRCLES}), param2={param2:.0} too loose"
            )));
        }

        // Convert all candidates
        let candidates: Vec<CircleFeature> = circles
            .iter()
            .map(|c| CircleFeature {
                center_x: c[0] as f64,
                center_y: c[1] as f64,
                radius: c[2] as f64,
            })
            .collect();

        // Model-based selection: find the best 10+1 subset
        match select_best_pattern(&candidates) {
            Some((clock, knobs, _score)) => {
                state.features = Some(DetectedFeatures {
                    clock,
                    knobs,
                    off_angles: vec![0.0; EXPECTED_KNOBS],
                });
                Ok(StageOutcome::Success)
            }
            None => Ok(StageOutcome::Retry(format!(
                "no valid 10+1 pattern in {count} candidates, param2={param2:.0}"
            ))),
        }
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

        // Also re-run HoughCircles to show ALL candidates in red (dim)
        {
            let mut gray = Mat::default();
            imgproc::cvt_color_def(&canvas, &mut gray, imgproc::COLOR_BGR2GRAY)?;
            let mut enhanced = Mat::default();
            let mut clahe = imgproc::create_clahe(2.0, Size::new(8, 8))?;
            clahe.apply(&gray, &mut enhanced)?;
            let mut blurred = Mat::default();
            imgproc::median_blur(&enhanced, &mut blurred, 5)?;

            let mut circles = Vector::<Vec3f>::new();
            // Use loose params to show all candidates
            let _ = imgproc::hough_circles(
                &blurred,
                &mut circles,
                imgproc::HOUGH_GRADIENT,
                1.0,
                8.0,
                100.0,
                15.0,
                MIN_RADIUS,
                MAX_RADIUS,
            );

            // Draw all candidates in dim red
            let dim_red = Scalar::new(0.0, 0.0, 120.0, 0.0);
            for c in &circles {
                imgproc::circle(
                    &mut canvas,
                    Point::new(c[0] as i32, c[1] as i32),
                    c[2] as i32,
                    dim_red,
                    1,
                    imgproc::LINE_8,
                    0,
                )?;
            }
        }

        // Draw selected clock in cyan (thick)
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

        // Draw selected knobs in bright green (thick) with indicator angles
        let green = Scalar::new(0.0, 255.0, 0.0, 0.0);
        let yellow = Scalar::new(0.0, 255.0, 255.0, 0.0);

        // Compute edges for radial scan
        let mut gray = Mat::default();
        imgproc::cvt_color_def(&canvas, &mut gray, imgproc::COLOR_BGR2GRAY)?;
        let mut edges = Mat::default();
        imgproc::canny(&gray, &mut edges, 10.0, 30.0, 3, false)?;

        for knob in &features.knobs {
            let cx = knob.center_x as i32;
            let cy = knob.center_y as i32;
            let r = knob.radius as i32;

            // Draw circle
            imgproc::circle(
                &mut canvas,
                Point::new(cx, cy),
                r,
                green,
                2,
                imgproc::LINE_8,
                0,
            )?;

            // Detect indicator angle via radial edge scan
            if let Ok((angle_deg, strength)) =
                radial_edge_scan(&edges, cx as u32, cy as u32, r as u32)
            {
                if strength > 0.05 {
                    let rad = angle_deg.to_radians();
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

/// A scored 10+1 candidate pattern.
struct PatternCandidate {
    clock: CircleFeature,
    knobs: Vec<CircleFeature>,
    score: f64,
}

/// Find the best 10+1 pattern from a pool of circle candidates.
///
/// Algorithm:
/// 1. Compute the median radius of smaller circles (likely knobs).
/// 2. Partition into "knob-sized" and "clock-sized" buckets.
/// 3. Among knob-sized circles, find the best Y-aligned subset of 10
///    with consistent radius and monotonic X-spacing.
/// 4. Among clock-sized circles, find the one left of the knob row.
/// 5. Score and return the best configuration.
fn select_best_pattern(
    candidates: &[CircleFeature],
) -> Option<(CircleFeature, Vec<CircleFeature>, f64)> {
    if candidates.len() < EXPECTED_KNOBS + 1 {
        return None;
    }

    // Sort all by radius to find the knob-size baseline.
    // The median of the smaller 80% is a robust knob-radius estimate
    // (ignoring the few large clock candidates).
    let mut radii: Vec<f64> = candidates.iter().map(|c| c.radius).collect();
    radii.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p80_idx = (radii.len() as f64 * 0.8) as usize;
    let knob_radius_est = radii[p80_idx.min(radii.len() - 1) / 2]; // median of lower 80%

    // Partition: knob-sized (within 3x of estimate) and clock-sized (larger)
    let radius_lo = knob_radius_est * 0.4;
    let radius_hi = knob_radius_est * 2.5;
    let clock_min = knob_radius_est * CLOCK_RADIUS_RATIO_MIN;

    let knob_pool: Vec<&CircleFeature> = candidates
        .iter()
        .filter(|c| c.radius >= radius_lo && c.radius <= radius_hi)
        .collect();

    let clock_pool: Vec<&CircleFeature> = candidates
        .iter()
        .filter(|c| c.radius >= clock_min)
        .collect();

    if knob_pool.len() < EXPECTED_KNOBS || clock_pool.is_empty() {
        return None;
    }

    // Find the best Y-aligned subset of 10 knobs.
    // Strategy: try each possible "reference Y" (each candidate's Y),
    // collect the 10 closest circles to that Y, score the set.
    let mut best: Option<PatternCandidate> = None;

    // Collect unique Y values to try as reference (dedup to within 3px)
    let mut ref_ys: Vec<f64> = knob_pool.iter().map(|c| c.center_y).collect();
    ref_ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ref_ys.dedup_by(|a, b| (*a - *b).abs() < 3.0);

    for &ref_y in &ref_ys {
        // Sort knob-pool by distance from ref_y
        let mut by_y_dist: Vec<(&CircleFeature, f64)> = knob_pool
            .iter()
            .map(|c| (*c, (c.center_y - ref_y).abs()))
            .collect();
        by_y_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Take the closest candidates (up to 15) and find the best 10 among them
        let close: Vec<&CircleFeature> = by_y_dist
            .iter()
            .take(15)
            .map(|(c, _)| *c)
            .collect();

        if close.len() < EXPECTED_KNOBS {
            continue;
        }

        // Sort by X for monotonic-spacing check
        let mut sorted_x: Vec<&CircleFeature> = close.clone();
        sorted_x.sort_by(|a, b| a.center_x.partial_cmp(&b.center_x).unwrap());

        // Greedy: pick 10 with best radius consistency from the X-sorted list
        // Remove outliers by radius first
        let median_r = {
            let mut rs: Vec<f64> = sorted_x.iter().map(|c| c.radius).collect();
            rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            rs[rs.len() / 2]
        };

        let radius_filtered: Vec<&CircleFeature> = sorted_x
            .into_iter()
            .filter(|c| {
                c.radius >= median_r * 0.5 && c.radius <= median_r * 2.0
            })
            .collect();

        if radius_filtered.len() < EXPECTED_KNOBS {
            continue;
        }

        // If more than 10, try all combinations of dropping extras.
        // For efficiency, just take the 10 most Y-aligned.
        let mut knobs: Vec<CircleFeature> = radius_filtered
            .iter()
            .map(|c| (*c).clone())
            .collect();
        if knobs.len() > EXPECTED_KNOBS {
            knobs.sort_by(|a, b| {
                let da = (a.center_y - ref_y).abs();
                let db = (b.center_y - ref_y).abs();
                da.partial_cmp(&db).unwrap()
            });
            knobs.truncate(EXPECTED_KNOBS);
        }

        // Re-sort by X
        knobs.sort_by(|a, b| a.center_x.partial_cmp(&b.center_x).unwrap());

        // --- Hard rejection based on reference proportions ---

        // Reject if knob radii are too inconsistent
        let mean_r: f64 = knobs.iter().map(|k| k.radius).sum::<f64>() / knobs.len() as f64;
        let var_r: f64 = knobs.iter().map(|k| (k.radius - mean_r).powi(2)).sum::<f64>()
            / knobs.len() as f64;
        let cv_r = var_r.sqrt() / mean_r.max(1.0);
        if cv_r > MAX_RADIUS_CV {
            continue;
        }

        // Reject if X spacing is too irregular
        let mut gaps: Vec<f64> = Vec::new();
        for i in 1..knobs.len() {
            gaps.push(knobs[i].center_x - knobs[i - 1].center_x);
        }
        let mean_gap: f64 = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let var_gap: f64 =
            gaps.iter().map(|g| (g - mean_gap).powi(2)).sum::<f64>() / gaps.len() as f64;
        let cv_gap = var_gap.sqrt() / mean_gap.max(1.0);
        if cv_gap > MAX_SPACING_CV {
            continue;
        }

        // Reject if any gap is negative or very small (non-monotonic)
        let min_gap = gaps.iter().cloned().fold(f64::INFINITY, f64::min);
        if min_gap < 5.0 {
            continue;
        }

        let knob_score = score_knob_set(&knobs);

        // --- Clock selection with reference-based constraints ---
        let leftmost_knob_x = knobs[0].center_x;
        let knob_median_y = {
            let mut ys: Vec<f64> = knobs.iter().map(|k| k.center_y).collect();
            ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ys[ys.len() / 2]
        };

        // Clock must be: left of knob row, correct radius ratio, same Y row
        let y_tol = mean_r * CLOCK_Y_TOLERANCE_RADII;
        let mut clock_candidates: Vec<(&CircleFeature, f64)> = clock_pool
            .iter()
            .filter(|c| {
                // Must be left of the knob row
                c.center_x < leftmost_knob_x
                // Radius must be in the expected range relative to knobs
                && c.radius >= mean_r * CLOCK_RADIUS_RATIO_MIN
                && c.radius <= mean_r * CLOCK_RADIUS_RATIO_MAX
                // Must be at the same Y height as the knob row
                && (c.center_y - knob_median_y).abs() <= y_tol
            })
            .map(|c| {
                let r_ratio = c.radius / mean_r;
                // Ideal ratio is ~1.9 from the Boretti reference photo
                let r_score = 1.0 - ((r_ratio - 1.9).abs() / 1.0).min(1.0);
                let y_dist = (c.center_y - knob_median_y).abs();
                let y_score = 1.0 - (y_dist / y_tol).min(1.0);
                (*c, r_score * 0.5 + y_score * 0.5)
            })
            .collect();

        if clock_candidates.is_empty() {
            continue;
        }

        clock_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let clock = clock_candidates[0].0.clone();
        let clock_score = clock_candidates[0].1;

        let total_score = knob_score * 0.8 + clock_score * 0.2;

        let dominated = best.as_ref().is_some_and(|b| b.score >= total_score);
        if !dominated {
            best = Some(PatternCandidate {
                clock,
                knobs,
                score: total_score,
            });
        }
    }

    best.map(|b| (b.clock, b.knobs, b.score))
}

/// Score a set of 10 knobs. Higher is better.
///
/// Based on the Boretti Toscana reference:
///   - Y alignment is the dominant signal (knobs share an exact Y)
///   - Radius consistency is strong (identical hardware)
///   - X spacing has 3 groups so we only check monotonic + min gap, not uniformity
fn score_knob_set(knobs: &[CircleFeature]) -> f64 {
    if knobs.len() < 2 {
        return 0.0;
    }

    // Y alignment: normalised inverse of max Y-deviation from median.
    // This is THE strongest constraint — knobs are manufactured at the same height.
    let mut ys: Vec<f64> = knobs.iter().map(|k| k.center_y).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_y = ys[ys.len() / 2];
    let max_y_dev = ys.iter().map(|y| (y - median_y).abs()).fold(0.0f64, f64::max);
    let y_score = 1.0 - (max_y_dev / 20.0).min(1.0); // 20px = worst acceptable

    // Radius consistency: inverse of coefficient of variation.
    // All 10 knobs are identical hardware — radii must be very consistent.
    let mean_r: f64 = knobs.iter().map(|k| k.radius).sum::<f64>() / knobs.len() as f64;
    let var_r: f64 = knobs.iter().map(|k| (k.radius - mean_r).powi(2)).sum::<f64>()
        / knobs.len() as f64;
    let cv_r = var_r.sqrt() / mean_r.max(1.0);
    let r_score = 1.0 - cv_r.min(1.0);

    // X monotonicity: just check all gaps are positive and above minimum.
    // Don't penalise uneven spacing — the Boretti has 3 control groups with
    // wider gaps between groups (~1.5× the within-group spacing).
    let mut gaps: Vec<f64> = Vec::new();
    for i in 1..knobs.len() {
        gaps.push(knobs[i].center_x - knobs[i - 1].center_x);
    }
    let min_gap = gaps.iter().cloned().fold(f64::INFINITY, f64::min);
    if min_gap < 5.0 {
        return 0.0; // non-monotonic or overlapping → reject
    }

    // Mild bonus for reasonable total span (knobs should cover a wide area)
    let span = knobs.last().unwrap().center_x - knobs.first().unwrap().center_x;
    let span_score = (span / 200.0).min(1.0); // normalise: 200px+ = full score

    // Weights: Y alignment dominates, radius consistency secondary, span tertiary
    0.55 * y_score + 0.35 * r_score + 0.10 * span_score
}
