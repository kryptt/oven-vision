use std::f64::consts::PI;

use opencv::core::{Mat, Rect, Size, Vec2f, Vec3f, Vector};
use opencv::imgproc;
use opencv::prelude::*;

use super::Stage;
use super::perspective::transform_to_mat;
use super::stage::{PipelineState, StageId, StageOutcome};

/// Maximum allowed Y-deviation (pixels) between any knob center and the
/// median knob Y. If any knob exceeds this, perspective correction is off.
const Y_TOLERANCE_PX: f64 = 25.0;

/// Minimum X-gap (pixels) between consecutive knob centers. Knobs closer
/// than this are likely duplicates or false positives.
const MIN_X_GAP_PX: f64 = 10.0;

/// Maximum allowed radius deviation: any knob radius must be within this
/// factor of the median knob radius (e.g., 2.0 = between 0.5× and 2× median).
const MAX_RADIUS_FACTOR: f64 = 2.0;

/// Expected knob count (excluding the clock).
const EXPECTED_KNOBS: usize = 10;

/// Stage 5: Validate the detected features are geometrically consistent.
///
/// Checks:
/// 1. Exactly 10 knobs detected.
/// 2. All knob centers are Y-aligned within ±Y_TOLERANCE_PX of the median.
/// 3. Knob X-positions are monotonically increasing with minimum gap.
/// 4. All knob radii are consistent (within MAX_RADIUS_FACTOR of median).
/// 5. Clock is to the left of all knobs.
///
/// Failure signals the pipeline to fall back to Stage 4 (FindFeatures) to
/// try different HoughCircles parameters.
pub struct SanityCheck;

impl SanityCheck {
    pub fn new() -> Self {
        Self
    }
}

impl Stage for SanityCheck {
    fn id(&self) -> StageId {
        StageId::SanityCheck
    }

    fn run(
        &self,
        state: &mut PipelineState,
        _frame: &Mat,
        _iteration: u32,
    ) -> Result<StageOutcome, opencv::Error> {
        let Some(features) = &state.features else {
            return Ok(StageOutcome::Exhausted("no features from Stage 4".into()));
        };

        // Check 1: knob count
        if features.knobs.len() != EXPECTED_KNOBS {
            return Ok(StageOutcome::Exhausted(format!(
                "expected {EXPECTED_KNOBS} knobs, got {}",
                features.knobs.len()
            )));
        }

        // Check 2: Y alignment — all knob centers within tolerance of median Y
        let mut ys: Vec<f64> = features.knobs.iter().map(|k| k.center_y).collect();
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_y = ys[ys.len() / 2];

        for (i, knob) in features.knobs.iter().enumerate() {
            let dev = (knob.center_y - median_y).abs();
            if dev > Y_TOLERANCE_PX {
                return Ok(StageOutcome::Exhausted(format!(
                    "knob {i} Y-deviation = {dev:.1}px (max {Y_TOLERANCE_PX}px)"
                )));
            }
        }

        // Check 3: X monotonically increasing with minimum gap
        for i in 1..features.knobs.len() {
            let gap = features.knobs[i].center_x - features.knobs[i - 1].center_x;
            if gap < MIN_X_GAP_PX {
                return Ok(StageOutcome::Exhausted(format!(
                    "knobs {}/{} X-gap = {gap:.1}px (min {MIN_X_GAP_PX}px)",
                    i - 1,
                    i
                )));
            }
        }

        // Check 4: radius consistency
        let mut radii: Vec<f64> = features.knobs.iter().map(|k| k.radius).collect();
        radii.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_r = radii[radii.len() / 2];
        let r_lo = median_r / MAX_RADIUS_FACTOR;
        let r_hi = median_r * MAX_RADIUS_FACTOR;

        for (i, knob) in features.knobs.iter().enumerate() {
            if knob.radius < r_lo || knob.radius > r_hi {
                return Ok(StageOutcome::Exhausted(format!(
                    "knob {i} radius = {:.1} outside [{r_lo:.1}, {r_hi:.1}] (median {median_r:.1})",
                    knob.radius
                )));
            }
        }

        // Check 5: clock is to the left of all knobs
        if let Some(first_knob) = features.knobs.first() {
            if features.clock.center_x >= first_knob.center_x {
                return Ok(StageOutcome::Exhausted(format!(
                    "clock x={:.1} is not left of first knob x={:.1}",
                    features.clock.center_x, first_knob.center_x
                )));
            }
        }

        // Check 6 (independent signal): verify that dominant lines in the
        // warped image are actually horizontal. This catches bad perspective
        // corrections that S4's HoughCircles might not notice.
        if let Err(reason) = verify_warp_horizontality(state, _frame) {
            return Ok(StageOutcome::Exhausted(format!(
                "warp horizontality check failed: {reason}"
            )));
        }

        state.validated = true;
        Ok(StageOutcome::Success)
    }

    fn max_retries(&self) -> u32 {
        // Sanity check is pass/fail — no parameter variation.
        // Failure triggers fallback to FindFeatures.
        1
    }
}

/// Maximum slope (degrees from horizontal) allowed for the strongest line
/// in the warped image. If the warp is correct, the chrome trim lines should
/// be nearly perfectly horizontal.
const MAX_WARP_SLOPE_DEG: f64 = 2.0;

/// Independent validation: run HoughLines on the warped image and verify
/// that the dominant lines are horizontal. This uses a different algorithm
/// path than Stages 2-4, breaking the circular validation chain.
fn verify_warp_horizontality(state: &PipelineState, frame: &Mat) -> Result<(), String> {
    let (Some(crop), Some(persp)) = (&state.crop, &state.perspective) else {
        return Err("missing crop or perspective state".into());
    };

    // Re-warp the frame
    let roi_rect = Rect::new(
        crop.x as i32,
        crop.y as i32,
        crop.width as i32,
        crop.height as i32,
    );
    let cropped = Mat::roi(frame, roi_rect).map_err(|e| format!("roi: {e}"))?;
    let mat = transform_to_mat(&persp.matrix).map_err(|e| format!("mat: {e}"))?;

    let mut warped = Mat::default();
    imgproc::warp_perspective_def(
        &cropped,
        &mut warped,
        &mat,
        Size::new(persp.output_width as i32, persp.output_height as i32),
    )
    .map_err(|e| format!("warp: {e}"))?;

    // Convert to grayscale + edge detection
    let mut gray = Mat::default();
    imgproc::cvt_color_def(&warped, &mut gray, imgproc::COLOR_BGR2GRAY)
        .map_err(|e| format!("gray: {e}"))?;

    let mut edges = Mat::default();
    imgproc::canny(&gray, &mut edges, 50.0, 150.0, 3, false).map_err(|e| format!("canny: {e}"))?;

    // Find lines in the warped image
    let mut lines = Vector::<Vec2f>::new();
    imgproc::hough_lines_def(&edges, &mut lines, 1.0, PI / 180.0, 100)
        .map_err(|e| format!("hough: {e}"))?;

    if lines.is_empty() {
        // No strong lines detected — can't validate, pass cautiously
        return Ok(());
    }

    // Check the strongest line (first returned by HoughLines) is horizontal.
    // In HoughLines, theta = PI/2 is horizontal.
    let strongest = lines.get(0).map_err(|e| format!("get: {e}"))?;
    let theta = strongest[1] as f64;
    let slope_from_horizontal = ((theta - PI / 2.0).abs() * 180.0 / PI).min(90.0);

    if slope_from_horizontal > MAX_WARP_SLOPE_DEG {
        return Err(format!(
            "strongest line slope = {slope_from_horizontal:.1}° (max {MAX_WARP_SLOPE_DEG}°)"
        ));
    }

    Ok(())
}

/// Minimum circles for the quick check to pass. We allow a bit of slack
/// compared to the full calibration check (which requires exactly 11).
const QUICK_MIN_CIRCLES: usize = 9;

/// Maximum circles before we consider the image too noisy.
const QUICK_MAX_CIRCLES: usize = 16;

/// Y tolerance for the quick check — slightly more lenient than calibration.
const QUICK_Y_TOLERANCE_PX: f64 = 10.0;

/// Minimum number of Y-aligned circles to pass.
const QUICK_MIN_ALIGNED: usize = 9;

/// Lightweight sanity check for detection mode.
///
/// Runs HoughCircles on the already-warped BGR image and verifies that we
/// still find roughly the right number of circles with Y-alignment. This is
/// much cheaper than a full recalibration and catches camera bumps or
/// obstructions.
///
/// Returns `Ok(true)` if the check passes, `Ok(false)` if the warp appears
/// invalid, or `Err` on OpenCV errors.
pub fn quick_sanity_check(warped: &Mat) -> Result<bool, opencv::Error> {
    let mut gray = Mat::default();
    imgproc::cvt_color_def(warped, &mut gray, imgproc::COLOR_BGR2GRAY)?;

    let mut enhanced = Mat::default();
    let mut clahe = imgproc::create_clahe(2.0, Size::new(8, 8))?;
    clahe.apply(&gray, &mut enhanced)?;

    let mut blurred = Mat::default();
    imgproc::median_blur(&enhanced, &mut blurred, 5)?;

    let mut circles = Vector::<Vec3f>::new();
    imgproc::hough_circles(
        &blurred,
        &mut circles,
        imgproc::HOUGH_GRADIENT,
        1.0,  // dp
        20.0, // min_dist
        100.0,
        30.0,
        5,  // min_radius
        50, // max_radius
    )?;

    let count = circles.len();
    if count < QUICK_MIN_CIRCLES || count > QUICK_MAX_CIRCLES {
        return Ok(false);
    }

    // Check Y alignment of the majority of circles
    let mut ys: Vec<f64> = circles.iter().map(|c| c[1] as f64).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_y = ys[ys.len() / 2];

    let aligned = ys
        .iter()
        .filter(|y| (*y - median_y).abs() <= QUICK_Y_TOLERANCE_PX)
        .count();

    Ok(aligned >= QUICK_MIN_ALIGNED)
}
