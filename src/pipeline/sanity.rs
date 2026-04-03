use opencv::core::{Mat, Size, Vec3f, Vector};
use opencv::imgproc;
use opencv::prelude::*;

use super::stage::{PipelineState, StageDescriptor, StageOutcome};
use super::{ImageOutput, Stage};

/// Maximum allowed Y-deviation (pixels) between any knob center and the
/// median knob Y. Strict — knobs are at the same manufactured height.
const Y_TOLERANCE_PX: f64 = 8.0;

/// Minimum X-gap (pixels) between consecutive knob centers. Knobs closer
/// than this are likely duplicates or false positives.
const MIN_X_GAP_PX: f64 = 15.0;

/// Maximum coefficient of variation of inter-knob X-gaps. The Boretti has
/// 3 groups with wider gaps between groups (~1.5× within-group), so the CV
/// should be moderate but not extreme. Reject if gaps are wildly uneven.
const MAX_GAP_CV: f64 = 0.45;

/// Maximum allowed radius deviation: any knob radius must be within this
/// factor of the median knob radius (e.g., 1.5 = between 0.67x and 1.5x median).
const MAX_RADIUS_FACTOR: f64 = 1.5;

/// Expected knob count (excluding the clock).
const EXPECTED_KNOBS: usize = 10;

/// Stage 9: Validate the detected features are geometrically consistent.
///
/// Checks:
/// 1. Exactly 10 knobs detected.
/// 2. All knob centers are Y-aligned within +/-Y_TOLERANCE_PX of the median.
/// 3. Knob X-positions are monotonically increasing with minimum gap.
/// 4. All knob radii are consistent (within MAX_RADIUS_FACTOR of median).
/// 5. Clock is to the left of all knobs.
///
/// Failure signals the pipeline to fall back to FindVerticals to try
/// different parameters.
pub struct SanityCheck;

impl SanityCheck {
    pub fn new() -> Self {
        Self
    }
}

pub(crate) const DESCRIPTOR: StageDescriptor = StageDescriptor {
    name: "SanityCheck",
    label: "S9:SanityCheck",
    fallback: Some("FindFeatures"),
};

impl Stage for SanityCheck {
    fn descriptor(&self) -> StageDescriptor {
        DESCRIPTOR
    }

    fn run(
        &self,
        state: &mut PipelineState,
        _src: &Mat,
        _dst: &mut Mat,
        _raw: &Mat,
        _iteration: u32,
    ) -> Result<(StageOutcome, ImageOutput), opencv::Error> {
        let Some(features) = &state.features else {
            return Ok((
                StageOutcome::Exhausted("no features from Stage 8".into()),
                ImageOutput::Passthrough,
            ));
        };

        // Check 1: knob count
        if features.knobs.len() != EXPECTED_KNOBS {
            return Ok((
                StageOutcome::Exhausted(format!(
                    "expected {EXPECTED_KNOBS} knobs, got {}",
                    features.knobs.len()
                )),
                ImageOutput::Passthrough,
            ));
        }

        // Check 2: Y alignment -- all knob centers within tolerance of median Y
        let mut ys: Vec<f64> = features.knobs.iter().map(|k| k.center_y).collect();
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_y = ys[ys.len() / 2];

        for (i, knob) in features.knobs.iter().enumerate() {
            let dev = (knob.center_y - median_y).abs();
            if dev > Y_TOLERANCE_PX {
                return Ok((
                    StageOutcome::Exhausted(format!(
                        "knob {i} Y-deviation = {dev:.1}px (max {Y_TOLERANCE_PX}px)"
                    )),
                    ImageOutput::Passthrough,
                ));
            }
        }

        // Check 3: X monotonically increasing with minimum gap
        for i in 1..features.knobs.len() {
            let gap = features.knobs[i].center_x - features.knobs[i - 1].center_x;
            if gap < MIN_X_GAP_PX {
                return Ok((
                    StageOutcome::Exhausted(format!(
                        "knobs {}/{} X-gap = {gap:.1}px (min {MIN_X_GAP_PX}px)",
                        i - 1,
                        i
                    )),
                    ImageOutput::Passthrough,
                ));
            }
        }

        // Check 3b: no overlapping knobs — distance between any two knob
        // centers must be greater than the sum of their radii.
        for i in 0..features.knobs.len() {
            for j in (i + 1)..features.knobs.len() {
                let dx = features.knobs[j].center_x - features.knobs[i].center_x;
                let dy = features.knobs[j].center_y - features.knobs[i].center_y;
                let dist = (dx * dx + dy * dy).sqrt();
                let min_dist = features.knobs[i].radius + features.knobs[j].radius;
                if dist < min_dist {
                    return Ok((
                        StageOutcome::Exhausted(format!(
                            "knobs {i}/{j} overlap: dist={dist:.1}px < radii_sum={min_dist:.1}px"
                        )),
                        ImageOutput::Passthrough,
                    ));
                }
            }
        }

        // Check 3c: X-gap regularity — spacing should be approximately even.
        // The Boretti has 3 control groups with wider gaps between them, so
        // we allow moderate variation but reject wildly uneven spacing.
        {
            let mut gaps: Vec<f64> = Vec::new();
            for i in 1..features.knobs.len() {
                gaps.push(features.knobs[i].center_x - features.knobs[i - 1].center_x);
            }
            let mean_gap: f64 = gaps.iter().sum::<f64>() / gaps.len() as f64;
            let var_gap: f64 =
                gaps.iter().map(|g| (g - mean_gap).powi(2)).sum::<f64>() / gaps.len() as f64;
            let cv_gap = var_gap.sqrt() / mean_gap.max(1.0);
            if cv_gap > MAX_GAP_CV {
                return Ok((
                    StageOutcome::Exhausted(format!(
                        "X-gap CV = {cv_gap:.2} (max {MAX_GAP_CV}), mean_gap={mean_gap:.1}px"
                    )),
                    ImageOutput::Passthrough,
                ));
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
                return Ok((
                    StageOutcome::Exhausted(format!(
                        "knob {i} radius = {:.1} outside [{r_lo:.1}, {r_hi:.1}] (median {median_r:.1})",
                        knob.radius
                    )),
                    ImageOutput::Passthrough,
                ));
            }
        }

        // Check 5: clock is to the left of all knobs
        if let Some(first_knob) = features.knobs.first() {
            if features.clock.center_x >= first_knob.center_x {
                return Ok((
                    StageOutcome::Exhausted(format!(
                        "clock x={:.1} is not left of first knob x={:.1}",
                        features.clock.center_x, first_knob.center_x
                    )),
                    ImageOutput::Passthrough,
                ));
            }
        }

        state.validated = true;
        Ok((StageOutcome::Success, ImageOutput::Passthrough))
    }

    fn max_retries(&self) -> u32 {
        // Sanity check is pass/fail -- no parameter variation.
        // Failure triggers fallback to FindVerticals.
        1
    }
}

/// Minimum circles for the quick check to pass. We allow a bit of slack
/// compared to the full calibration check (which requires exactly 11).
const QUICK_MIN_CIRCLES: usize = 9;

/// Maximum circles before we consider the image too noisy.
const QUICK_MAX_CIRCLES: usize = 16;

/// Y tolerance for the quick check -- slightly more lenient than calibration.
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
