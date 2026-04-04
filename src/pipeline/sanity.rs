use opencv::core::Mat;
use opencv::imgproc;
use opencv::prelude::*;

use tracing::info;

use super::stage::{PipelineState, StageDescriptor, StageOutcome};
use super::util::{enhance_gray, validate_features, SanityThresholds};
use super::{ImageOutput, Stage};

/// Coarse-pass thresholds — just good enough to locate features for
/// the corner-based perspective refinement (S10-S11). The strict check
/// is in S13 (FinalCheck).
const THRESHOLDS: SanityThresholds = SanityThresholds {
    y_tolerance_px: 25.0,
    min_x_gap_px: 4.0,
    max_gap_cv: 0.65,
    max_radius_factor: 2.0,
    expected_knobs: 10,
    skip_overlap: true,
};

/// Stage 9: Validate the detected features are geometrically consistent.
///
/// Checks knob count, Y-alignment, X-ordering, overlap, gap regularity,
/// radius consistency, and clock position using coarse thresholds.
///
/// Failure signals the pipeline to fall back to FindFeatures to try
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

        match validate_features(features, &THRESHOLDS) {
            Ok(m) => {
                info!(
                    max_y_dev = format!("{:.1}", m.max_y_dev),
                    min_x_gap = format!("{:.1}", m.min_x_gap),
                    gap_cv = format!("{:.2}", m.gap_cv),
                    min_pair_dist = format!("{:.1}", m.min_pair_dist),
                    median_r = format!("{:.1}", m.median_r),
                    r_cv = format!("{:.2}", m.r_cv),
                    "S9 passed — measured values"
                );
                Ok((StageOutcome::Success, ImageOutput::Passthrough))
            }
            Err(reason) => Ok((StageOutcome::Exhausted(reason), ImageOutput::Passthrough)),
        }
    }

    fn max_retries(&self) -> u32 {
        // Sanity check is pass/fail -- no parameter variation.
        // Failure triggers fallback to FindFeatures.
        1
    }
}

/// Minimum edge density ratio for the quick sanity check to pass.
/// A well-warped image of the knob panel has consistent edge structure.
const QUICK_MIN_EDGE_DENSITY: f64 = 0.02;

/// Maximum edge density — too many edges means noise / bad warp.
const QUICK_MAX_EDGE_DENSITY: f64 = 0.40;

/// Lightweight sanity check for detection mode.
///
/// Computes edge density on the warped image using Canny. A valid warp
/// produces consistent edge structure from the chrome panel and knobs.
/// This is more robust than HoughCircles which is sensitive to param2.
///
/// Returns `Ok(true)` if the check passes, `Ok(false)` if the warp appears
/// invalid, or `Err` on OpenCV errors.
pub fn quick_sanity_check(warped: &Mat) -> Result<bool, opencv::Error> {
    let blurred = enhance_gray(warped, 5)?;

    let mut edges = Mat::default();
    imgproc::canny(&blurred, &mut edges, 50.0, 150.0, 3, false)?;

    let total_pixels = (edges.rows() * edges.cols()) as f64;
    if total_pixels < 1.0 {
        return Ok(false);
    }

    let edge_count = opencv::core::count_non_zero(&edges)? as f64;
    let density = edge_count / total_pixels;

    Ok(density >= QUICK_MIN_EDGE_DENSITY && density <= QUICK_MAX_EDGE_DENSITY)
}
