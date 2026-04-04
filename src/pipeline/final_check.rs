use opencv::core::Mat;
use tracing::info;

use super::stage::{PipelineState, StageDescriptor, StageOutcome};
use super::util::{validate_features, SanityThresholds};
use super::{ImageOutput, Stage};

/// Strict thresholds for the final validation pass.
const THRESHOLDS: SanityThresholds = SanityThresholds {
    y_tolerance_px: 6.0,
    min_x_gap_px: 30.0,
    max_gap_cv: 0.20,
    max_radius_factor: 1.2,
    expected_knobs: 10,
    skip_overlap: false,
};

pub(crate) const DESCRIPTOR: StageDescriptor = StageDescriptor {
    name: "FinalCheck",
    label: "S13:FinalCheck",
    fallback: Some("FinalDetect"),
};

/// Stage S13: Strict sanity check on the final refined feature detection.
///
/// Same checks as S9 but with tighter thresholds. This is the gate that
/// determines whether the calibration succeeded.
pub struct FinalCheck;

impl FinalCheck {
    pub fn new() -> Self {
        Self
    }
}

impl Stage for FinalCheck {
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
                StageOutcome::Exhausted("no features from S12".into()),
                ImageOutput::Passthrough,
            ));
        };

        match validate_features(features, &THRESHOLDS) {
            Ok(m) => {
                info!(
                    max_y_dev = format!("{:.1}", m.max_y_dev),
                    min_x_gap = format!("{:.1}", m.min_x_gap),
                    gap_cv = format!("{:.2}", m.gap_cv),
                    median_r = format!("{:.1}", m.median_r),
                    "S13 FINAL passed"
                );
                Ok((StageOutcome::Success, ImageOutput::Passthrough))
            }
            Err(reason) => Ok((StageOutcome::Exhausted(reason), ImageOutput::Passthrough)),
        }
    }

    fn max_retries(&self) -> u32 {
        1
    }
}
