use std::sync::Arc;

use opencv::core::Mat;

use super::find_features::FindFeatures;
use super::stage::{PipelineState, StageDescriptor, StageOutcome};
use super::{DebugImage, ImageOutput, Stage};

pub(crate) const DESCRIPTOR: StageDescriptor = StageDescriptor {
    name: "FinalDetect",
    label: "S12:FinalDetect",
    fallback: Some("RefineWarp"),
};

/// Stage S12: Final feature detection on the refined image.
///
/// Delegates to FindFeatures' template matching logic but operates on
/// the cropped image from S11. Shares the same pre-computed edge templates
/// as S8 to avoid duplicating the 288-template cache.
///
/// S11 updates knob_search offsets (y_min, x_min) to reflect the S11 crop
/// origin, so FindFeatures translates coordinates correctly.
pub struct FinalDetect {
    inner: Arc<FindFeatures>,
}

impl FinalDetect {
    pub fn new(inner: Arc<FindFeatures>) -> Self {
        Self { inner }
    }
}

impl Stage for FinalDetect {
    fn descriptor(&self) -> StageDescriptor {
        DESCRIPTOR
    }

    fn run(
        &self,
        state: &mut PipelineState,
        src: &Mat,
        dst: &mut Mat,
        raw: &Mat,
        iteration: u32,
    ) -> Result<(StageOutcome, ImageOutput), opencv::Error> {
        // Clear previous features so FindFeatures runs fresh
        state.features = None;

        // Delegate to the shared FindFeatures.
        // S11 (RefineWarp) has already updated knob_search.y_min and
        // knob_search.x_min to reflect the S11 crop origin, so
        // FindFeatures will translate coordinates correctly.
        self.inner.run(state, src, dst, raw, iteration)
    }

    fn max_retries(&self) -> u32 {
        10
    }

    fn debug_image(
        &self,
        state: &PipelineState,
        working: &Mat,
        raw: &Mat,
    ) -> Result<Option<DebugImage>, opencv::Error> {
        // Use the shared FindFeatures debug image but with our label
        if let Some((_, jpeg)) = self.inner.debug_image(state, working, raw)? {
            let n_knobs = state
                .features
                .as_ref()
                .map(|f| f.knobs.len())
                .unwrap_or(0);
            let clk_x = state
                .features
                .as_ref()
                .map(|f| f.clock.center_x)
                .unwrap_or(0.0);
            Ok(Some((
                format!("S12:FinalDetect_{n_knobs}k_clk{clk_x:.0}"),
                jpeg,
            )))
        } else {
            Ok(None)
        }
    }
}
