use opencv::core::{Mat, Point, Rect, Scalar, Size};
use opencv::imgproc;
use opencv::prelude::*;

use super::find_features::FindFeatures;
use super::stage::{PipelineState, StageDescriptor, StageOutcome};
use super::{DebugImage, ImageOutput, Stage};

pub(crate) const DESCRIPTOR: StageDescriptor = StageDescriptor {
    name: "FinalDetect",
    label: "S12:FinalDetect",
    fallback: Some("RefineWarp"),
};

/// Stage S12: Final feature detection on the refined, inverted image.
///
/// Delegates to FindFeatures' template matching logic but operates on
/// the inverted+cropped image from S11. Since the image is already inverted,
/// the edge-based matching should have improved contrast. Overwrites
/// state.features with the refined positions.
pub struct FinalDetect {
    inner: FindFeatures,
}

impl FinalDetect {
    pub fn new() -> Self {
        Self {
            inner: FindFeatures::new(),
        }
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

        // Delegate to the inner FindFeatures
        self.inner.run(state, src, dst, raw, iteration)
    }

    fn max_retries(&self) -> u32 {
        30
    }

    fn debug_image(
        &self,
        state: &PipelineState,
        working: &Mat,
        raw: &Mat,
    ) -> Result<Option<DebugImage>, opencv::Error> {
        // Use the inner FindFeatures debug image but with our label
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
