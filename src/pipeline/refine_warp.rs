use opencv::core::{Mat, Rect};
use opencv::prelude::*;

use super::stage::{PipelineState, StageDescriptor, StageOutcome};
use super::util::median_radius;
use super::{DebugImage, ImageOutput, Stage};
use crate::annotate::encode_jpeg;

pub(crate) const DESCRIPTOR: StageDescriptor = StageDescriptor {
    name: "RefineWarp",
    label: "S11:RefineWarp",
    fallback: Some("FindCorner"),
};

/// Stage S11: Crop to the oven panel using the detected corner.
///
/// Uses the top-right corner (chrome bar × wall edge) from S10 to crop the
/// image to [0..corner_x, 0..knob_row_bottom]. The tightly-cropped image
/// goes to S12 for final feature detection.
pub struct RefineWarp;

impl RefineWarp {
    pub fn new() -> Self {
        Self
    }
}

impl Stage for RefineWarp {
    fn descriptor(&self) -> StageDescriptor {
        DESCRIPTOR
    }

    fn run(
        &self,
        state: &mut PipelineState,
        src: &Mat,
        dst: &mut Mat,
        _raw: &Mat,
        _iteration: u32,
    ) -> Result<(StageOutcome, ImageOutput), opencv::Error> {
        let Some(ks) = &state.knob_search else {
            return Ok((
                StageOutcome::Exhausted("no knob search area".into()),
                ImageOutput::Passthrough,
            ));
        };
        let Some(features) = &state.features else {
            return Ok((
                StageOutcome::Exhausted("no features".into()),
                ImageOutput::Passthrough,
            ));
        };

        let corner_x = ks.corner_x.unwrap_or(src.cols() as f64);
        let corner_y = ks.corner_y.unwrap_or(0.0);

        if features.knobs.is_empty() {
            return Ok((
                StageOutcome::Exhausted("no knobs in features".into()),
                ImageOutput::Passthrough,
            ));
        }

        // Compute the bottom of the knob row (lowest knob center + radius + margin)
        let median_r = median_radius(&features.knobs);
        let max_knob_y = features
            .knobs
            .iter()
            .map(|k| k.center_y + k.radius)
            .fold(0.0f64, f64::max);
        let crop_bottom = (max_knob_y + median_r * 2.0).min(src.rows() as f64);
        let crop_top = (corner_y - median_r).max(0.0);
        let crop_right = (corner_x + median_r).min(src.cols() as f64);

        let crop_w = crop_right as i32;
        let crop_h = (crop_bottom - crop_top) as i32;

        if crop_w < 50 || crop_h < 20 {
            return Ok((
                StageOutcome::Retry(format!("refined crop too small: {crop_w}x{crop_h}")),
                ImageOutput::Passthrough,
            ));
        }

        // Crop
        let crop_rect = Rect::new(0, crop_top as i32, crop_w, crop_h);
        let cropped = Mat::roi(src, crop_rect)?;
        cropped.copy_to(dst)?;

        // Update knob_search offsets so S12 (FinalDetect) translates
        // coordinates back to warped-image space correctly.
        // The S11 crop starts at (0, crop_top) in warped-image coords,
        // and x_min=0 since we crop from x=0.
        if let Some(ks) = state.knob_search.as_mut() {
            ks.y_min = crop_top;
            ks.x_min = 0.0;
        }

        Ok((StageOutcome::Success, ImageOutput::Transformed))
    }

    fn max_retries(&self) -> u32 {
        1
    }

    fn debug_image(
        &self,
        _state: &PipelineState,
        working: &Mat,
        _raw: &Mat,
    ) -> Result<Option<DebugImage>, opencv::Error> {
        let jpeg = encode_jpeg(working, 90)?;
        Ok(Some(("S11:RefineWarp".into(), jpeg)))
    }
}
