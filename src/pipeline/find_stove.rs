use opencv::core::{Mat, Rect, Scalar};
use opencv::imgproc;
use opencv::prelude::*;

use super::stage::{CropRegion, PipelineState, StageDescriptor, StageOutcome};
use super::{DebugImage, ImageOutput, Stage};
use crate::annotate::encode_jpeg;

/// Default crop region covering the stove panel with generous margins.
/// Derived from the Python spike: dials sit around x=1350-2120, y=980-1100
/// in the 2560x1440 frame. Extra horizontal margin to capture panel edges
/// for vertical line detection.
const DEFAULT_CROP: CropRegion = CropRegion {
    x: 1300,
    y: 830,
    width: 870,
    height: 450,
};

/// On each retry, expand the crop by this many pixels per side.
const MARGIN_STEP: u32 = 20;

/// Stage 1: Crop the frame to the stove panel area.
///
/// Uses a configurable initial crop region. On retries, progressively expands
/// the margins to capture more of the panel if subsequent stages fail and
/// fall back here.
pub struct FindStove {
    initial_crop: CropRegion,
}

impl FindStove {
    pub fn new() -> Self {
        Self {
            initial_crop: DEFAULT_CROP,
        }
    }

    pub fn with_crop(crop: CropRegion) -> Self {
        Self { initial_crop: crop }
    }

    /// Compute the crop region for a given iteration, expanding margins.
    fn crop_for_iteration(&self, iteration: u32, frame_w: u32, frame_h: u32) -> CropRegion {
        let expand = iteration * MARGIN_STEP;
        let x = self.initial_crop.x.saturating_sub(expand);
        let y = self.initial_crop.y.saturating_sub(expand);
        let right = (self.initial_crop.x + self.initial_crop.width + expand).min(frame_w);
        let bottom = (self.initial_crop.y + self.initial_crop.height + expand).min(frame_h);
        CropRegion {
            x,
            y,
            width: right - x,
            height: bottom - y,
        }
    }
}

pub(crate) const DESCRIPTOR: StageDescriptor = StageDescriptor {
    name: "FindStove",
    label: "S1:FindStove",
    fallback: None,
};

impl Stage for FindStove {
    fn descriptor(&self) -> StageDescriptor {
        DESCRIPTOR
    }

    fn run(
        &self,
        state: &mut PipelineState,
        _src: &Mat,
        dst: &mut Mat,
        raw: &Mat,
        iteration: u32,
    ) -> Result<(StageOutcome, ImageOutput), opencv::Error> {
        let frame_w = raw.cols() as u32;
        let frame_h = raw.rows() as u32;

        if frame_w == 0 || frame_h == 0 {
            return Ok((
                StageOutcome::Exhausted("empty frame (0x0)".into()),
                ImageOutput::Passthrough,
            ));
        }

        let crop = self.crop_for_iteration(iteration, frame_w, frame_h);

        // Validate the crop fits within the frame
        if crop.x + crop.width > frame_w || crop.y + crop.height > frame_h {
            return Ok((
                StageOutcome::Retry(format!(
                    "crop {}x{}+{}+{} exceeds frame {}x{}",
                    crop.width, crop.height, crop.x, crop.y, frame_w, frame_h
                )),
                ImageOutput::Passthrough,
            ));
        }

        if crop.width < 100 || crop.height < 50 {
            return Ok((
                StageOutcome::Exhausted("crop region too small for meaningful detection".into()),
                ImageOutput::Passthrough,
            ));
        }

        // Extract the ROI and copy to dst
        let roi = Rect::new(
            crop.x as i32,
            crop.y as i32,
            crop.width as i32,
            crop.height as i32,
        );
        let cropped = Mat::roi(raw, roi)?;
        cropped.copy_to(dst)?;

        state.crop = Some(crop);
        Ok((StageOutcome::Success, ImageOutput::Transformed))
    }

    fn max_retries(&self) -> u32 {
        20 // 20 x 20px = 400px max expansion
    }

    fn debug_image(
        &self,
        state: &PipelineState,
        _working: &Mat,
        raw: &Mat,
    ) -> Result<Option<DebugImage>, opencv::Error> {
        let Some(crop) = &state.crop else {
            return Ok(None);
        };

        let mut canvas = raw.try_clone()?;
        let rect = Rect::new(
            crop.x as i32,
            crop.y as i32,
            crop.width as i32,
            crop.height as i32,
        );
        // Draw green rectangle around crop region
        imgproc::rectangle(
            &mut canvas,
            rect,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;

        let jpeg = encode_jpeg(&canvas, 80)?;
        Ok(Some(("S1:FindStove".into(), jpeg)))
    }
}
