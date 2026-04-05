use opencv::core::{Mat, Size, BORDER_CONSTANT};
use opencv::imgproc;
use opencv::prelude::*;

use crate::config::KnobDetection;
use super::{FrameState, Stage, StageError};

pub struct Enhance {
    clip_limit: f64,
    grid_size: i32,
}

impl Enhance {
    pub fn new(kd: &KnobDetection) -> Self {
        Self {
            clip_limit: kd.clahe_clip_limit,
            grid_size: kd.clahe_grid_size,
        }
    }
}

impl Stage for Enhance {
    fn name(&self) -> &'static str { "enhance" }

    fn process(&self, state: &mut FrameState) -> Result<(), StageError> {
        let warped = state.warped.as_ref()
            .ok_or(StageError { stage: self.name(), message: "no warped frame".into() })?;

        let mut gray = Mat::default();
        imgproc::cvt_color(warped, &mut gray, imgproc::COLOR_BGR2GRAY, 0)
            .map_err(|e| StageError { stage: self.name(), message: e.to_string() })?;

        let mut clahe = imgproc::create_clahe(
            self.clip_limit,
            Size::new(self.grid_size, self.grid_size),
        ).map_err(|e| StageError { stage: self.name(), message: e.to_string() })?;

        let mut enhanced = Mat::default();
        clahe.apply(&gray, &mut enhanced)
            .map_err(|e| StageError { stage: self.name(), message: e.to_string() })?;

        let mut blurred = Mat::default();
        imgproc::gaussian_blur(
            &enhanced, &mut blurred,
            Size::new(5, 5), 1.5, 1.5, BORDER_CONSTANT,
        ).map_err(|e| StageError { stage: self.name(), message: e.to_string() })?;

        state.enhanced = Some(blurred);
        Ok(())
    }
}
