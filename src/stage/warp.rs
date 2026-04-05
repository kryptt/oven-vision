use opencv::core::{Mat, Point2f, Scalar, Size, BORDER_CONSTANT};
use opencv::imgproc;
use opencv::prelude::*;

use crate::config::Calibration;
use super::{FrameState, Stage, StageError};

pub struct Warp {
    transform: Mat,
    dst_w: i32,
    dst_h: i32,
}

impl Warp {
    pub fn new(calib: &Calibration, dst_w: i32, dst_h: i32) -> Result<Self, Box<dyn std::error::Error>> {
        let sp = &calib.source_points;
        let src_pts = Mat::from_slice_2d(&[
            &[sp.top_left[0] as f32, sp.top_left[1] as f32],
            &[sp.top_right[0] as f32, sp.top_right[1] as f32],
            &[sp.bottom_right[0] as f32, sp.bottom_right[1] as f32],
            &[sp.bottom_left[0] as f32, sp.bottom_left[1] as f32],
        ])?;
        let dst_pts = Mat::from_slice_2d(&[
            &[0.0f32, 0.0],
            &[dst_w as f32, 0.0],
            &[dst_w as f32, dst_h as f32],
            &[0.0, dst_h as f32],
        ])?;
        let transform = imgproc::get_perspective_transform(&src_pts, &dst_pts, 0)?;
        Ok(Self { transform, dst_w, dst_h })
    }
}

impl Stage for Warp {
    fn name(&self) -> &'static str { "warp" }

    fn process(&self, state: &mut FrameState) -> Result<(), StageError> {
        let input = state.undistorted.as_ref().unwrap_or(&state.raw);
        let mut out = Mat::default();
        imgproc::warp_perspective(
            input,
            &mut out,
            &self.transform,
            Size::new(self.dst_w, self.dst_h),
            imgproc::INTER_LINEAR,
            BORDER_CONSTANT,
            Scalar::default(),
        )
        .map_err(|e| StageError { stage: self.name(), message: e.to_string() })?;
        state.warped = Some(out);
        Ok(())
    }
}
