use opencv::core::Mat;
use opencv::prelude::*;

use crate::config::Calibration;
use super::{FrameState, Stage, StageError};

pub struct Undistort {
    camera_matrix: Mat,
    dist_coeffs: Mat,
}

impl Undistort {
    pub fn new(calib: &Calibration) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        let k1 = calib.distortion_k1;
        if k1.abs() < 1e-9 {
            return Ok(None);
        }

        let fw = calib.frame_size[0];
        let fh = calib.frame_size[1];
        let camera_matrix = Mat::from_slice_2d(&[
            &[fw, 0.0, fw / 2.0],
            &[0.0, fw, fh / 2.0],
            &[0.0, 0.0, 1.0],
        ])?;
        let coeffs = [k1, 0.0, 0.0, 0.0, 0.0];
        let dist_coeffs = Mat::from_slice(&coeffs)?.try_clone()?;

        Ok(Some(Self { camera_matrix, dist_coeffs }))
    }
}

impl Stage for Undistort {
    fn name(&self) -> &'static str { "undistort" }

    fn process(&self, state: &mut FrameState) -> Result<(), StageError> {
        let mut out = Mat::default();
        opencv::calib3d::undistort(
            &state.raw,
            &mut out,
            &self.camera_matrix,
            &self.dist_coeffs,
            &self.camera_matrix,
        )
        .map_err(|e| StageError { stage: self.name(), message: e.to_string() })?;
        state.undistorted = Some(out);
        Ok(())
    }
}
