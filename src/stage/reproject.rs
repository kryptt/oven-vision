use opencv::calib3d;
use opencv::core::{CV_32FC1, Mat, Point2f, Point3f, Scalar, Size, Vector};
use opencv::imgproc;
use opencv::prelude::*;

use super::{FrameState, Stage, StageError};
use crate::config::Calibration;

/// 3D-aware reprojection that corrects both lens distortion and foreshortening
/// of protruding knob caps in a single remap pass.
///
/// Uses solvePnP to recover the camera pose from the 4 panel calibration points,
/// then builds a remap table where the knob row is projected from Z=protrusion
/// instead of Z=0 (the panel surface). This makes protruding spherical knob caps
/// appear more circular rather than foreshortened ovals.
pub struct Reproject {
    map_x: Mat,
    map_y: Mat,
    dst_w: i32,
    dst_h: i32,
}

impl Reproject {
    pub fn new(
        calib: &Calibration,
        dst_w: i32,
        dst_h: i32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let sp = &calib.source_points;
        let fw = calib.frame_size[0];
        let fh = calib.frame_size[1];
        let k1 = calib.distortion_k1;

        // Camera intrinsics — focal_length overrides fw if set
        let fx = if calib.focal_length > 0.0 {
            calib.focal_length
        } else {
            fw
        };
        let camera_matrix =
            Mat::from_slice_2d(&[&[fx, 0.0, fw / 2.0], &[0.0, fx, fh / 2.0], &[0.0, 0.0, 1.0]])?;

        eprintln!("  reproject: fx={} cx={} cy={}", fx, fw / 2.0, fh / 2.0);
        let dist_arr = [k1, 0.0, 0.0, 0.0, 0.0];
        let dist_coeffs = Mat::from_slice(&dist_arr)?.try_clone()?;

        // Object points: panel corners in world coordinates.
        // Use output pixel coordinates so 1 world unit = 1 output pixel.
        let obj_pts = Mat::from_slice_2d(&[
            &[0.0f32, 0.0, 0.0],                // TL
            &[dst_w as f32, 0.0, 0.0],          // TR
            &[dst_w as f32, dst_h as f32, 0.0], // BR
            &[0.0, dst_h as f32, 0.0],          // BL
        ])?;

        // Image points: where these corners appear in the raw camera frame
        let img_pts = Mat::from_slice_2d(&[
            &[sp.top_left[0] as f32, sp.top_left[1] as f32],
            &[sp.top_right[0] as f32, sp.top_right[1] as f32],
            &[sp.bottom_right[0] as f32, sp.bottom_right[1] as f32],
            &[sp.bottom_left[0] as f32, sp.bottom_left[1] as f32],
        ])?;

        // Recover camera pose (rotation + translation from world to camera)
        let mut rvec = Mat::default();
        let mut tvec = Mat::default();
        calib3d::solve_pnp(
            &obj_pts,
            &img_pts,
            &camera_matrix,
            &dist_coeffs,
            &mut rvec,
            &mut tvec,
            false,
            calib3d::SOLVEPNP_IPPE, // designed for coplanar points
        )?;

        eprintln!(
            "  reproject: solvePnP converged, building {}x{} remap",
            dst_w, dst_h
        );

        // Build remap: for each output pixel, compute 3D world point and project
        // to source image. The knob row gets Z=protrusion_z, panel surface gets Z=0.
        // Collect all 3D world points on the panel plane (Z=0).
        // solvePnP + projectPoints handles undistortion and perspective
        // in a single remap pass — one interpolation instead of two.
        let total = (dst_w * dst_h) as usize;
        let mut world_points: Vec<Point3f> = Vec::with_capacity(total);

        for v in 0..dst_h {
            for u in 0..dst_w {
                world_points.push(Point3f::new(u as f32, v as f32, 0.0));
            }
        }

        let world_mat = Vector::<Point3f>::from_iter(world_points);

        // Project all points at once
        let mut projected = Vector::<Point2f>::new();
        let mut jacobian = Mat::default();
        calib3d::project_points(
            &world_mat,
            &rvec,
            &tvec,
            &camera_matrix,
            &dist_coeffs,
            &mut projected,
            &mut jacobian,
            0.0,
        )?;

        // Fill remap tables
        let mut map_x = Mat::new_rows_cols_with_default(dst_h, dst_w, CV_32FC1, Scalar::all(0.0))?;
        let mut map_y = Mat::new_rows_cols_with_default(dst_h, dst_w, CV_32FC1, Scalar::all(0.0))?;

        for v in 0..dst_h {
            for u in 0..dst_w {
                let idx = (v * dst_w + u) as usize;
                let pt = projected.get(idx)?;
                *map_x.at_2d_mut::<f32>(v, u)? = pt.x;
                *map_y.at_2d_mut::<f32>(v, u)? = pt.y;
            }
        }

        eprintln!("  reproject: remap built");

        Ok(Self {
            map_x,
            map_y,
            dst_w,
            dst_h,
        })
    }
}

impl Stage for Reproject {
    fn name(&self) -> &'static str {
        "reproject"
    }

    fn process(&self, state: &mut FrameState) -> Result<(), StageError> {
        let mut warped = Mat::default();
        imgproc::remap(
            &state.raw,
            &mut warped,
            &self.map_x,
            &self.map_y,
            imgproc::INTER_LINEAR,
            opencv::core::BORDER_CONSTANT,
            Scalar::default(),
        )
        .map_err(|e| StageError {
            stage: self.name(),
            message: e.to_string(),
        })?;

        state.warped = Some(warped);
        Ok(())
    }
}
