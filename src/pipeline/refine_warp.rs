use opencv::core::{Mat, Point2f, Rect, Size};
use opencv::imgproc;
use opencv::prelude::*;

use super::stage::{PipelineState, StageDescriptor, StageOutcome};
use super::util::median_radius;
use super::{DebugImage, ImageOutput, Stage};
use crate::annotate::encode_jpeg;

pub(crate) const DESCRIPTOR: StageDescriptor = StageDescriptor {
    name: "RefineWarp",
    label: "S10:RefineWarp",
    fallback: Some("FindFeatures"),
};

/// Minimum ratio of max/min per-knob radius to attempt keystone correction.
/// Below this, the scale variation is too small to compute a meaningful transform.
const MIN_SCALE_RATIO: f64 = 1.05;

/// Stage S10: Keystone correction using per-knob bounding boxes.
///
/// Each knob's template-matched radius encodes its apparent size, which varies
/// with perspective (closer knobs appear larger). We map the 4 bounding-box
/// corners of each detected knob (variable-size rectangles) to ideal
/// destinations (equal-size, equal-spacing rectangles) and compute a homography
/// with RANSAC. This gives a well-conditioned system with 40 points spanning
/// both X and Y axes.
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
        let Some(_) = &state.knob_search else {
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

        if features.knobs.len() < 4 {
            return Ok((
                StageOutcome::Exhausted(format!(
                    "need >= 4 knobs for homography, got {}",
                    features.knobs.len()
                )),
                ImageOutput::Passthrough,
            ));
        }

        let median_r = median_radius(&features.knobs);

        // Sort knobs left-to-right by center_x
        let mut sorted_knobs: Vec<_> = features.knobs.iter().collect();
        sorted_knobs.sort_by(|a, b| a.center_x.total_cmp(&b.center_x));

        let n = sorted_knobs.len();
        let leftmost_x = sorted_knobs[0].center_x;
        let rightmost_x = sorted_knobs[n - 1].center_x;

        // Check if per-knob radii vary enough to encode perspective
        let min_r = sorted_knobs.iter().map(|k| k.radius).fold(f64::INFINITY, f64::min);
        let max_r = sorted_knobs.iter().map(|k| k.radius).fold(0.0f64, f64::max);
        let scale_ratio = max_r / min_r.max(1.0);

        tracing::debug!(
            n_knobs = n,
            min_r = format!("{:.1}", min_r),
            max_r = format!("{:.1}", max_r),
            scale_ratio = format!("{:.2}", scale_ratio),
            "per-knob scale variation"
        );

        if scale_ratio < MIN_SCALE_RATIO {
            return Ok((
                StageOutcome::Retry(format!(
                    "scale variation too small: ratio={scale_ratio:.2} (need {MIN_SCALE_RATIO})"
                )),
                ImageOutput::Passthrough,
            ));
        }

        // Ideal target: all knobs on the same Y, equally spaced, equal radius
        let ideal_y = {
            let mut ys: Vec<f64> = sorted_knobs.iter().map(|k| k.center_y).collect();
            ys.sort_by(f64::total_cmp);
            ys[ys.len() / 2]
        };
        let ideal_gap = (rightmost_x - leftmost_x) / (n - 1) as f64;
        let ideal_r = median_r;

        // Build source (detected, variable-size) and destination (ideal, uniform)
        // point arrays: 4 bounding-box corners per knob = 4N points total.
        let n_pts = (n * 4) as i32;
        let mut src_pts = Mat::zeros(n_pts, 1, opencv::core::CV_32FC2)?.to_mat()?;
        let mut dst_pts = Mat::zeros(n_pts, 1, opencv::core::CV_32FC2)?.to_mat()?;

        for (i, knob) in sorted_knobs.iter().enumerate() {
            let cx = knob.center_x as f32;
            let cy = knob.center_y as f32;
            let r = knob.radius as f32;

            let ideal_cx = (leftmost_x + i as f64 * ideal_gap) as f32;
            let ideal_cy = ideal_y as f32;
            let ir = ideal_r as f32;

            let base = (i * 4) as i32;
            // Top-left, top-right, bottom-right, bottom-left
            *src_pts.at_2d_mut::<Point2f>(base, 0)? = Point2f::new(cx - r, cy - r);
            *src_pts.at_2d_mut::<Point2f>(base + 1, 0)? = Point2f::new(cx + r, cy - r);
            *src_pts.at_2d_mut::<Point2f>(base + 2, 0)? = Point2f::new(cx + r, cy + r);
            *src_pts.at_2d_mut::<Point2f>(base + 3, 0)? = Point2f::new(cx - r, cy + r);

            *dst_pts.at_2d_mut::<Point2f>(base, 0)? = Point2f::new(ideal_cx - ir, ideal_cy - ir);
            *dst_pts.at_2d_mut::<Point2f>(base + 1, 0)? = Point2f::new(ideal_cx + ir, ideal_cy - ir);
            *dst_pts.at_2d_mut::<Point2f>(base + 2, 0)? = Point2f::new(ideal_cx + ir, ideal_cy + ir);
            *dst_pts.at_2d_mut::<Point2f>(base + 3, 0)? = Point2f::new(ideal_cx - ir, ideal_cy + ir);
        }

        tracing::debug!(
            n_points = n_pts,
            ideal_y = format!("{:.1}", ideal_y),
            ideal_gap = format!("{:.1}", ideal_gap),
            ideal_r = format!("{:.1}", ideal_r),
            "computing homography from {n_pts} bounding-box corners"
        );

        // Compute homography with RANSAC
        let mut mask = Mat::default();
        let homography = opencv::calib3d::find_homography(
            &src_pts,
            &dst_pts,
            &mut mask,
            opencv::calib3d::RANSAC,
            3.0,
        )?;

        if homography.empty() || homography.rows() != 3 || homography.cols() != 3 {
            tracing::warn!(
                rows = homography.rows(),
                cols = homography.cols(),
                empty = homography.empty(),
                "homography computation failed"
            );
            return Ok((
                StageOutcome::Retry("homography computation failed".into()),
                ImageOutput::Passthrough,
            ));
        }

        // Log the computed matrix for debugging
        tracing::info!(
            h00 = format!("{:.4}", *homography.at_2d::<f64>(0, 0)?),
            h01 = format!("{:.4}", *homography.at_2d::<f64>(0, 1)?),
            h02 = format!("{:.4}", *homography.at_2d::<f64>(0, 2)?),
            h10 = format!("{:.4}", *homography.at_2d::<f64>(1, 0)?),
            h11 = format!("{:.4}", *homography.at_2d::<f64>(1, 1)?),
            h12 = format!("{:.4}", *homography.at_2d::<f64>(1, 2)?),
            h20 = format!("{:.6}", *homography.at_2d::<f64>(2, 0)?),
            h21 = format!("{:.6}", *homography.at_2d::<f64>(2, 1)?),
            h22 = format!("{:.4}", *homography.at_2d::<f64>(2, 2)?),
            "computed keystone homography"
        );

        // Warp the image
        let mut warped = Mat::default();
        imgproc::warp_perspective(
            src,
            &mut warped,
            &homography,
            Size::new(src.cols(), src.rows()),
            imgproc::INTER_CUBIC,
            opencv::core::BORDER_CONSTANT,
            opencv::core::Scalar::default(),
        )?;

        // Crop to the panel region using ideal knob positions
        let min_knob_y = sorted_knobs.iter().map(|k| k.center_y).fold(f64::INFINITY, f64::min);
        let crop_top = (min_knob_y - median_r * 3.0).max(0.0);
        let max_knob_y = ideal_y + median_r * 3.0;
        let crop_bottom = max_knob_y.min(warped.rows() as f64);
        let crop_right = (rightmost_x + median_r * 3.0).min(warped.cols() as f64);
        let crop_w = crop_right as i32;
        let crop_h = (crop_bottom - crop_top) as i32;

        if crop_w < 50 || crop_h < 20 {
            return Ok((
                StageOutcome::Retry(format!("refined crop too small: {crop_w}x{crop_h}")),
                ImageOutput::Passthrough,
            ));
        }

        let crop_rect = Rect::new(0, crop_top as i32, crop_w, crop_h);
        let cropped = Mat::roi(&warped, crop_rect)?;
        cropped.copy_to(dst)?;

        // Update knob_search offsets for S11
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
        Ok(Some(("S10:RefineWarp".into(), jpeg)))
    }
}
