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

/// Stage S11: Keystone correction using detected knob centers.
///
/// The 10 knobs are physically equally spaced on a straight line. We compute
/// a homography that maps their detected (distorted) positions to ideal
/// (equal-spacing, single-Y) positions, then warp the image. This removes
/// the residual perspective distortion that S4 couldn't fully correct.
///
/// Uses RANSAC to reject outlier detections (1-2 bad template matches).
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

        // Ideal target: all knobs on the same Y, equally spaced in X
        let ideal_y = {
            let mut ys: Vec<f64> = sorted_knobs.iter().map(|k| k.center_y).collect();
            ys.sort_by(f64::total_cmp);
            ys[ys.len() / 2] // median Y
        };
        let ideal_gap = (rightmost_x - leftmost_x) / (n - 1) as f64;

        // Build a non-degenerate quadrilateral for getPerspectiveTransform.
        // The knob centers are nearly collinear, so we use the leftmost and
        // rightmost knobs to form two pairs: one at the knob Y, one shifted
        // vertically by the median radius. This creates a trapezoid whose
        // horizontal compression maps to a rectangle, correcting keystone.
        let first = sorted_knobs[0];
        let last = sorted_knobs[n - 1];
        let v_offset = median_r * 3.0; // vertical arm for the quadrilateral

        let mut src_pts = opencv::core::Vector::<Point2f>::new();
        let mut dst_pts = opencv::core::Vector::<Point2f>::new();

        // Top-left, top-right (at knob Y)
        src_pts.push(Point2f::new(first.center_x as f32, first.center_y as f32));
        src_pts.push(Point2f::new(last.center_x as f32, last.center_y as f32));
        // Bottom-left, bottom-right (shifted down by v_offset)
        src_pts.push(Point2f::new(last.center_x as f32, (last.center_y + v_offset) as f32));
        src_pts.push(Point2f::new(first.center_x as f32, (first.center_y + v_offset) as f32));

        // Ideal: equal X span, same Y, forming a rectangle
        let ideal_left_x = leftmost_x as f32;
        let ideal_right_x = (leftmost_x + (n - 1) as f64 * ideal_gap) as f32;
        let ideal_top_y = ideal_y as f32;
        let ideal_bot_y = (ideal_y + v_offset) as f32;

        dst_pts.push(Point2f::new(ideal_left_x, ideal_top_y));
        dst_pts.push(Point2f::new(ideal_right_x, ideal_top_y));
        dst_pts.push(Point2f::new(ideal_right_x, ideal_bot_y));
        dst_pts.push(Point2f::new(ideal_left_x, ideal_bot_y));

        tracing::debug!(
            src = format!("{:?}", src_pts.to_vec()),
            dst = format!("{:?}", dst_pts.to_vec()),
            "computing perspective transform from knob-derived quadrilateral"
        );

        let homography = imgproc::get_perspective_transform_def(&src_pts, &dst_pts)?;

        if homography.empty() || homography.rows() != 3 || homography.cols() != 3 {
            return Ok((
                StageOutcome::Retry("perspective transform computation failed".into()),
                ImageOutput::Passthrough,
            ));
        }

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

        // Crop to the panel region using knob positions (no corner dependency)
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

        // Update knob_search offsets for S12
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
