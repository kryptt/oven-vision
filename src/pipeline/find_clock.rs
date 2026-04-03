use opencv::core::{Mat, Point, Rect, Scalar, Size};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;

use super::perspective::transform_to_mat;
use super::stage::{PipelineState, StageDescriptor, StageOutcome};
use super::{DebugImage, Stage};
use crate::annotate::encode_jpeg;

/// Scale factors for clock template matching.
const SCALE_FACTORS: [f64; 6] = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40];

/// Match threshold for clock detection.
const CLOCK_THRESHOLD: f64 = 0.25;

/// Path to the clock template.
const CLOCK_TEMPLATE_PATH: &str = "/templates/clock.jpg";

/// Stage 3c: Find the analog clock in the extracted band.
///
/// The clock is the leftmost large feature on the panel. Its right edge
/// defines the start of the knob area. This stage updates `knob_search.x_min`
/// so that S4 only looks for knobs to the right of the clock.
pub struct FindClock {
    template: Mat,
}

impl FindClock {
    pub fn new() -> Self {
        let template = imgcodecs::imread(CLOCK_TEMPLATE_PATH, imgcodecs::IMREAD_GRAYSCALE)
            .unwrap_or_else(|e| {
                tracing::warn!(%e, "failed to load clock template");
                Mat::default()
            });
        Self { template }
    }
}

pub(crate) const DESCRIPTOR: StageDescriptor = StageDescriptor {
    name: "FindClock",
    label: "S3c:FindClock",
    fallback: Some("ExtractBand"),
};

impl Stage for FindClock {
    fn descriptor(&self) -> StageDescriptor {
        DESCRIPTOR
    }

    fn run(
        &self,
        state: &mut PipelineState,
        frame: &Mat,
        iteration: u32,
    ) -> Result<StageOutcome, opencv::Error> {
        let (Some(crop), Some(persp), Some(search)) =
            (&state.crop, &state.perspective, &state.knob_search)
        else {
            return Ok(StageOutcome::Exhausted("missing prior state".into()));
        };

        if self.template.empty() {
            return Ok(StageOutcome::Exhausted("clock template not loaded".into()));
        }

        // Warp frame and extract the band
        let roi_rect = Rect::new(
            crop.x as i32,
            crop.y as i32,
            crop.width as i32,
            crop.height as i32,
        );
        let cropped = Mat::roi(frame, roi_rect)?;
        let mat = transform_to_mat(&persp.matrix)?;

        let mut warped = Mat::default();
        imgproc::warp_perspective_def(
            &cropped,
            &mut warped,
            &mat,
            Size::new(persp.output_width as i32, persp.output_height as i32),
        )?;

        let band_rect = Rect::new(
            0,
            search.y_min as i32,
            persp.output_width as i32,
            (search.y_max - search.y_min) as i32,
        );
        let band = Mat::roi(&warped, band_rect)?;

        // Convert to grayscale + enhance
        let mut gray = Mat::default();
        imgproc::cvt_color_def(&band, &mut gray, imgproc::COLOR_BGR2GRAY)?;
        let mut clahe_obj = imgproc::create_clahe(2.0, Size::new(8, 8))?;
        let mut enhanced = Mat::default();
        clahe_obj.apply(&gray, &mut enhanced)?;

        // Pick scale from iteration
        let scale_idx = (iteration as usize) % SCALE_FACTORS.len();
        let scale = SCALE_FACTORS[scale_idx];

        // Scale the template
        let new_w = (self.template.cols() as f64 * scale) as i32;
        let new_h = (self.template.rows() as f64 * scale) as i32;
        if new_w < 5 || new_h < 5 {
            return Ok(StageOutcome::Retry(format!(
                "clock template too small at scale {scale:.2}"
            )));
        }
        let mut scaled = Mat::default();
        imgproc::resize(
            &self.template,
            &mut scaled,
            Size::new(new_w, new_h),
            0.0,
            0.0,
            imgproc::INTER_AREA,
        )?;

        // Enhance + invert both
        let mut templ_enhanced = Mat::default();
        clahe_obj.apply(&scaled, &mut templ_enhanced)?;

        let mut img_inv = Mat::default();
        opencv::core::bitwise_not(&enhanced, &mut img_inv, &Mat::default())?;
        let mut templ_inv = Mat::default();
        opencv::core::bitwise_not(&templ_enhanced, &mut templ_inv, &Mat::default())?;

        if templ_inv.cols() >= img_inv.cols() || templ_inv.rows() >= img_inv.rows() {
            return Ok(StageOutcome::Retry(format!(
                "clock template larger than band at scale {scale:.2}"
            )));
        }

        // Template match — search only the left third of the band (clock is on the left)
        let search_w = (img_inv.cols() as f64 * 0.35) as i32;
        let left_roi = Rect::new(0, 0, search_w, img_inv.rows());
        let left_region = Mat::roi(&img_inv, left_roi)?;

        let mut result = Mat::default();
        imgproc::match_template(
            &left_region,
            &templ_inv,
            &mut result,
            imgproc::TM_CCOEFF_NORMED,
            &Mat::default(),
        )?;

        // Find the best match
        let mut min_val = 0.0;
        let mut max_val = 0.0;
        let mut min_loc = opencv::core::Point::default();
        let mut max_loc = opencv::core::Point::default();
        opencv::core::min_max_loc(
            &result,
            Some(&mut min_val),
            Some(&mut max_val),
            Some(&mut min_loc),
            Some(&mut max_loc),
            &Mat::default(),
        )?;

        if max_val < CLOCK_THRESHOLD {
            return Ok(StageOutcome::Retry(format!(
                "clock match too low: {max_val:.2} (need {CLOCK_THRESHOLD}), scale={scale:.2}"
            )));
        }

        // Clock center in band coordinates
        let clock_cx = max_loc.x as f64 + new_w as f64 / 2.0;
        let clock_cy = max_loc.y as f64 + new_h as f64 / 2.0;
        let clock_r = new_w.max(new_h) as f64 / 2.0;

        // The knob area starts to the right of the clock (with some margin)
        let knob_x_start = clock_cx + clock_r * 1.5;

        // Update knob_search with clock info (convert band-local Y to warped-image Y)
        let ks = state.knob_search.as_mut().unwrap();
        ks.x_min = knob_x_start;
        ks.clock_center_x = clock_cx;
        ks.clock_center_y = clock_cy + ks.y_min; // convert to warped-image coords
        ks.clock_radius = clock_r;

        Ok(StageOutcome::Success)
    }

    fn max_retries(&self) -> u32 {
        12 // 6 scales × 2
    }

    fn debug_image(
        &self,
        state: &PipelineState,
        frame: &Mat,
    ) -> Result<Option<DebugImage>, opencv::Error> {
        let (Some(crop), Some(persp), Some(search)) =
            (&state.crop, &state.perspective, &state.knob_search)
        else {
            return Ok(None);
        };

        // Warp and extract band
        let roi_rect = Rect::new(
            crop.x as i32,
            crop.y as i32,
            crop.width as i32,
            crop.height as i32,
        );
        let cropped = Mat::roi(frame, roi_rect)?;
        let mat = transform_to_mat(&persp.matrix)?;

        let mut warped = Mat::default();
        imgproc::warp_perspective_def(
            &cropped,
            &mut warped,
            &mat,
            Size::new(persp.output_width as i32, persp.output_height as i32),
        )?;

        let band_rect = Rect::new(
            0,
            search.y_min as i32,
            persp.output_width as i32,
            (search.y_max - search.y_min) as i32,
        );
        let mut canvas = Mat::roi(&warped, band_rect)?.try_clone()?;

        // Draw clock circle in cyan
        let cyan = Scalar::new(255.0, 255.0, 0.0, 0.0);
        let clock_y_in_band = (search.clock_center_y - search.y_min) as i32;
        imgproc::circle(
            &mut canvas,
            Point::new(search.clock_center_x as i32, clock_y_in_band),
            search.clock_radius as i32,
            cyan,
            2,
            imgproc::LINE_8,
            0,
        )?;

        // Draw vertical line at x_min (knob area boundary) in yellow
        let yellow = Scalar::new(0.0, 255.0, 255.0, 0.0);
        let canvas_h = canvas.rows();
        imgproc::line(
            &mut canvas,
            Point::new(search.x_min as i32, 0),
            Point::new(search.x_min as i32, canvas_h),
            yellow,
            2,
            imgproc::LINE_8,
            0,
        )?;

        let jpeg = encode_jpeg(&canvas, 90)?;
        Ok(Some(("S3c:FindClock".into(), jpeg)))
    }
}
