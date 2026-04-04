use std::sync::Arc;

use opencv::core::{Mat, Point, Rect, Scalar, Size};
use opencv::imgproc;
use opencv::prelude::*;

use super::stage::{PipelineState, StageDescriptor, StageOutcome};
use super::util::{Templates, enhance_gray};
use super::{DebugImage, ImageOutput, Stage};
use crate::annotate::encode_jpeg;

/// Scale factors for clock template matching.
const SCALE_FACTORS: [f64; 6] = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40];

/// Match threshold for grayscale TM_CCORR_NORMED clock detection.
const CLOCK_THRESHOLD: f64 = 0.70;

/// Stage 7: Find the analog clock in the extracted band.
///
/// The clock is the leftmost large feature on the panel. Its right edge
/// defines the start of the knob area. This stage reads `src` (the band),
/// finds the clock, and copies the region right of the clock into `dst`.
/// It also stores the x_offset in state for coordinate translation.
pub struct FindClock {
    templates: Arc<Templates>,
}

impl FindClock {
    pub fn new(templates: Arc<Templates>) -> Self {
        Self { templates }
    }
}

pub(crate) const DESCRIPTOR: StageDescriptor = StageDescriptor {
    name: "FindClock",
    label: "S7:FindClock",
    fallback: Some("ExtractBand"),
};

impl Stage for FindClock {
    fn descriptor(&self) -> StageDescriptor {
        DESCRIPTOR
    }

    fn run(
        &self,
        state: &mut PipelineState,
        src: &Mat,
        dst: &mut Mat,
        _raw: &Mat,
        iteration: u32,
    ) -> Result<(StageOutcome, ImageOutput), opencv::Error> {
        if state.knob_search.is_none() {
            return Ok((
                StageOutcome::Exhausted("missing prior state".into()),
                ImageOutput::Passthrough,
            ));
        }

        if self.templates.clock.empty() {
            return Ok((
                StageOutcome::Exhausted("clock template not loaded".into()),
                ImageOutput::Passthrough,
            ));
        }

        // src is the band image from ExtractBand

        // Grayscale + CLAHE + blur (no Canny — preserves texture for better matching)
        let blurred = enhance_gray(src, 3)?;

        // Pick scale from iteration
        let scale_idx = (iteration as usize) % SCALE_FACTORS.len();
        let scale = SCALE_FACTORS[scale_idx];

        // Scale the template (already CLAHE-enhanced via shared Templates)
        let new_w = (self.templates.clock.cols() as f64 * scale) as i32;
        let new_h = (self.templates.clock.rows() as f64 * scale) as i32;
        if new_w < 5 || new_h < 5 {
            return Ok((
                StageOutcome::Retry(format!("clock template too small at scale {scale:.2}")),
                ImageOutput::Passthrough,
            ));
        }
        let mut scaled = Mat::default();
        imgproc::resize(
            &self.templates.clock,
            &mut scaled,
            Size::new(new_w, new_h),
            0.0,
            0.0,
            imgproc::INTER_AREA,
        )?;

        if scaled.cols() >= blurred.cols() || scaled.rows() >= blurred.rows() {
            return Ok((
                StageOutcome::Retry(format!(
                    "clock template larger than band at scale {scale:.2}"
                )),
                ImageOutput::Passthrough,
            ));
        }

        // Grayscale template matching — search only the left 35% of the band
        let search_w = (blurred.cols() as f64 * 0.35) as i32;
        let left_roi = Rect::new(0, 0, search_w, blurred.rows());
        let left_region = Mat::roi(&blurred, left_roi)?;

        let mut result = Mat::default();
        imgproc::match_template(
            &left_region,
            &scaled,
            &mut result,
            imgproc::TM_CCORR_NORMED,
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
            return Ok((
                StageOutcome::Retry(format!(
                    "clock match too low: {max_val:.2} (need {CLOCK_THRESHOLD}), scale={scale:.2}"
                )),
                ImageOutput::Passthrough,
            ));
        }

        // Clock center in band coordinates
        let clock_cx = max_loc.x as f64 + new_w as f64 / 2.0;
        let clock_cy = max_loc.y as f64 + new_h as f64 / 2.0;
        let clock_r = new_w.max(new_h) as f64 / 2.0;

        // The knob area starts to the right of the clock (with some margin)
        let knob_x_start = (clock_cx + clock_r * 1.5) as i32;
        let band_w = src.cols();
        let band_h = src.rows();

        if knob_x_start >= band_w {
            return Ok((
                StageOutcome::Retry(format!(
                    "clock right edge {knob_x_start} exceeds band width {band_w}"
                )),
                ImageOutput::Passthrough,
            ));
        }

        // Copy the region right of clock into dst
        let right_roi = Rect::new(knob_x_start, 0, band_w - knob_x_start, band_h);
        let right_region = Mat::roi(src, right_roi)?;
        right_region.copy_to(dst)?;

        // Update knob_search with clock info (convert band-local Y to warped-image Y)
        let ks = state.knob_search.as_mut().unwrap();
        ks.x_min = knob_x_start as f64;
        ks.clock_center_x = clock_cx;
        ks.clock_center_y = clock_cy + ks.y_min; // convert to warped-image coords
        ks.clock_radius = clock_r;

        Ok((StageOutcome::Success, ImageOutput::Transformed))
    }

    fn max_retries(&self) -> u32 {
        12 // 6 scales x 2
    }

    fn debug_image(
        &self,
        state: &PipelineState,
        working: &Mat,
        _raw: &Mat,
    ) -> Result<Option<DebugImage>, opencv::Error> {
        if state.knob_search.is_none() {
            return Ok(None);
        }

        // working is the knob area (right of clock)
        // For debug, we want to show the full band with annotations.
        // Since we only have the knob area, we annotate what we have.
        let mut canvas = working.try_clone()?;

        // Draw a label indicating the x_offset
        let yellow = Scalar::new(0.0, 255.0, 255.0, 0.0);
        let canvas_h = canvas.rows();
        imgproc::line(
            &mut canvas,
            Point::new(0, 0),
            Point::new(0, canvas_h),
            yellow,
            2,
            imgproc::LINE_8,
            0,
        )?;

        let jpeg = encode_jpeg(&canvas, 90)?;
        Ok(Some(("S7:FindClock".into(), jpeg)))
    }
}
