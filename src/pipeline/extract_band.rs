use opencv::core::{Mat, Point, Rect, Scalar, Size};
use opencv::imgproc;
use opencv::prelude::*;

use super::perspective::transform_to_mat;
use super::stage::{PipelineState, StageDescriptor, StageOutcome};
use super::{DebugImage, Stage};
use crate::annotate::encode_jpeg;

/// Extra padding above the top trim line and below the bottom trim line,
/// as a fraction of the inter-line distance. This ensures the knob row
/// (which sits just below the bottom trim) is fully included.
const BAND_PADDING_TOP: f64 = 0.3;
const BAND_PADDING_BOTTOM: f64 = 4.0;

/// Stage 3b: Extract the horizontal band between the two chrome trim lines
/// from the perspective-corrected image.
///
/// Computes the Y range of the knob panel in the warped image and stores it
/// in `PipelineState::knob_search`. Subsequent stages only search within
/// this band, eliminating false matches on the stovetop, oven doors, etc.
pub struct ExtractBand;

impl ExtractBand {
    pub fn new() -> Self {
        Self
    }
}

pub(crate) const DESCRIPTOR: StageDescriptor = StageDescriptor {
    name: "ExtractBand",
    label: "S3b:ExtractBand",
    fallback: Some("Perspective"),
};

impl Stage for ExtractBand {
    fn descriptor(&self) -> StageDescriptor {
        DESCRIPTOR
    }

    fn run(
        &self,
        state: &mut PipelineState,
        _frame: &Mat,
        _iteration: u32,
    ) -> Result<StageOutcome, opencv::Error> {
        let (Some(lines), Some(verts), Some(persp)) =
            (&state.lines, &state.verticals, &state.perspective)
        else {
            return Ok(StageOutcome::Exhausted(
                "missing lines, verticals, or perspective".into(),
            ));
        };

        // Compute the trim line Y positions in the warped image.
        // The perspective stage maps the 4 corners to a rectangle where:
        //   top trim → Y = top_pad
        //   bottom trim → Y = top_pad + inter_line
        let left_dist = (lines.bottom.y1 - lines.top.y1).abs();
        let right_dist = (lines.bottom.y2 - lines.top.y2).abs();
        let inter_line = (left_dist + right_dist) / 2.0;
        let top_pad = inter_line * 0.5; // matches perspective.rs TOP_PADDING_FRAC

        let top_trim_y = top_pad;
        let bottom_trim_y = top_pad + inter_line;

        // The band extends from above the top trim to well below the bottom trim
        // (where the knobs sit).
        let band_top = (top_trim_y - inter_line * BAND_PADDING_TOP).max(0.0);
        let band_bottom =
            (bottom_trim_y + inter_line * BAND_PADDING_BOTTOM).min(persp.output_height as f64);

        if band_bottom - band_top < 10.0 {
            return Ok(StageOutcome::Retry(format!(
                "band too narrow: {:.0}px",
                band_bottom - band_top
            )));
        }

        // Initialize knob_search with the Y band; x_min will be set by FindClock
        state.knob_search = Some(super::stage::KnobSearchArea {
            y_min: band_top,
            y_max: band_bottom,
            x_min: 0.0, // will be updated by FindClock
            clock_center_x: 0.0,
            clock_center_y: 0.0,
            clock_radius: 0.0,
        });

        Ok(StageOutcome::Success)
    }

    fn max_retries(&self) -> u32 {
        1
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

        // Warp the frame
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

        // Extract just the band
        let band_rect = Rect::new(
            0,
            search.y_min as i32,
            persp.output_width as i32,
            (search.y_max - search.y_min) as i32,
        );
        let band = Mat::roi(&warped, band_rect)?.try_clone()?;

        // Draw the trim line positions within the band
        let green = Scalar::new(0.0, 255.0, 0.0, 0.0);

        // The trim lines relative to the band top
        let lines = state.lines.as_ref().unwrap();
        let left_dist = (lines.bottom.y1 - lines.top.y1).abs();
        let right_dist = (lines.bottom.y2 - lines.top.y2).abs();
        let inter_line = (left_dist + right_dist) / 2.0;
        let top_pad = inter_line * 0.5;
        let top_y = (top_pad - search.y_min) as i32;
        let bot_y = (top_pad + inter_line - search.y_min) as i32;

        let mut canvas = band;
        let w = persp.output_width as i32;
        imgproc::line(
            &mut canvas,
            Point::new(0, top_y),
            Point::new(w, top_y),
            green,
            1,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::line(
            &mut canvas,
            Point::new(0, bot_y),
            Point::new(w, bot_y),
            green,
            1,
            imgproc::LINE_8,
            0,
        )?;

        let jpeg = encode_jpeg(&canvas, 90)?;
        Ok(Some(("S3b:ExtractBand".into(), jpeg)))
    }
}
