use opencv::core::{Mat, Point, Rect, Scalar, Size, CV_16S};
use opencv::imgproc;
use opencv::prelude::*;

use super::stage::{KnobSearchArea, PipelineState, StageDescriptor, StageOutcome};
use super::{DebugImage, ImageOutput, Stage};
use crate::annotate::encode_jpeg;

/// How far above the knob row to look for the chrome bar (fraction of knob radius).
const BAR_SEARCH_ABOVE: f64 = 3.0;

/// How far right of the last knob to look for the wall edge (fraction of knob radius).
const WALL_SEARCH_RIGHT: f64 = 5.0;

pub(crate) const DESCRIPTOR: StageDescriptor = StageDescriptor {
    name: "FindCorner",
    label: "S10:FindCorner",
    fallback: Some("FindFeatures"),
};

/// Stage S10: Find the top-right corner of the oven panel.
///
/// Uses the detected knob positions from S8 to locate:
/// 1. The chrome bar: horizontal edge just above the knob row
/// 2. The wall edge: vertical brightness transition right of the last knob
///
/// Their intersection is the top-right corner. This corner is stored in
/// state for the subsequent perspective refinement stage.
pub struct FindCorner;

impl FindCorner {
    pub fn new() -> Self {
        Self
    }
}

impl Stage for FindCorner {
    fn descriptor(&self) -> StageDescriptor {
        DESCRIPTOR
    }

    fn run(
        &self,
        state: &mut PipelineState,
        src: &Mat,
        _dst: &mut Mat,
        _raw: &Mat,
        _iteration: u32,
    ) -> Result<(StageOutcome, ImageOutput), opencv::Error> {
        let Some(features) = &state.features else {
            return Ok((
                StageOutcome::Exhausted("no features from S8".into()),
                ImageOutput::Passthrough,
            ));
        };

        if features.knobs.is_empty() {
            return Ok((
                StageOutcome::Exhausted("no knobs detected".into()),
                ImageOutput::Passthrough,
            ));
        }

        let img_w = src.cols() as f64;
        let img_h = src.rows() as f64;

        // Find the rightmost knob and the knob row Y
        let rightmost = features
            .knobs
            .iter()
            .max_by(|a, b| {
                a.center_x
                    .partial_cmp(&b.center_x)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        let mut ys: Vec<f64> = features.knobs.iter().map(|k| k.center_y).collect();
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_y = ys[ys.len() / 2];
        let median_r = {
            let mut rs: Vec<f64> = features.knobs.iter().map(|k| k.radius).collect();
            rs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            rs[rs.len() / 2]
        };

        // --- Find the chrome bar Y: horizontal edge above the knob row ---
        // Search in a band above the knob centers
        let bar_search_top = (median_y - median_r * BAR_SEARCH_ABOVE).max(0.0) as i32;
        let bar_search_bottom = (median_y - median_r * 0.5).max(0.0) as i32;
        if bar_search_bottom <= bar_search_top {
            return Ok((
                StageOutcome::Retry("bar search band too narrow".into()),
                ImageOutput::Passthrough,
            ));
        }

        let mut gray = Mat::default();
        imgproc::cvt_color_def(src, &mut gray, imgproc::COLOR_BGR2GRAY)?;

        // Sobel-Y to find horizontal edges
        let mut sobel_y = Mat::default();
        imgproc::sobel(
            &gray,
            &mut sobel_y,
            CV_16S,
            0,
            1,
            3,
            1.0,
            0.0,
            opencv::core::BORDER_DEFAULT,
        )?;

        // Row-wise mean of Sobel-Y in the search band
        let band_h = bar_search_bottom - bar_search_top;
        let mut row_profile = vec![0.0f64; band_h as usize];
        for y in 0..band_h {
            let abs_y = bar_search_top + y;
            let mut sum = 0.0f64;
            let w = sobel_y.cols();
            for x in 0..w {
                sum += (*sobel_y.at_2d::<i16>(abs_y, x)? as f64).abs();
            }
            row_profile[y as usize] = sum / w as f64;
        }

        // Find the row with the strongest horizontal edge = the chrome bar
        let bar_row = row_profile
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let bar_y = (bar_search_top + bar_row as i32) as f64;

        // --- Find the wall edge X: vertical edge right of the last knob ---
        // Use Sobel-X column profiling on a narrow horizontal band around bar_y
        let wall_search_left = (rightmost.center_x + median_r).min(img_w) as i32;
        let wall_search_right = (rightmost.center_x + median_r * WALL_SEARCH_RIGHT)
            .min(img_w) as i32;

        if wall_search_right <= wall_search_left + 5 {
            return Ok((
                StageOutcome::Retry("wall search band too narrow".into()),
                ImageOutput::Passthrough,
            ));
        }

        let mut sobel_x = Mat::default();
        imgproc::sobel(
            &gray,
            &mut sobel_x,
            CV_16S,
            1,
            0,
            3,
            1.0,
            0.0,
            opencv::core::BORDER_DEFAULT,
        )?;

        // Column-wise mean of Sobel-X in the wall search region
        // Look for a strong positive gradient (dark stove → bright wall)
        let search_band_top = (bar_y - median_r).max(0.0) as i32;
        let search_band_bottom = (median_y + median_r).min(img_h) as i32;
        let band_height = search_band_bottom - search_band_top;

        let mut col_profile = vec![0.0f64; (wall_search_right - wall_search_left) as usize];
        for cx in 0..col_profile.len() {
            let abs_x = wall_search_left + cx as i32;
            let mut sum = 0.0f64;
            for y in search_band_top..search_band_bottom {
                sum += *sobel_x.at_2d::<i16>(y, abs_x)? as f64;
            }
            col_profile[cx] = sum / band_height as f64;
        }

        // Strongest positive gradient = dark→light = stove→wall edge
        let wall_col = col_profile
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let wall_x = (wall_search_left + wall_col as i32) as f64;

        // The corner is at (wall_x, bar_y)
        // Store in knob_search for the refinement stage
        if let Some(ks) = state.knob_search.as_mut() {
            ks.corner_x = Some(wall_x);
            ks.corner_y = Some(bar_y);
        }

        Ok((StageOutcome::Success, ImageOutput::Passthrough))
    }

    fn max_retries(&self) -> u32 {
        5
    }

    fn debug_image(
        &self,
        state: &PipelineState,
        working: &Mat,
        _raw: &Mat,
    ) -> Result<Option<DebugImage>, opencv::Error> {
        let Some(ks) = &state.knob_search else {
            return Ok(None);
        };
        let (Some(cx), Some(cy)) = (ks.corner_x, ks.corner_y) else {
            return Ok(None);
        };

        let mut canvas = working.try_clone()?;

        // Draw crosshair at the detected corner
        let red = Scalar::new(0.0, 0.0, 255.0, 0.0);
        let x = cx as i32;
        let y = cy as i32;
        imgproc::line(
            &mut canvas,
            Point::new(x - 20, y),
            Point::new(x + 20, y),
            red,
            2,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::line(
            &mut canvas,
            Point::new(x, y - 20),
            Point::new(x, y + 20),
            red,
            2,
            imgproc::LINE_8,
            0,
        )?;

        let jpeg = encode_jpeg(&canvas, 90)?;
        Ok(Some((format!("S10:FindCorner_{x}_{y}"), jpeg)))
    }
}
