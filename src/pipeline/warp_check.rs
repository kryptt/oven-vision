use std::f64::consts::PI;

use opencv::core::{Mat, Size, Vec2f, Vector};
use opencv::imgproc;
use opencv::prelude::*;

use super::stage::{PipelineState, StageDescriptor, StageOutcome};
use super::{ImageOutput, Stage};

/// Maximum allowed mean angle deviation (degrees) from horizontal for the
/// top-3 strongest horizontal lines in the warped image.
const MAX_MEAN_SLOPE_DEG: f64 = 2.0;

/// Angular tolerance (degrees) for considering a line "near-horizontal".
const HORIZONTAL_TOLERANCE_DEG: f64 = 10.0;

/// Stage 5: Validate that the warped image has horizontal lines that are
/// actually horizontal. This catches bad perspective corrections early,
/// before we waste time on band extraction and template matching.
///
/// Reads the warped image from `src` directly (no re-warp). Converts to
/// gray, applies CLAHE(3.0) + median_blur, then Canny(50,150), then
/// HoughLines(thresh=150). Filters for near-horizontal lines (within 10
/// degrees) and checks that the mean angle of the top-3 strongest is
/// within 2 degrees of horizontal.
pub struct WarpCheck;

impl WarpCheck {
    pub fn new() -> Self {
        Self
    }
}

pub(crate) const DESCRIPTOR: StageDescriptor = StageDescriptor {
    name: "WarpCheck",
    label: "S5:WarpCheck",
    fallback: Some("FindVerticals"),
};

impl Stage for WarpCheck {
    fn descriptor(&self) -> StageDescriptor {
        DESCRIPTOR
    }

    fn run(
        &self,
        _state: &mut PipelineState,
        src: &Mat,
        _dst: &mut Mat,
        _raw: &Mat,
        _iteration: u32,
    ) -> Result<(StageOutcome, ImageOutput), opencv::Error> {
        if src.empty() {
            return Ok((
                StageOutcome::Exhausted("empty warped image".into()),
                ImageOutput::Passthrough,
            ));
        }

        // Convert to grayscale
        let mut gray = Mat::default();
        imgproc::cvt_color_def(src, &mut gray, imgproc::COLOR_BGR2GRAY)?;

        // CLAHE for contrast enhancement
        let mut enhanced = Mat::default();
        let mut clahe = imgproc::create_clahe(3.0, Size::new(8, 8))?;
        clahe.apply(&gray, &mut enhanced)?;

        // Median blur to reduce noise
        let mut blurred = Mat::default();
        imgproc::median_blur(&enhanced, &mut blurred, 5)?;

        // Edge detection
        let mut edges = Mat::default();
        imgproc::canny(&blurred, &mut edges, 50.0, 150.0, 3, false)?;

        // Find lines
        let mut lines = Vector::<Vec2f>::new();
        imgproc::hough_lines_def(&edges, &mut lines, 1.0, PI / 180.0, 150)?;

        if lines.is_empty() {
            // No strong lines detected -- can't validate, pass cautiously
            return Ok((StageOutcome::Success, ImageOutput::Passthrough));
        }

        // Filter for near-horizontal lines (theta near PI/2)
        let horizontal_center = PI / 2.0;
        let tolerance_rad = HORIZONTAL_TOLERANCE_DEG * PI / 180.0;

        let mut horizontal_lines: Vec<(f64, f64)> = Vec::new(); // (rho, theta)
        for v in &lines {
            let theta = v[1] as f64;
            if (theta - horizontal_center).abs() <= tolerance_rad {
                horizontal_lines.push((v[0] as f64, theta));
            }
        }

        if horizontal_lines.is_empty() {
            return Ok((
                StageOutcome::Exhausted("no near-horizontal lines found in warped image".into()),
                ImageOutput::Passthrough,
            ));
        }

        // Take the top-3 strongest (HoughLines returns sorted by votes)
        let top_n = horizontal_lines.len().min(3);
        let top_lines = &horizontal_lines[..top_n];

        // Compute mean angle deviation from perfectly horizontal
        let mean_slope: f64 = top_lines
            .iter()
            .map(|(_rho, theta)| (theta - horizontal_center).abs() * 180.0 / PI)
            .sum::<f64>()
            / top_n as f64;

        if mean_slope > MAX_MEAN_SLOPE_DEG {
            return Ok((
                StageOutcome::Exhausted(format!(
                    "mean horizontal slope = {mean_slope:.1} deg (max {MAX_MEAN_SLOPE_DEG} deg)"
                )),
                ImageOutput::Passthrough,
            ));
        }

        Ok((StageOutcome::Success, ImageOutput::Passthrough))
    }

    fn max_retries(&self) -> u32 {
        1 // Pass/fail -- no parameter variation
    }
}
