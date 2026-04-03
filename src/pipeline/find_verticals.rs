use std::f64::consts::PI;

use opencv::core::{CV_16S, Mat, Point, Rect, Scalar, Size, Vec4i, Vector};
use opencv::imgproc;
use opencv::prelude::*;

use super::stage::{Line, PipelineState, StageDescriptor, StageOutcome, VerticalPair};
use super::util::draw_line;
use super::{DebugImage, ImageOutput, Stage};
use crate::annotate::encode_jpeg;

/// Minimum horizontal separation (pixels) between left and right boundaries.
const MIN_SEPARATION: f64 = 150.0;

/// How far into the image to search for the left boundary (fraction of width).
const LEFT_SEARCH_FRAC: f64 = 0.40;

/// How far from the right edge to start searching for the right boundary.
const RIGHT_SEARCH_FRAC: f64 = 0.60;

/// Vertical band of the image to use for column profiling (skip occluded
/// top/bottom). The iteration varies these to explore different vertical slices.
const BAND_TOP_FRAC: f64 = 0.15;
const BAND_BOTTOM_FRAC: f64 = 0.85;

/// Minimum absolute gradient strength for a peak to be accepted.
const MIN_GRADIENT_STRENGTH: f64 = 5.0;

/// Stage S3: Detect the left and right boundaries of the stove panel using
/// Sobel-X column profiling with gradient polarity.
///
/// The left boundary is cream cabinet → dark stove (negative horizontal gradient).
/// The right boundary is dark stove → white wall (positive horizontal gradient).
///
/// Uses directional gradient (Sobel-X) instead of Canny+HoughLines because
/// the boundaries are material transitions (not geometric lines) that may be
/// partially occluded.
///
/// On iteration, varies the vertical band used for column profiling and the
/// bilateral filter parameters to find boundaries under different conditions.
pub struct FindVerticals;

impl FindVerticals {
    pub fn new() -> Self {
        Self
    }
}

pub(crate) const DESCRIPTOR: StageDescriptor = StageDescriptor {
    name: "FindVerticals",
    label: "S3:FindVerticals",
    fallback: Some("FindLines"),
};

impl Stage for FindVerticals {
    fn descriptor(&self) -> StageDescriptor {
        DESCRIPTOR
    }

    fn run(
        &self,
        state: &mut PipelineState,
        src: &Mat,
        _dst: &mut Mat,
        _raw: &Mat,
        iteration: u32,
    ) -> Result<(StageOutcome, ImageOutput), opencv::Error> {
        let Some(crop) = &state.crop else {
            return Ok((
                StageOutcome::Exhausted("no crop region from Stage 1".into()),
                ImageOutput::Passthrough,
            ));
        };

        let w = src.cols();
        let h = src.rows();
        if w < 50 || h < 50 {
            return Ok((
                StageOutcome::Exhausted("source image too small".into()),
                ImageOutput::Passthrough,
            ));
        }

        // Convert to grayscale
        let mut gray = Mat::default();
        imgproc::cvt_color_def(src, &mut gray, imgproc::COLOR_BGR2GRAY)?;

        // Bilateral filter: preserves material boundaries while smoothing texture.
        // Vary sigma on iteration for robustness.
        let sigma = 75.0 + (iteration as f64 * 5.0).min(50.0);
        let mut smooth = Mat::default();
        imgproc::bilateral_filter(
            &gray,
            &mut smooth,
            9,
            sigma,
            sigma,
            opencv::core::BORDER_DEFAULT,
        )?;

        // Sobel in X direction only — finds vertical edges, ignores horizontal.
        // Signed 16-bit to preserve gradient polarity.
        let mut sobel_x = Mat::default();
        imgproc::sobel(
            &smooth,
            &mut sobel_x,
            CV_16S,
            1,
            0,
            3,
            1.0,
            0.0,
            opencv::core::BORDER_DEFAULT,
        )?;

        // Vary the vertical band used for column profiling on each iteration
        let band_shift = (iteration as f64 * 0.03).min(0.15);
        let y_start = ((BAND_TOP_FRAC + band_shift) * h as f64) as i32;
        let y_end = ((BAND_BOTTOM_FRAC - band_shift) * h as f64) as i32;
        if y_end <= y_start + 10 {
            return Ok((
                StageOutcome::Retry("vertical band too narrow".into()),
                ImageOutput::Passthrough,
            ));
        }

        let roi = Rect::new(0, y_start, w, y_end - y_start);
        let sobel_roi = Mat::roi(&sobel_x, roi)?;

        // Column-wise mean of Sobel-X: 1D gradient profile
        let mut column_profile = vec![0.0f64; w as usize];
        for x in 0..w {
            let mut sum = 0.0f64;
            let roi_h = sobel_roi.rows();
            for y in 0..roi_h {
                sum += *sobel_roi.at_2d::<i16>(y, x)? as f64;
            }
            column_profile[x as usize] = sum / roi_h as f64;
        }

        // LEFT edge: cream (bright) → dark stove = NEGATIVE gradient peak
        let left_search_end = (w as f64 * LEFT_SEARCH_FRAC) as usize;
        let left_result = column_profile[..left_search_end]
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let Some((left_x, &left_val)) = left_result else {
            return Ok((
                StageOutcome::Retry("no left boundary candidate".into()),
                ImageOutput::Passthrough,
            ));
        };

        if left_val.abs() < MIN_GRADIENT_STRENGTH {
            return Ok((
                StageOutcome::Retry(format!(
                    "left gradient too weak: {left_val:.1} (need {MIN_GRADIENT_STRENGTH})"
                )),
                ImageOutput::Passthrough,
            ));
        }

        // RIGHT edge: dark stove → white wall = POSITIVE gradient peak
        let right_search_start = (w as f64 * RIGHT_SEARCH_FRAC) as usize;
        let right_result = column_profile[right_search_start..]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let Some((right_offset, &right_val)) = right_result else {
            return Ok((
                StageOutcome::Retry("no right boundary candidate".into()),
                ImageOutput::Passthrough,
            ));
        };
        let right_x = right_search_start + right_offset;

        if right_val.abs() < MIN_GRADIENT_STRENGTH {
            return Ok((
                StageOutcome::Retry(format!(
                    "right gradient too weak: {right_val:.1} (need {MIN_GRADIENT_STRENGTH})"
                )),
                ImageOutput::Passthrough,
            ));
        }

        // Check separation
        let separation = right_x as f64 - left_x as f64;
        if separation < MIN_SEPARATION {
            return Ok((
                StageOutcome::Retry(format!(
                    "boundary separation {separation:.0}px < {MIN_SEPARATION}"
                )),
                ImageOutput::Passthrough,
            ));
        }

        // Optional: refine slope with HoughLinesP on narrow strips around each boundary
        let left_line = refine_boundary_slope(&sobel_x, left_x as i32, w, h, true)?;
        let right_line = refine_boundary_slope(&sobel_x, right_x as i32, w, h, false)?;

        state.verticals = Some(VerticalPair {
            left: left_line,
            right: right_line,
        });

        Ok((StageOutcome::Success, ImageOutput::Passthrough))
    }

    fn max_retries(&self) -> u32 {
        20
    }

    fn debug_image(
        &self,
        state: &PipelineState,
        working: &Mat,
        _raw: &Mat,
    ) -> Result<Option<DebugImage>, opencv::Error> {
        let Some(verticals) = &state.verticals else {
            return Ok(None);
        };

        let mut canvas = working.try_clone()?;

        draw_line(
            &mut canvas,
            &verticals.left,
            Scalar::new(255.0, 0.0, 255.0, 0.0), // magenta
        )?;
        draw_line(
            &mut canvas,
            &verticals.right,
            Scalar::new(255.0, 255.0, 0.0, 0.0), // cyan
        )?;

        let jpeg = encode_jpeg(&canvas, 80)?;
        let label = format!(
            "S3:FindVerticals_L{:.0}_R{:.0}",
            verticals.left.x1, verticals.right.x1
        );
        Ok(Some((label, jpeg)))
    }
}

/// Refine the slope of a detected boundary using HoughLinesP on a narrow
/// vertical strip around the detected X column.
///
/// If HoughLinesP finds a segment, returns its endpoints as a Line.
/// Otherwise, returns a vertical line at the detected X.
fn refine_boundary_slope(
    sobel_x: &Mat,
    center_x: i32,
    img_w: i32,
    img_h: i32,
    is_negative: bool,
) -> Result<Line, opencv::Error> {
    let strip_half = 25;
    let x0 = (center_x - strip_half).max(0);
    let x1 = (center_x + strip_half).min(img_w);
    let strip_w = x1 - x0;

    if strip_w < 10 {
        // Fallback: vertical line
        return Ok(Line {
            x1: center_x as f64,
            y1: 0.0,
            x2: center_x as f64,
            y2: img_h as f64,
        });
    }

    let strip_roi = Rect::new(x0, 0, strip_w, img_h);
    let strip = Mat::roi(sobel_x, strip_roi)?;

    // Convert to absolute 8-bit
    let mut abs_strip = Mat::default();
    opencv::core::convert_scale_abs(&strip, &mut abs_strip, 1.0, 0.0)?;

    // Threshold to keep only strong gradients
    let mut thresh = Mat::default();
    imgproc::threshold(&abs_strip, &mut thresh, 30.0, 255.0, imgproc::THRESH_BINARY)?;

    // HoughLinesP on the narrow strip
    let mut lines = Vector::<Vec4i>::new();
    imgproc::hough_lines_p(
        &thresh,
        &mut lines,
        1.0,
        PI / 180.0,
        20,   // threshold (votes)
        30.0, // min line length
        15.0, // max line gap
    )?;

    // Filter to positive-slope segments only (x increases as y increases,
    // matching the known camera angle where vertical edges tilt right going down).
    // Ensure the segment goes top-to-bottom by normalizing so y1 < y2.
    let positive_slope: Vec<Vec4i> = lines
        .iter()
        .map(|l| {
            if l[1] <= l[3] {
                l // already top-to-bottom
            } else {
                Vec4i::from([l[2], l[3], l[0], l[1]]) // flip
            }
        })
        .filter(|l| l[2] >= l[0]) // x2 >= x1 = positive slope (right-leaning)
        .collect();

    if positive_slope.is_empty() {
        return Ok(Line {
            x1: center_x as f64,
            y1: 0.0,
            x2: center_x as f64,
            y2: img_h as f64,
        });
    }

    // Pick the longest segment
    let mut best_len = 0.0f64;
    let mut best_line = positive_slope[0];
    for l in &positive_slope {
        let dx = (l[2] - l[0]) as f64;
        let dy = (l[3] - l[1]) as f64;
        let len = (dx * dx + dy * dy).sqrt();
        if len > best_len {
            best_len = len;
            best_line = *l;
        }
    }

    // Convert strip-local coordinates back to image coordinates
    Ok(Line {
        x1: (best_line[0] + x0) as f64,
        y1: best_line[1] as f64,
        x2: (best_line[2] + x0) as f64,
        y2: best_line[3] as f64,
    })
}
