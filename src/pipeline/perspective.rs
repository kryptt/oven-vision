use opencv::core::{Mat, Point, Point2f, Rect, Scalar, Size};
use opencv::imgproc;
use opencv::prelude::*;

use super::stage::{
    Line, PerspectiveCorrection, PipelineState, StageDescriptor, StageOutcome, TransformMatrix,
};
use super::{DebugImage, Stage};
use crate::annotate::encode_jpeg;

/// Padding above the top line, as a fraction of the inter-line distance.
const TOP_PADDING_FRAC: f64 = 0.5;

/// Padding below the bottom line, as a fraction of the inter-line distance.
/// The knobs sit 6-8x the band height below the bottom chrome trim.
const BOTTOM_PADDING_FRAC: f64 = 8.0;

/// Per-retry pixel jitter applied to source points to explore slight
/// variations in the perspective transform.
const JITTER_PX: f64 = 1.5;

/// Maximum total jitter (pixels). Beyond this, source points are jittered
/// past the reference lines and produce garbage transforms.
const MAX_JITTER_PX: f64 = 22.5;

/// Stage 3: Compute and apply a perspective transform from the 4 corner
/// points formed by the intersection of horizontal lines (S2) and vertical
/// lines (S2b).
///
/// The 4 corners define the full trapezoid of the stove panel as seen by the
/// camera. Mapping them to a rectangle corrects both vertical tilt AND
/// horizontal convergence, producing equal-sized knobs across the row.
pub struct Perspective;

impl Perspective {
    pub fn new() -> Self {
        Self
    }
}

pub(crate) const DESCRIPTOR: StageDescriptor = StageDescriptor {
    name: "Perspective",
    label: "S3:Perspective",
    fallback: Some("FindVerticals"),
};

impl Stage for Perspective {
    fn descriptor(&self) -> StageDescriptor {
        DESCRIPTOR
    }

    fn run(
        &self,
        state: &mut PipelineState,
        frame: &Mat,
        iteration: u32,
    ) -> Result<StageOutcome, opencv::Error> {
        let (Some(crop), Some(lines), Some(verts)) = (&state.crop, &state.lines, &state.verticals)
        else {
            return Ok(StageOutcome::Exhausted(
                "missing crop, lines, or verticals from previous stages".into(),
            ));
        };

        // Compute jitter for this iteration (0 on first attempt)
        let jitter = if iteration == 0 {
            0.0
        } else {
            let sign = if iteration % 2 == 0 { 1.0 } else { -1.0 };
            let magnitude = (((iteration + 1) / 2) as f64 * JITTER_PX).min(MAX_JITTER_PX);
            sign * magnitude
        };

        // 4 source corners: intersections of horizontal and vertical lines.
        // Each intersection is computed from one horizontal + one vertical line.
        let tl = line_intersection(&lines.top, &verts.left);
        let tr = line_intersection(&lines.top, &verts.right);
        let bl = line_intersection(&lines.bottom, &verts.left);
        let br = line_intersection(&lines.bottom, &verts.right);

        let (Some(tl), Some(tr), Some(bl), Some(br)) = (tl, tr, bl, br) else {
            return Ok(StageOutcome::Retry(
                "horizontal and vertical lines are parallel — no intersection".into(),
            ));
        };

        let src_pts: [Point2f; 4] = [
            Point2f::new(tl.0 as f32, (tl.1 + jitter) as f32),
            Point2f::new(tr.0 as f32, (tr.1 + jitter) as f32),
            Point2f::new(br.0 as f32, (br.1 - jitter) as f32),
            Point2f::new(bl.0 as f32, (bl.1 - jitter) as f32),
        ];

        // Output uses full crop width — the verticals define the perspective
        // correction but NOT the output extent. This way even if the verticals
        // detect an internal edge (oven door divider), the full panel is preserved.
        let out_w = crop.width as i32;

        // Inter-line distance (average of left and right side distances)
        let left_dist = ((bl.0 - tl.0).powi(2) + (bl.1 - tl.1).powi(2)).sqrt();
        let right_dist = ((br.0 - tr.0).powi(2) + (br.1 - tr.1).powi(2)).sqrt();
        let inter_line = (left_dist + right_dist) / 2.0;

        let top_pad = inter_line * TOP_PADDING_FRAC;
        let bot_pad = inter_line * BOTTOM_PADDING_FRAC;
        let out_h = (top_pad + inter_line + bot_pad) as i32;

        if out_w <= 0 || out_h <= 0 {
            return Ok(StageOutcome::Retry(format!(
                "invalid output dimensions: {out_w}x{out_h}"
            )));
        }

        // Destination: the panel corners map to a rectangle at their x positions
        // within the full-width output. The warp extrapolates to fill the rest.
        let left_x = ((tl.0 + bl.0) / 2.0) as f32;
        let right_x = ((tr.0 + br.0) / 2.0) as f32;
        let dst_pts: [Point2f; 4] = [
            Point2f::new(left_x, top_pad as f32),                 // TL
            Point2f::new(right_x, top_pad as f32),                // TR
            Point2f::new(right_x, (top_pad + inter_line) as f32), // BR
            Point2f::new(left_x, (top_pad + inter_line) as f32),  // BL
        ];

        let mat = imgproc::get_perspective_transform_slice_def(&src_pts, &dst_pts)?;

        // Extract the cropped region and warp it
        let roi_rect = Rect::new(
            crop.x as i32,
            crop.y as i32,
            crop.width as i32,
            crop.height as i32,
        );
        let cropped = Mat::roi(frame, roi_rect)?;

        let mut warped = Mat::default();
        imgproc::warp_perspective_def(&cropped, &mut warped, &mat, Size::new(out_w, out_h))?;

        if warped.empty() {
            return Ok(StageOutcome::Retry("warp produced empty image".into()));
        }

        let matrix = mat_to_transform(&mat)?;

        state.perspective = Some(PerspectiveCorrection {
            matrix,
            output_width: out_w as u32,
            output_height: out_h as u32,
        });

        Ok(StageOutcome::Success)
    }

    fn max_retries(&self) -> u32 {
        30
    }

    fn debug_image(
        &self,
        state: &PipelineState,
        frame: &Mat,
    ) -> Result<Option<DebugImage>, opencv::Error> {
        let (Some(crop), Some(lines), Some(verts), Some(persp)) = (
            &state.crop,
            &state.lines,
            &state.verticals,
            &state.perspective,
        ) else {
            return Ok(None);
        };

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

        // Draw horizontal guide lines where the trim should now be level
        let w = persp.output_width as f64;
        let left_dist = {
            let tl = line_intersection(&lines.top, &verts.left);
            let bl = line_intersection(&lines.bottom, &verts.left);
            match (tl, bl) {
                (Some(t), Some(b)) => ((b.0 - t.0).powi(2) + (b.1 - t.1).powi(2)).sqrt(),
                _ => (lines.bottom.y1 - lines.top.y1).abs(),
            }
        };
        let right_dist = {
            let tr = line_intersection(&lines.top, &verts.right);
            let br = line_intersection(&lines.bottom, &verts.right);
            match (tr, br) {
                (Some(t), Some(b)) => ((b.0 - t.0).powi(2) + (b.1 - t.1).powi(2)).sqrt(),
                _ => (lines.bottom.y2 - lines.top.y2).abs(),
            }
        };
        let inter_line = (left_dist + right_dist) / 2.0;
        let top_pad = inter_line * TOP_PADDING_FRAC;
        let y_top = top_pad as i32;
        let y_bot = (top_pad + inter_line) as i32;

        let green = Scalar::new(0.0, 255.0, 0.0, 0.0);
        imgproc::line(
            &mut warped,
            Point::new(0, y_top),
            Point::new(w as i32, y_top),
            green,
            1,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::line(
            &mut warped,
            Point::new(0, y_bot),
            Point::new(w as i32, y_bot),
            green,
            1,
            imgproc::LINE_8,
            0,
        )?;

        let jpeg = encode_jpeg(&warped, 80)?;
        let label = format!(
            "S3:Perspective_{}x{}",
            persp.output_width, persp.output_height
        );
        Ok(Some((label, jpeg)))
    }
}

/// Compute the intersection of two lines given in endpoint form.
///
/// Returns `Some((x, y))` or `None` if the lines are parallel.
fn line_intersection(a: &Line, b: &Line) -> Option<(f64, f64)> {
    let x1 = a.x1;
    let y1 = a.y1;
    let x2 = a.x2;
    let y2 = a.y2;
    let x3 = b.x1;
    let y3 = b.y1;
    let x4 = b.x2;
    let y4 = b.y2;

    let denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if denom.abs() < 1e-10 {
        return None;
    }

    let t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;

    let x = x1 + t * (x2 - x1);
    let y = y1 + t * (y2 - y1);

    Some((x, y))
}

/// Extract a 3x3 f64 Mat into our serializable TransformMatrix.
fn mat_to_transform(mat: &Mat) -> Result<TransformMatrix, opencv::Error> {
    let mut rows = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            rows[r][c] = *mat.at_2d::<f64>(r as i32, c as i32)?;
        }
    }
    Ok(TransformMatrix(rows))
}

/// Reconstruct a 3x3 f64 Mat from our TransformMatrix.
pub fn transform_to_mat(tm: &TransformMatrix) -> Result<Mat, opencv::Error> {
    let mut mat = Mat::zeros(3, 3, opencv::core::CV_64FC1)?.to_mat()?;
    for r in 0..3 {
        for c in 0..3 {
            *mat.at_2d_mut::<f64>(r as i32, c as i32)? = tm.0[r][c];
        }
    }
    Ok(mat)
}
