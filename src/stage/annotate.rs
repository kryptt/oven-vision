use opencv::core::{Mat, Point, Scalar};
use opencv::imgproc;
use opencv::prelude::*;

use super::{FrameState, Stage, StageError};

pub struct Annotate;

impl Stage for Annotate {
    fn name(&self) -> &'static str { "annotate" }

    fn process(&self, state: &mut FrameState) -> Result<(), StageError> {
        let warped = state.warped.as_ref()
            .ok_or(StageError { stage: self.name(), message: "no warped frame".into() })?;
        let mut out = warped.clone();
        let h = out.rows();

        let pass = state.sanity.as_ref().map_or(false, |s| s.ok);
        let green = Scalar::new(0.0, 255.0, 0.0, 0.0);
        let orange = Scalar::new(0.0, 128.0, 255.0, 0.0);
        let red = Scalar::new(0.0, 0.0, 255.0, 0.0);

        for knob in &state.knobs {
            let center = Point::new(knob.x as i32, knob.y as i32);
            let r = knob.radius as i32;

            if knob.synthetic {
                // Synthetic: red dashed (draw as thin dotted circle)
                imgproc::circle(&mut out, center, r, red, 1, imgproc::LINE_AA, 0)
                    .map_err(|e| StageError { stage: self.name(), message: e.to_string() })?;
                let label = format!("?{}", knob.slot + 1);
                imgproc::put_text(
                    &mut out, &label,
                    Point::new(knob.x as i32 - 10, knob.y as i32 - r - 6),
                    imgproc::FONT_HERSHEY_SIMPLEX, 0.4, red, 1, imgproc::LINE_AA, false,
                ).map_err(|e| StageError { stage: self.name(), message: e.to_string() })?;
            } else {
                // Real: green/orange solid circle
                let color = if pass { green } else { orange };
                imgproc::circle(&mut out, center, r, color, 2, imgproc::LINE_AA, 0)
                    .map_err(|e| StageError { stage: self.name(), message: e.to_string() })?;
                imgproc::circle(&mut out, center, 3, color, -1, imgproc::LINE_AA, 0)
                    .map_err(|e| StageError { stage: self.name(), message: e.to_string() })?;
                let label = format!("{}", knob.slot + 1);
                imgproc::put_text(
                    &mut out, &label,
                    Point::new(knob.x as i32 - 6, knob.y as i32 - r - 6),
                    imgproc::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, imgproc::LINE_AA, false,
                ).map_err(|e| StageError { stage: self.name(), message: e.to_string() })?;
            }
        }

        // Status line
        let sanity = state.sanity.as_ref();
        let syn = sanity.map_or(0, |s| s.synthetic_count);
        let status = format!(
            "{}/{} real | {} | y:{} sz:{} sp:{}",
            state.knobs.iter().filter(|k| !k.synthetic).count(),
            state.knobs.len(),
            if pass { "PASS" } else { "FAIL" },
            sanity.map_or("?", |s| if s.y_aligned { "ok" } else { "FAIL" }),
            sanity.map_or("?", |s| if s.size_uniform { "ok" } else { "FAIL" }),
            sanity.map_or("?", |s| if s.spacing_uniform { "ok" } else { "FAIL" }),
        );
        let color = if pass { green } else { orange };
        imgproc::put_text(
            &mut out, &status,
            Point::new(10, h - 10),
            imgproc::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, imgproc::LINE_AA, false,
        ).map_err(|e| StageError { stage: self.name(), message: e.to_string() })?;

        state.annotated = Some(out);
        Ok(())
    }
}
