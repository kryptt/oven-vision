use opencv::core::{Mat, Rect, Scalar, Size, CV_8UC1};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;

use crate::config::KnobDetection;
use super::{FrameState, Stage, StageError};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Confidence {
    High,
    Low,
    Ambiguous,
}

#[derive(Debug, Clone)]
pub struct AngleReading {
    pub angle_deg: f32,
    pub confidence: Confidence,
    pub match_score: f32,
}

/// Template for a known knob angle.
struct Template {
    angle_deg: f32,
    image: Mat,
}

pub struct Angle {
    templates: Vec<Template>,
    centers_x: Vec<f32>,
    center_y: f32,
    crop_size: i32,
}

impl Angle {
    pub fn new(kd: &KnobDetection, template_dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut templates = Vec::new();

        // Load template images from disk
        for angle in [0, 30, 60, 90, 120, 150] {
            let path = format!("{}/template_{}deg.jpg", template_dir, angle);
            let img = imgcodecs::imread(&path, imgcodecs::IMREAD_GRAYSCALE)?;
            if img.empty() {
                eprintln!("  angle: warning: template {} not found", path);
                continue;
            }
            templates.push(Template {
                angle_deg: angle as f32,
                image: img,
            });
        }

        eprintln!("  angle: loaded {} templates", templates.len());

        Ok(Self {
            templates,
            centers_x: kd.knob_centers_x.clone(),
            center_y: kd.prior_y,
            crop_size: kd.knob_crop_size,
        })
    }
}

impl Stage for Angle {
    fn name(&self) -> &'static str { "angle" }

    fn process(&self, state: &mut FrameState) -> Result<(), StageError> {
        if self.templates.is_empty() {
            state.angle_readings = vec![None; state.knobs.len()];
            return Ok(());
        }

        // Get the binary mask (debug_edges) for template matching
        let mask = state.debug_edges.as_ref()
            .ok_or(StageError { stage: self.name(), message: "no binary mask".into() })?;

        let img_w = mask.cols();
        let img_h = mask.rows();
        let half = self.crop_size / 2;

        let mut readings: Vec<Option<AngleReading>> = Vec::new();

        for (i, knob) in state.knobs.iter().enumerate() {
            if knob.synthetic {
                readings.push(None);
                continue;
            }

            // Use calibrated center X if available, otherwise use detected X
            let cx = if i < self.centers_x.len() {
                self.centers_x[i] as i32
            } else {
                knob.x as i32
            };
            let cy = self.center_y as i32;

            // Crop region
            let x0 = (cx - half).max(0);
            let y0 = (cy - half).max(0);
            let x1 = (cx + half).min(img_w);
            let y1 = (cy + half).min(img_h);
            let w = x1 - x0;
            let h = y1 - y0;

            if w < self.crop_size / 2 || h < self.crop_size / 2 {
                readings.push(None);
                continue;
            }

            let roi = Rect::new(x0, y0, w, h);
            let crop = Mat::roi(mask, roi)
                .map_err(|e| StageError { stage: self.name(), message: e.to_string() })?;

            // Resize crop to match template size if needed (edge knobs may be smaller)
            let mut sized_crop = Mat::default();
            if crop.cols() != self.crop_size || crop.rows() != self.crop_size {
                // Pad with black to full size
                sized_crop = Mat::new_rows_cols_with_default(
                    self.crop_size, self.crop_size, CV_8UC1, Scalar::all(0.0),
                ).map_err(|e| StageError { stage: self.name(), message: e.to_string() })?;
                let paste_roi = Rect::new(0, 0, w.min(self.crop_size), h.min(self.crop_size));
                let mut paste_region = Mat::roi_mut(&mut sized_crop, paste_roi)
                    .map_err(|e| StageError { stage: self.name(), message: e.to_string() })?;
                let src_roi = Rect::new(0, 0, w.min(self.crop_size), h.min(self.crop_size));
                let src_region = Mat::roi(&crop, src_roi)
                    .map_err(|e| StageError { stage: self.name(), message: e.to_string() })?;
                src_region.copy_to(&mut paste_region)
                    .map_err(|e| StageError { stage: self.name(), message: e.to_string() })?;
            } else {
                sized_crop = crop.try_clone()
                    .map_err(|e| StageError { stage: self.name(), message: e.to_string() })?;
            }

            // Match against all templates
            let mut best_angle = 0.0f32;
            let mut best_score = -1.0f32;
            let mut second_score = -1.0f32;

            for tmpl in &self.templates {
                // Direct comparison: both are same size, so matchTemplate gives 1x1 result
                let mut result = Mat::default();
                imgproc::match_template(
                    &sized_crop, &tmpl.image, &mut result,
                    imgproc::TM_CCOEFF_NORMED,
                    &Mat::default(),
                ).map_err(|e| StageError { stage: self.name(), message: e.to_string() })?;

                let score = *result.at_2d::<f32>(0, 0)
                    .map_err(|e| StageError { stage: self.name(), message: e.to_string() })?;

                if score > best_score {
                    second_score = best_score;
                    best_score = score;
                    best_angle = tmpl.angle_deg;
                } else if score > second_score {
                    second_score = score;
                }
            }

            let gap = best_score - second_score;
            let confidence = if best_score > 0.7 && gap > 0.1 {
                Confidence::High
            } else if best_score > 0.4 {
                Confidence::Low
            } else {
                Confidence::Ambiguous
            };

            eprintln!(
                "    knob {}: best={}° score={:.2} gap={:.2} {:?}",
                i + 1, best_angle, best_score, gap, confidence
            );

            readings.push(Some(AngleReading {
                angle_deg: best_angle,
                confidence,
                match_score: best_score,
            }));
        }

        state.angle_readings = readings;
        Ok(())
    }
}
