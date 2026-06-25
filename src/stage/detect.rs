use std::collections::HashSet;

use opencv::core::{Mat, Point};
use opencv::imgproc;
use opencv::prelude::*;

use super::{FrameState, Stage, StageError};
use crate::config::KnobDetection;

#[derive(Debug, Clone)]
pub struct Knob {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub slot: usize,
    pub synthetic: bool,
}

pub struct Detect {
    min_radius: f32,
    max_radius: f32,
    y_band_top: f32,
    y_band_bottom: f32,
    expected_count: usize,
    prior_x_start: f32,
    prior_x_end: f32,
    prior_y: f32,
    slot_capture_radius: f32,
}

impl Detect {
    pub fn new(kd: &KnobDetection) -> Self {
        Self {
            min_radius: kd.hough_min_radius as f32,
            max_radius: kd.hough_max_radius as f32,
            y_band_top: kd.y_band_top,
            y_band_bottom: kd.y_band_bottom,
            expected_count: kd.expected_count,
            prior_x_start: kd.prior_x_start,
            prior_x_end: kd.prior_x_end,
            prior_y: kd.prior_y,
            slot_capture_radius: kd.slot_capture_radius,
        }
    }
}

impl Stage for Detect {
    fn name(&self) -> &'static str {
        "detect"
    }

    fn process(&self, state: &mut FrameState) -> Result<(), StageError> {
        let input = state.enhanced.as_ref().ok_or(StageError {
            stage: self.name(),
            message: "no enhanced frame".into(),
        })?;

        let img_h = input.rows() as f32;
        let y_top = self.y_band_top * img_h;
        let y_bot = self.y_band_bottom * img_h;

        // Adaptive Canny thresholds from gradient magnitude
        let mut grad_x = Mat::default();
        let mut grad_y = Mat::default();
        imgproc::sobel(
            input,
            &mut grad_x,
            opencv::core::CV_16S,
            1,
            0,
            3,
            1.0,
            0.0,
            opencv::core::BORDER_DEFAULT,
        )
        .map_err(|e| StageError {
            stage: self.name(),
            message: e.to_string(),
        })?;
        imgproc::sobel(
            input,
            &mut grad_y,
            opencv::core::CV_16S,
            0,
            1,
            3,
            1.0,
            0.0,
            opencv::core::BORDER_DEFAULT,
        )
        .map_err(|e| StageError {
            stage: self.name(),
            message: e.to_string(),
        })?;
        let mut abs_x = Mat::default();
        let mut abs_y = Mat::default();
        opencv::core::convert_scale_abs(&grad_x, &mut abs_x, 1.0, 0.0).map_err(|e| StageError {
            stage: self.name(),
            message: e.to_string(),
        })?;
        opencv::core::convert_scale_abs(&grad_y, &mut abs_y, 1.0, 0.0).map_err(|e| StageError {
            stage: self.name(),
            message: e.to_string(),
        })?;
        let mean_x = opencv::core::mean(&abs_x, &Mat::default()).map_err(|e| StageError {
            stage: self.name(),
            message: e.to_string(),
        })?;
        let mean_y_val = opencv::core::mean(&abs_y, &Mat::default()).map_err(|e| StageError {
            stage: self.name(),
            message: e.to_string(),
        })?;
        let mean_grad = (mean_x[0] + mean_y_val[0]) / 2.0;

        let mut edges = Mat::default();
        imgproc::canny(
            input,
            &mut edges,
            mean_grad * 0.5,
            mean_grad * 1.5,
            3,
            false,
        )
        .map_err(|e| StageError {
            stage: self.name(),
            message: e.to_string(),
        })?;

        state.debug_edges = Some(edges.clone());

        // Find contours
        let mut contours = opencv::core::Vector::<opencv::core::Vector<Point>>::new();
        imgproc::find_contours(
            &edges,
            &mut contours,
            imgproc::RETR_LIST,
            imgproc::CHAIN_APPROX_NONE,
            Point::new(0, 0),
        )
        .map_err(|e| StageError {
            stage: self.name(),
            message: e.to_string(),
        })?;

        // Filter contours into candidates
        let min_perim = std::f64::consts::PI * self.min_radius as f64;
        let max_perim = 2.0 * std::f64::consts::PI * self.max_radius as f64 * 2.5;

        let mut candidates: Vec<Knob> = Vec::new();

        for i in 0..contours.len() {
            let contour = contours.get(i).map_err(|e| StageError {
                stage: self.name(),
                message: e.to_string(),
            })?;
            if contour.len() < 5 {
                continue;
            }

            let perim = imgproc::arc_length(&contour, true).map_err(|e| StageError {
                stage: self.name(),
                message: e.to_string(),
            })?;
            if perim < min_perim || perim > max_perim {
                continue;
            }

            let ellipse = imgproc::fit_ellipse(&contour).map_err(|e| StageError {
                stage: self.name(),
                message: e.to_string(),
            })?;

            let r_avg = (ellipse.size.width + ellipse.size.height) / 4.0;
            if r_avg < self.min_radius * 0.6 || r_avg > self.max_radius * 1.5 {
                continue;
            }

            // Y-band filter
            if ellipse.center.y < y_top || ellipse.center.y > y_bot {
                continue;
            }

            // Axis ratio: reject very elongated shapes
            let (major, minor) = if ellipse.size.width > ellipse.size.height {
                (ellipse.size.width, ellipse.size.height)
            } else {
                (ellipse.size.height, ellipse.size.width)
            };
            if minor > 0.0 && major / minor > 2.5 {
                continue;
            }

            candidates.push(Knob {
                x: ellipse.center.x,
                y: ellipse.center.y,
                radius: r_avg,
                slot: 0,
                synthetic: false,
            });
        }

        // --- Spatial prior: assign candidates to expected slot positions ---
        let n = self.expected_count;
        let slot_xs: Vec<f32> = (0..n)
            .map(|i| {
                self.prior_x_start
                    + i as f32 * (self.prior_x_end - self.prior_x_start) / (n - 1) as f32
            })
            .collect();

        let median_radius: f32 = {
            let mut rs: Vec<f32> = candidates.iter().map(|c| c.radius).collect();
            rs.sort_by(f32::total_cmp);
            if rs.is_empty() {
                (self.min_radius + self.max_radius) / 2.0
            } else {
                rs[rs.len() / 2]
            }
        };

        let capture_r = self.slot_capture_radius;
        let prior_y = self.prior_y;
        let mut assigned: Vec<Option<Knob>> = vec![None; n];
        let mut used: HashSet<usize> = HashSet::new();

        for (slot_idx, &sx) in slot_xs.iter().enumerate() {
            let best = candidates
                .iter()
                .enumerate()
                .filter(|(ci, c)| {
                    if used.contains(ci) {
                        return false;
                    }
                    let dx = c.x - sx;
                    let dy = c.y - prior_y;
                    (dx * dx + dy * dy).sqrt() < capture_r
                })
                .min_by(|(_, a), (_, b)| {
                    let score = |c: &Knob| -> f32 {
                        let dx = c.x - sx;
                        let dy = c.y - prior_y;
                        let dist = (dx * dx + dy * dy).sqrt() / capture_r;
                        let r_dev = (c.radius - median_radius).abs() / median_radius.max(1.0);
                        // Equal weight: position accuracy + radius consistency
                        dist * 0.5 + r_dev * 0.5
                    };
                    score(a).total_cmp(&score(b))
                });

            if let Some((ci, knob)) = best {
                used.insert(ci);
                let mut k = knob.clone();
                k.slot = slot_idx;
                assigned[slot_idx] = Some(k);
            }
        }

        // Compute median radius and Y from assigned (real) knobs for normalization
        let real_knobs: Vec<&Knob> = assigned.iter().filter_map(|a| a.as_ref()).collect();

        let norm_radius = if real_knobs.is_empty() {
            median_radius
        } else {
            let mut sorted: Vec<f32> = real_knobs.iter().map(|k| k.radius).collect();
            sorted.sort_by(f32::total_cmp);
            sorted[sorted.len() / 2]
        };

        let norm_y = if real_knobs.is_empty() {
            prior_y
        } else {
            let mut sorted: Vec<f32> = real_knobs.iter().map(|k| k.y).collect();
            sorted.sort_by(f32::total_cmp);
            sorted[sorted.len() / 2]
        };

        // Synthetic fallback for empty slots + normalize radius and Y
        // All knobs are physically identical and sit on the same horizontal rail
        let knobs: Vec<Knob> = (0..n)
            .map(|i| {
                let mut k = assigned[i].clone().unwrap_or(Knob {
                    x: slot_xs[i],
                    y: norm_y,
                    radius: norm_radius,
                    slot: i,
                    synthetic: true,
                });
                k.radius = norm_radius;
                k.y = norm_y;
                // Use detected X but snap toward the prior slot position
                // to smooth out contour-center jitter while preserving real shifts
                if !k.synthetic {
                    let slot_x = slot_xs[i];
                    k.x = slot_x * 0.5 + k.x * 0.5; // blend: 50% detected, 50% prior
                }
                k
            })
            .collect();

        let synthetic_count = knobs.iter().filter(|k| k.synthetic).count();
        eprintln!(
            "  detect: contours={} candidates={} assigned={}/{}",
            contours.len(),
            candidates.len(),
            n - synthetic_count,
            n
        );

        state.knobs = knobs;
        Ok(())
    }
}
