use std::f64::consts::PI;

use opencv::core::{Mat, Point, Scalar, Size, Vec2f, Vector};
use opencv::imgproc;
use opencv::prelude::*;

use super::stage::{Line, LinePair, PipelineState, StageDescriptor, StageOutcome};
use super::util::{cluster_average, cluster_by_rho, draw_line, enhance_gray};
use super::{DebugImage, ImageOutput, Stage};
use crate::annotate::encode_jpeg;

/// Maximum angular deviation from horizontal (in degrees) for a line to be
/// considered "near-horizontal". In HoughLines, theta=pi/2 is horizontal.
const HORIZONTAL_TOLERANCE_DEG: f64 = 15.0;

/// Minimum vertical separation (in pixels of rho) between the two reference
/// lines. Prevents picking two lines from the same chrome strip.
const MIN_LINE_SEPARATION: f64 = 15.0;

/// Rho clustering distance -- lines with rho within this range are merged.
const CLUSTER_RHO_THRESHOLD: f64 = 10.0;

/// Stage 2: Detect horizontal reference lines and select a pair using
/// ranked heuristics. Each retry `iteration` selects the next-best pair,
/// so downstream failures naturally explore different line pairs via the
/// fallback mechanism.
///
/// Pair ranking heuristics (weighted sum):
///   1. **Span** -- wider vertical separation is better.
///   2. **Position** -- pair in the upper portion of the crop is better
///      (chrome trim is in the upper half, not the oven door area).
///   3. **Parallelism** -- similar slope is better.
///   4. **Strength** -- more Hough votes is better.
pub struct FindLines;

impl FindLines {
    pub fn new() -> Self {
        Self
    }

    /// Canny / Hough thresholds. Relax every 5 pair-selection iterations.
    fn thresholds_for_iteration(iteration: u32) -> (f64, f64, i32) {
        let thresh_step = iteration / 3;
        let canny_low = (80.0 - thresh_step as f64 * 6.0).max(20.0);
        let canny_high = (200.0 - thresh_step as f64 * 14.0).max(60.0);
        let hough_threshold = (200 - thresh_step as i32 * 15).max(80);
        (canny_low, canny_high, hough_threshold)
    }
}

pub(crate) const DESCRIPTOR: StageDescriptor = StageDescriptor {
    name: "FindLines",
    label: "S2:FindLines",
    fallback: Some("FindStove"),
};

impl Stage for FindLines {
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

        // src is already the cropped image from FindStove
        let blurred = enhance_gray(src, 5)?;

        let (canny_low, canny_high, hough_threshold) = Self::thresholds_for_iteration(iteration);

        let mut edges = Mat::default();
        imgproc::canny(&blurred, &mut edges, canny_low, canny_high, 3, false)?;

        // Morphological close with horizontal kernel to bridge fragmented horizontal edges
        let kernel_h = imgproc::get_structuring_element(
            imgproc::MORPH_RECT,
            Size::new(3, 1),
            Point::new(-1, -1),
        )?;
        let mut closed = Mat::default();
        imgproc::morphology_ex(
            &edges,
            &mut closed,
            imgproc::MORPH_CLOSE,
            &kernel_h,
            Point::new(-1, -1),
            1,
            opencv::core::BORDER_CONSTANT,
            Scalar::default(),
        )?;

        let mut lines_raw = Vector::<Vec2f>::new();
        imgproc::hough_lines_def(&closed, &mut lines_raw, 1.0, PI / 180.0, hough_threshold)?;

        // Filter for near-horizontal lines
        let horizontal_center = PI / 2.0;
        let tolerance_rad = HORIZONTAL_TOLERANCE_DEG * PI / 180.0;

        let mut horizontal: Vec<(f64, f64)> = Vec::new();
        for v in &lines_raw {
            let rho = v[0] as f64;
            let theta = v[1] as f64;
            if (theta - horizontal_center).abs() <= tolerance_rad {
                horizontal.push((rho, theta));
            }
        }

        if horizontal.len() < 2 {
            return Ok((
                StageOutcome::Retry(format!(
                    "found {} horizontal lines (need 2), canny={canny_low}/{canny_high}, hough={hough_threshold}",
                    horizontal.len()
                )),
                ImageOutput::Passthrough,
            ));
        }

        // Cluster lines by rho
        let clusters = cluster_by_rho(&horizontal, CLUSTER_RHO_THRESHOLD);

        if clusters.len() < 2 {
            return Ok((
                StageOutcome::Retry(format!(
                    "only {} line cluster(s) after merging (need 2)",
                    clusters.len()
                )),
                ImageOutput::Passthrough,
            ));
        }

        // Summarise each cluster: (avg_rho, avg_theta, vote_count)
        let summaries: Vec<(f64, f64, usize)> = clusters
            .iter()
            .map(|c| {
                let (rho, theta) = cluster_average(c);
                (rho, theta, c.len())
            })
            .collect();

        // Generate all candidate pairs, score them, sort descending
        let h = crop.height as f64;
        let max_votes = summaries.iter().map(|s| s.2).max().unwrap_or(1).max(1) as f64;

        let mut candidates: Vec<CandidatePair> = Vec::new();
        for i in 0..summaries.len() {
            for j in (i + 1)..summaries.len() {
                let (rho_a, theta_a, votes_a) = summaries[i];
                let (rho_b, theta_b, votes_b) = summaries[j];

                // Order: smaller rho = top line
                let (top, bot) = if rho_a <= rho_b {
                    ((rho_a, theta_a, votes_a), (rho_b, theta_b, votes_b))
                } else {
                    ((rho_b, theta_b, votes_b), (rho_a, theta_a, votes_a))
                };

                let span = bot.0 - top.0;
                if span < MIN_LINE_SEPARATION {
                    continue;
                }

                let score = score_pair(top, bot, h, max_votes);
                candidates.push(CandidatePair {
                    top_rho: top.0,
                    top_theta: top.1,
                    bot_rho: bot.0,
                    bot_theta: bot.1,
                    score,
                });
            }
        }

        if candidates.is_empty() {
            return Ok((
                StageOutcome::Retry(format!(
                    "{} clusters but no pair with separation >= {MIN_LINE_SEPARATION}",
                    summaries.len()
                )),
                ImageOutput::Passthrough,
            ));
        }

        // Sort by score descending (best first)
        candidates.sort_by(|a, b| {
            b.score.total_cmp(&a.score)
        });

        // Pick the pair at index = iteration % 5 (cycles within each threshold group)
        let pair_idx = (iteration % 5) as usize;
        let pick = if pair_idx < candidates.len() {
            &candidates[pair_idx]
        } else {
            &candidates[candidates.len() - 1]
        };

        let w = crop.width as f64;
        let top_line = rho_theta_to_line(pick.top_rho, pick.top_theta, w);
        let bot_line = rho_theta_to_line(pick.bot_rho, pick.bot_theta, w);

        let avg_theta = (pick.top_theta + pick.bot_theta) / 2.0;
        state.lines = Some(LinePair {
            top: top_line,
            bottom: bot_line,
            avg_theta,
        });

        Ok((StageOutcome::Success, ImageOutput::Passthrough))
    }

    fn max_retries(&self) -> u32 {
        15
    }

    fn debug_image(
        &self,
        state: &PipelineState,
        working: &Mat,
        _raw: &Mat,
    ) -> Result<Option<DebugImage>, opencv::Error> {
        let (Some(_crop), Some(lines)) = (&state.crop, &state.lines) else {
            return Ok(None);
        };

        let mut canvas = working.try_clone()?;

        // Draw top line in green
        draw_line(&mut canvas, &lines.top, Scalar::new(0.0, 255.0, 0.0, 0.0))?;
        // Draw bottom line in blue
        draw_line(
            &mut canvas,
            &lines.bottom,
            Scalar::new(255.0, 0.0, 0.0, 0.0),
        )?;

        let label = format!("S2:FindLines_T{:.0}_B{:.0}", lines.top.y1, lines.bottom.y1);
        let jpeg = encode_jpeg(&canvas, 80)?;
        Ok(Some((label, jpeg)))
    }
}

struct CandidatePair {
    top_rho: f64,
    top_theta: f64,
    bot_rho: f64,
    bot_theta: f64,
    score: f64,
}

/// Score a horizontal pair. Higher is better.
///
/// Components (each normalised to 0..1, then weighted):
///   - position:    1 - avg_rho / crop_height          (upper half = better -- chrome trim)
///   - strength:    avg_votes / max_votes              (stronger = better)
///   - parallelism: 1 - |delta_theta| / max_delta      (parallel = better)
///   - span:        separation / crop_height            (wider vertical gap = better)
fn score_pair(top: (f64, f64, usize), bot: (f64, f64, usize), crop_h: f64, max_votes: f64) -> f64 {
    let span = bot.0 - top.0;
    let span_norm = (span / crop_h).min(1.0);

    // Prefer lines in the upper portion of the crop (chrome trim area)
    let avg_rho = (top.0 + bot.0) / 2.0;
    let position_norm = 1.0 - (avg_rho / crop_h).min(1.0);

    let delta_theta = (top.1 - bot.1).abs();
    let max_delta = 15.0_f64.to_radians();
    let parallel_norm = 1.0 - (delta_theta / max_delta).min(1.0);

    let avg_votes = (top.2 + bot.2) as f64 / 2.0;
    let strength_norm = (avg_votes / max_votes).min(1.0);

    // Weights: position matters most, then parallelism, then strength, then span
    0.40 * position_norm + 0.25 * parallel_norm + 0.20 * strength_norm + 0.15 * span_norm
}

/// Convert a Hough (rho, theta) line to endpoint form spanning the full width.
fn rho_theta_to_line(rho: f64, theta: f64, width: f64) -> Line {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let sin_safe = if sin_t.abs() < 1e-6 { 1e-6 } else { sin_t };

    let y_at_0 = rho / sin_safe;
    let y_at_w = (rho - width * cos_t) / sin_safe;

    Line {
        x1: 0.0,
        y1: y_at_0,
        x2: width,
        y2: y_at_w,
    }
}
