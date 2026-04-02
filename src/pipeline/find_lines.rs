use std::f64::consts::PI;

use opencv::core::{Mat, Point, Rect, Scalar, Size, Vec2f, Vector};
use opencv::imgproc;
use opencv::prelude::*;

use super::stage::{Line, LinePair, PipelineState, StageId, StageOutcome};
use super::{DebugImage, Stage};
use crate::annotate::encode_jpeg;

/// Maximum angular deviation from horizontal (in degrees) for a line to be
/// considered "near-horizontal". In HoughLines, theta=π/2 is horizontal.
const HORIZONTAL_TOLERANCE_DEG: f64 = 15.0;

/// Minimum vertical separation (in pixels of rho) between the two reference
/// lines. Prevents picking two lines from the same chrome strip.
const MIN_LINE_SEPARATION: f64 = 15.0;

/// Rho clustering distance — lines with rho within this range are merged.
const CLUSTER_RHO_THRESHOLD: f64 = 10.0;

/// Stage 2: Detect horizontal reference lines and select a pair using
/// ranked heuristics. Each retry `iteration` selects the next-best pair,
/// so downstream failures naturally explore different line pairs via the
/// fallback mechanism.
///
/// Pair ranking heuristics (weighted sum):
///   1. **Span** — wider vertical separation is better.
///   2. **Position** — pair in the upper portion of the crop is better
///      (chrome trim is in the upper half, not the oven door area).
///   3. **Parallelism** — similar slope is better.
///   4. **Strength** — more Hough votes is better.
pub struct FindLines;

impl FindLines {
    pub fn new() -> Self {
        Self
    }

    /// Canny / Hough thresholds. Relax every 5 pair-selection iterations.
    fn thresholds_for_iteration(iteration: u32) -> (f64, f64, i32) {
        let thresh_step = iteration / 5;
        let canny_low = (80.0 - thresh_step as f64 * 5.0).max(20.0);
        let canny_high = (200.0 - thresh_step as f64 * 10.0).max(60.0);
        let hough_threshold = (200 - thresh_step as i32 * 10).max(50);
        (canny_low, canny_high, hough_threshold)
    }
}

impl Stage for FindLines {
    fn id(&self) -> StageId {
        StageId::FindLines
    }

    fn run(
        &self,
        state: &mut PipelineState,
        frame: &Mat,
        iteration: u32,
    ) -> Result<StageOutcome, opencv::Error> {
        let Some(crop) = &state.crop else {
            return Ok(StageOutcome::Exhausted(
                "no crop region from Stage 1".into(),
            ));
        };

        let roi_rect = Rect::new(
            crop.x as i32,
            crop.y as i32,
            crop.width as i32,
            crop.height as i32,
        );
        let cropped = Mat::roi(frame, roi_rect)?;

        let mut gray = Mat::default();
        imgproc::cvt_color_def(&cropped, &mut gray, imgproc::COLOR_BGR2GRAY)?;

        let mut enhanced = Mat::default();
        let mut clahe = imgproc::create_clahe(2.0, Size::new(8, 8))?;
        clahe.apply(&gray, &mut enhanced)?;

        let (canny_low, canny_high, hough_threshold) = Self::thresholds_for_iteration(iteration);

        let mut edges = Mat::default();
        imgproc::canny(&enhanced, &mut edges, canny_low, canny_high, 3, false)?;

        let mut lines_raw = Vector::<Vec2f>::new();
        imgproc::hough_lines_def(&edges, &mut lines_raw, 1.0, PI / 180.0, hough_threshold)?;

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
            return Ok(StageOutcome::Retry(format!(
                "found {} horizontal lines (need 2), canny={canny_low}/{canny_high}, hough={hough_threshold}",
                horizontal.len()
            )));
        }

        // Cluster lines by rho
        let clusters = cluster_by_rho(&horizontal, CLUSTER_RHO_THRESHOLD);

        if clusters.len() < 2 {
            return Ok(StageOutcome::Retry(format!(
                "only {} line cluster(s) after merging (need 2)",
                clusters.len()
            )));
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
            return Ok(StageOutcome::Retry(format!(
                "{} clusters but no pair with separation >= {MIN_LINE_SEPARATION}",
                summaries.len()
            )));
        }

        // Sort by score descending (best first)
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

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

        state.lines = Some(LinePair {
            top: top_line,
            bottom: bot_line,
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
        let (Some(crop), Some(lines)) = (&state.crop, &state.lines) else {
            return Ok(None);
        };

        let roi_rect = Rect::new(
            crop.x as i32,
            crop.y as i32,
            crop.width as i32,
            crop.height as i32,
        );
        let mut canvas = Mat::roi(frame, roi_rect)?.try_clone()?;

        // Draw top line in green
        draw_line(&mut canvas, &lines.top, Scalar::new(0.0, 255.0, 0.0, 0.0))?;
        // Draw bottom line in blue
        draw_line(
            &mut canvas,
            &lines.bottom,
            Scalar::new(255.0, 0.0, 0.0, 0.0),
        )?;

        let label = format!(
            "S2:FindLines_T{:.0}_B{:.0}",
            lines.top.y1, lines.bottom.y1
        );
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
///   - span:        separation / crop_height          (wider vertical gap = better)
///   - position:    1 - avg_rho / crop_height          (upper half = better — chrome trim)
///   - parallelism: 1 - |Δtheta| / max_Δtheta         (parallel = better)
///   - strength:    avg_votes / max_votes              (stronger = better)
fn score_pair(
    top: (f64, f64, usize),
    bot: (f64, f64, usize),
    crop_h: f64,
    max_votes: f64,
) -> f64 {
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

    // Weights: strength matters most (strong chrome edges), then position,
    // then parallelism, then span
    0.30 * strength_norm + 0.30 * position_norm + 0.20 * parallel_norm + 0.20 * span_norm
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

/// Cluster lines by rho proximity.
fn cluster_by_rho(lines: &[(f64, f64)], threshold: f64) -> Vec<Vec<(f64, f64)>> {
    let mut sorted: Vec<(f64, f64)> = lines.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut clusters: Vec<Vec<(f64, f64)>> = Vec::new();
    for &(rho, theta) in &sorted {
        let mut added = false;
        for cluster in &mut clusters {
            let avg_rho: f64 = cluster.iter().map(|&(r, _)| r).sum::<f64>() / cluster.len() as f64;
            if (rho - avg_rho).abs() <= threshold {
                cluster.push((rho, theta));
                added = true;
                break;
            }
        }
        if !added {
            clusters.push(vec![(rho, theta)]);
        }
    }
    clusters
}

/// Average rho and theta for a cluster.
fn cluster_average(cluster: &[(f64, f64)]) -> (f64, f64) {
    let n = cluster.len() as f64;
    let rho = cluster.iter().map(|&(r, _)| r).sum::<f64>() / n;
    let theta = cluster.iter().map(|&(_, t)| t).sum::<f64>() / n;
    (rho, theta)
}

fn draw_line(canvas: &mut Mat, line: &Line, color: Scalar) -> Result<(), opencv::Error> {
    imgproc::line(
        canvas,
        Point::new(line.x1 as i32, line.y1 as i32),
        Point::new(line.x2 as i32, line.y2 as i32),
        color,
        2,
        imgproc::LINE_8,
        0,
    )
}
