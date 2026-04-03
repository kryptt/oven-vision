use std::f64::consts::PI;

use opencv::core::{Mat, Rect, Scalar, Size, Vec2f, Vector};
use opencv::imgproc;
use opencv::prelude::*;

use super::stage::{Line, PipelineState, StageDescriptor, StageOutcome, VerticalPair};
use super::util::{cluster_average, cluster_by_rho, draw_line};
use super::{DebugImage, Stage};
use crate::annotate::encode_jpeg;

/// Maximum angular deviation from vertical (in degrees) for a line to be
/// considered "near-vertical". In HoughLines, theta=0 is vertical.
const VERTICAL_TOLERANCE_DEG: f64 = 20.0;

/// Minimum horizontal separation (in pixels) between the two vertical
/// reference lines. Prevents picking two lines from the same edge.
const MIN_LINE_SEPARATION: f64 = 100.0;

/// Rho clustering distance — lines with rho within this range are merged.
const CLUSTER_RHO_THRESHOLD: f64 = 15.0;

/// Stage 2b: Detect vertical reference lines and select a pair using
/// ranked heuristics. Each retry `iteration` selects the next-best pair,
/// so downstream failures (S5 sanity) naturally explore different pairs
/// via the fallback mechanism.
///
/// Pair ranking heuristics (weighted sum):
///   1. **Span** — wider separation is better (captures more panel).
///   2. **Symmetry** — pair centred on the crop is better.
///   3. **Parallelism** — similar slope is better (real panel edges are parallel).
///   4. **Strength** — more Hough votes is better.
pub struct FindVerticals;

impl FindVerticals {
    pub fn new() -> Self {
        Self
    }

    /// Canny / Hough thresholds.  Lower iterations are stricter.
    /// We use the upper half of the iteration range for threshold relaxation
    /// and the lower half for pair selection (same thresholds, different pair).
    fn thresholds_for_iteration(iteration: u32) -> (f64, f64, i32) {
        // Relax thresholds every 5 pair-selection iterations
        let thresh_step = iteration / 5;
        let canny_low = (80.0 - thresh_step as f64 * 5.0).max(20.0);
        let canny_high = (200.0 - thresh_step as f64 * 10.0).max(60.0);
        let hough_threshold = (150 - thresh_step as i32 * 10).max(30);
        (canny_low, canny_high, hough_threshold)
    }
}

pub(crate) const DESCRIPTOR: StageDescriptor = StageDescriptor {
    name: "FindVerticals",
    label: "S2b:FindVerticals",
    fallback: Some("FindLines"),
};

impl Stage for FindVerticals {
    fn descriptor(&self) -> StageDescriptor {
        DESCRIPTOR
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

        // Blur before edge detection to reduce noise from chrome reflections.
        let mut blurred = Mat::default();
        imgproc::gaussian_blur(
            &gray,
            &mut blurred,
            Size::new(5, 5),
            1.5,
            1.5,
            opencv::core::BORDER_DEFAULT,
        )?;

        let mut enhanced = Mat::default();
        let mut clahe = imgproc::create_clahe(2.0, Size::new(8, 8))?;
        clahe.apply(&blurred, &mut enhanced)?;

        let (canny_low, canny_high, hough_threshold) = Self::thresholds_for_iteration(iteration);

        let mut edges = Mat::default();
        imgproc::canny(&enhanced, &mut edges, canny_low, canny_high, 3, false)?;

        let mut lines_raw = Vector::<Vec2f>::new();
        imgproc::hough_lines_def(&edges, &mut lines_raw, 1.0, PI / 180.0, hough_threshold)?;

        // Filter for near-vertical lines.
        let tolerance_rad = VERTICAL_TOLERANCE_DEG * PI / 180.0;

        let mut vertical: Vec<(f64, f64)> = Vec::new();
        for v in &lines_raw {
            let rho = v[0] as f64;
            let theta = v[1] as f64;
            if theta <= tolerance_rad || theta >= (PI - tolerance_rad) {
                let (norm_rho, norm_theta) = if theta > PI / 2.0 {
                    (-rho, theta - PI)
                } else {
                    (rho, theta)
                };
                vertical.push((norm_rho, norm_theta));
            }
        }

        if vertical.len() < 2 {
            return Ok(StageOutcome::Retry(format!(
                "found {} vertical lines (need 2), canny={canny_low}/{canny_high}, hough={hough_threshold}",
                vertical.len()
            )));
        }

        // Cluster by rho (x-position)
        let clusters = cluster_by_rho(&vertical, CLUSTER_RHO_THRESHOLD);

        if clusters.len() < 2 {
            return Ok(StageOutcome::Retry(format!(
                "only {} vertical cluster(s) after merging (need 2)",
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
        let w = crop.width as f64;
        let max_votes = summaries.iter().map(|s| s.2).max().unwrap_or(1).max(1) as f64;

        let mut candidates: Vec<CandidatePair> = Vec::new();
        for i in 0..summaries.len() {
            for j in (i + 1)..summaries.len() {
                let (l_rho, l_theta, l_votes) = summaries[i];
                let (r_rho, r_theta, r_votes) = summaries[j];

                // Ensure left < right by rho
                let (left, right) = if l_rho <= r_rho {
                    ((l_rho, l_theta, l_votes), (r_rho, r_theta, r_votes))
                } else {
                    ((r_rho, r_theta, r_votes), (l_rho, l_theta, l_votes))
                };

                let span = right.0 - left.0;
                if span < MIN_LINE_SEPARATION {
                    continue;
                }

                let score = score_pair(left, right, w, max_votes);
                candidates.push(CandidatePair {
                    left_rho: left.0,
                    left_theta: left.1,
                    right_rho: right.0,
                    right_theta: right.1,
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
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Pick the pair at index = iteration % num_candidates
        // This way each retry from a downstream fallback tries the next pair
        let pair_idx = (iteration % 5) as usize; // pair selection cycles within each threshold group
        let pick = if pair_idx < candidates.len() {
            &candidates[pair_idx]
        } else {
            &candidates[candidates.len() - 1]
        };

        let h = crop.height as f64;
        let left_line = rho_theta_to_vertical_line(pick.left_rho, pick.left_theta, h);
        let right_line = rho_theta_to_vertical_line(pick.right_rho, pick.right_theta, h);

        state.verticals = Some(VerticalPair {
            left: left_line,
            right: right_line,
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
        let (Some(crop), Some(verticals)) = (&state.crop, &state.verticals) else {
            return Ok(None);
        };

        let roi_rect = Rect::new(
            crop.x as i32,
            crop.y as i32,
            crop.width as i32,
            crop.height as i32,
        );
        let mut canvas = Mat::roi(frame, roi_rect)?.try_clone()?;

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
        // Include a unique suffix so each iteration's image is kept
        let label = format!(
            "S2b:FindVerticals_L{:.0}_R{:.0}",
            verticals.left.x1, verticals.right.x1
        );
        Ok(Some((label, jpeg)))
    }
}

struct CandidatePair {
    left_rho: f64,
    left_theta: f64,
    right_rho: f64,
    right_theta: f64,
    score: f64,
}

/// Score a vertical pair.  Higher is better.
///
/// Components (each normalised to 0..1, then weighted):
///   - span:        separation / crop_width           (wider = better)
///   - symmetry:    1 - |centre_offset| / half_width  (centred = better)
///   - parallelism: 1 - |Δtheta| / max_Δtheta         (parallel = better)
///   - strength:    avg_votes / max_votes              (stronger = better)
fn score_pair(
    left: (f64, f64, usize), // (rho, theta, votes)
    right: (f64, f64, usize),
    crop_w: f64,
    max_votes: f64,
) -> f64 {
    let span = right.0 - left.0;
    let span_norm = (span / crop_w).min(1.0);

    let centre = (left.0 + right.0) / 2.0;
    let half_w = crop_w / 2.0;
    let symmetry_norm = 1.0 - ((centre - half_w).abs() / half_w).min(1.0);

    let delta_theta = (left.1 - right.1).abs();
    let max_delta = 30.0_f64.to_radians(); // beyond this, parallelism score = 0
    let parallel_norm = 1.0 - (delta_theta / max_delta).min(1.0);

    let avg_votes = (left.2 + right.2) as f64 / 2.0;
    let strength_norm = (avg_votes / max_votes).min(1.0);

    // Weights: span matters most, then symmetry, then parallelism, then strength
    0.40 * span_norm + 0.25 * symmetry_norm + 0.20 * parallel_norm + 0.15 * strength_norm
}

/// Convert a Hough (rho, theta) vertical line to endpoint form spanning
/// the full height.
fn rho_theta_to_vertical_line(rho: f64, theta: f64, height: f64) -> Line {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let cos_safe = if cos_t.abs() < 1e-6 { 1e-6 } else { cos_t };

    let x_at_0 = rho / cos_safe;
    let x_at_h = (rho - height * sin_t) / cos_safe;

    Line {
        x1: x_at_0,
        y1: 0.0,
        x2: x_at_h,
        y2: height,
    }
}
