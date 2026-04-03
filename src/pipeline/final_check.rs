use opencv::core::Mat;
use opencv::prelude::*;
use tracing::info;

use super::stage::{PipelineState, StageDescriptor, StageOutcome};
use super::{ImageOutput, Stage};

/// Strict Y tolerance for the final pass.
const Y_TOLERANCE_PX: f64 = 6.0;
const MIN_X_GAP_PX: f64 = 30.0;
const MAX_GAP_CV: f64 = 0.20;
const MAX_RADIUS_FACTOR: f64 = 1.1;
const EXPECTED_KNOBS: usize = 10;

pub(crate) const DESCRIPTOR: StageDescriptor = StageDescriptor {
    name: "FinalCheck",
    label: "S13:FinalCheck",
    fallback: Some("FinalDetect"),
};

/// Stage S13: Strict sanity check on the final refined feature detection.
///
/// Same checks as S9 but with tighter thresholds. This is the gate that
/// determines whether the calibration succeeded.
pub struct FinalCheck;

impl FinalCheck {
    pub fn new() -> Self {
        Self
    }
}

impl Stage for FinalCheck {
    fn descriptor(&self) -> StageDescriptor {
        DESCRIPTOR
    }

    fn run(
        &self,
        state: &mut PipelineState,
        _src: &Mat,
        _dst: &mut Mat,
        _raw: &Mat,
        _iteration: u32,
    ) -> Result<(StageOutcome, ImageOutput), opencv::Error> {
        let Some(features) = &state.features else {
            return Ok((
                StageOutcome::Exhausted("no features from S12".into()),
                ImageOutput::Passthrough,
            ));
        };

        // Check 1: knob count
        if features.knobs.len() != EXPECTED_KNOBS {
            return Ok((
                StageOutcome::Exhausted(format!(
                    "expected {EXPECTED_KNOBS} knobs, got {}",
                    features.knobs.len()
                )),
                ImageOutput::Passthrough,
            ));
        }

        // Check 2: Y alignment
        let mut ys: Vec<f64> = features.knobs.iter().map(|k| k.center_y).collect();
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_y = ys[ys.len() / 2];

        for (i, knob) in features.knobs.iter().enumerate() {
            let dev = (knob.center_y - median_y).abs();
            if dev > Y_TOLERANCE_PX {
                return Ok((
                    StageOutcome::Exhausted(format!(
                        "knob {i} Y-deviation = {dev:.1}px (max {Y_TOLERANCE_PX}px)"
                    )),
                    ImageOutput::Passthrough,
                ));
            }
        }

        // Check 3: X monotonically increasing with minimum gap
        for i in 1..features.knobs.len() {
            let gap = features.knobs[i].center_x - features.knobs[i - 1].center_x;
            if gap < MIN_X_GAP_PX {
                return Ok((
                    StageOutcome::Exhausted(format!(
                        "knobs {}/{} X-gap = {gap:.1}px (min {MIN_X_GAP_PX}px)",
                        i - 1,
                        i
                    )),
                    ImageOutput::Passthrough,
                ));
            }
        }

        // Check 3b: no overlapping knobs
        for i in 0..features.knobs.len() {
            for j in (i + 1)..features.knobs.len() {
                let dx = features.knobs[j].center_x - features.knobs[i].center_x;
                let dy = features.knobs[j].center_y - features.knobs[i].center_y;
                let dist = (dx * dx + dy * dy).sqrt();
                let min_dist = features.knobs[i].radius + features.knobs[j].radius;
                if dist < min_dist {
                    return Ok((
                        StageOutcome::Exhausted(format!(
                            "knobs {i}/{j} overlap: dist={dist:.1}px < radii_sum={min_dist:.1}px"
                        )),
                        ImageOutput::Passthrough,
                    ));
                }
            }
        }

        // Check 3c: X-gap regularity
        {
            let mut gaps: Vec<f64> = Vec::new();
            for i in 1..features.knobs.len() {
                gaps.push(features.knobs[i].center_x - features.knobs[i - 1].center_x);
            }
            let mean_gap: f64 = gaps.iter().sum::<f64>() / gaps.len() as f64;
            let var_gap: f64 =
                gaps.iter().map(|g| (g - mean_gap).powi(2)).sum::<f64>() / gaps.len() as f64;
            let cv_gap = var_gap.sqrt() / mean_gap.max(1.0);
            if cv_gap > MAX_GAP_CV {
                return Ok((
                    StageOutcome::Exhausted(format!(
                        "X-gap CV = {cv_gap:.2} (max {MAX_GAP_CV}), mean_gap={mean_gap:.1}px"
                    )),
                    ImageOutput::Passthrough,
                ));
            }
        }

        // Check 4: radius consistency
        let mut radii: Vec<f64> = features.knobs.iter().map(|k| k.radius).collect();
        radii.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_r = radii[radii.len() / 2];
        let r_lo = median_r / MAX_RADIUS_FACTOR;
        let r_hi = median_r * MAX_RADIUS_FACTOR;

        for (i, knob) in features.knobs.iter().enumerate() {
            if knob.radius < r_lo || knob.radius > r_hi {
                return Ok((
                    StageOutcome::Exhausted(format!(
                        "knob {i} radius = {:.1} outside [{r_lo:.1}, {r_hi:.1}]",
                        knob.radius
                    )),
                    ImageOutput::Passthrough,
                ));
            }
        }

        // Log measured values
        let max_y_dev = ys
            .iter()
            .map(|y| (y - median_y).abs())
            .fold(0.0f64, f64::max);
        let min_gap = {
            let mut mg = f64::INFINITY;
            for i in 1..features.knobs.len() {
                let g = features.knobs[i].center_x - features.knobs[i - 1].center_x;
                if g < mg {
                    mg = g;
                }
            }
            mg
        };
        let gap_cv = {
            let mut gaps: Vec<f64> = Vec::new();
            for i in 1..features.knobs.len() {
                gaps.push(features.knobs[i].center_x - features.knobs[i - 1].center_x);
            }
            let mean: f64 = gaps.iter().sum::<f64>() / gaps.len() as f64;
            let var: f64 =
                gaps.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / gaps.len() as f64;
            var.sqrt() / mean.max(1.0)
        };
        info!(
            max_y_dev = format!("{max_y_dev:.1}"),
            min_x_gap = format!("{min_gap:.1}"),
            gap_cv = format!("{gap_cv:.2}"),
            median_r = format!("{median_r:.1}"),
            "S13 FINAL passed"
        );

        state.validated = true;
        Ok((StageOutcome::Success, ImageOutput::Passthrough))
    }

    fn max_retries(&self) -> u32 {
        1
    }
}
