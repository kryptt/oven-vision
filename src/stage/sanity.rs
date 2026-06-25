use super::detect::Knob;
use super::{FrameState, Stage, StageError};
use crate::config::KnobDetection;

#[derive(Debug)]
pub struct SanityResult {
    pub ok: bool,
    pub count_ok: bool,
    pub y_aligned: bool,
    pub size_uniform: bool,
    pub spacing_uniform: bool,
    pub synthetic_count: usize,
    pub details: String,
}

pub struct Sanity {
    expected_count: usize,
    y_tolerance: f32,
    size_tolerance: f32,
    spacing_tolerance: f32,
    max_synthetic: usize,
}

impl Sanity {
    pub fn new(kd: &KnobDetection) -> Self {
        Self {
            expected_count: kd.expected_count,
            y_tolerance: kd.y_tolerance_px,
            size_tolerance: kd.size_tolerance_pct,
            spacing_tolerance: kd.spacing_tolerance_pct,
            max_synthetic: 3, // allow up to 3 synthetic slots
        }
    }

    fn check(&self, knobs: &[Knob]) -> SanityResult {
        let mut details = Vec::new();

        // Synthetic count check (replaces raw count check)
        let synthetic_count = knobs.iter().filter(|k| k.synthetic).count();
        let count_ok = synthetic_count <= self.max_synthetic;
        if !count_ok {
            details.push(format!("synthetic={}/{}", synthetic_count, knobs.len()));
        }

        // Only check real (non-synthetic) knobs for alignment/size/spacing
        let real: Vec<&Knob> = knobs.iter().filter(|k| !k.synthetic).collect();

        if real.len() < 2 {
            return SanityResult {
                ok: false,
                count_ok,
                y_aligned: false,
                size_uniform: false,
                spacing_uniform: false,
                synthetic_count,
                details: details.join("; "),
            };
        }

        // Y alignment of real knobs
        let mean_y: f32 = real.iter().map(|k| k.y).sum::<f32>() / real.len() as f32;
        let max_y_dev = real
            .iter()
            .map(|k| (k.y - mean_y).abs())
            .fold(0.0f32, f32::max);
        let y_aligned = max_y_dev <= self.y_tolerance;
        if !y_aligned {
            details.push(format!("y_dev={:.1}px", max_y_dev));
        }

        // Size uniformity of real knobs
        let mut radii: Vec<f32> = real.iter().map(|k| k.radius).collect();
        radii.sort_by(f32::total_cmp);
        let median_r = radii[radii.len() / 2];
        let max_r_dev = real
            .iter()
            .map(|k| ((k.radius - median_r) / median_r).abs())
            .fold(0.0f32, f32::max);
        let size_uniform = max_r_dev <= self.size_tolerance;
        if !size_uniform {
            details.push(format!("size_dev={:.0}%", max_r_dev * 100.0));
        }

        // Spacing uniformity (use all knobs including synthetic — they're at prior positions)
        let spacings: Vec<f32> = knobs.windows(2).map(|w| w[1].x - w[0].x).collect();
        let mean_sp: f32 = spacings.iter().sum::<f32>() / spacings.len() as f32;
        let max_sp_dev = spacings
            .iter()
            .map(|s| ((s - mean_sp) / mean_sp).abs())
            .fold(0.0f32, f32::max);
        let spacing_uniform = max_sp_dev <= self.spacing_tolerance;
        if !spacing_uniform {
            details.push(format!("spacing_dev={:.0}%", max_sp_dev * 100.0));
        }

        let ok = count_ok && y_aligned && size_uniform && spacing_uniform;
        if ok {
            details.push("PASS".into());
        }

        SanityResult {
            ok,
            count_ok,
            y_aligned,
            size_uniform,
            spacing_uniform,
            synthetic_count,
            details: details.join("; "),
        }
    }
}

impl Stage for Sanity {
    fn name(&self) -> &'static str {
        "sanity"
    }

    fn process(&self, state: &mut FrameState) -> Result<(), StageError> {
        state.sanity = Some(self.check(&state.knobs));
        Ok(())
    }
}
