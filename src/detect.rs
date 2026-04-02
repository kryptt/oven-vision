use opencv::core::Mat;
use opencv::imgproc;
use opencv::prelude::*;

use crate::config::DialConfig;
use crate::pipeline::stage::DetectedFeatures;

/// Coarse heat level bands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeatLevel {
    Low,
    Medium,
    High,
    Max,
}

/// The state of a single dial — illegal states are unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DialState {
    Off,
    On(HeatLevel),
    Unavailable,
}

/// Result of analyzing one dial in a frame.
pub struct DialReading {
    pub label: String,
    pub state: DialState,
    pub angle_deg: Option<f64>,
    pub confidence: f64,
}

/// Manages state-machine-based dial detection with hysteresis.
///
/// Each dial tracks its last confirmed state. A state transition only occurs
/// when N consecutive raw readings agree on the new state. This prevents
/// flapping from frame-to-frame noise in the radial edge scan.
pub struct DialDetector {
    configs: Vec<DialConfig>,
    /// Last confirmed state per dial
    confirmed: Vec<DialState>,
    /// Count of consecutive frames agreeing on a state different from confirmed
    transition_count: Vec<u32>,
    /// The candidate state being accumulated
    candidate: Vec<DialState>,
    /// Latest raw angle and confidence (for reporting)
    latest_angle: Vec<Option<f64>>,
    latest_confidence: Vec<f64>,
}

/// Maximum distinguishable angular distance (degrees).
const MAX_ROTATION_DEG: f64 = 180.0;

/// Minimum normalized edge strength to consider a detection valid.
const MIN_EDGE_STRENGTH: f64 = 0.05;

/// Number of radial sample points per angle.
const RADIAL_SAMPLES: u32 = 15;

/// Canny edge detection thresholds.
const CANNY_LOW: f64 = 10.0;
const CANNY_HIGH: f64 = 30.0;

/// Number of consecutive frames required to confirm a state change.
/// At 1 fps with 5s poll interval, this means ~15 seconds of consistent
/// readings before flipping state.
const HYSTERESIS_FRAMES: u32 = 3;

/// Default dial labels in left-to-right order (matching the stove layout).
const DEFAULT_LABELS: [&str; 10] = [
    "oven_left_mode",
    "burner_top_left",
    "burner_bottom_left",
    "oven_left_temp",
    "burner_top_center",
    "burner_bottom_center",
    "burner_top_right",
    "burner_bottom_right",
    "oven_right_temp",
    "oven_right_mode",
];

impl DialDetector {
    pub fn new(configs: Vec<DialConfig>) -> Self {
        let n = configs.len();
        Self {
            configs,
            confirmed: vec![DialState::Off; n],
            transition_count: vec![0; n],
            candidate: vec![DialState::Off; n],
            latest_angle: vec![None; n],
            latest_confidence: vec![0.0; n],
        }
    }

    /// Create a detector from pipeline-discovered features.
    ///
    /// Knob positions and radii come from the pipeline's `DetectedFeatures`.
    /// Labels are taken from `labels` (if provided and long enough), falling
    /// back to the hardcoded stove layout names, then generic "dial_N" names.
    pub fn from_features(features: &DetectedFeatures, labels: Option<&[String]>) -> Self {
        let configs: Vec<DialConfig> = features
            .knobs
            .iter()
            .enumerate()
            .map(|(i, knob)| {
                let label = labels.and_then(|l| l.get(i)).cloned().unwrap_or_else(|| {
                    DEFAULT_LABELS
                        .get(i)
                        .map(|s| (*s).to_string())
                        .unwrap_or_else(|| format!("dial_{}", i + 1))
                });
                DialConfig {
                    label,
                    center_x: knob.center_x.round() as u32,
                    center_y: knob.center_y.round() as u32,
                    radius: knob.radius.round().max(1.0) as u32,
                    off_angle_deg: features.off_angles.get(i).copied().unwrap_or(0.0),
                    off_tolerance_deg: 25.0,
                }
            })
            .collect();
        Self::new(configs)
    }

    /// Access the underlying dial configs (useful for MQTT discovery).
    pub fn configs(&self) -> &[DialConfig] {
        &self.configs
    }

    /// Detect all dials in the preprocessed grayscale image.
    /// Uses hysteresis: state only changes after HYSTERESIS_FRAMES consecutive
    /// frames agree on a different state from the current confirmed state.
    pub fn detect_all(&mut self, preprocessed: &Mat) -> Result<Vec<DialReading>, opencv::Error> {
        let mut edges = Mat::default();
        imgproc::canny(preprocessed, &mut edges, CANNY_LOW, CANNY_HIGH, 3, false)?;

        let mut results = Vec::with_capacity(self.configs.len());

        for (i, cfg) in self.configs.iter().enumerate() {
            let raw = detect_single(&edges, cfg)?;

            self.latest_angle[i] = raw.angle_deg;
            self.latest_confidence[i] = raw.confidence;

            // Hysteresis state machine
            let raw_state = raw.state;
            let same_category = state_category(raw_state) == state_category(self.confirmed[i]);

            if same_category {
                // Raw agrees with confirmed — reset transition counter
                self.transition_count[i] = 0;
            } else if state_category(raw_state) == state_category(self.candidate[i]) {
                // Raw agrees with the candidate (different from confirmed) — accumulate
                self.transition_count[i] += 1;
                if self.transition_count[i] >= HYSTERESIS_FRAMES {
                    // Transition confirmed
                    self.confirmed[i] = raw_state;
                    self.transition_count[i] = 0;
                }
            } else {
                // New candidate — reset counter
                self.candidate[i] = raw_state;
                self.transition_count[i] = 1;
            }

            results.push(DialReading {
                label: cfg.label.clone(),
                state: self.confirmed[i],
                angle_deg: self.latest_angle[i],
                confidence: self.latest_confidence[i],
            });
        }

        Ok(results)
    }
}

/// Collapse DialState into a category for hysteresis comparison.
/// We only care about Off vs On vs Unavailable — not the specific heat level.
fn state_category(state: DialState) -> u8 {
    match state {
        DialState::Off => 0,
        DialState::On(_) => 1,
        DialState::Unavailable => 2,
    }
}

/// Detect a single dial from the edge image and its configuration.
fn detect_single(edges: &Mat, cfg: &DialConfig) -> Result<DialReading, opencv::Error> {
    let (angle, strength) = radial_edge_scan(edges, cfg.center_x, cfg.center_y, cfg.radius)?;

    if strength < MIN_EDGE_STRENGTH {
        return Ok(DialReading {
            label: cfg.label.clone(),
            state: DialState::Unavailable,
            angle_deg: None,
            confidence: strength,
        });
    }

    let state = classify_dial(angle, cfg.off_angle_deg, cfg.off_tolerance_deg);

    Ok(DialReading {
        label: cfg.label.clone(),
        state,
        angle_deg: Some(angle),
        confidence: strength,
    })
}

/// Core detection: scan radial lines on the edge image and find the angle
/// with the strongest mean edge response.
///
/// Returns `(best_angle_degrees, max_normalized_strength)`.
pub fn radial_edge_scan(
    edges: &Mat,
    cx: u32,
    cy: u32,
    radius: u32,
) -> Result<(f64, f64), opencv::Error> {
    let w = edges.cols() as u32;
    let h = edges.rows() as u32;
    let step = edges.step1(0)? as u32;
    let data = edges.data_bytes()?;

    let mut profile = [0.0f64; 360];

    for deg in 0u32..360 {
        let rad = (deg as f64).to_radians();
        let cos_a = rad.cos();
        let sin_a = rad.sin();

        let mut sum = 0.0f64;
        let mut count = 0u32;

        for s in 0..RADIAL_SAMPLES {
            // Sample from 50% to 100% of radius
            let frac = 0.5 + 0.5 * (s as f64) / ((RADIAL_SAMPLES - 1) as f64);
            let r = (radius as f64) * frac;

            let px = (cx as f64 + r * cos_a).round() as i64;
            let py = (cy as f64 + r * sin_a).round() as i64;

            if px >= 0 && py >= 0 && (px as u32) < w && (py as u32) < h {
                let idx = (py as u32) * step + (px as u32);
                sum += data[idx as usize] as f64;
                count += 1;
            }
        }

        profile[deg as usize] = if count > 0 { sum / count as f64 } else { 0.0 };
    }

    // Smooth the angular profile with a small Gaussian kernel (sigma ~2°, width 7)
    let kernel = [0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006];
    let half = (kernel.len() / 2) as i32;
    let mut smoothed = [0.0f64; 360];

    for i in 0..360 {
        let mut val = 0.0;
        for (k, &weight) in kernel.iter().enumerate() {
            let j = ((i as i32 + k as i32 - half) % 360 + 360) % 360;
            val += profile[j as usize] * weight;
        }
        smoothed[i] = val;
    }

    // Find the peak
    let mut best_angle = 0usize;
    let mut best_val = 0.0f64;
    for (i, &val) in smoothed.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_angle = i;
        }
    }

    // Normalize strength to 0.0-1.0 (edge pixels are 255 in Canny output)
    let normalized = best_val / 255.0;

    Ok((best_angle as f64, normalized))
}

/// Classify a detected angle into a `DialState` based on the calibrated
/// off-angle and tolerance.
pub fn classify_dial(angle_deg: f64, off_angle_deg: f64, off_tolerance_deg: f64) -> DialState {
    let diff = angular_distance(angle_deg, off_angle_deg);

    if diff.abs() <= off_tolerance_deg {
        return DialState::Off;
    }

    // Map angular distance from off position to heat level.
    // The distance ranges from tolerance to MAX_ROTATION_DEG.
    let effective = diff.abs() - off_tolerance_deg;
    let range = MAX_ROTATION_DEG - off_tolerance_deg;
    let frac = (effective / range).clamp(0.0, 1.0);

    let level = if frac < 0.25 {
        HeatLevel::Low
    } else if frac < 0.50 {
        HeatLevel::Medium
    } else if frac < 0.75 {
        HeatLevel::High
    } else {
        HeatLevel::Max
    };

    DialState::On(level)
}

/// Signed angular difference in [-180, 180].
fn angular_distance(a: f64, b: f64) -> f64 {
    ((a - b) % 360.0 + 540.0) % 360.0 - 180.0
}

impl std::fmt::Display for HeatLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
            Self::Max => write!(f, "max"),
        }
    }
}

impl std::fmt::Display for DialState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Off => write!(f, "off"),
            Self::On(level) => write!(f, "on ({level:?})"),
            Self::Unavailable => write!(f, "unavailable"),
        }
    }
}

impl std::fmt::Display for DialReading {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.angle_deg {
            Some(angle) => write!(
                f,
                "{}: {} (angle={:.1}, confidence={:.2})",
                self.label, self.state, angle, self.confidence
            ),
            None => write!(
                f,
                "{}: {} (confidence={:.2})",
                self.label, self.state, self.confidence
            ),
        }
    }
}
