use opencv::core::{BORDER_CONSTANT, Mat, Point2f, Rect, Scalar, Size};
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

/// Number of consecutive frames required to confirm a state change.
/// At 1 fps with 5s poll interval, this means ~15 seconds of consistent
/// readings before flipping state.
const HYSTERESIS_FRAMES: u32 = 3;

/// Pre-computed rotation angles for template matching (0-340 in 20-degree steps).
const TEMPLATE_ANGLES: [f64; 18] = [
    0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 240.0, 260.0,
    280.0, 300.0, 320.0, 340.0,
];

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

    /// Detect all dials using template matching on the warped grayscale image.
    ///
    /// For each knob, extracts an ROI and matches the knob template at several
    /// rotation angles. The best-matching angle gives the handle position.
    /// angle=0° is the template's native orientation (OFF position).
    ///
    /// `knob_template` should be a grayscale `Mat` of the reference knob photo,
    /// pre-scaled to match the warped image resolution.
    pub fn detect_all_template(
        &mut self,
        warped_gray: &Mat,
        knob_template: &Mat,
    ) -> Result<Vec<DialReading>, opencv::Error> {
        let mut results = Vec::with_capacity(self.configs.len());

        for i in 0..self.configs.len() {
            let raw = detect_single_template(
                warped_gray,
                knob_template,
                &self.configs[i],
                &TEMPLATE_ANGLES,
            )?;

            self.latest_angle[i] = raw.angle_deg;
            self.latest_confidence[i] = raw.confidence;

            self.apply_hysteresis(i, raw.state);

            results.push(DialReading {
                label: self.configs[i].label.clone(),
                state: self.confirmed[i],
                angle_deg: self.latest_angle[i],
                confidence: self.latest_confidence[i],
            });
        }

        Ok(results)
    }

    fn apply_hysteresis(&mut self, i: usize, raw_state: DialState) {
        let same_category = state_category(raw_state) == state_category(self.confirmed[i]);
        if same_category {
            self.transition_count[i] = 0;
        } else if state_category(raw_state) == state_category(self.candidate[i]) {
            self.transition_count[i] += 1;
            if self.transition_count[i] >= HYSTERESIS_FRAMES {
                self.confirmed[i] = raw_state;
                self.transition_count[i] = 0;
            }
        } else {
            self.candidate[i] = raw_state;
            self.transition_count[i] = 1;
        }
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

/// Detect a single knob's state by template matching at multiple rotations.
///
/// Returns the best-matching rotation angle (0° = template as-is = OFF)
/// and the match confidence. The off_angle from `cfg` is the calibration
/// baseline (always 0° for template-based detection since the template
/// IS the off position).
fn detect_single_template(
    warped_gray: &Mat,
    knob_template: &Mat,
    cfg: &DialConfig,
    angles: &[f64],
) -> Result<DialReading, opencv::Error> {
    let cx = cfg.center_x as i32;
    let cy = cfg.center_y as i32;
    let r = cfg.radius as i32;
    let margin = r * 2;

    // Extract ROI around the knob (with margin for rotation)
    let img_w = warped_gray.cols();
    let img_h = warped_gray.rows();
    let x0 = (cx - margin).max(0);
    let y0 = (cy - margin).max(0);
    let x1 = (cx + margin).min(img_w);
    let y1 = (cy + margin).min(img_h);

    if x1 - x0 < knob_template.cols() || y1 - y0 < knob_template.rows() {
        return Ok(DialReading {
            label: cfg.label.clone(),
            state: DialState::Unavailable,
            angle_deg: None,
            confidence: 0.0,
        });
    }

    let roi = Mat::roi(warped_gray, Rect::new(x0, y0, x1 - x0, y1 - y0))?;

    let mut best_angle = 0.0f64;
    let mut best_score = -1.0f64;

    let templ_cx = knob_template.cols() as f64 / 2.0;
    let templ_cy = knob_template.rows() as f64 / 2.0;

    for &angle in angles {
        // Rotate the template
        let rot_mat = imgproc::get_rotation_matrix_2d(
            Point2f::new(templ_cx as f32, templ_cy as f32),
            -angle,
            1.0,
        )?;
        let mut rotated = Mat::default();
        imgproc::warp_affine(
            knob_template,
            &mut rotated,
            &rot_mat,
            Size::new(knob_template.cols(), knob_template.rows()),
            imgproc::INTER_LINEAR,
            BORDER_CONSTANT,
            Scalar::default(),
        )?;

        if rotated.cols() > roi.cols() || rotated.rows() > roi.rows() {
            continue;
        }

        let mut result = Mat::default();
        imgproc::match_template(
            &roi,
            &rotated,
            &mut result,
            imgproc::TM_CCOEFF_NORMED,
            &Mat::default(),
        )?;

        // Find max value
        let mut min_val = 0.0;
        let mut max_val = 0.0;
        opencv::core::min_max_loc(
            &result,
            Some(&mut min_val),
            Some(&mut max_val),
            None,
            None,
            &Mat::default(),
        )?;

        if max_val > best_score {
            best_score = max_val;
            best_angle = angle;
        }
    }

    // Classify: angle 0 = OFF (template position), increasing angle = more ON
    // The off_angle_deg should be 0 for template-based detection
    let state = classify_dial(best_angle, cfg.off_angle_deg, cfg.off_tolerance_deg);

    Ok(DialReading {
        label: cfg.label.clone(),
        state,
        angle_deg: Some(best_angle),
        confidence: best_score.max(0.0),
    })
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
