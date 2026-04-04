use serde::{Deserialize, Serialize};

/// A rectangular crop region in the full frame.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CropRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// A line defined by two endpoints (x1,y1) → (x2,y2).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Line {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

/// The two dominant horizontal reference lines from the chrome trim.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LinePair {
    pub top: Line,
    pub bottom: Line,
    /// Average Hough theta of the pair (radians). Horizontal = PI/2.
    /// Used by S3 to determine the expected perpendicular direction.
    #[serde(default = "default_avg_theta")]
    pub avg_theta: f64,
}

fn default_avg_theta() -> f64 {
    std::f64::consts::FRAC_PI_2
}

/// The two dominant vertical reference lines from the stove panel edges.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VerticalPair {
    pub left: Line,
    pub right: Line,
}

/// A 3x3 perspective transform matrix, stored row-major.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TransformMatrix(pub [[f64; 3]; 3]);

/// Perspective correction output: the transform matrix and the warped image dimensions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PerspectiveCorrection {
    pub matrix: TransformMatrix,
    pub output_width: u32,
    pub output_height: u32,
}

/// The search area for knob detection, computed from the band + clock stages.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KnobSearchArea {
    /// Top of the knob band in warped-image Y coordinates.
    pub y_min: f64,
    /// Bottom of the knob band in warped-image Y coordinates.
    pub y_max: f64,
    /// Left edge of the knob area (right side of the clock).
    pub x_min: f64,
    /// Clock center position in warped-image coordinates.
    pub clock_center_x: f64,
    pub clock_center_y: f64,
    pub clock_radius: f64,
    /// Top-right corner of the oven panel (set by FindCorner).
    #[serde(default)]
    pub corner_x: Option<f64>,
    #[serde(default)]
    pub corner_y: Option<f64>,
}

/// A detected circular feature (knob or clock).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CircleFeature {
    pub center_x: f64,
    pub center_y: f64,
    pub radius: f64,
}

/// All detected features in the perspective-corrected image.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DetectedFeatures {
    pub clock: CircleFeature,
    pub knobs: Vec<CircleFeature>,
    /// Per-knob off-angle (degrees), indexed same as `knobs`.
    pub off_angles: Vec<f64>,
}

/// Cumulative pipeline state — each stage populates its field.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PipelineState {
    pub crop: Option<CropRegion>,
    pub lines: Option<LinePair>,
    pub verticals: Option<VerticalPair>,
    pub perspective: Option<PerspectiveCorrection>,
    pub knob_search: Option<KnobSearchArea>,
    pub features: Option<DetectedFeatures>,
    /// Set to true after the sanity check stage passes.
    pub validated: bool,
}

/// Metadata that each stage declares about itself.
#[derive(Debug, Clone, Copy)]
pub struct StageDescriptor {
    /// Unique name, e.g. "FindStove". Used for cache hashing and fallback resolution.
    pub name: &'static str,
    /// Display label, e.g. "S1:FindStove". Used in logs and debug image filenames.
    pub label: &'static str,
    /// Name of the stage to fall back to on exhaustion. None = pipeline fails.
    pub fallback: Option<&'static str>,
}

/// Outcome of running a single stage attempt.
pub enum StageOutcome {
    /// Stage succeeded — advance to the next stage.
    Success,
    /// Stage failed but can be retried with different parameters.
    Retry(String),
    /// Stage exhausted all retries — fall back to the previous stage.
    Exhausted(String),
}
