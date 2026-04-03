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
    pub features: Option<DetectedFeatures>,
    /// Set to true after the sanity check stage passes.
    pub validated: bool,
}

/// The 6 pipeline stages, in execution order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StageId {
    FindStove = 0,
    FindLines = 1,
    FindVerticals = 2,
    Perspective = 3,
    FindFeatures = 4,
    SanityCheck = 5,
}

impl StageId {
    /// The previous stage to fall back to, if any.
    pub fn fallback(self) -> Option<StageId> {
        match self {
            StageId::FindStove => None,
            StageId::FindLines => Some(StageId::FindStove),
            StageId::FindVerticals => Some(StageId::FindLines),
            StageId::Perspective => Some(StageId::FindVerticals),
            StageId::FindFeatures => Some(StageId::Perspective),
            // SanityCheck failures (Y-deviation, X-gap) indicate a perspective
            // or vertical selection problem, not a circle detection problem.
            // Fall back to FindVerticals so a different pair is tried.
            StageId::SanityCheck => Some(StageId::FindVerticals),
        }
    }

    /// Numeric index of this stage (matches enum discriminant).
    pub fn index(self) -> usize {
        self as usize
    }
}

impl std::fmt::Display for StageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StageId::FindStove => write!(f, "S1:FindStove"),
            StageId::FindLines => write!(f, "S2:FindLines"),
            StageId::FindVerticals => write!(f, "S2b:FindVerticals"),
            StageId::Perspective => write!(f, "S3:Perspective"),
            StageId::FindFeatures => write!(f, "S4:FindFeatures"),
            StageId::SanityCheck => write!(f, "S5:SanityCheck"),
        }
    }
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
