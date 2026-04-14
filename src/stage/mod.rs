pub mod annotate;
pub mod detect;
pub mod enhance;
pub mod reproject;
pub mod sanity;
pub mod undistort;
pub mod warp;

use opencv::core::Mat;

/// Accumulated state as a frame passes through pipeline stages.
pub struct FrameState {
    pub raw: Mat,
    pub undistorted: Option<Mat>,
    pub warped: Option<Mat>,
    pub enhanced: Option<Mat>,
    pub knobs: Vec<detect::Knob>,
    pub sanity: Option<sanity::SanityResult>,
    pub annotated: Option<Mat>,
    pub debug_edges: Option<Mat>,
}

impl FrameState {
    pub fn new(raw: Mat) -> Self {
        Self {
            raw,
            undistorted: None,
            warped: None,
            enhanced: None,
            knobs: Vec::new(),
            sanity: None,
            annotated: None,
            debug_edges: None,
        }
    }

    /// The best available image at the current pipeline position.
    pub fn current_image(&self) -> &Mat {
        self.warped
            .as_ref()
            .or(self.undistorted.as_ref())
            .unwrap_or(&self.raw)
    }
}

/// A pipeline stage that transforms frame state in place.
pub trait Stage: Send + Sync {
    fn name(&self) -> &'static str;
    fn process(&self, state: &mut FrameState) -> Result<(), StageError>;
}

#[derive(Debug)]
pub struct StageError {
    pub stage: &'static str,
    pub message: String,
}

impl std::fmt::Display for StageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.stage, self.message)
    }
}

impl std::error::Error for StageError {}
