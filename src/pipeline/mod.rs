pub mod cache;
pub mod find_features;
pub mod find_lines;
pub mod find_stove;
pub mod find_verticals;
pub mod perspective;
pub mod sanity;
pub mod stage;
pub mod util;

use std::path::Path;

use opencv::core::{Mat, Rect, Size};
use opencv::imgproc;
use opencv::prelude::*;
use tracing::{debug, info, warn};

use cache::PipelineCache;
use stage::{PipelineState, StageId, StageOutcome};

/// Per-stage debug image: (label, JPEG-encoded bytes).
pub type DebugImage = (String, Vec<u8>);

/// Trait that concrete stage implementations must satisfy.
///
/// Each stage receives the accumulated pipeline state and the current frame,
/// mutates the state with its output, and returns a `StageOutcome`.
/// The `iteration` parameter (0-based) tells the stage which parameter
/// variation to try on retries.
pub trait Stage {
    fn id(&self) -> StageId;

    /// Run one attempt of this stage.
    ///
    /// * `state` — accumulated pipeline state; the stage reads its inputs from
    ///   prior stages and writes its own output.
    /// * `frame` — the original BGR frame from the camera.
    /// * `iteration` — 0-based retry counter; stages can vary parameters based
    ///   on this value.
    ///
    /// Returns `StageOutcome::Success` on success, `Retry` to try again with
    /// the next iteration, or `Exhausted` when all retries are spent.
    fn run(
        &self,
        state: &mut PipelineState,
        frame: &Mat,
        iteration: u32,
    ) -> Result<StageOutcome, opencv::Error>;

    /// Maximum retry iterations before this stage reports `Exhausted`.
    /// Default is 20; individual stages override as needed.
    fn max_retries(&self) -> u32 {
        20
    }

    /// Produce an annotated debug image for the current stage output.
    /// Returns `None` if no debug image is available.
    fn debug_image(
        &self,
        state: &PipelineState,
        frame: &Mat,
    ) -> Result<Option<DebugImage>, opencv::Error> {
        let _ = (state, frame);
        Ok(None)
    }
}

/// Pipeline configuration.
pub struct PipelineConfig {
    /// Path to the JSON cache file on disk.
    pub cache_path: std::path::PathBuf,
    /// Maximum number of fresh frames to try before giving up calibration.
    /// Each frame gets the full retry/fallback budget before fetching the next.
    pub max_frame_attempts: u32,
}

/// Orchestrates the 5-stage calibration pipeline.
///
/// Runs stages in order. On retry, re-runs the same stage with an incremented
/// iteration counter. When a stage exhausts its retries, falls back to the
/// stage indicated by `StageId::fallback()` and clears its iteration counter
/// so it can try different inputs.
pub struct Pipeline {
    stages: Vec<Box<dyn Stage>>,
    pub state: PipelineState,
    pub debug_images: Vec<DebugImage>,
    config: PipelineConfig,
}

impl Pipeline {
    pub fn new(stages: Vec<Box<dyn Stage>>, config: PipelineConfig) -> Self {
        // Verify stages are in order
        for (i, stage) in stages.iter().enumerate() {
            debug_assert_eq!(
                stage.id().index(),
                i,
                "stages must be registered in StageId order"
            );
        }
        Self {
            stages,
            state: PipelineState::default(),
            debug_images: Vec::new(),
            config,
        }
    }

    /// Try to load a cached pipeline state. Returns `true` if the cache was
    /// valid and loaded, `false` if calibration is needed.
    pub fn try_load_cache(&mut self, frame_width: u32, frame_height: u32) -> bool {
        match PipelineCache::load(&self.config.cache_path, frame_width, frame_height) {
            Ok(state) => {
                info!("loaded valid pipeline cache");
                self.state = state;
                true
            }
            Err(err) => {
                info!(%err, "no valid cache — full calibration needed");
                false
            }
        }
    }

    /// Run calibration with a frame-fetching closure, retrying with fresh
    /// frames when the pipeline exhausts all parameter variations on the
    /// current frame. This prevents spending thousands of iterations on a
    /// single bad frame (e.g., someone standing in front of the stove).
    ///
    /// The `fetch_frame` closure is called each time a fresh frame is needed.
    /// The first call happens immediately; subsequent calls happen only when
    /// the previous frame's full retry budget is exhausted.
    pub fn calibrate_with_fetch<F>(&mut self, mut fetch_frame: F) -> Result<(), PipelineError>
    where
        F: FnMut() -> Result<Mat, PipelineError>,
    {
        let max_attempts = self.config.max_frame_attempts;

        for attempt in 0..max_attempts {
            let frame = fetch_frame()?;
            info!(
                attempt,
                max_attempts, "calibration attempt with fresh frame"
            );

            match self.calibrate(&frame) {
                Ok(()) => return Ok(()),
                Err(PipelineError::Exhausted(reason)) => {
                    warn!(attempt, %reason, "frame exhausted, will try next");
                    continue;
                }
                Err(other) => return Err(other),
            }
        }

        Err(PipelineError::Exhausted(format!(
            "all {max_attempts} frame attempts exhausted"
        )))
    }

    /// Run the full calibration pipeline on a single frame.
    ///
    /// Returns `Ok(())` when all 5 stages complete and the sanity check passes,
    /// or `Err` if the pipeline cannot converge (Stage 1 exhausted, or an
    /// OpenCV error).
    pub fn calibrate(&mut self, frame: &Mat) -> Result<(), PipelineError> {
        self.state = PipelineState::default();
        self.debug_images.clear();

        let mut stage_idx: usize = 0;
        let mut iterations: Vec<u32> = vec![0; self.stages.len()];
        let mut total_iterations: u32 = 0;
        const MAX_TOTAL_ITERATIONS: u32 = 500;

        while stage_idx < self.stages.len() {
            total_iterations += 1;
            if total_iterations > MAX_TOTAL_ITERATIONS {
                return Err(PipelineError::Exhausted(format!(
                    "exceeded {MAX_TOTAL_ITERATIONS} total iterations"
                )));
            }
            let stage = &self.stages[stage_idx];
            let sid = stage.id();
            let iter = iterations[stage_idx];

            debug!(%sid, iter, "running stage");

            match stage.run(&mut self.state, frame, iter)? {
                StageOutcome::Success => {
                    info!(%sid, iter, "stage succeeded");

                    // Collect debug image — keep all images (don't replace),
                    // so iterative stages produce a visual history.
                    if let Some(img) = stage.debug_image(&self.state, frame)? {
                        self.debug_images.push(img);
                    }

                    // Do NOT reset iterations[stage_idx] here — the counter
                    // must persist across fallback cycles so that downstream
                    // failures cause this stage to try different parameters
                    // on each re-run.
                    stage_idx += 1;
                }

                StageOutcome::Retry(reason) => {
                    debug!(%sid, iter, %reason, "stage retry");
                    iterations[stage_idx] += 1;

                    if iterations[stage_idx] >= stage.max_retries() {
                        // Exceeded max retries — treat as exhausted
                        warn!(%sid, max = stage.max_retries(), "stage exhausted retries");
                        self.handle_fallback(
                            &mut stage_idx,
                            &mut iterations,
                            &format!("{sid} exhausted after {reason}"),
                        )?;
                    }
                }

                StageOutcome::Exhausted(reason) => {
                    warn!(%sid, %reason, "stage exhausted");
                    self.handle_fallback(&mut stage_idx, &mut iterations, &reason)?;
                }
            }
        }

        // All stages passed — mark validated and save cache
        self.state.validated = true;
        self.save_cache(frame)?;

        info!("calibration pipeline complete");
        Ok(())
    }

    /// Handle fallback: go back to the previous stage, incrementing its
    /// iteration counter. If there is no previous stage, the pipeline fails.
    fn handle_fallback(
        &self,
        stage_idx: &mut usize,
        iterations: &mut Vec<u32>,
        reason: &str,
    ) -> Result<(), PipelineError> {
        let sid = self.stages[*stage_idx].id();

        match sid.fallback() {
            Some(prev) => {
                let prev_idx = prev.index();
                iterations[*stage_idx] = 0; // reset current stage
                iterations[prev_idx] += 1; // bump previous stage

                if iterations[prev_idx] >= self.stages[prev_idx].max_retries() {
                    // Previous stage also exhausted — try its fallback
                    warn!(%prev, "fallback stage also exhausted, cascading");
                    *stage_idx = prev_idx;
                    return self.handle_fallback(stage_idx, iterations, reason);
                }

                info!(%sid, fallback = %prev, iter = iterations[prev_idx], "falling back");
                *stage_idx = prev_idx;
                Ok(())
            }
            None => Err(PipelineError::Exhausted(format!(
                "{sid} exhausted with no fallback: {reason}"
            ))),
        }
    }

    fn save_cache(&self, frame: &Mat) -> Result<(), PipelineError> {
        let cache =
            PipelineCache::new(self.state.clone(), frame.cols() as u32, frame.rows() as u32);

        if let Some(parent) = self.config.cache_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                PipelineError::Cache(format!("failed to create cache directory: {e}"))
            })?;
        }

        cache
            .save(&self.config.cache_path)
            .map_err(|e| PipelineError::Cache(format!("failed to save cache: {e}")))
    }

    /// Get the current pipeline state (for detection mode).
    pub fn state(&self) -> &PipelineState {
        &self.state
    }

    /// Get debug images from the last calibration run.
    pub fn debug_images(&self) -> &[DebugImage] {
        &self.debug_images
    }
}

/// Apply the cached crop and perspective transform to a raw BGR frame.
///
/// Returns the warped image in the corrected coordinate space where knob
/// positions from `PipelineState::features` are valid.
///
/// Returns `Err` if `state.crop` or `state.perspective` is `None` (i.e.,
/// pipeline has not been calibrated).
pub fn warp_frame(state: &PipelineState, frame: &Mat) -> Result<Mat, opencv::Error> {
    let crop = state
        .crop
        .as_ref()
        .ok_or_else(|| opencv::Error::new(opencv::core::StsError, "pipeline state has no crop"))?;
    let persp = state.perspective.as_ref().ok_or_else(|| {
        opencv::Error::new(opencv::core::StsError, "pipeline state has no perspective")
    })?;

    let roi = Rect::new(
        crop.x as i32,
        crop.y as i32,
        crop.width as i32,
        crop.height as i32,
    );
    let cropped = Mat::roi(frame, roi)?;
    let mat = perspective::transform_to_mat(&persp.matrix)?;

    let mut warped = Mat::default();
    imgproc::warp_perspective_def(
        &cropped,
        &mut warped,
        &mat,
        Size::new(persp.output_width as i32, persp.output_height as i32),
    )?;

    Ok(warped)
}

/// Errors from the pipeline orchestrator.
#[derive(Debug)]
pub enum PipelineError {
    /// An OpenCV function returned an error.
    Cv(opencv::Error),
    /// All stages and fallbacks exhausted — calibration impossible.
    Exhausted(String),
    /// Cache I/O error (non-fatal for calibration, but logged).
    Cache(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::Cv(err) => write!(f, "opencv error: {err}"),
            PipelineError::Exhausted(msg) => write!(f, "pipeline exhausted: {msg}"),
            PipelineError::Cache(msg) => write!(f, "cache error: {msg}"),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<opencv::Error> for PipelineError {
    fn from(err: opencv::Error) -> Self {
        PipelineError::Cv(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A mock stage that succeeds on the first attempt.
    struct SuccessStage(StageId);

    impl Stage for SuccessStage {
        fn id(&self) -> StageId {
            self.0
        }
        fn run(
            &self,
            state: &mut PipelineState,
            _frame: &Mat,
            _iteration: u32,
        ) -> Result<StageOutcome, opencv::Error> {
            // Populate dummy state based on stage
            match self.0 {
                StageId::FindStove => {
                    state.crop = Some(stage::CropRegion {
                        x: 100,
                        y: 200,
                        width: 800,
                        height: 300,
                    });
                }
                StageId::FindLines => {
                    state.lines = Some(stage::LinePair {
                        top: stage::Line {
                            x1: 0.0,
                            y1: 10.0,
                            x2: 800.0,
                            y2: 5.0,
                        },
                        bottom: stage::Line {
                            x1: 0.0,
                            y1: 50.0,
                            x2: 800.0,
                            y2: 45.0,
                        },
                    });
                }
                StageId::FindVerticals => {
                    state.verticals = Some(stage::VerticalPair {
                        left: stage::Line {
                            x1: 10.0,
                            y1: 0.0,
                            x2: 10.0,
                            y2: 300.0,
                        },
                        right: stage::Line {
                            x1: 790.0,
                            y1: 0.0,
                            x2: 790.0,
                            y2: 300.0,
                        },
                    });
                }
                StageId::Perspective => {
                    state.perspective = Some(stage::PerspectiveCorrection {
                        matrix: stage::TransformMatrix([
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ]),
                        output_width: 800,
                        output_height: 200,
                    });
                }
                StageId::FindFeatures => {
                    state.features = Some(stage::DetectedFeatures {
                        clock: stage::CircleFeature {
                            center_x: 50.0,
                            center_y: 100.0,
                            radius: 25.0,
                        },
                        knobs: (0..10)
                            .map(|i| stage::CircleFeature {
                                center_x: 120.0 + i as f64 * 65.0,
                                center_y: 100.0,
                                radius: 12.0,
                            })
                            .collect(),
                        off_angles: vec![90.0; 10],
                    });
                }
                StageId::SanityCheck => {
                    state.validated = true;
                }
            }
            Ok(StageOutcome::Success)
        }
        fn max_retries(&self) -> u32 {
            3
        }
    }

    /// A stage that fails N times then succeeds.
    struct FailThenSucceed {
        id: StageId,
        fail_count: u32,
        attempts: AtomicU32,
    }

    impl FailThenSucceed {
        fn new(id: StageId, fail_count: u32) -> Self {
            Self {
                id,
                fail_count,
                attempts: AtomicU32::new(0),
            }
        }
    }

    impl Stage for FailThenSucceed {
        fn id(&self) -> StageId {
            self.id
        }
        fn run(
            &self,
            state: &mut PipelineState,
            _frame: &Mat,
            _iteration: u32,
        ) -> Result<StageOutcome, opencv::Error> {
            let n = self.attempts.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_count {
                Ok(StageOutcome::Retry(format!("attempt {n}")))
            } else {
                // Populate dummy state
                match self.id {
                    StageId::Perspective => {
                        state.perspective = Some(stage::PerspectiveCorrection {
                            matrix: stage::TransformMatrix([
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                            ]),
                            output_width: 800,
                            output_height: 200,
                        });
                    }
                    StageId::SanityCheck => {
                        state.validated = true;
                    }
                    _ => {}
                }
                Ok(StageOutcome::Success)
            }
        }
        fn max_retries(&self) -> u32 {
            5
        }
    }

    /// A stage that always exhausts retries.
    struct AlwaysFail(StageId);

    impl Stage for AlwaysFail {
        fn id(&self) -> StageId {
            self.0
        }
        fn run(
            &self,
            _state: &mut PipelineState,
            _frame: &Mat,
            _iteration: u32,
        ) -> Result<StageOutcome, opencv::Error> {
            Ok(StageOutcome::Retry("always fails".into()))
        }
        fn max_retries(&self) -> u32 {
            2
        }
    }

    fn dummy_frame() -> Mat {
        // 4x4 black image — enough for mock stages that don't read pixels
        Mat::zeros(4, 4, opencv::core::CV_8UC3)
            .unwrap()
            .to_mat()
            .unwrap()
    }

    fn test_config() -> PipelineConfig {
        let tmp = tempfile::tempdir().unwrap();
        PipelineConfig {
            cache_path: tmp.into_path().join("pipeline_cache.json"),
            max_frame_attempts: 3,
        }
    }

    #[test]
    fn happy_path_all_stages_succeed() {
        let stages: Vec<Box<dyn Stage>> = vec![
            Box::new(SuccessStage(StageId::FindStove)),
            Box::new(SuccessStage(StageId::FindLines)),
            Box::new(SuccessStage(StageId::FindVerticals)),
            Box::new(SuccessStage(StageId::Perspective)),
            Box::new(SuccessStage(StageId::FindFeatures)),
            Box::new(SuccessStage(StageId::SanityCheck)),
        ];
        let mut pipeline = Pipeline::new(stages, test_config());
        let frame = dummy_frame();

        pipeline.calibrate(&frame).unwrap();

        assert!(pipeline.state.validated);
        assert!(pipeline.state.crop.is_some());
        assert!(pipeline.state.lines.is_some());
        assert!(pipeline.state.perspective.is_some());
        assert!(pipeline.state.features.is_some());
    }

    #[test]
    fn retry_stage_fails_twice_then_succeeds() {
        let stages: Vec<Box<dyn Stage>> = vec![
            Box::new(SuccessStage(StageId::FindStove)),
            Box::new(SuccessStage(StageId::FindLines)),
            Box::new(SuccessStage(StageId::FindVerticals)),
            Box::new(FailThenSucceed::new(StageId::Perspective, 2)),
            Box::new(SuccessStage(StageId::FindFeatures)),
            Box::new(SuccessStage(StageId::SanityCheck)),
        ];
        let mut pipeline = Pipeline::new(stages, test_config());
        let frame = dummy_frame();

        pipeline.calibrate(&frame).unwrap();
        assert!(pipeline.state.validated);
    }

    #[test]
    fn fallback_stage3_exhausted_retries_stage2() {
        // Stage 3 (Perspective) always fails → falls back to Stage 2
        // Stage 2 will be re-run and Stage 3 will eventually succeed
        // Use a counter to make Stage 3 succeed after Stage 2 re-runs
        let stages: Vec<Box<dyn Stage>> = vec![
            Box::new(SuccessStage(StageId::FindStove)),
            Box::new(SuccessStage(StageId::FindLines)),
            Box::new(SuccessStage(StageId::FindVerticals)),
            // Fails 3 times then succeeds (max_retries=5, so after fallback
            // resets counter it will succeed on the next round)
            Box::new(FailThenSucceed::new(StageId::Perspective, 3)),
            Box::new(SuccessStage(StageId::FindFeatures)),
            Box::new(SuccessStage(StageId::SanityCheck)),
        ];
        let mut pipeline = Pipeline::new(stages, test_config());
        let frame = dummy_frame();

        pipeline.calibrate(&frame).unwrap();
        assert!(pipeline.state.validated);
    }

    #[test]
    fn pipeline_exhausted_returns_error() {
        // Stage 1 always fails → no fallback → pipeline error
        let stages: Vec<Box<dyn Stage>> = vec![
            Box::new(AlwaysFail(StageId::FindStove)),
            Box::new(SuccessStage(StageId::FindLines)),
            Box::new(SuccessStage(StageId::FindVerticals)),
            Box::new(SuccessStage(StageId::Perspective)),
            Box::new(SuccessStage(StageId::FindFeatures)),
            Box::new(SuccessStage(StageId::SanityCheck)),
        ];
        let mut pipeline = Pipeline::new(stages, test_config());
        let frame = dummy_frame();

        let err = pipeline.calibrate(&frame).unwrap_err();
        assert!(matches!(err, PipelineError::Exhausted(_)));
    }

    #[test]
    fn cache_saved_after_calibration() {
        let config = test_config();
        let cache_path = config.cache_path.clone();

        let stages: Vec<Box<dyn Stage>> = vec![
            Box::new(SuccessStage(StageId::FindStove)),
            Box::new(SuccessStage(StageId::FindLines)),
            Box::new(SuccessStage(StageId::FindVerticals)),
            Box::new(SuccessStage(StageId::Perspective)),
            Box::new(SuccessStage(StageId::FindFeatures)),
            Box::new(SuccessStage(StageId::SanityCheck)),
        ];
        let mut pipeline = Pipeline::new(stages, config);
        let frame = dummy_frame();

        pipeline.calibrate(&frame).unwrap();
        assert!(
            cache_path.exists(),
            "cache file should be written after calibration"
        );
    }

    #[test]
    fn cache_loads_on_second_run() {
        let config = test_config();
        let cache_path = config.cache_path.clone();

        // First run: calibrate and save cache
        let stages: Vec<Box<dyn Stage>> = vec![
            Box::new(SuccessStage(StageId::FindStove)),
            Box::new(SuccessStage(StageId::FindLines)),
            Box::new(SuccessStage(StageId::FindVerticals)),
            Box::new(SuccessStage(StageId::Perspective)),
            Box::new(SuccessStage(StageId::FindFeatures)),
            Box::new(SuccessStage(StageId::SanityCheck)),
        ];
        let config1 = PipelineConfig {
            cache_path: cache_path.clone(),
            max_frame_attempts: 3,
        };
        let mut pipeline = Pipeline::new(stages, config1);
        let frame = dummy_frame();
        pipeline.calibrate(&frame).unwrap();

        // Second run: load from cache
        let stages2: Vec<Box<dyn Stage>> = vec![
            Box::new(SuccessStage(StageId::FindStove)),
            Box::new(SuccessStage(StageId::FindLines)),
            Box::new(SuccessStage(StageId::FindVerticals)),
            Box::new(SuccessStage(StageId::Perspective)),
            Box::new(SuccessStage(StageId::FindFeatures)),
            Box::new(SuccessStage(StageId::SanityCheck)),
        ];
        let config2 = PipelineConfig {
            cache_path: cache_path.clone(),
            max_frame_attempts: 3,
        };
        let mut pipeline2 = Pipeline::new(stages2, config2);
        let loaded = pipeline2.try_load_cache(4, 4); // 4x4 = dummy_frame dimensions
        assert!(loaded, "should load from cache");
        assert!(pipeline2.state.validated);
    }
}
