pub mod cache;
pub mod extract_band;
pub mod final_check;
pub mod final_detect;
pub mod find_clock;
pub mod find_corner;
pub mod find_features;
pub mod find_lines;
pub mod find_stove;
pub mod find_verticals;
pub mod perspective;
pub mod refine_warp;
pub mod sanity;
pub mod stage;
pub mod util;
pub mod warp_check;

use opencv::core::{Mat, Rect, Size};
use opencv::imgproc;
use opencv::prelude::*;
use tracing::{debug, info, warn};

use cache::PipelineCache;
use stage::{PipelineState, StageDescriptor, StageOutcome};

/// Per-stage debug image: (label, JPEG-encoded bytes).
pub type DebugImage = (String, Vec<u8>);

/// Whether a stage produced a new image in dst or passed through src unchanged.
pub enum ImageOutput {
    /// Stage wrote a transformed image into dst. Orchestrator swaps buffers.
    Transformed,
    /// Stage did not modify the image. Orchestrator keeps src as working image.
    Passthrough,
}

/// Trait that concrete stage implementations must satisfy.
///
/// Each stage receives the accumulated pipeline state and a double-buffer pair
/// (src/dst), mutates the state with its output, and returns a `StageOutcome`
/// plus an `ImageOutput` indicating whether it wrote into dst.
///
/// The `raw` parameter is the original full-resolution frame from the camera.
/// The `iteration` parameter (0-based) tells the stage which parameter
/// variation to try on retries.
pub trait Stage {
    fn descriptor(&self) -> StageDescriptor;

    /// Run one attempt of this stage.
    ///
    /// * `state` — accumulated pipeline state; the stage reads its inputs from
    ///   prior stages and writes its own output.
    /// * `src` — the current working image from the previous stage.
    /// * `dst` — the output buffer; write here if transforming the image.
    /// * `raw` — the original BGR frame from the camera.
    /// * `iteration` — 0-based retry counter; stages can vary parameters based
    ///   on this value.
    ///
    /// Returns `(StageOutcome, ImageOutput)` — outcome indicates success/retry/
    /// exhausted, and image output indicates whether dst was written.
    fn run(
        &self,
        state: &mut PipelineState,
        src: &Mat,
        dst: &mut Mat,
        raw: &Mat,
        iteration: u32,
    ) -> Result<(StageOutcome, ImageOutput), opencv::Error>;

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
        working: &Mat,
        raw: &Mat,
    ) -> Result<Option<DebugImage>, opencv::Error> {
        let _ = (state, working, raw);
        Ok(None)
    }
}

/// Blanket impl so `Arc<T: Stage>` can be stored in `Box<dyn Stage>`.
impl<T: Stage> Stage for std::sync::Arc<T> {
    fn descriptor(&self) -> StageDescriptor {
        (**self).descriptor()
    }
    fn run(
        &self,
        state: &mut PipelineState,
        src: &Mat,
        dst: &mut Mat,
        raw: &Mat,
        iteration: u32,
    ) -> Result<(StageOutcome, ImageOutput), opencv::Error> {
        (**self).run(state, src, dst, raw, iteration)
    }
    fn max_retries(&self) -> u32 {
        (**self).max_retries()
    }
    fn debug_image(
        &self,
        state: &PipelineState,
        working: &Mat,
        raw: &Mat,
    ) -> Result<Option<DebugImage>, opencv::Error> {
        (**self).debug_image(state, working, raw)
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
/// stage indicated by its `fallback` descriptor and clears its iteration counter
/// so it can try different inputs.
pub struct Pipeline {
    stages: Vec<Box<dyn Stage>>,
    /// Resolved fallback index for each stage. `None` = no fallback (pipeline fails).
    fallback_idx: Vec<Option<usize>>,
    /// Cache version computed from stage descriptors.
    cache_version: u64,
    pub state: PipelineState,
    pub debug_images: Vec<DebugImage>,
    config: PipelineConfig,
    /// Comma-separated stage filters from DEBUG_STAGES env var.
    /// Empty = all stages. "none" = no stages.
    debug_filter: Vec<String>,
    /// Optional callback fired each time a debug image is produced.
    /// Arguments: (sequence index, label, jpeg bytes).
    on_debug_image: Option<Box<dyn Fn(usize, &str, &[u8])>>,
}

impl Pipeline {
    pub fn new(stages: Vec<Box<dyn Stage>>, config: PipelineConfig) -> Self {
        // Build name-to-index map from stage descriptors
        let name_to_idx: std::collections::HashMap<&str, usize> = stages
            .iter()
            .enumerate()
            .map(|(i, s)| (s.descriptor().name, i))
            .collect();

        // Resolve each stage's fallback name to an index
        let fallback_idx: Vec<Option<usize>> = stages
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let desc = s.descriptor();
                desc.fallback.map(|fb_name| {
                    let idx = *name_to_idx.get(fb_name).unwrap_or_else(|| {
                        panic!(
                            "stage '{}' references unknown fallback '{}'",
                            desc.name, fb_name
                        )
                    });
                    assert!(
                        idx < i,
                        "stage '{}' fallback '{}' (idx {}) must precede it (idx {})",
                        desc.name,
                        fb_name,
                        idx,
                        i
                    );
                    idx
                })
            })
            .collect();

        let cache_version = cache::compute_cache_version(&stages);

        let debug_filter = std::env::var("DEBUG_STAGES")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        Self {
            stages,
            fallback_idx,
            cache_version,
            state: PipelineState::default(),
            debug_images: Vec::new(),
            config,
            debug_filter,
            on_debug_image: None,
        }
    }

    /// Set a callback that fires each time a debug image is produced during
    /// calibration. The callback receives (sequence_index, label, jpeg_bytes).
    pub fn set_debug_image_callback<F: Fn(usize, &str, &[u8]) + 'static>(&mut self, cb: F) {
        self.on_debug_image = Some(Box::new(cb));
    }

    /// Try to load a cached pipeline state. Returns `true` if the cache was
    /// valid and loaded, `false` if calibration is needed.
    pub fn try_load_cache(&mut self, frame_width: u32, frame_height: u32) -> bool {
        match PipelineCache::load(
            &self.config.cache_path,
            frame_width,
            frame_height,
            self.cache_version,
        ) {
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
        F: FnMut() -> Result<Mat, PipelineError> + Send + 'static,
    {
        let max_attempts = self.config.max_frame_attempts;

        // Fetch the first frame synchronously.
        let mut current_frame = fetch_frame()?;

        for attempt in 0..max_attempts {
            info!(
                attempt,
                max_attempts, "calibration attempt with fresh frame"
            );

            // Prefetch the next frame on a background thread while we process
            // the current one. If we succeed, the prefetched frame is discarded.
            let prefetch = if attempt + 1 < max_attempts {
                Some(std::thread::spawn(move || {
                    let result = fetch_frame();
                    (result, fetch_frame)
                }))
            } else {
                None
            };

            match self.calibrate(&current_frame) {
                Ok(()) => return Ok(()),
                Err(PipelineError::Exhausted(reason)) => {
                    warn!(attempt, %reason, "frame exhausted, will try next");
                    match prefetch {
                        Some(handle) => {
                            let (result, returned_fetch) = handle.join().unwrap_or_else(|e| {
                                std::panic::resume_unwind(e);
                            });
                            current_frame = result?;
                            fetch_frame = returned_fetch;
                        }
                        None => break,
                    }
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
    /// Uses an image stack: each stage that returns `ImageOutput::Transformed`
    /// pushes a new image onto the stack. The current working image is always
    /// the top of the stack. Passthrough stages leave the stack unchanged.
    ///
    /// On fallback, the stack is truncated to the depth it had when the
    /// fallback target stage originally ran, restoring the correct src image.
    /// This is safe because only stages *after* the fallback target pushed
    /// images, and those are the ones being discarded.
    ///
    /// Returns `Ok(())` when all stages complete and the sanity check passes,
    /// or `Err` if the pipeline cannot converge.
    pub fn calibrate(&mut self, frame: &Mat) -> Result<(), PipelineError> {
        self.state = PipelineState::default();
        self.debug_images.clear();

        // Image stack: Transformed stages push, fallback truncates.
        // The raw frame is the implicit bottom of the stack.
        let mut image_stack: Vec<Mat> = Vec::new();
        let mut dst_buf = Mat::default(); // reusable scratch buffer for dst

        let mut stage_idx: usize = 0;
        let mut iterations: Vec<u32> = vec![0; self.stages.len()];
        // Stack depth at entry to each stage — used to restore on fallback.
        let mut entry_depth: Vec<usize> = vec![0; self.stages.len()];
        while stage_idx < self.stages.len() {
            let stage = &self.stages[stage_idx];
            let label = stage.descriptor().label;
            let iter = iterations[stage_idx];

            // Record stack depth at entry for fallback recovery
            entry_depth[stage_idx] = image_stack.len();

            // Current working image: top of stack, or raw frame if stack is empty
            let src = image_stack.last().unwrap_or(frame);

            let debug_seq = self.debug_images.len();
            debug!(stage = label, iter, seq = debug_seq, "running stage");

            let (outcome, img_out) = stage.run(&mut self.state, src, &mut dst_buf, frame, iter)?;

            match outcome {
                StageOutcome::Success => {
                    // Push the new image if the stage transformed
                    if matches!(img_out, ImageOutput::Transformed) {
                        image_stack.push(std::mem::take(&mut dst_buf));
                    }

                    // Collect debug image and fire callback immediately
                    let img_seq = if self.should_debug(stage_idx) {
                        let working = image_stack.last().unwrap_or(frame);
                        if let Some(img) = stage.debug_image(&self.state, working, frame)? {
                            let idx = self.debug_images.len();
                            if let Some(cb) = &self.on_debug_image {
                                cb(idx, &img.0, &img.1);
                            }
                            self.debug_images.push(img);
                            Some(idx)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    info!(stage = label, iter, seq = debug_seq, img = ?img_seq, "stage succeeded");

                    stage_idx += 1;
                }

                StageOutcome::Retry(reason) => {
                    debug!(stage = label, iter, seq = debug_seq, %reason, "stage retry");
                    iterations[stage_idx] += 1;

                    if iterations[stage_idx] >= stage.max_retries() {
                        warn!(
                            stage = label,
                            max = stage.max_retries(),
                            seq = debug_seq,
                            "stage exhausted retries"
                        );
                        let fallback_target = self.resolve_fallback(stage_idx)?;
                        self.apply_fallback(
                            &mut stage_idx,
                            &mut iterations,
                            &mut image_stack,
                            &entry_depth,
                            fallback_target,
                            &format!("{label} exhausted after {reason}"),
                        )?;
                    }
                }

                StageOutcome::Exhausted(reason) => {
                    warn!(stage = label, seq = debug_seq, %reason, "stage exhausted");
                    let fallback_target = self.resolve_fallback(stage_idx)?;
                    self.apply_fallback(
                        &mut stage_idx,
                        &mut iterations,
                        &mut image_stack,
                        &entry_depth,
                        fallback_target,
                        &reason,
                    )?;
                }
            }
        }

        // All stages passed — mark validated and save cache
        self.state.validated = true;
        self.save_cache(frame)?;

        info!("calibration pipeline complete");
        Ok(())
    }

    /// Resolve the fallback target for a given stage. Returns the fallback
    /// stage index or an error if there is no fallback.
    fn resolve_fallback(&self, stage_idx: usize) -> Result<usize, PipelineError> {
        let label = self.stages[stage_idx].descriptor().label;
        match self.fallback_idx[stage_idx] {
            Some(prev_idx) => Ok(prev_idx),
            None => Err(PipelineError::Exhausted(format!(
                "{label} exhausted with no fallback"
            ))),
        }
    }

    /// Apply fallback: truncate the image stack to the depth the fallback
    /// target had at entry, reset the current stage's iteration counter, and
    /// bump the fallback target's iteration. Cascades if the fallback target
    /// is also exhausted.
    fn apply_fallback(
        &self,
        stage_idx: &mut usize,
        iterations: &mut Vec<u32>,
        image_stack: &mut Vec<Mat>,
        entry_depth: &[usize],
        fallback_target: usize,
        reason: &str,
    ) -> Result<(), PipelineError> {
        let label = self.stages[*stage_idx].descriptor().label;
        let prev_label = self.stages[fallback_target].descriptor().label;
        iterations[*stage_idx] = 0; // reset current stage
        iterations[fallback_target] += 1; // bump fallback target

        // Restore the image stack to the depth the fallback target originally saw
        image_stack.truncate(entry_depth[fallback_target]);

        if iterations[fallback_target] >= self.stages[fallback_target].max_retries() {
            // Fallback target also exhausted — cascade
            warn!(
                stage = prev_label,
                "fallback stage also exhausted, cascading"
            );
            *stage_idx = fallback_target;
            let next_fallback = self.resolve_fallback(fallback_target)?;
            return self.apply_fallback(
                stage_idx,
                iterations,
                image_stack,
                entry_depth,
                next_fallback,
                reason,
            );
        }

        info!(
            stage = label,
            fallback = prev_label,
            iter = iterations[fallback_target],
            "falling back"
        );
        *stage_idx = fallback_target;
        Ok(())
    }

    fn save_cache(&self, frame: &Mat) -> Result<(), PipelineError> {
        let cache = PipelineCache::new(
            self.state.clone(),
            frame.cols() as u32,
            frame.rows() as u32,
            self.cache_version,
        );

        if let Some(parent) = self.config.cache_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                PipelineError::Cache(format!("failed to create cache directory: {e}"))
            })?;
        }

        cache
            .save(&self.config.cache_path)
            .map_err(|e| PipelineError::Cache(format!("failed to save cache: {e}")))
    }

    /// Check if a stage should produce debug images.
    fn should_debug(&self, stage_idx: usize) -> bool {
        if self.debug_filter.is_empty() {
            return true; // no filter = all stages
        }
        if self.debug_filter.iter().any(|f| f == "none") {
            return false;
        }
        if self.debug_filter.iter().any(|f| f == "all") {
            return true;
        }
        let desc = self.stages[stage_idx].descriptor();
        self.debug_filter
            .iter()
            .any(|f| desc.label.contains(f.as_str()) || f.eq_ignore_ascii_case(desc.name))
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

    /// A mock stage that always succeeds, populating state via a closure.
    struct MockStage {
        desc: StageDescriptor,
        populate: fn(&mut PipelineState),
        retries: u32,
    }

    impl Stage for MockStage {
        fn descriptor(&self) -> StageDescriptor {
            self.desc.clone()
        }
        fn run(
            &self,
            state: &mut PipelineState,
            _src: &Mat,
            _dst: &mut Mat,
            _raw: &Mat,
            _iteration: u32,
        ) -> Result<(StageOutcome, ImageOutput), opencv::Error> {
            (self.populate)(state);
            Ok((StageOutcome::Success, ImageOutput::Passthrough))
        }
        fn max_retries(&self) -> u32 {
            self.retries
        }
    }

    /// A stage that fails N times then succeeds.
    struct FailNThenSucceed {
        desc: StageDescriptor,
        fail_count: u32,
        attempts: AtomicU32,
        populate: fn(&mut PipelineState),
        retries: u32,
    }

    impl Stage for FailNThenSucceed {
        fn descriptor(&self) -> StageDescriptor {
            self.desc.clone()
        }
        fn run(
            &self,
            state: &mut PipelineState,
            _src: &Mat,
            _dst: &mut Mat,
            _raw: &Mat,
            _iteration: u32,
        ) -> Result<(StageOutcome, ImageOutput), opencv::Error> {
            let n = self.attempts.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_count {
                Ok((
                    StageOutcome::Retry(format!("attempt {n}")),
                    ImageOutput::Passthrough,
                ))
            } else {
                (self.populate)(state);
                Ok((StageOutcome::Success, ImageOutput::Passthrough))
            }
        }
        fn max_retries(&self) -> u32 {
            self.retries
        }
    }

    /// A stage that always fails (exhausts retries).
    struct AlwaysFailStage {
        desc: StageDescriptor,
        retries: u32,
    }

    impl Stage for AlwaysFailStage {
        fn descriptor(&self) -> StageDescriptor {
            self.desc.clone()
        }
        fn run(
            &self,
            _state: &mut PipelineState,
            _src: &Mat,
            _dst: &mut Mat,
            _raw: &Mat,
            _iteration: u32,
        ) -> Result<(StageOutcome, ImageOutput), opencv::Error> {
            Ok((
                StageOutcome::Retry("always fails".into()),
                ImageOutput::Passthrough,
            ))
        }
        fn max_retries(&self) -> u32 {
            self.retries
        }
    }

    fn mock_success(
        name: &'static str,
        label: &'static str,
        fallback: Option<&'static str>,
        populate: fn(&mut PipelineState),
    ) -> Box<dyn Stage> {
        Box::new(MockStage {
            desc: StageDescriptor {
                name,
                label,
                fallback,
            },
            populate,
            retries: 3,
        })
    }

    fn all_success_stages() -> Vec<Box<dyn Stage>> {
        vec![
            mock_success("FindStove", "S1:FindStove", None, |s| {
                s.crop = Some(stage::CropRegion {
                    x: 100,
                    y: 200,
                    width: 800,
                    height: 300,
                });
            }),
            mock_success("FindLines", "S2:FindLines", Some("FindStove"), |s| {
                s.lines = Some(stage::LinePair {
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
                    avg_theta: std::f64::consts::FRAC_PI_2,
                });
            }),
            mock_success(
                "FindVerticals",
                "S3:FindVerticals",
                Some("FindLines"),
                |s| {
                    s.verticals = Some(stage::VerticalPair {
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
                },
            ),
            mock_success(
                "Perspective",
                "S4:Perspective",
                Some("FindVerticals"),
                |s| {
                    s.perspective = Some(stage::PerspectiveCorrection {
                        matrix: stage::TransformMatrix([
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ]),
                        output_width: 800,
                        output_height: 200,
                    });
                },
            ),
            mock_success("WarpCheck", "S5:WarpCheck", Some("FindVerticals"), |_s| {}),
            mock_success("ExtractBand", "S6:ExtractBand", Some("Perspective"), |s| {
                s.knob_search = Some(stage::KnobSearchArea {
                    y_min: 10.0,
                    y_max: 190.0,
                    x_min: 0.0,
                    clock_center_x: 0.0,
                    clock_center_y: 0.0,
                    clock_radius: 0.0,
                    corner_x: None,
                    corner_y: None,
                });
            }),
            mock_success("FindClock", "S7:FindClock", Some("ExtractBand"), |s| {
                if let Some(ks) = s.knob_search.as_mut() {
                    ks.x_min = 80.0;
                    ks.clock_center_x = 50.0;
                    ks.clock_center_y = 100.0;
                    ks.clock_radius = 25.0;
                }
            }),
            mock_success("FindFeatures", "S8:FindFeatures", Some("FindClock"), |s| {
                s.features = Some(stage::DetectedFeatures {
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
            }),
            mock_success(
                "SanityCheck",
                "S9:SanityCheck",
                Some("FindFeatures"),
                |_s| {},
            ),
            mock_success("FindCorner", "S10:FindCorner", Some("FindFeatures"), |s| {
                if let Some(ks) = s.knob_search.as_mut() {
                    ks.corner_x = Some(700.0);
                    ks.corner_y = Some(10.0);
                }
            }),
            mock_success(
                "RefineWarp",
                "S11:RefineWarp",
                Some("FindCorner"),
                |_s| {},
            ),
            mock_success(
                "FinalDetect",
                "S12:FinalDetect",
                Some("RefineWarp"),
                |s| {
                    s.features = Some(stage::DetectedFeatures {
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
                },
            ),
            mock_success(
                "FinalCheck",
                "S13:FinalCheck",
                Some("FinalDetect"),
                |s| {
                    s.validated = true;
                },
            ),
        ]
    }

    /// Replace a stage at the given index in the stage list.
    fn replace_stage(stages: &mut Vec<Box<dyn Stage>>, idx: usize, new_stage: Box<dyn Stage>) {
        stages[idx] = new_stage;
    }

    fn dummy_frame() -> Mat {
        // 4x4 black image -- enough for mock stages that don't read pixels
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
        let stages = all_success_stages();
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
        let mut stages = all_success_stages();
        replace_stage(
            &mut stages,
            3, // Perspective
            Box::new(FailNThenSucceed {
                desc: StageDescriptor {
                    name: "Perspective",
                    label: "S4:Perspective",
                    fallback: Some("FindVerticals"),
                },
                fail_count: 2,
                attempts: AtomicU32::new(0),
                populate: |s| {
                    s.perspective = Some(stage::PerspectiveCorrection {
                        matrix: stage::TransformMatrix([
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ]),
                        output_width: 800,
                        output_height: 200,
                    });
                },
                retries: 5,
            }),
        );
        let mut pipeline = Pipeline::new(stages, test_config());
        let frame = dummy_frame();

        pipeline.calibrate(&frame).unwrap();
        assert!(pipeline.state.validated);
    }

    #[test]
    fn fallback_stage4_exhausted_retries_stage3() {
        // Stage 4 (Perspective) fails 3 times then succeeds
        // Falls back to FindVerticals which re-runs, then Perspective succeeds
        let mut stages = all_success_stages();
        replace_stage(
            &mut stages,
            3, // Perspective
            Box::new(FailNThenSucceed {
                desc: StageDescriptor {
                    name: "Perspective",
                    label: "S4:Perspective",
                    fallback: Some("FindVerticals"),
                },
                fail_count: 3,
                attempts: AtomicU32::new(0),
                populate: |s| {
                    s.perspective = Some(stage::PerspectiveCorrection {
                        matrix: stage::TransformMatrix([
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ]),
                        output_width: 800,
                        output_height: 200,
                    });
                },
                retries: 5,
            }),
        );
        let mut pipeline = Pipeline::new(stages, test_config());
        let frame = dummy_frame();

        pipeline.calibrate(&frame).unwrap();
        assert!(pipeline.state.validated);
    }

    #[test]
    fn pipeline_exhausted_returns_error() {
        // Stage 1 always fails, no fallback -> pipeline error
        let mut stages = all_success_stages();
        replace_stage(
            &mut stages,
            0, // FindStove
            Box::new(AlwaysFailStage {
                desc: StageDescriptor {
                    name: "FindStove",
                    label: "S1:FindStove",
                    fallback: None,
                },
                retries: 2,
            }),
        );
        let mut pipeline = Pipeline::new(stages, test_config());
        let frame = dummy_frame();

        let err = pipeline.calibrate(&frame).unwrap_err();
        assert!(matches!(err, PipelineError::Exhausted(_)));
    }

    #[test]
    fn cache_saved_after_calibration() {
        let config = test_config();
        let cache_path = config.cache_path.clone();

        let stages = all_success_stages();
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
        let stages = all_success_stages();
        let config1 = PipelineConfig {
            cache_path: cache_path.clone(),
            max_frame_attempts: 3,
        };
        let mut pipeline = Pipeline::new(stages, config1);
        let frame = dummy_frame();
        pipeline.calibrate(&frame).unwrap();

        // Second run: load from cache
        let stages2 = all_success_stages();
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
