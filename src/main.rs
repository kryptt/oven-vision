use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use async_stream::stream;
use opencv::prelude::*;
use tokio_stream::{Stream, StreamExt};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

use oven_vision::annotate::{annotate_frame, encode_jpeg};
use oven_vision::capture::fetch_frame;
use oven_vision::capture_store::CaptureStore;
use oven_vision::config::{self, Config};
use oven_vision::debug_server::{DebugState, run_debug_server};
use oven_vision::detect::{DialDetector, DialReading};
use oven_vision::led::detect_leds;
use oven_vision::mqtt::MqttPublisher;
use oven_vision::pipeline::extract_band::ExtractBand;
use oven_vision::pipeline::final_check::FinalCheck;
use oven_vision::pipeline::final_detect::FinalDetect;
use oven_vision::pipeline::find_clock::FindClock;
use oven_vision::pipeline::find_corner::FindCorner;
use oven_vision::pipeline::find_features::FindFeatures;
use oven_vision::pipeline::find_lines::FindLines;
use oven_vision::pipeline::find_stove::FindStove;
use oven_vision::pipeline::find_verticals::FindVerticals;
use oven_vision::pipeline::perspective::Perspective;
use oven_vision::pipeline::refine_warp::RefineWarp;
use oven_vision::pipeline::sanity::{SanityCheck, quick_sanity_check};
use oven_vision::pipeline::stage::CropRegion;
use oven_vision::pipeline::util::Templates;
use oven_vision::pipeline::warp_check::WarpCheck;
use oven_vision::pipeline::{self, Pipeline, PipelineError};
use oven_vision::preprocess::preprocess;
use oven_vision::types::LedState;

/// How often to run the quick sanity check (every Nth frame).
const SANITY_CHECK_INTERVAL: u64 = 10;

/// Duration of consecutive sanity failures before triggering recalibration.
const RECALIBRATION_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// Number of initial frames used to auto-calibrate off-angles.
const CALIBRATION_FRAMES: u64 = 5;

/// Result of processing a single frame through the CV pipeline.
///
/// Bundles all data produced by the CPU-bound detection stages so that
/// downstream async sinks can consume it without touching OpenCV.
struct FrameResult {
    readings: Vec<DialReading>,
    led_states: Vec<(String, LedState)>,
    jpeg: Vec<u8>,
    frame_count: u64,
}

/// Produce a stream of raw BGR frames from go2rtc, yielding one frame
/// per `interval`. Failed fetches are logged and skipped (filtered out).
fn frame_stream(
    client: reqwest::Client,
    url: String,
    interval: Duration,
) -> impl Stream<Item = opencv::core::Mat> {
    stream! {
        loop {
            match fetch_frame(&client, &url).await {
                Ok(frame) => yield frame,
                Err(err) => error!(%err, "failed to capture frame"),
            }
            tokio::time::sleep(interval).await;
        }
    }
}

/// CPU-bound processing: warp, preprocess, detect dials, detect LEDs,
/// annotate, and encode JPEG.
///
/// Called inside `block_in_place` so OpenCV work does not stall the async
/// runtime. Returns `None` on any intermediate failure (the frame is skipped).
fn process_frame(
    state: &pipeline::stage::PipelineState,
    frame: &opencv::core::Mat,
    detector: &mut DialDetector,
    knob_template: &opencv::core::Mat,
    led_configs: &[oven_vision::config::LedConfig],
) -> Option<(Vec<DialReading>, Vec<(String, LedState)>, Vec<u8>)> {
    let warped = match pipeline::warp_frame(state, frame) {
        Ok(w) => w,
        Err(err) => {
            error!(%err, "warp failed");
            return None;
        }
    };

    let gray = match preprocess(&warped) {
        Ok(g) => g,
        Err(err) => {
            error!(%err, "preprocessing failed");
            return None;
        }
    };

    let readings = match detector.detect_all_template(&gray, knob_template) {
        Ok(r) => r,
        Err(err) => {
            error!(%err, "dial detection failed");
            return None;
        }
    };

    for reading in &readings {
        info!(%reading, "dial reading");
    }

    // LED detection runs on the raw (unwarped) frame
    let led_readings = match detect_leds(frame, led_configs) {
        Ok(r) => r,
        Err(err) => {
            error!(%err, "LED detection failed");
            return None;
        }
    };
    for led in &led_readings {
        info!(label = %led.label, state = %led.state, "led reading");
    }

    let led_states: Vec<(String, LedState)> = led_readings
        .iter()
        .map(|r| (r.label.clone(), r.state))
        .collect();

    // Annotate the warped image with dial overlays
    let annotated = match annotate_frame(&warped, &readings, &[], detector.configs(), &[]) {
        Ok(a) => a,
        Err(err) => {
            error!(%err, "annotation failed");
            return None;
        }
    };

    let jpeg = match encode_jpeg(&annotated, 80) {
        Ok(j) => j,
        Err(err) => {
            error!(%err, "JPEG encoding failed");
            return None;
        }
    };

    Some((readings, led_states, jpeg))
}

/// Mutable state threaded through the detection stream.
///
/// Encapsulates the pipeline, detector, and calibration accumulators so
/// the stream generator owns all mutation in one place.
struct DetectionState {
    pipe: Pipeline,
    detector: DialDetector,
    knob_template: opencv::core::Mat,
    off_angle_samples: Vec<Vec<f64>>,
    off_calibrated: bool,
    features: pipeline::stage::DetectedFeatures,
    labels: Vec<String>,
    sanity_fail_start: Option<Instant>,
    frame_count: u64,
    client: reqwest::Client,
    go2rtc_url: String,
    cfg: Config,
    capture_dir: PathBuf,
}

impl DetectionState {
    /// Apply off-angle calibration from the first N frames.
    fn calibrate_off_angles(&mut self, readings: &[DialReading]) {
        if self.off_calibrated || self.frame_count >= CALIBRATION_FRAMES {
            return;
        }

        for (i, reading) in readings.iter().enumerate() {
            if let Some(angle) = reading.angle_deg {
                if i < self.off_angle_samples.len() {
                    self.off_angle_samples[i].push(angle);
                }
            }
        }

        if self.frame_count + 1 >= CALIBRATION_FRAMES {
            let off_angles: Vec<f64> = self
                .off_angle_samples
                .iter_mut()
                .map(|samples| {
                    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    if samples.is_empty() {
                        0.0
                    } else {
                        samples[samples.len() / 2]
                    }
                })
                .collect();

            info!(?off_angles, "off-angle calibration complete");

            let mut new_features = self.features.clone();
            new_features.off_angles = off_angles;

            let labels_ref = if self.labels.is_empty() {
                None
            } else {
                Some(self.labels.as_slice())
            };
            self.detector = DialDetector::from_features(&new_features, labels_ref);
            self.features = new_features;
            self.off_calibrated = true;
        }
    }

    /// Handle sanity check result: track failures and trigger recalibration
    /// after the timeout expires.
    fn handle_sanity(&mut self, ok: bool) {
        if ok {
            if self.sanity_fail_start.take().is_some() {
                info!("sanity check recovered");
            }
            return;
        }

        let start = *self.sanity_fail_start.get_or_insert_with(|| {
            warn!("sanity check failed, starting recalibration timer");
            Instant::now()
        });

        if start.elapsed() >= RECALIBRATION_TIMEOUT {
            warn!("sanity failures exceeded 15 min, recalibrating");
            self.recalibrate();
            self.sanity_fail_start = None;
        }
    }

    /// Rebuild the pipeline and re-run calibration.
    fn recalibrate(&mut self) {
        self.pipe = build_pipeline(&self.cfg);
        match run_calibration(&mut self.pipe, &self.client, &self.go2rtc_url) {
            Ok(()) => {
                save_calibration_debug_images(&self.pipe, &self.capture_dir);
                let f = self.pipe.state().features.as_ref().unwrap();
                let labels_ref = if self.labels.is_empty() {
                    None
                } else {
                    Some(self.labels.as_slice())
                };
                self.detector = DialDetector::from_features(f, labels_ref);
                self.features = f.clone();
                info!("recalibration complete");
            }
            Err(err) => {
                save_calibration_debug_images(&self.pipe, &self.capture_dir);
                error!(%err, "recalibration failed, keeping old state");
            }
        }
    }

    /// Run the sanity check on the warped frame (CPU-bound).
    fn run_sanity_check(&self, frame: &opencv::core::Mat) -> Option<bool> {
        let warped = pipeline::warp_frame(self.pipe.state(), frame).ok()?;
        match quick_sanity_check(&warped) {
            Ok(ok) => Some(ok),
            Err(err) => {
                warn!(%err, "sanity check error");
                None
            }
        }
    }
}

/// Build the detection stream: frames -> warp -> detect -> annotate -> encode.
///
/// Each yielded `FrameResult` contains everything the async sinks need
/// (JPEG bytes, dial readings, LED states). CPU-bound OpenCV work runs
/// inside `block_in_place` so it does not block the async executor.
fn detection_stream(
    client: reqwest::Client,
    url: String,
    interval: Duration,
    mut det: DetectionState,
) -> impl Stream<Item = FrameResult> {
    stream! {
        let frames = frame_stream(client, url, interval);
        tokio::pin!(frames);

        while let Some(frame) = frames.next().await {
            // CPU-bound work via block_in_place (Mat is !Send, so
            // spawn_blocking is not an option).
            let result = tokio::task::block_in_place(|| {
                process_frame(
                    det.pipe.state(),
                    &frame,
                    &mut det.detector,
                    &det.knob_template,
                    &det.cfg.leds,
                )
            });

            let Some((readings, led_states, jpeg)) = result else {
                det.frame_count += 1;
                continue;
            };

            // Off-angle calibration (first N frames)
            det.calibrate_off_angles(&readings);

            // Periodic sanity check (CPU-bound)
            if det.frame_count > 0 && det.frame_count % SANITY_CHECK_INTERVAL == 0 {
                if let Some(ok) = tokio::task::block_in_place(|| {
                    det.run_sanity_check(&frame)
                }) {
                    det.handle_sanity(ok);
                }
            }

            let frame_count = det.frame_count;
            det.frame_count += 1;

            yield FrameResult {
                readings,
                led_states,
                jpeg,
                frame_count,
            };
        }
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("oven_vision=info")),
        )
        .init();

    let cfg = match config::load() {
        Ok(cfg) => {
            info!(
                dials = cfg.dials.len(),
                leds = cfg.leds.len(),
                poll_interval_secs = cfg.poll_interval_secs,
                mqtt_host = %cfg.mqtt.host,
                "configuration loaded"
            );
            cfg
        }
        Err(err) => {
            error!(%err, "failed to load configuration");
            std::process::exit(1);
        }
    };

    // Initialize MQTT publisher
    let mut publisher = match MqttPublisher::new(&cfg.mqtt) {
        Ok(p) => p,
        Err(err) => {
            error!(%err, "failed to create MQTT publisher");
            std::process::exit(1);
        }
    };

    let debug_state = DebugState::new();
    tokio::spawn(run_debug_server(debug_state.clone(), cfg.debug_port));

    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(2)
        .build()
        .expect("failed to build HTTP client");

    let capture_dir = PathBuf::from(&cfg.capture.dir);
    let mut capture_store = CaptureStore::new(
        capture_dir.clone(),
        cfg.capture.max_files,
        cfg.capture.min_interval_secs,
        cfg.capture.confidence_threshold,
    );

    let poll_interval = Duration::from_secs(cfg.poll_interval_secs);

    // --- Pipeline calibration ---
    let mut pipe = build_pipeline(&cfg);

    let initial_frame = match fetch_frame(&client, &cfg.go2rtc_url).await {
        Ok(f) => f,
        Err(err) => {
            error!(%err, "failed to fetch initial frame");
            std::process::exit(1);
        }
    };
    let (frame_w, frame_h) = (initial_frame.cols() as u32, initial_frame.rows() as u32);
    info!(frame_w, frame_h, "initial frame captured");

    if !pipe.try_load_cache(frame_w, frame_h) {
        info!("running full calibration pipeline");
        if let Err(err) = run_calibration(&mut pipe, &client, &cfg.go2rtc_url) {
            save_calibration_debug_images(&pipe, &capture_dir);
            error!(%err, "calibration failed");
            std::process::exit(1);
        }
        save_calibration_debug_images(&pipe, &capture_dir);
    }

    // Build detector from pipeline-discovered knob positions
    let labels: Vec<String> = cfg.dials.iter().map(|d| d.label.clone()).collect();
    let labels_ref = if labels.is_empty() {
        None
    } else {
        Some(labels.as_slice())
    };
    let features = match pipe.state().features.clone() {
        Some(f) => f,
        None => {
            error!("pipeline has no features after calibration");
            std::process::exit(1);
        }
    };
    let detector = DialDetector::from_features(&features, labels_ref);

    // Load and scale knob template
    let knob_template = load_knob_template(&features);

    // Start MQTT with pipeline-derived dial configs
    if let Err(err) = publisher.start(detector.configs(), &cfg.leds).await {
        error!(%err, "failed to start MQTT publisher");
        std::process::exit(1);
    }
    debug_state.set_mqtt_connected(true);

    // --- Build detection state and stream pipeline ---
    let det_state = DetectionState {
        pipe,
        detector,
        knob_template,
        off_angle_samples: vec![Vec::new(); features.knobs.len()],
        off_calibrated: false,
        features,
        labels,
        sanity_fail_start: None,
        frame_count: 0,
        client: client.clone(),
        go2rtc_url: cfg.go2rtc_url.clone(),
        cfg,
        capture_dir: capture_dir.clone(),
    };

    let go2rtc_url = det_state.go2rtc_url.clone();
    let results = detection_stream(client, go2rtc_url, poll_interval, det_state);
    tokio::pin!(results);

    // --- Sink: consume stream results ---
    while let Some(result) = results.next().await {
        debug_state.update_frame(result.jpeg.clone());
        debug_state.set_frames_flowing(true);

        write_debug_latest(&capture_dir, &result.jpeg);

        if result.frame_count < 10 {
            save_runtime_frame(&capture_dir, result.frame_count, &result.jpeg);
        }
        if result.frame_count == 10 {
            info!("10 runtime frames captured, exiting for evaluation");
            std::process::exit(0);
        }

        // Low-confidence capture (decode JPEG back to Mat for CaptureStore)
        if let Ok(decoded) = opencv::imgcodecs::imdecode(
            &opencv::core::Vector::from_slice(&result.jpeg),
            opencv::imgcodecs::IMREAD_COLOR,
        ) {
            match capture_store.maybe_capture(&decoded, &result.readings) {
                Ok(Some(path)) => info!(path = %path.display(), "low-confidence frame saved"),
                Ok(None) => {}
                Err(err) => warn!(%err, "failed to save low-confidence capture"),
            }
        }

        if let Err(err) = publisher
            .publish_states(&result.readings, &result.led_states)
            .await
        {
            error!(%err, "failed to publish MQTT states");
        }
    }
}

/// Load the knob template image and scale it to match the detected knob radius.
fn load_knob_template(features: &pipeline::stage::DetectedFeatures) -> opencv::core::Mat {
    let raw =
        opencv::imgcodecs::imread("/templates/knob.jpg", opencv::imgcodecs::IMREAD_GRAYSCALE)
            .expect("failed to load /templates/knob.jpg");
    let target_size = (features.knobs[0].radius * 2.0) as i32;
    let scale = target_size as f64 / raw.cols().max(raw.rows()) as f64;
    let new_w = (raw.cols() as f64 * scale) as i32;
    let new_h = (raw.rows() as f64 * scale) as i32;
    let mut scaled = opencv::core::Mat::default();
    opencv::imgproc::resize(
        &raw,
        &mut scaled,
        opencv::core::Size::new(new_w, new_h),
        0.0,
        0.0,
        opencv::imgproc::INTER_AREA,
    )
    .expect("failed to scale knob template");
    info!(original = %format!("{}x{}", raw.cols(), raw.rows()),
          scaled = %format!("{}x{}", new_w, new_h),
          "knob template loaded for per-frame detection");
    scaled
}

/// Write the latest debug JPEG to disk.
fn write_debug_latest(capture_dir: &Path, jpeg: &[u8]) {
    if let Err(err) = std::fs::create_dir_all(capture_dir) {
        warn!(%err, "failed to create capture directory");
        return;
    }
    let debug_path = capture_dir.join("debug_latest.jpg");
    if let Err(err) = std::fs::write(&debug_path, jpeg) {
        warn!(%err, "failed to write debug_latest.jpg");
    }
}

/// Save a numbered runtime debug frame.
fn save_runtime_frame(capture_dir: &Path, frame_count: u64, jpeg: &[u8]) {
    let path = capture_dir.join(format!("runtime_frame_{:02}.jpg", frame_count));
    if let Err(err) = std::fs::write(&path, jpeg) {
        warn!(%err, "failed to save runtime frame");
    } else {
        info!(path = %path.display(), "saved runtime frame");
    }
}

/// Build the 13-stage calibration pipeline from config.
fn build_pipeline(cfg: &Config) -> Pipeline {
    let templates = std::sync::Arc::new(Templates::load());
    let find_features = std::sync::Arc::new(FindFeatures::new(templates.clone()));
    let stages: Vec<Box<dyn pipeline::Stage>> = vec![
        Box::new(match &cfg.pipeline.initial_crop {
            Some(crop) => FindStove::with_crop(CropRegion {
                x: crop.x,
                y: crop.y,
                width: crop.width,
                height: crop.height,
            }),
            None => FindStove::new(),
        }),
        Box::new(FindLines::new()),
        Box::new(FindVerticals::new()),
        Box::new(Perspective::new()),
        Box::new(WarpCheck::new()),
        Box::new(ExtractBand::new()),
        Box::new(FindClock::new(templates.clone())),
        Box::new(find_features.clone()),
        Box::new(SanityCheck::new()),
        Box::new(FindCorner::new()),
        Box::new(RefineWarp::new()),
        Box::new(FinalDetect::new(find_features)),
        Box::new(FinalCheck::new()),
    ];

    Pipeline::new(
        stages,
        pipeline::PipelineConfig {
            cache_path: PathBuf::from(&cfg.pipeline.cache_path),
            max_frame_attempts: cfg.pipeline.max_frame_attempts,
        },
    )
}

/// Run calibration using async frame fetching bridged into the sync pipeline.
fn run_calibration(
    pipe: &mut Pipeline,
    client: &reqwest::Client,
    url: &str,
) -> Result<(), PipelineError> {
    let handle = tokio::runtime::Handle::current();
    let client = client.clone();
    let url = url.to_owned();
    pipe.calibrate_with_fetch(move || {
        tokio::task::block_in_place(|| handle.block_on(fetch_frame(&client, &url)))
            .map_err(|e| PipelineError::Exhausted(format!("frame fetch: {e}")))
    })
}

/// Write per-stage debug images from the last calibration run to disk.
fn save_calibration_debug_images(pipe: &Pipeline, capture_dir: &Path) {
    let images = pipe.debug_images();
    if images.is_empty() {
        info!("no calibration debug images to save");
        return;
    }

    if let Err(err) = std::fs::create_dir_all(capture_dir) {
        warn!(%err, "failed to create capture directory for debug images");
        return;
    }

    for (idx, (label, jpeg)) in images.iter().enumerate() {
        let filename = format!(
            "calibration_{:03}_{}.jpg",
            idx,
            label.replace(':', "_")
        );
        let path = capture_dir.join(&filename);
        match std::fs::write(&path, jpeg) {
            Ok(()) => info!(path = %path.display(), label, "saved calibration debug image"),
            Err(err) => warn!(%err, label, "failed to save calibration debug image"),
        }
    }
}
