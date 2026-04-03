use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use opencv::prelude::*;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

use oven_vision::annotate::{annotate_frame, encode_jpeg};
use oven_vision::capture::fetch_frame;
use oven_vision::capture_store::CaptureStore;
use oven_vision::config::{self, Config};
use oven_vision::debug_server::{DebugState, run_debug_server};
use oven_vision::detect::DialDetector;
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
use oven_vision::pipeline::sanity::{SanityCheck, quick_sanity_check};
use oven_vision::pipeline::stage::CropRegion;
use oven_vision::pipeline::refine_warp::RefineWarp;
use oven_vision::pipeline::warp_check::WarpCheck;
use oven_vision::pipeline::{self, Pipeline, PipelineError};
use oven_vision::preprocess::preprocess;

/// How often to run the quick sanity check (every Nth frame).
const SANITY_CHECK_INTERVAL: u64 = 10;

/// Duration of consecutive sanity failures before triggering recalibration.
const RECALIBRATION_TIMEOUT: Duration = Duration::from_secs(15 * 60);

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

    // Initialize MQTT publisher (connection starts later after calibration)
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

    // --- v2 Pipeline: calibrate or load cache ---
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
    let mut detector = DialDetector::from_features(&features, labels_ref);

    // Load and scale the knob template for per-frame template matching.
    // Scale it to match the knob radius in the warped image.
    let knob_template = {
        let raw =
            opencv::imgcodecs::imread("/templates/knob.jpg", opencv::imgcodecs::IMREAD_GRAYSCALE)
                .expect("failed to load /templates/knob.jpg");
        // The template knob is ~100px diameter. Scale to 2× the detected knob radius.
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
    };

    // Start MQTT with pipeline-derived dial configs
    if let Err(err) = publisher.start(detector.configs(), &cfg.leds).await {
        error!(%err, "failed to start MQTT publisher");
        std::process::exit(1);
    }
    debug_state.set_mqtt_connected(true);

    // --- Off-angle auto-calibration ---
    // The first few frames calibrate the off-angles (stove assumed off at startup).
    const CALIBRATION_FRAMES: u64 = 5;
    let mut off_angle_samples: Vec<Vec<f64>> = vec![Vec::new(); detector.configs().len()];
    let mut off_calibrated = false;

    // --- Detection loop ---
    let mut sanity_fail_start: Option<Instant> = None;
    let mut frame_count: u64 = 0;

    loop {
        match fetch_frame(&client, &cfg.go2rtc_url).await {
            Ok(frame) => {
                // Apply cached perspective warp
                let warped = match pipeline::warp_frame(pipe.state(), &frame) {
                    Ok(w) => w,
                    Err(err) => {
                        error!(%err, "warp failed");
                        continue;
                    }
                };

                // Convert warped image to grayscale for template matching
                let gray = match preprocess(&warped) {
                    Ok(g) => g,
                    Err(err) => {
                        error!(%err, "preprocessing failed");
                        continue;
                    }
                };

                let readings = match detector.detect_all_template(&gray, &knob_template) {
                    Ok(r) => r,
                    Err(err) => {
                        error!(%err, "dial detection failed");
                        continue;
                    }
                };

                // Auto-calibrate off-angles from the first N frames
                if !off_calibrated && frame_count < CALIBRATION_FRAMES {
                    for (i, reading) in readings.iter().enumerate() {
                        if let Some(angle) = reading.angle_deg {
                            if i < off_angle_samples.len() {
                                off_angle_samples[i].push(angle);
                            }
                        }
                    }
                    if frame_count + 1 >= CALIBRATION_FRAMES {
                        // Compute median off-angle per knob and update detector
                        let mut off_angles: Vec<f64> = Vec::new();
                        for samples in &mut off_angle_samples {
                            samples.sort_by(|a, b| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let median = if samples.is_empty() {
                                0.0
                            } else {
                                samples[samples.len() / 2]
                            };
                            off_angles.push(median);
                        }
                        info!(?off_angles, "off-angle calibration complete");
                        // Rebuild detector with calibrated off-angles
                        let mut new_features = features.clone();
                        new_features.off_angles = off_angles;
                        detector = DialDetector::from_features(&new_features, labels_ref);
                        off_calibrated = true;
                    }
                }

                for reading in &readings {
                    info!(%reading, "dial reading");
                }

                // Save runtime debug frames for the first 10 frames, then exit.
                // This allows visual inspection of the detection results.
                if frame_count < 10 {
                    let runtime_path =
                        capture_dir.join(format!("runtime_frame_{:02}.jpg", frame_count));
                    // Draw detections on the warped image
                    if let Ok(annotated) =
                        annotate_frame(&warped, &readings, &[], detector.configs(), &[])
                    {
                        if let Ok(jpeg) = encode_jpeg(&annotated, 80) {
                            if let Err(err) = std::fs::write(&runtime_path, &jpeg) {
                                warn!(%err, "failed to save runtime frame");
                            } else {
                                info!(path = %runtime_path.display(), "saved runtime frame");
                            }
                        }
                    }
                }
                if frame_count == 10 {
                    info!("10 runtime frames captured, exiting for evaluation");
                    save_calibration_debug_images(&pipe, &capture_dir);
                    std::process::exit(0);
                }

                // LED detection runs on the raw (unwarped) frame
                let led_readings = match detect_leds(&frame, &cfg.leds) {
                    Ok(r) => r,
                    Err(err) => {
                        error!(%err, "LED detection failed");
                        continue;
                    }
                };
                for led in &led_readings {
                    info!(label = %led.label, state = %led.state, "led reading");
                }

                // Annotate the warped image with dial overlays (LEDs are on the
                // raw frame so we pass empty slices for them here)
                let annotated =
                    match annotate_frame(&warped, &readings, &[], detector.configs(), &[]) {
                        Ok(a) => a,
                        Err(err) => {
                            error!(%err, "annotation failed");
                            continue;
                        }
                    };

                let jpeg = match encode_jpeg(&annotated, 80) {
                    Ok(j) => j,
                    Err(err) => {
                        error!(%err, "JPEG encoding failed");
                        continue;
                    }
                };

                debug_state.update_frame(jpeg.clone());
                debug_state.set_frames_flowing(true);

                // Write latest debug frame to disk
                if let Err(err) = std::fs::create_dir_all(&capture_dir) {
                    warn!(%err, "failed to create capture directory");
                } else {
                    let debug_path = capture_dir.join("debug_latest.jpg");
                    if let Err(err) = std::fs::write(&debug_path, &jpeg) {
                        warn!(%err, "failed to write debug_latest.jpg");
                    }
                }

                // Save low-confidence captures
                match capture_store.maybe_capture(&annotated, &readings) {
                    Ok(Some(path)) => info!(path = %path.display(), "low-confidence frame saved"),
                    Ok(None) => {}
                    Err(err) => warn!(%err, "failed to save low-confidence capture"),
                }

                // Periodic sanity check
                frame_count += 1;
                if frame_count % SANITY_CHECK_INTERVAL == 0 {
                    match quick_sanity_check(&warped) {
                        Ok(true) => {
                            if sanity_fail_start.take().is_some() {
                                info!("sanity check recovered");
                            }
                        }
                        Ok(false) => {
                            let start = *sanity_fail_start.get_or_insert_with(|| {
                                warn!("sanity check failed, starting recalibration timer");
                                Instant::now()
                            });
                            if start.elapsed() >= RECALIBRATION_TIMEOUT {
                                warn!("sanity failures exceeded 15 min, recalibrating");
                                pipe = build_pipeline(&cfg);
                                match run_calibration(&mut pipe, &client, &cfg.go2rtc_url) {
                                    Ok(()) => {
                                        save_calibration_debug_images(&pipe, &capture_dir);
                                        let f = pipe.state().features.as_ref().unwrap();
                                        detector = DialDetector::from_features(f, labels_ref);
                                        info!("recalibration complete");
                                    }
                                    Err(err) => {
                                        save_calibration_debug_images(&pipe, &capture_dir);
                                        error!(%err, "recalibration failed, keeping old state");
                                    }
                                }
                                sanity_fail_start = None;
                            }
                        }
                        Err(err) => {
                            warn!(%err, "sanity check error");
                        }
                    }
                }

                // Publish MQTT
                let led_states: Vec<(String, _)> = led_readings
                    .iter()
                    .map(|r| (r.label.clone(), r.state))
                    .collect();
                if let Err(err) = publisher.publish_states(&readings, &led_states).await {
                    error!(%err, "failed to publish MQTT states");
                }
            }
            Err(err) => {
                error!(%err, "failed to capture frame");
                debug_state.set_frames_flowing(false);
            }
        }

        tokio::time::sleep(poll_interval).await;
    }
}

/// Build the 6-stage calibration pipeline from config.
fn build_pipeline(cfg: &Config) -> Pipeline {
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
        Box::new(FindClock::new()),
        Box::new(FindFeatures::new()),
        Box::new(SanityCheck::new()),
        Box::new(FindCorner::new()),
        Box::new(RefineWarp::new()),
        Box::new(FinalDetect::new()),
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
    pipe.calibrate_with_fetch(|| {
        tokio::task::block_in_place(|| handle.block_on(fetch_frame(client, url)))
            .map_err(|e| PipelineError::Exhausted(format!("frame fetch: {e}")))
    })
}

/// Write per-stage debug images from the last calibration run to disk.
/// Called after both successful and failed calibration so the images are
/// always available for inspection.
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
        // Prefix with sequence number so no overwrites across iterations.
        // e.g., "calibration_003_S3_FindVerticals_L92_R735.jpg"
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
