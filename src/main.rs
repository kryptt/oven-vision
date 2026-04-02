use std::path::PathBuf;
use std::time::Duration;

use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

use oven_vision::annotate::{annotate_frame, encode_jpeg};
use oven_vision::capture::fetch_frame;
use oven_vision::capture_store::CaptureStore;
use oven_vision::config;
use oven_vision::debug_server::{run_debug_server, DebugState};
use oven_vision::detect::DialDetector;
use oven_vision::led::detect_leds;
use oven_vision::mqtt::MqttPublisher;
use oven_vision::preprocess::preprocess;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("oven_vision=info")),
        )
        .init();

    let cfg = match config::load() {
        Ok(cfg) => {
            info!(
                dials = cfg.dials.len(),
                leds = cfg.leds.len(),
                poll_interval_secs = cfg.poll_interval_secs,
                mqtt_host = %cfg.mqtt.host,
                mqtt_user = ?cfg.mqtt.user,
                mqtt_has_pass = cfg.mqtt.pass.is_some(),
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

    if let Err(err) = publisher.start(&cfg.dials, &cfg.leds).await {
        error!(%err, "failed to start MQTT publisher");
        std::process::exit(1);
    }

    debug_state.set_mqtt_connected(true);
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
    let mut detector = DialDetector::new(cfg.dials.clone());

    loop {
        match fetch_frame(&client, &cfg.go2rtc_url).await {
            Ok(frame) => {
                let gray = preprocess(&frame);
                info!(
                    width = gray.width(),
                    height = gray.height(),
                    "frame captured and preprocessed"
                );

                let readings = detector.detect_all(&gray);
                for reading in &readings {
                    info!(%reading, "dial reading");
                }

                let led_readings = detect_leds(&frame, &cfg.leds);
                for led in &led_readings {
                    info!(label = %led.label, state = %led.state, "led reading");
                }

                // Build annotated debug frame
                let annotated = annotate_frame(
                    &frame,
                    &readings,
                    &led_readings,
                    &cfg.dials,
                    &cfg.leds,
                );

                // Build JPEG for debug frame
                let jpeg = encode_jpeg(&annotated, 80);

                // Update shared debug state
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
                    Ok(Some(path)) => {
                        info!(path = %path.display(), "low-confidence frame saved");
                    }
                    Ok(None) => {}
                    Err(err) => {
                        warn!(%err, "failed to save low-confidence capture");
                    }
                }

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
