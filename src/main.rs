#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use opencv::core::{Mat, Vector};
use opencv::imgcodecs;
use opencv::prelude::*;
use tokio_stream::StreamExt;

use oven_vision::config::Calibration;
use oven_vision::stage::annotate::Annotate;
use oven_vision::stage::detect::Detect;
use oven_vision::stage::enhance::Enhance;
use oven_vision::stage::reproject::Reproject;
use oven_vision::stage::sanity::Sanity;
use oven_vision::stage::{FrameState, Stage};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let url = std::env::var("GO2RTC_URL")
        .unwrap_or_else(|_| "http://localhost:1984/api/frame.jpeg?src=kitchen".into());
    let calib_path = std::env::var("CALIBRATION_FILE")
        .unwrap_or_else(|_| "/data/calibration-points.json".into());
    let output_dir = std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "/data/output".into());
    let duration_secs: u64 = std::env::var("DURATION_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let save_stages = std::env::var("SAVE_STAGES").unwrap_or_default() == "1";

    let calib = Calibration::load(&PathBuf::from(&calib_path))?;
    eprintln!(
        "Loaded calibration: k1={}, {} knobs expected",
        calib.distortion_k1, calib.knob_detection.expected_count
    );

    // Build pipeline — solvePnP handles undistortion + perspective in one remap pass
    let mut stages: Vec<Box<dyn Stage>> = Vec::new();
    stages.push(Box::new(Reproject::new(&calib, 1200, 250)?));
    stages.push(Box::new(Enhance::new(&calib.knob_detection)));
    stages.push(Box::new(Detect::new(&calib.knob_detection)));
    stages.push(Box::new(Sanity::new(&calib.knob_detection)));
    stages.push(Box::new(Annotate));

    eprintln!(
        "Pipeline: {}",
        stages
            .iter()
            .enumerate()
            .map(|(i, s)| format!("s{}_{}", i + 1, s.name()))
            .collect::<Vec<_>>()
            .join(" → ")
    );

    fs::create_dir_all(&output_dir)?;

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;

    // Frame stream: yields raw Mats at fixed interval
    let interval = Duration::from_millis(500);
    let frame_stream = frame_stream(client, url, interval);
    tokio::pin!(frame_stream);

    // Take frames for the configured duration
    let deadline = tokio::time::Instant::now() + Duration::from_secs(duration_secs);
    let mut frame_num = 0u32;

    eprintln!("Streaming for {}s", duration_secs);

    while tokio::time::Instant::now() < deadline {
        let frame = match frame_stream.next().await {
            Some(Ok(f)) => f,
            Some(Err(e)) => {
                eprintln!("fetch: {e}");
                continue;
            }
            None => break,
        };

        // Run pipeline stages (CPU-bound, so block_in_place)
        let result = tokio::task::block_in_place(|| {
            run_pipeline(&stages, frame, frame_num, &output_dir, save_stages)
        });

        match result {
            Ok(summary) => eprintln!("f{:02}: {}", frame_num, summary),
            Err(e) => eprintln!("f{:02}: pipeline error: {}", frame_num, e),
        }

        frame_num += 1;
    }

    eprintln!("\n{frame_num} frames → {output_dir}/");
    Ok(())
}

/// Async stream that yields frames at a fixed interval.
fn frame_stream(
    client: reqwest::Client,
    url: String,
    interval: Duration,
) -> impl tokio_stream::Stream<Item = Result<Mat, Box<dyn std::error::Error + Send>>> {
    async_stream::stream! {
        let mut ticker = tokio::time::interval(interval);
        loop {
            ticker.tick().await;
            match fetch_frame(&client, &url).await {
                Ok(frame) => yield Ok(frame),
                Err(e) => yield Err(e),
            }
        }
    }
}

async fn fetch_frame(
    client: &reqwest::Client,
    url: &str,
) -> Result<Mat, Box<dyn std::error::Error + Send>> {
    let bytes = client
        .get(url)
        .send()
        .await
        .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?
        .bytes()
        .await
        .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?;
    let buf = Vector::from_slice(&bytes);
    let frame = imgcodecs::imdecode(&buf, imgcodecs::IMREAD_COLOR)
        .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?;
    if frame.empty() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "empty frame",
        )));
    }
    Ok(frame)
}

/// Run all pipeline stages, optionally saving each stage's output.
fn run_pipeline(
    stages: &[Box<dyn Stage>],
    frame: Mat,
    frame_num: u32,
    output_dir: &str,
    save_stages: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut state = FrameState::new(frame);
    let params = Vector::from_slice(&[imgcodecs::IMWRITE_JPEG_QUALITY, 92]);

    for (i, stage) in stages.iter().enumerate() {
        stage.process(&mut state)?;

        if save_stages {
            let prefix = format!(
                "{}/f{:02}_s{}_{}",
                output_dir,
                frame_num,
                i + 1,
                stage.name()
            );
            save_stage_output(&state, stage.name(), &prefix, &params);
        }
    }

    // Always save final annotated frame
    if let Some(ref annotated) = state.annotated {
        let path = format!("{}/frame_{:04}.jpg", output_dir, frame_num);
        imgcodecs::imwrite(&path, annotated, &params)?;
    }

    let sanity = state.sanity.as_ref();
    Ok(format!(
        "{} knobs | {}",
        state.knobs.len(),
        sanity.map_or("no sanity".to_string(), |s| s.details.clone()),
    ))
}

/// Save the relevant image from the current stage.
fn save_stage_output(state: &FrameState, stage_name: &str, prefix: &str, params: &Vector<i32>) {
    let img: Option<&Mat> = match stage_name {
        "undistort" => state.undistorted.as_ref(),
        "warp" => state.warped.as_ref(),
        "enhance" => state.enhanced.as_ref(),
        "annotate" => state.annotated.as_ref(),
        "reproject" => state.warped.as_ref(),
        "detect" => state.debug_edges.as_ref(),
        _ => None,
    };
    if let Some(mat) = img {
        let path = format!("{}.jpg", prefix);
        let _ = imgcodecs::imwrite(&path, mat, params);
    }
}
