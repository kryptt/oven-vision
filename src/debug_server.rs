use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use tokio::net::TcpListener;
use tracing::info;

#[derive(Clone)]
pub struct DebugState {
    pub latest_frame: Arc<RwLock<Option<Vec<u8>>>>,
    pub frames_flowing: Arc<AtomicBool>,
    pub mqtt_connected: Arc<AtomicBool>,
}

impl DebugState {
    pub fn new() -> Self {
        Self {
            latest_frame: Arc::new(RwLock::new(None)),
            frames_flowing: Arc::new(AtomicBool::new(false)),
            mqtt_connected: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn update_frame(&self, jpeg_bytes: Vec<u8>) {
        let mut lock = self.latest_frame.write().expect("frame lock poisoned");
        *lock = Some(jpeg_bytes);
    }

    pub fn set_frames_flowing(&self, v: bool) {
        self.frames_flowing.store(v, Ordering::Relaxed);
    }

    pub fn set_mqtt_connected(&self, v: bool) {
        self.mqtt_connected.store(v, Ordering::Relaxed);
    }
}

pub async fn run_debug_server(state: DebugState, port: u16) {
    let app = Router::new()
        .route("/debug/frame.jpg", get(debug_frame))
        .route("/health", get(health))
        .with_state(state);

    let listener = TcpListener::bind(("0.0.0.0", port))
        .await
        .expect("failed to bind debug server");

    info!(port, "debug server listening");
    axum::serve(listener, app).await.unwrap();
}

async fn debug_frame(State(state): State<DebugState>) -> impl IntoResponse {
    let lock = state.latest_frame.read().expect("frame lock poisoned");
    match lock.as_ref() {
        Some(bytes) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "image/jpeg")],
            bytes.clone(),
        )
            .into_response(),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            "no frame available yet",
        )
            .into_response(),
    }
}

async fn health(State(state): State<DebugState>) -> impl IntoResponse {
    let frames = state.frames_flowing.load(Ordering::Relaxed);
    let mqtt = state.mqtt_connected.load(Ordering::Relaxed);

    if frames && mqtt {
        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            r#"{"status":"ok"}"#.to_string(),
        )
    } else {
        let reason = match (frames, mqtt) {
            (false, false) => "no frames captured yet, MQTT not connected",
            (false, true) => "no frames captured yet",
            (true, false) => "MQTT not connected",
            _ => unreachable!(),
        };
        (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::CONTENT_TYPE, "application/json")],
            format!(r#"{{"status":"unhealthy","reason":"{reason}"}}"#),
        )
    }
}
