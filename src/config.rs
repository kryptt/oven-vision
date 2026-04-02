use serde::Deserialize;
use std::{fmt, io};

#[derive(Debug, Deserialize)]
pub struct Config {
    pub go2rtc_url: String,
    pub mqtt: MqttConfig,
    #[serde(default = "default_poll_interval")]
    pub poll_interval_secs: u64,
    pub dials: Vec<DialConfig>,
    #[serde(default)]
    pub leds: Vec<LedConfig>,
    #[serde(default)]
    pub capture: CaptureConfig,
    #[serde(default = "default_debug_port")]
    pub debug_port: u16,
    #[serde(default)]
    pub pipeline: PipelineConfig,
}

#[derive(Debug, Deserialize)]
pub struct PipelineConfig {
    /// Initial crop region hint (x, y, width, height). If omitted, uses a
    /// built-in default covering the known stove panel area.
    pub initial_crop: Option<CropConfig>,
    /// Maximum fresh frames to try during calibration before giving up.
    #[serde(default = "default_max_frame_attempts")]
    pub max_frame_attempts: u32,
    /// Path for the pipeline cache file.
    #[serde(default = "default_cache_path")]
    pub cache_path: String,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            initial_crop: None,
            max_frame_attempts: default_max_frame_attempts(),
            cache_path: default_cache_path(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CropConfig {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

fn default_max_frame_attempts() -> u32 {
    5
}

fn default_cache_path() -> String {
    "/data/captures/pipeline_cache.json".to_string()
}

#[derive(Debug, Deserialize)]
pub struct MqttConfig {
    pub host: String,
    #[serde(default = "default_mqtt_port")]
    pub port: u16,
    pub user: Option<String>,
    pub pass: Option<String>,
    #[serde(default = "default_keepalive")]
    pub keepalive_secs: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DialConfig {
    pub label: String,
    pub center_x: u32,
    pub center_y: u32,
    pub radius: u32,
    pub off_angle_deg: f64,
    #[serde(default = "default_tolerance")]
    pub off_tolerance_deg: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LedConfig {
    pub label: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

fn default_poll_interval() -> u64 {
    5
}

fn default_mqtt_port() -> u16 {
    1883
}

fn default_keepalive() -> u64 {
    30
}

fn default_tolerance() -> f64 {
    25.0
}

fn default_capture_dir() -> String {
    "/data/captures".to_string()
}

fn default_max_files() -> usize {
    200
}

fn default_capture_interval() -> u64 {
    300
}

fn default_confidence_threshold() -> f64 {
    0.25
}

fn default_debug_port() -> u16 {
    8080
}

#[derive(Debug, Deserialize)]
pub struct CaptureConfig {
    #[serde(default = "default_capture_dir")]
    pub dir: String,
    #[serde(default = "default_max_files")]
    pub max_files: usize,
    #[serde(default = "default_capture_interval")]
    pub min_interval_secs: u64,
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f64,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            dir: default_capture_dir(),
            max_files: default_max_files(),
            min_interval_secs: default_capture_interval(),
            confidence_threshold: default_confidence_threshold(),
        }
    }
}

#[derive(Debug)]
pub enum ConfigError {
    EnvMissing,
    ReadFile(io::Error),
    Parse(serde_yaml::Error),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EnvMissing => {
                write!(f, "environment variable OVEN_VISION_CONFIG is not set")
            }
            Self::ReadFile(err) => write!(f, "failed to read config file: {err}"),
            Self::Parse(err) => write!(f, "failed to parse config YAML: {err}"),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::EnvMissing => None,
            Self::ReadFile(err) => Some(err),
            Self::Parse(err) => Some(err),
        }
    }
}

pub fn load() -> Result<Config, ConfigError> {
    let path = std::env::var("OVEN_VISION_CONFIG").map_err(|_| ConfigError::EnvMissing)?;
    let contents = std::fs::read_to_string(&path).map_err(ConfigError::ReadFile)?;
    let mut config: Config = serde_yaml::from_str(&contents).map_err(ConfigError::Parse)?;

    // Override MQTT credentials from environment if set (for K8s Secret injection)
    if let Ok(user) = std::env::var("MQTT_USER") {
        config.mqtt.user = Some(user);
    }
    if let Ok(pass) = std::env::var("MQTT_PASS") {
        config.mqtt.pass = Some(pass);
    }

    Ok(config)
}
