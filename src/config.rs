use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Calibration {
    pub frame_size: [f64; 2],
    #[serde(default)]
    pub distortion_k1: f64,
    pub source_points: SourcePoints,
    #[serde(default)]
    pub knob_detection: KnobDetection,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SourcePoints {
    pub top_left: [f64; 2],
    pub bottom_left: [f64; 2],
    pub top_right: [f64; 2],
    pub bottom_right: [f64; 2],
}

#[derive(Debug, Clone, Deserialize)]
pub struct KnobDetection {
    #[serde(default = "default_clahe_clip")]
    pub clahe_clip_limit: f64,
    #[serde(default = "default_clahe_grid")]
    pub clahe_grid_size: i32,
    #[serde(default = "default_dp")]
    pub hough_dp: f64,
    #[serde(default = "default_min_dist")]
    pub hough_min_dist: f64,
    #[serde(default = "default_param1")]
    pub hough_param1: f64,
    #[serde(default = "default_param2")]
    pub hough_param2: f64,
    #[serde(default = "default_min_r")]
    pub hough_min_radius: i32,
    #[serde(default = "default_max_r")]
    pub hough_max_radius: i32,
    #[serde(default = "default_count")]
    pub expected_count: usize,
    #[serde(default = "default_y_tol")]
    pub y_tolerance_px: f32,
    #[serde(default = "default_size_tol")]
    pub size_tolerance_pct: f32,
    #[serde(default = "default_spacing_tol")]
    pub spacing_tolerance_pct: f32,
    #[serde(default = "default_y_band_top")]
    pub y_band_top: f32,
    #[serde(default = "default_y_band_bottom")]
    pub y_band_bottom: f32,
    #[serde(default = "default_prior_x_start")]
    pub prior_x_start: f32,
    #[serde(default = "default_prior_x_end")]
    pub prior_x_end: f32,
    #[serde(default = "default_prior_y")]
    pub prior_y: f32,
    #[serde(default = "default_slot_capture_radius")]
    pub slot_capture_radius: f32,
}

impl Default for KnobDetection {
    fn default() -> Self {
        Self {
            clahe_clip_limit: 3.0,
            clahe_grid_size: 8,
            hough_dp: 1.2,
            hough_min_dist: 60.0,
            hough_param1: 80.0,
            hough_param2: 35.0,
            hough_min_radius: 20,
            hough_max_radius: 50,
            expected_count: 10,
            y_tolerance_px: 20.0,
            size_tolerance_pct: 0.4,
            spacing_tolerance_pct: 0.35,
            y_band_top: 0.25,
            y_band_bottom: 0.75,
            prior_x_start: 50.0,
            prior_x_end: 1150.0,
            prior_y: 125.0,
            slot_capture_radius: 55.0,
        }
    }
}

fn default_clahe_clip() -> f64 { 3.0 }
fn default_clahe_grid() -> i32 { 8 }
fn default_dp() -> f64 { 1.2 }
fn default_min_dist() -> f64 { 60.0 }
fn default_param1() -> f64 { 80.0 }
fn default_param2() -> f64 { 35.0 }
fn default_min_r() -> i32 { 20 }
fn default_max_r() -> i32 { 50 }
fn default_count() -> usize { 10 }
fn default_y_tol() -> f32 { 20.0 }
fn default_size_tol() -> f32 { 0.4 }
fn default_spacing_tol() -> f32 { 0.35 }
fn default_y_band_top() -> f32 { 0.25 }
fn default_y_band_bottom() -> f32 { 0.75 }
fn default_prior_x_start() -> f32 { 50.0 }
fn default_prior_x_end() -> f32 { 1150.0 }
fn default_prior_y() -> f32 { 125.0 }
fn default_slot_capture_radius() -> f32 { 55.0 }

impl Calibration {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let text = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }
}
