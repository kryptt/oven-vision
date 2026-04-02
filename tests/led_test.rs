use opencv::core::{Mat, CV_8UC3};
use opencv::prelude::*;

use oven_vision::config::LedConfig;
use oven_vision::led::{detect_leds, rgb_to_hsv};
use oven_vision::types::LedState;

/// Create a solid-color BGR Mat of the given dimensions.
fn make_solid_frame(r: u8, g: u8, b: u8, width: u32, height: u32) -> Mat {
    // OpenCV uses BGR ordering
    Mat::new_rows_cols_with_default(
        height as i32,
        width as i32,
        CV_8UC3,
        opencv::core::Scalar::new(b as f64, g as f64, r as f64, 0.0),
    )
    .expect("failed to create test Mat")
}

fn led_config() -> LedConfig {
    LedConfig {
        label: "test".to_string(),
        x: 0,
        y: 0,
        width: 10,
        height: 10,
    }
}

#[test]
fn rgb_to_hsv_pure_red() {
    let (h, s, v) = rgb_to_hsv(255, 0, 0);
    assert!((h - 0.0).abs() < 1.0);
    assert!((s - 255.0).abs() < 1.0);
    assert!((v - 255.0).abs() < 1.0);
}

#[test]
fn rgb_to_hsv_pure_green() {
    let (h, s, v) = rgb_to_hsv(0, 255, 0);
    assert!((h - 120.0).abs() < 1.0);
    assert!((s - 255.0).abs() < 1.0);
    assert!((v - 255.0).abs() < 1.0);
}

#[test]
fn rgb_to_hsv_pure_blue() {
    let (h, s, v) = rgb_to_hsv(0, 0, 255);
    assert!((h - 240.0).abs() < 1.0);
    assert!((s - 255.0).abs() < 1.0);
    assert!((v - 255.0).abs() < 1.0);
}

#[test]
fn rgb_to_hsv_white() {
    let (_, s, v) = rgb_to_hsv(255, 255, 255);
    assert!((s - 0.0).abs() < 1.0);
    assert!((v - 255.0).abs() < 1.0);
}

#[test]
fn rgb_to_hsv_black() {
    let (h, s, v) = rgb_to_hsv(0, 0, 0);
    assert!((h - 0.0).abs() < 1.0);
    assert!((s - 0.0).abs() < 1.0);
    assert!((v - 0.0).abs() < 1.0);
}

#[test]
fn classify_green_led_as_heating() {
    // Bright green: H~120, S=255, V=255
    let frame = make_solid_frame(0, 255, 0, 10, 10);
    let readings = detect_leds(&frame, &[led_config()]).unwrap();
    assert_eq!(readings.len(), 1);
    assert_eq!(readings[0].state, LedState::Heating);
}

#[test]
fn classify_orange_led_as_on() {
    // Orange: RGB (255, 140, 0) -> H~33
    let frame = make_solid_frame(255, 140, 0, 10, 10);
    let readings = detect_leds(&frame, &[led_config()]).unwrap();
    assert_eq!(readings.len(), 1);
    assert_eq!(readings[0].state, LedState::On);
}

#[test]
fn classify_dark_led_as_off() {
    // Very dark pixel
    let frame = make_solid_frame(5, 5, 5, 10, 10);
    let readings = detect_leds(&frame, &[led_config()]).unwrap();
    assert_eq!(readings.len(), 1);
    assert_eq!(readings[0].state, LedState::Off);
}

#[test]
fn classify_desaturated_bright_as_off() {
    // Bright but desaturated (grayish white)
    let frame = make_solid_frame(200, 200, 200, 10, 10);
    let readings = detect_leds(&frame, &[led_config()]).unwrap();
    assert_eq!(readings.len(), 1);
    assert_eq!(readings[0].state, LedState::Off);
}
