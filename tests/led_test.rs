use image::{DynamicImage, RgbImage};

use oven_vision::config::LedConfig;
use oven_vision::led::{detect_leds, rgb_to_hsv};
use oven_vision::types::LedState;

fn make_solid_frame(r: u8, g: u8, b: u8, width: u32, height: u32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);
    for pixel in img.pixels_mut() {
        pixel[0] = r;
        pixel[1] = g;
        pixel[2] = b;
    }
    DynamicImage::ImageRgb8(img)
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
    let readings = detect_leds(&frame, &[led_config()]);
    assert_eq!(readings.len(), 1);
    assert_eq!(readings[0].state, LedState::Heating);
}

#[test]
fn classify_orange_led_as_on() {
    // Orange: RGB (255, 140, 0) -> H~33
    let frame = make_solid_frame(255, 140, 0, 10, 10);
    let readings = detect_leds(&frame, &[led_config()]);
    assert_eq!(readings.len(), 1);
    assert_eq!(readings[0].state, LedState::On);
}

#[test]
fn classify_dark_led_as_off() {
    // Very dark pixel
    let frame = make_solid_frame(5, 5, 5, 10, 10);
    let readings = detect_leds(&frame, &[led_config()]);
    assert_eq!(readings.len(), 1);
    assert_eq!(readings[0].state, LedState::Off);
}

#[test]
fn classify_desaturated_bright_as_off() {
    // Bright but desaturated (grayish white)
    let frame = make_solid_frame(200, 200, 200, 10, 10);
    let readings = detect_leds(&frame, &[led_config()]);
    assert_eq!(readings.len(), 1);
    assert_eq!(readings[0].state, LedState::Off);
}
