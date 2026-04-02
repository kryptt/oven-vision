use std::io::Cursor;

use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, Rgba, RgbaImage};
use imageproc::drawing::{draw_hollow_circle_mut, draw_hollow_rect_mut, draw_line_segment_mut};
use imageproc::rect::Rect;

use crate::config::{DialConfig, LedConfig};
use crate::detect::{DialReading, DialState, HeatLevel};
use crate::types::{LedReading, LedState};

/// Green for dial circles and "off" state.
const COLOR_GREEN: Rgba<u8> = Rgba([0, 255, 0, 255]);
/// Red for angle indicator lines.
const COLOR_RED: Rgba<u8> = Rgba([255, 0, 0, 255]);
/// Yellow for "on" dials (low/medium).
const COLOR_YELLOW: Rgba<u8> = Rgba([255, 255, 0, 255]);
/// Orange for high heat.
const COLOR_ORANGE: Rgba<u8> = Rgba([255, 165, 0, 255]);
/// Bright red for max heat.
const COLOR_MAX: Rgba<u8> = Rgba([255, 50, 50, 255]);
/// Magenta for LED rectangles.
const COLOR_MAGENTA: Rgba<u8> = Rgba([255, 0, 255, 255]);
/// Cyan for LED "on" state marker.
const COLOR_CYAN: Rgba<u8> = Rgba([0, 255, 255, 255]);
/// White for unavailable / neutral.
const COLOR_WHITE: Rgba<u8> = Rgba([255, 255, 255, 255]);

/// Draw debug annotations onto a color frame.
///
/// Draws dial circles, angle indicator lines, LED rectangles, and colored
/// state markers. Uses shapes (not text) to keep the dependency footprint small.
pub fn annotate_frame(
    frame: &DynamicImage,
    dial_readings: &[DialReading],
    led_readings: &[LedReading],
    dial_configs: &[DialConfig],
    led_configs: &[LedConfig],
) -> DynamicImage {
    let mut canvas: RgbaImage = frame.to_rgba8();

    // Annotate dials
    for (reading, cfg) in dial_readings.iter().zip(dial_configs.iter()) {
        let cx = cfg.center_x as i32;
        let cy = cfg.center_y as i32;
        let r = cfg.radius as i32;

        // Draw ROI circle in green
        draw_hollow_circle_mut(&mut canvas, (cx, cy), r, COLOR_GREEN);

        // Draw angle indicator line in red (if we have an angle)
        if let Some(angle_deg) = reading.angle_deg {
            let rad = angle_deg.to_radians();
            let end_x = cx as f32 + (r as f32) * rad.cos() as f32;
            let end_y = cy as f32 + (r as f32) * rad.sin() as f32;
            draw_line_segment_mut(
                &mut canvas,
                (cx as f32, cy as f32),
                (end_x, end_y),
                COLOR_RED,
            );
        }

        // Draw a colored state dot above the circle (5px filled circle)
        let marker_y = cy - r - 12;
        let state_color = match reading.state {
            DialState::Off => COLOR_GREEN,
            DialState::On(HeatLevel::Low) => COLOR_YELLOW,
            DialState::On(HeatLevel::Medium) => COLOR_YELLOW,
            DialState::On(HeatLevel::High) => COLOR_ORANGE,
            DialState::On(HeatLevel::Max) => COLOR_MAX,
            DialState::Unavailable => COLOR_WHITE,
        };
        // Draw filled marker as concentric circles
        for dr in 0..=5 {
            draw_hollow_circle_mut(&mut canvas, (cx, marker_y), dr, state_color);
        }

        // Draw confidence bar: a horizontal line proportional to confidence
        // below the state dot, from cx-r to cx-r + 2*r*confidence
        let bar_y = cy - r - 4;
        let bar_len = (2.0 * r as f64 * reading.confidence) as f32;
        let bar_color = if reading.confidence < 0.25 {
            COLOR_MAX
        } else {
            COLOR_GREEN
        };
        draw_line_segment_mut(
            &mut canvas,
            (cx as f32 - r as f32, bar_y as f32),
            (cx as f32 - r as f32 + bar_len, bar_y as f32),
            bar_color,
        );
    }

    // Annotate LEDs
    for (reading, cfg) in led_readings.iter().zip(led_configs.iter()) {
        let rect = Rect::at(cfg.x as i32, cfg.y as i32)
            .of_size(cfg.width, cfg.height);

        // Draw ROI rectangle in magenta
        draw_hollow_rect_mut(&mut canvas, rect, COLOR_MAGENTA);

        // Draw state marker inside the rectangle (top-left corner, small filled circle)
        let marker_x = cfg.x as i32 + 6;
        let marker_y = cfg.y as i32 + 6;
        let led_color = match reading.state {
            LedState::Off => COLOR_WHITE,
            LedState::On => COLOR_CYAN,
            LedState::Heating => COLOR_GREEN,
        };
        for dr in 0..=4 {
            draw_hollow_circle_mut(&mut canvas, (marker_x, marker_y), dr, led_color);
        }
    }

    DynamicImage::ImageRgba8(canvas)
}

/// Encode a `DynamicImage` to JPEG bytes at the given quality (0-100).
pub fn encode_jpeg(image: &DynamicImage, quality: u8) -> Vec<u8> {
    let rgb = image.to_rgb8();
    let mut buf = Cursor::new(Vec::new());
    let encoder = JpegEncoder::new_with_quality(&mut buf, quality);
    rgb.write_with_encoder(encoder)
        .expect("JPEG encoding should not fail for valid RGB data");
    buf.into_inner()
}
