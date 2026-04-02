use opencv::core::{Mat, Point, Scalar, Vector};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;

use crate::config::{DialConfig, LedConfig};
use crate::detect::{DialReading, DialState, HeatLevel};
use crate::types::{LedReading, LedState};

/// Green for dial circles and "off" state (BGR).
const COLOR_GREEN: Scalar = Scalar::new(0.0, 255.0, 0.0, 255.0);
/// Red for angle indicator lines (BGR).
const COLOR_RED: Scalar = Scalar::new(0.0, 0.0, 255.0, 255.0);
/// Yellow for "on" dials low/medium (BGR).
const COLOR_YELLOW: Scalar = Scalar::new(0.0, 255.0, 255.0, 255.0);
/// Orange for high heat (BGR).
const COLOR_ORANGE: Scalar = Scalar::new(0.0, 165.0, 255.0, 255.0);
/// Bright red for max heat (BGR).
const COLOR_MAX: Scalar = Scalar::new(50.0, 50.0, 255.0, 255.0);
/// Magenta for LED rectangles (BGR).
const COLOR_MAGENTA: Scalar = Scalar::new(255.0, 0.0, 255.0, 255.0);
/// Cyan for LED "on" state marker (BGR).
const COLOR_CYAN: Scalar = Scalar::new(255.0, 255.0, 0.0, 255.0);
/// White for unavailable / neutral (BGR).
const COLOR_WHITE: Scalar = Scalar::new(255.0, 255.0, 255.0, 255.0);

/// Draw debug annotations onto a BGR color frame.
///
/// Draws dial circles, angle indicator lines, LED rectangles, and colored
/// state markers. Uses shapes (not text) to keep the dependency footprint small.
pub fn annotate_frame(
    frame: &Mat,
    dial_readings: &[DialReading],
    led_readings: &[LedReading],
    dial_configs: &[DialConfig],
    led_configs: &[LedConfig],
) -> Result<Mat, opencv::Error> {
    let mut canvas = frame.clone();

    // Annotate dials
    for (reading, cfg) in dial_readings.iter().zip(dial_configs.iter()) {
        let cx = cfg.center_x as i32;
        let cy = cfg.center_y as i32;
        let r = cfg.radius as i32;
        let center = Point::new(cx, cy);

        // Draw ROI circle in green
        imgproc::circle(&mut canvas, center, r, COLOR_GREEN, 1, imgproc::LINE_8, 0)?;

        // Draw angle indicator line in red (if we have an angle)
        if let Some(angle_deg) = reading.angle_deg {
            let rad = angle_deg.to_radians();
            let end_x = cx + (r as f64 * rad.cos()) as i32;
            let end_y = cy + (r as f64 * rad.sin()) as i32;
            imgproc::line(
                &mut canvas,
                center,
                Point::new(end_x, end_y),
                COLOR_RED,
                1,
                imgproc::LINE_8,
                0,
            )?;
        }

        // Draw a colored state dot above the circle (radius 5 filled circle)
        let marker_y = cy - r - 12;
        let state_color = match reading.state {
            DialState::Off => COLOR_GREEN,
            DialState::On(HeatLevel::Low) => COLOR_YELLOW,
            DialState::On(HeatLevel::Medium) => COLOR_YELLOW,
            DialState::On(HeatLevel::High) => COLOR_ORANGE,
            DialState::On(HeatLevel::Max) => COLOR_MAX,
            DialState::Unavailable => COLOR_WHITE,
        };
        imgproc::circle(
            &mut canvas,
            Point::new(cx, marker_y),
            5,
            state_color,
            imgproc::FILLED,
            imgproc::LINE_8,
            0,
        )?;

        // Draw confidence bar: a horizontal line proportional to confidence
        // below the state dot, from cx-r to cx-r + 2*r*confidence
        let bar_y = cy - r - 4;
        let bar_len = (2.0 * r as f64 * reading.confidence) as i32;
        let bar_color = if reading.confidence < 0.25 {
            COLOR_MAX
        } else {
            COLOR_GREEN
        };
        imgproc::line(
            &mut canvas,
            Point::new(cx - r, bar_y),
            Point::new(cx - r + bar_len, bar_y),
            bar_color,
            1,
            imgproc::LINE_8,
            0,
        )?;
    }

    // Annotate LEDs
    for (reading, cfg) in led_readings.iter().zip(led_configs.iter()) {
        // Draw ROI rectangle in magenta
        imgproc::rectangle(
            &mut canvas,
            opencv::core::Rect::new(
                cfg.x as i32,
                cfg.y as i32,
                cfg.width as i32,
                cfg.height as i32,
            ),
            COLOR_MAGENTA,
            1,
            imgproc::LINE_8,
            0,
        )?;

        // Draw state marker inside the rectangle (top-left corner, small filled circle)
        let marker_x = cfg.x as i32 + 6;
        let marker_y = cfg.y as i32 + 6;
        let led_color = match reading.state {
            LedState::Off => COLOR_WHITE,
            LedState::On => COLOR_CYAN,
            LedState::Heating => COLOR_GREEN,
        };
        imgproc::circle(
            &mut canvas,
            Point::new(marker_x, marker_y),
            4,
            led_color,
            imgproc::FILLED,
            imgproc::LINE_8,
            0,
        )?;
    }

    Ok(canvas)
}

/// Encode a BGR `Mat` to JPEG bytes at the given quality (0-100).
pub fn encode_jpeg(mat: &Mat, quality: i32) -> Result<Vec<u8>, opencv::Error> {
    let mut buf = Vector::new();
    let params = Vector::from_slice(&[imgcodecs::IMWRITE_JPEG_QUALITY, quality]);
    imgcodecs::imencode(".jpg", mat, &mut buf, &params)?;
    Ok(buf.to_vec())
}
