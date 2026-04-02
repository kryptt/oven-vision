use image::DynamicImage;

use crate::config::LedConfig;
use crate::types::{LedReading, LedState};

/// Convert an RGB pixel to HSV.
///
/// Returns `(h, s, v)` where H is in 0..360, S in 0..255, V in 0..255.
pub fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
    let rf = r as f64 / 255.0;
    let gf = g as f64 / 255.0;
    let bf = b as f64 / 255.0;

    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let delta = max - min;

    let v = max * 255.0;

    if max == 0.0 {
        return (0.0, 0.0, v);
    }

    let s = (delta / max) * 255.0;

    if delta == 0.0 {
        return (0.0, s, v);
    }

    let h = if max == rf {
        60.0 * (((gf - bf) / delta) % 6.0)
    } else if max == gf {
        60.0 * (((bf - rf) / delta) + 2.0)
    } else {
        60.0 * (((rf - gf) / delta) + 4.0)
    };

    let h = if h < 0.0 { h + 360.0 } else { h };

    (h, s, v)
}

/// Detect LED states from a color frame for each configured LED region.
pub fn detect_leds(frame: &DynamicImage, led_configs: &[LedConfig]) -> Vec<LedReading> {
    let rgb = frame.to_rgb8();
    let (img_w, img_h) = (rgb.width(), rgb.height());

    led_configs
        .iter()
        .map(|cfg| {
            let x_end = (cfg.x + cfg.width).min(img_w);
            let y_end = (cfg.y + cfg.height).min(img_h);

            let mut sum_h_sin = 0.0f64;
            let mut sum_h_cos = 0.0f64;
            let mut sum_s = 0.0f64;
            let mut sum_v = 0.0f64;
            let mut count = 0u32;

            for py in cfg.y..y_end {
                for px in cfg.x..x_end {
                    let pixel = rgb.get_pixel(px, py);
                    let (h, s, v) = rgb_to_hsv(pixel[0], pixel[1], pixel[2]);
                    sum_h_sin += h.to_radians().sin();
                    sum_h_cos += h.to_radians().cos();
                    sum_s += s;
                    sum_v += v;
                    count += 1;
                }
            }

            if count == 0 {
                return LedReading {
                    label: cfg.label.clone(),
                    state: LedState::Off,
                };
            }

            let n = count as f64;
            let mean_h = (sum_h_sin.atan2(sum_h_cos).to_degrees() + 360.0) % 360.0;
            let mean_s = sum_s / n;
            let mean_v = sum_v / n;

            let state = classify_led(mean_h, mean_s, mean_v);

            LedReading {
                label: cfg.label.clone(),
                state,
            }
        })
        .collect()
}

/// Classify an LED state from mean HSV values.
///
/// H in 0..360, S in 0..255, V in 0..255.
fn classify_led(h: f64, s: f64, v: f64) -> LedState {
    if v < 50.0 || s < 50.0 {
        return LedState::Off;
    }

    // Green: H 110-160 (full-scale 0-360, corresponding to ~55-80 in OpenCV half-scale)
    if (110.0..=160.0).contains(&h) {
        return LedState::Heating;
    }

    // Orange: H 10-60 (full-scale, corresponding to ~5-30 in OpenCV half-scale)
    if (10.0..=60.0).contains(&h) {
        return LedState::On;
    }

    LedState::Off
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hsv_pure_red() {
        let (h, s, v) = rgb_to_hsv(255, 0, 0);
        assert!((h - 0.0).abs() < 1.0, "h={h}");
        assert!((s - 255.0).abs() < 1.0, "s={s}");
        assert!((v - 255.0).abs() < 1.0, "v={v}");
    }

    #[test]
    fn hsv_pure_green() {
        let (h, s, v) = rgb_to_hsv(0, 255, 0);
        assert!((h - 120.0).abs() < 1.0, "h={h}");
        assert!((s - 255.0).abs() < 1.0, "s={s}");
        assert!((v - 255.0).abs() < 1.0, "v={v}");
    }

    #[test]
    fn hsv_pure_blue() {
        let (h, s, v) = rgb_to_hsv(0, 0, 255);
        assert!((h - 240.0).abs() < 1.0, "h={h}");
        assert!((s - 255.0).abs() < 1.0, "s={s}");
        assert!((v - 255.0).abs() < 1.0, "v={v}");
    }

    #[test]
    fn hsv_white() {
        let (h, s, v) = rgb_to_hsv(255, 255, 255);
        assert!((s - 0.0).abs() < 1.0, "s={s}");
        assert!((v - 255.0).abs() < 1.0, "v={v}");
        // h is undefined for white, just check s and v
    }

    #[test]
    fn hsv_black() {
        let (h, s, v) = rgb_to_hsv(0, 0, 0);
        assert!((h - 0.0).abs() < 1.0, "h={h}");
        assert!((s - 0.0).abs() < 1.0, "s={s}");
        assert!((v - 0.0).abs() < 1.0, "v={v}");
    }
}
