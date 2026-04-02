use std::path::PathBuf;

use opencv::core::Vector;
use opencv::imgcodecs;
use opencv::prelude::*;

use oven_vision::capture::fetch_frame;
use oven_vision::config::{DialConfig, LedConfig};
use oven_vision::detect::radial_edge_scan;
use oven_vision::preprocess::preprocess;

const DEFAULT_URL: &str = "http://192.168.2.52:1984/api/frame.jpeg?src=kitchen";

/// Hardcoded defaults from the feasibility spike — validated ROI coordinates
/// at 2560x1440 resolution from the kitchen camera.
fn default_dials() -> Vec<DialConfig> {
    vec![
        // Position 1: Left oven mode (drives left LED)
        // Precise per-dial coordinates from HoughCircles detection targeting inner knobs
        DialConfig { label: "oven_left_mode".into(),       center_x: 1430, center_y: 1059, radius: 10, off_angle_deg: 0.0, off_tolerance_deg: 25.0 },
        DialConfig { label: "burner_top_left".into(),      center_x: 1491, center_y: 1097, radius: 14, off_angle_deg: 0.0, off_tolerance_deg: 25.0 },
        DialConfig { label: "burner_bottom_left".into(),   center_x: 1551, center_y: 1112, radius: 14, off_angle_deg: 0.0, off_tolerance_deg: 25.0 },
        DialConfig { label: "oven_left_temp".into(),       center_x: 1609, center_y: 1103, radius: 14, off_angle_deg: 0.0, off_tolerance_deg: 25.0 },
        DialConfig { label: "burner_top_center".into(),    center_x: 1677, center_y: 1101, radius: 14, off_angle_deg: 0.0, off_tolerance_deg: 25.0 },
        DialConfig { label: "burner_bottom_center".into(), center_x: 1753, center_y: 1116, radius:  9, off_angle_deg: 0.0, off_tolerance_deg: 25.0 },
        DialConfig { label: "burner_top_right".into(),     center_x: 1844, center_y: 1077, radius: 14, off_angle_deg: 0.0, off_tolerance_deg: 25.0 },
        DialConfig { label: "burner_bottom_right".into(),  center_x: 1893, center_y: 1077, radius: 14, off_angle_deg: 0.0, off_tolerance_deg: 25.0 },
        // Position 9: Right oven temp (same CCW scale)
        DialConfig { label: "oven_right_temp".into(),   center_x: 1965, center_y: 1085, radius: 28, off_angle_deg: 0.0, off_tolerance_deg: 20.0 },
        // Position 10: Right oven mode (drives right LED)
        DialConfig { label: "oven_right_mode".into(),   center_x: 2030, center_y: 1085, radius: 28, off_angle_deg: 0.0, off_tolerance_deg: 20.0 },
    ]
}

fn default_leds() -> Vec<LedConfig> {
    vec![
        LedConfig { label: "oven_left_led".into(),  x: 1860, y: 1010, width: 20, height: 15 },
        LedConfig { label: "oven_right_led".into(), x: 1995, y: 1010, width: 20, height: 15 },
    ]
}

fn print_usage() {
    eprintln!("Usage: calibrate [OPTIONS]");
    eprintln!();
    eprintln!("Fetches a frame from go2rtc, detects dial angles (assuming dials are");
    eprintln!("currently at the OFF position), and outputs a calibration YAML file.");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --url <URL>       go2rtc snapshot URL (default: {DEFAULT_URL})");
    eprintln!("  --output <PATH>   write YAML to file instead of stdout");
    eprintln!("  --reference <PATH> save fetched frame to this path (default: calibration_reference.jpg)");
    eprintln!("  --help            show this help");
}

struct Args {
    url: String,
    output: Option<PathBuf>,
    reference_path: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut url = DEFAULT_URL.to_string();
    let mut output = None;
    let mut reference_path = PathBuf::from("calibration_reference.jpg");

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            "--url" => {
                url = args
                    .next()
                    .ok_or_else(|| "--url requires a value".to_string())?;
            }
            "--output" => {
                output = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--output requires a value".to_string())?,
                ));
            }
            "--reference" => {
                reference_path = PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--reference requires a value".to_string())?,
                );
            }
            other => {
                return Err(format!("unknown argument: {other}"));
            }
        }
    }

    Ok(Args {
        url,
        output,
        reference_path,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error: {e}");
            eprintln!();
            print_usage();
            std::process::exit(1);
        }
    };

    eprintln!("Fetching frame from {}...", args.url);
    let client = reqwest::Client::new();
    let frame = fetch_frame(&client, &args.url).await?;
    eprintln!(
        "Frame captured: {}x{} pixels",
        frame.cols(),
        frame.rows()
    );

    // Save reference image
    let ref_path = args.reference_path.to_str().unwrap_or("calibration_reference.jpg");
    imgcodecs::imwrite(ref_path, &frame, &Vector::new())?;
    eprintln!("Reference image saved to {:?}", args.reference_path);

    // Preprocess for edge detection (no perspective correction in calibration)
    let gray = preprocess(&frame)?;

    // Use Canny edge detection (same params as the detector)
    let mut edges = opencv::core::Mat::default();
    opencv::imgproc::canny(&gray, &mut edges, 10.0, 30.0, 3, false)?;

    // Detect angles for each default dial
    let dials = default_dials();
    let leds = default_leds();

    eprintln!();
    eprintln!("Detecting dial angles (ensure all dials are in the OFF position)...");
    eprintln!();

    let mut calibrated_dials = Vec::with_capacity(dials.len());

    for dial in &dials {
        let (angle, strength) = radial_edge_scan(&edges, dial.center_x, dial.center_y, dial.radius)?;
        eprintln!(
            "  {:16} => angle={:6.1} deg, strength={:.3}",
            dial.label, angle, strength
        );

        calibrated_dials.push(DialConfig {
            label: dial.label.clone(),
            center_x: dial.center_x,
            center_y: dial.center_y,
            radius: dial.radius,
            off_angle_deg: angle,
            off_tolerance_deg: dial.off_tolerance_deg,
        });
    }

    eprintln!();

    // Build YAML output
    let yaml = build_yaml(&args.url, &calibrated_dials, &leds);

    match &args.output {
        Some(path) => {
            std::fs::write(path, &yaml)?;
            eprintln!("Calibration written to {:?}", path);
        }
        None => {
            eprintln!("--- calibration YAML ---");
            print!("{yaml}");
        }
    }

    Ok(())
}

fn build_yaml(url: &str, dials: &[DialConfig], leds: &[LedConfig]) -> String {
    use std::fmt::Write;

    let mut out = String::new();

    writeln!(out, "go2rtc_url: \"{url}\"").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "mqtt:").unwrap();
    writeln!(out, "  host: mosquitto.home").unwrap();
    writeln!(out, "  port: 1883").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "poll_interval_secs: 5").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "dials:").unwrap();

    for d in dials {
        writeln!(out, "  - label: \"{}\"", d.label).unwrap();
        writeln!(out, "    center_x: {}", d.center_x).unwrap();
        writeln!(out, "    center_y: {}", d.center_y).unwrap();
        writeln!(out, "    radius: {}", d.radius).unwrap();
        writeln!(out, "    off_angle_deg: {:.1}", d.off_angle_deg).unwrap();
        writeln!(out, "    off_tolerance_deg: {:.1}", d.off_tolerance_deg).unwrap();
    }

    writeln!(out).unwrap();
    writeln!(out, "leds:").unwrap();

    for l in leds {
        writeln!(out, "  - label: \"{}\"", l.label).unwrap();
        writeln!(out, "    x: {}", l.x).unwrap();
        writeln!(out, "    y: {}", l.y).unwrap();
        writeln!(out, "    width: {}", l.width).unwrap();
        writeln!(out, "    height: {}", l.height).unwrap();
    }

    out
}
