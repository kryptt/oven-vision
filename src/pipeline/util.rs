use opencv::core::{Mat, Point, Scalar, Size};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;

use super::stage::{DetectedFeatures, Line, LinePair};

/// Padding above the top line, as a fraction of the inter-line distance.
/// Shared by S4 (Perspective) and S6 (ExtractBand).
pub const TOP_PADDING_FRAC: f64 = 0.5;

/// Path to template images (bundled in container at build time).
const KNOB_TEMPLATE_PATH: &str = "/templates/knob.jpg";
const CLOCK_TEMPLATE_PATH: &str = "/templates/clock.jpg";

/// Shared CLAHE-enhanced templates loaded once and reused by S7 and S8.
pub struct Templates {
    pub knob: Mat,
    pub clock: Mat,
}

impl Templates {
    pub fn load() -> Self {
        let knob = Self::load_and_enhance(KNOB_TEMPLATE_PATH);
        let clock = Self::load_and_enhance(CLOCK_TEMPLATE_PATH);
        Self { knob, clock }
    }

    /// Load templates from a custom directory (for use outside Docker).
    pub fn load_from(dir: &std::path::Path) -> Self {
        let knob = Self::load_and_enhance(&dir.join("knob.jpg").to_string_lossy());
        let clock = Self::load_and_enhance(&dir.join("clock.jpg").to_string_lossy());
        Self { knob, clock }
    }

    fn load_and_enhance(path: &str) -> Mat {
        let raw = imgcodecs::imread(path, imgcodecs::IMREAD_GRAYSCALE).unwrap_or_else(|e| {
            tracing::warn!(%e, path, "failed to load template, using empty");
            Mat::default()
        });
        if raw.empty() {
            return raw;
        }
        let mut clahe = match imgproc::create_clahe(3.0, Size::new(8, 8)) {
            Ok(c) => c,
            Err(_) => return raw,
        };
        let mut enhanced = Mat::default();
        clahe.apply(&raw, &mut enhanced).unwrap_or_else(|e| {
            tracing::warn!(%e, path, "failed to apply CLAHE to template");
        });
        if enhanced.empty() { raw } else { enhanced }
    }
}

/// Cluster lines by rho proximity.
pub(super) fn cluster_by_rho(lines: &[(f64, f64)], threshold: f64) -> Vec<Vec<(f64, f64)>> {
    let mut sorted: Vec<(f64, f64)> = lines.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut clusters: Vec<Vec<(f64, f64)>> = Vec::new();
    for &(rho, theta) in &sorted {
        let mut added = false;
        for cluster in &mut clusters {
            let avg_rho: f64 = cluster.iter().map(|&(r, _)| r).sum::<f64>() / cluster.len() as f64;
            if (rho - avg_rho).abs() <= threshold {
                cluster.push((rho, theta));
                added = true;
                break;
            }
        }
        if !added {
            clusters.push(vec![(rho, theta)]);
        }
    }
    clusters
}

/// Average rho and theta for a cluster.
pub(super) fn cluster_average(cluster: &[(f64, f64)]) -> (f64, f64) {
    let n = cluster.len() as f64;
    let rho = cluster.iter().map(|&(r, _)| r).sum::<f64>() / n;
    let theta = cluster.iter().map(|&(_, t)| t).sum::<f64>() / n;
    (rho, theta)
}

/// Draw a line on a canvas.
pub(super) fn draw_line(canvas: &mut Mat, line: &Line, color: Scalar) -> Result<(), opencv::Error> {
    imgproc::line(
        canvas,
        Point::new(line.x1 as i32, line.y1 as i32),
        Point::new(line.x2 as i32, line.y2 as i32),
        color,
        2,
        imgproc::LINE_8,
        0,
    )
}

/// Thresholds for feature validation (shared by S9 coarse and S13 strict checks).
pub(super) struct SanityThresholds {
    pub y_tolerance_px: f64,
    pub min_x_gap_px: f64,
    pub max_gap_cv: f64,
    pub max_radius_factor: f64,
    pub expected_knobs: usize,
    /// Skip the pairwise overlap check (coarse pass only — S13 enforces it).
    pub skip_overlap: bool,
}

/// Measurements computed during validation, returned on success for logging.
pub(super) struct SanityMeasurements {
    pub max_y_dev: f64,
    pub min_x_gap: f64,
    pub gap_cv: f64,
    pub min_pair_dist: f64,
    pub median_r: f64,
    pub r_cv: f64,
}

/// Validate detected features against thresholds.
/// Returns `Ok(measurements)` on success or `Err(reason)` on failure.
pub(super) fn validate_features(
    features: &DetectedFeatures,
    t: &SanityThresholds,
) -> Result<SanityMeasurements, String> {
    // Check 1: knob count
    if features.knobs.len() != t.expected_knobs {
        return Err(format!(
            "expected {} knobs, got {}",
            t.expected_knobs,
            features.knobs.len()
        ));
    }

    // Check 2: Y alignment
    let mut ys: Vec<f64> = features.knobs.iter().map(|k| k.center_y).collect();
    ys.sort_by(f64::total_cmp);
    let median_y = ys[ys.len() / 2];

    let max_y_dev = ys.iter().map(|y| (y - median_y).abs()).fold(0.0f64, f64::max);
    if max_y_dev > t.y_tolerance_px {
        return Err(format!(
            "Y-deviation = {max_y_dev:.1}px (max {}px)",
            t.y_tolerance_px
        ));
    }

    // Check 3: X monotonically increasing with minimum gap
    let mut min_x_gap = f64::INFINITY;
    for i in 1..features.knobs.len() {
        let gap = features.knobs[i].center_x - features.knobs[i - 1].center_x;
        if gap < min_x_gap {
            min_x_gap = gap;
        }
        if gap < t.min_x_gap_px {
            return Err(format!(
                "knobs {}/{} X-gap = {gap:.1}px (min {}px)",
                i - 1,
                i,
                t.min_x_gap_px
            ));
        }
    }

    // Check 3b: no overlapping knobs (skipped in coarse pass)
    let mut min_pair_dist = f64::INFINITY;
    for i in 0..features.knobs.len() {
        for j in (i + 1)..features.knobs.len() {
            let dx = features.knobs[j].center_x - features.knobs[i].center_x;
            let dy = features.knobs[j].center_y - features.knobs[i].center_y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < min_pair_dist {
                min_pair_dist = dist;
            }
            if !t.skip_overlap {
                let radii_sum = features.knobs[i].radius + features.knobs[j].radius;
                if dist < radii_sum {
                    return Err(format!(
                        "knobs {i}/{j} overlap: dist={dist:.1}px < radii_sum={radii_sum:.1}px"
                    ));
                }
            }
        }
    }

    // Check 3c: X-gap regularity
    let gap_cv = {
        let mut gaps: Vec<f64> = Vec::new();
        for i in 1..features.knobs.len() {
            gaps.push(features.knobs[i].center_x - features.knobs[i - 1].center_x);
        }
        let mean: f64 = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let var: f64 = gaps.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / gaps.len() as f64;
        var.sqrt() / mean.max(1.0)
    };
    if gap_cv > t.max_gap_cv {
        return Err(format!(
            "X-gap CV = {gap_cv:.2} (max {})",
            t.max_gap_cv
        ));
    }

    // Check 4: radius consistency
    let mut radii: Vec<f64> = features.knobs.iter().map(|k| k.radius).collect();
    radii.sort_by(f64::total_cmp);
    let median_r = radii[radii.len() / 2];
    let r_lo = median_r / t.max_radius_factor;
    let r_hi = median_r * t.max_radius_factor;

    for (i, knob) in features.knobs.iter().enumerate() {
        if knob.radius < r_lo || knob.radius > r_hi {
            return Err(format!(
                "knob {i} radius = {:.1} outside [{r_lo:.1}, {r_hi:.1}]",
                knob.radius
            ));
        }
    }

    // Check 5: clock is to the left of all knobs
    if let Some(first_knob) = features.knobs.first() {
        if features.clock.center_x >= first_knob.center_x {
            return Err(format!(
                "clock x={:.1} is not left of first knob x={:.1}",
                features.clock.center_x, first_knob.center_x
            ));
        }
    }

    // Compute radius CV for logging
    let r_cv = {
        let mean: f64 = radii.iter().sum::<f64>() / radii.len() as f64;
        let var: f64 = radii.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / radii.len() as f64;
        var.sqrt() / mean.max(1.0)
    };

    Ok(SanityMeasurements {
        max_y_dev,
        min_x_gap,
        gap_cv,
        min_pair_dist,
        median_r,
        r_cv,
    })
}

/// Convert BGR to grayscale, apply CLAHE, and median blur.
///
/// Common preprocessing shared by S2, S5, S7, S8, and S12.
/// `blur_ksize` is the median blur kernel size (3 for fine features, 5 for
/// coarse line detection).
pub(super) fn enhance_gray(src: &Mat, blur_ksize: i32) -> Result<Mat, opencv::Error> {
    let mut gray = Mat::default();
    imgproc::cvt_color_def(src, &mut gray, imgproc::COLOR_BGR2GRAY)?;

    // Scale CLAHE tile size to image dimensions (min 4×4 tiles, max 8×8)
    let tile_w = (gray.cols() / 32).clamp(4, 8);
    let tile_h = (gray.rows() / 32).clamp(4, 8);
    let mut clahe = imgproc::create_clahe(3.0, Size::new(tile_w, tile_h))?;
    let mut enhanced = Mat::default();
    clahe.apply(&gray, &mut enhanced)?;

    let mut blurred = Mat::default();
    imgproc::median_blur(&enhanced, &mut blurred, blur_ksize)?;
    Ok(blurred)
}

/// Average inter-line distance from a `LinePair`.
pub(super) fn inter_line_distance(lines: &LinePair) -> f64 {
    let left_dist = (lines.bottom.y1 - lines.top.y1).abs();
    let right_dist = (lines.bottom.y2 - lines.top.y2).abs();
    (left_dist + right_dist) / 2.0
}

/// Median radius from a slice of knobs.
pub(super) fn median_radius(knobs: &[super::stage::CircleFeature]) -> f64 {
    let mut rs: Vec<f64> = knobs.iter().map(|k| k.radius).collect();
    rs.sort_by(f64::total_cmp);
    rs[rs.len() / 2]
}
