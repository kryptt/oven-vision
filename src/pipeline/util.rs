use opencv::core::{Mat, Point, Scalar};
use opencv::imgproc;

use super::stage::Line;

/// Cluster lines by rho proximity.
pub(super) fn cluster_by_rho(lines: &[(f64, f64)], threshold: f64) -> Vec<Vec<(f64, f64)>> {
    let mut sorted: Vec<(f64, f64)> = lines.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

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
