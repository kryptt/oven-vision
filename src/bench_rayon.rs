//! Benchmark: rayon parallelism for S8 template matching.
//!
//! Tests the template-matching inner loop at varying rayon thread counts
//! and OpenCV internal thread settings to find the sweet spot.

use opencv::core::Mat;
use opencv::imgproc;
use opencv::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

/// Simulate the S8 inner loop: match N rotated edge templates against an edge image.
/// Returns Vec of (angle, max_score) per template.
fn match_templates_sequential(
    edge_img: &Mat,
    templates: &[(Mat, f64)], // (edge_template, angle)
) -> Vec<(f64, f64)> {
    let mut results = Vec::with_capacity(templates.len());
    for (tmpl, angle) in templates {
        if tmpl.cols() >= edge_img.cols() || tmpl.rows() >= edge_img.rows() {
            continue;
        }
        let mut result = Mat::default();
        imgproc::match_template(
            edge_img,
            tmpl,
            &mut result,
            imgproc::TM_CCOEFF_NORMED,
            &Mat::default(),
        )
        .unwrap();

        // Find max score
        let mut min_val = 0.0;
        let mut max_val = 0.0;
        opencv::core::min_max_loc(
            &result, Some(&mut min_val), Some(&mut max_val), None, None, &Mat::default(),
        )
        .unwrap();

        results.push((*angle, max_val));
    }
    results
}

fn match_templates_parallel(
    edge_img: &Mat,
    templates: &[(Mat, f64)],
) -> Vec<(f64, f64)> {
    templates
        .par_iter()
        .filter_map(|(tmpl, angle)| {
            if tmpl.cols() >= edge_img.cols() || tmpl.rows() >= edge_img.rows() {
                return None;
            }
            let mut result = Mat::default();
            imgproc::match_template(
                edge_img,
                tmpl,
                &mut result,
                imgproc::TM_CCOEFF_NORMED,
                &Mat::default(),
            )
            .ok()?;

            let mut min_val = 0.0;
            let mut max_val = 0.0;
            opencv::core::min_max_loc(
                &result, Some(&mut min_val), Some(&mut max_val), None, None, &Mat::default(),
            )
            .ok()?;

            Some((*angle, max_val))
        })
        .collect()
}

/// Create a synthetic edge image and rotated templates that approximate
/// the real S8 workload.
fn create_test_data() -> (Mat, Vec<(Mat, f64)>) {
    // Approximate the knob-area image size after FindClock crops it:
    // warped image ~870px wide, clock takes ~80px, so knob area ~790 x ~100
    let img_w = 790;
    let img_h = 100;

    // Create a synthetic edge image with some structure
    let mut img = Mat::zeros(img_h, img_w, opencv::core::CV_8UC1)
        .unwrap()
        .to_mat()
        .unwrap();

    // Draw some circles to simulate knob edges
    for i in 0..10 {
        let cx = 40 + i * 75;
        let cy = 50;
        imgproc::circle(
            &mut img,
            opencv::core::Point::new(cx, cy),
            15,
            opencv::core::Scalar::new(255.0, 0.0, 0.0, 0.0),
            1,
            imgproc::LINE_8,
            0,
        )
        .unwrap();
    }

    // Create rotated templates (36 rotations like real S8)
    // Template size ~20x20 (scale 0.20 of ~100px source)
    let tmpl_size = 20;
    let mut base_tmpl = Mat::zeros(tmpl_size, tmpl_size, opencv::core::CV_8UC1)
        .unwrap()
        .to_mat()
        .unwrap();
    imgproc::circle(
        &mut base_tmpl,
        opencv::core::Point::new(tmpl_size / 2, tmpl_size / 2),
        tmpl_size / 2 - 2,
        opencv::core::Scalar::new(255.0, 0.0, 0.0, 0.0),
        1,
        imgproc::LINE_8,
        0,
    )
    .unwrap();

    let n_rotations = 36;
    let mut templates = Vec::with_capacity(n_rotations);
    for i in 0..n_rotations {
        let angle = i as f64 * 10.0;
        let cx = tmpl_size as f64 / 2.0;
        let cy = tmpl_size as f64 / 2.0;
        let rot_mat = imgproc::get_rotation_matrix_2d(
            opencv::core::Point2f::new(cx as f32, cy as f32),
            -angle,
            1.0,
        )
        .unwrap();
        let mut rotated = Mat::default();
        imgproc::warp_affine(
            &base_tmpl,
            &mut rotated,
            &rot_mat,
            opencv::core::Size::new(tmpl_size, tmpl_size),
            imgproc::INTER_NEAREST,
            opencv::core::BORDER_CONSTANT,
            opencv::core::Scalar::default(),
        )
        .unwrap();
        templates.push((rotated, angle));
    }

    (img, templates)
}

fn bench_config(
    label: &str,
    edge_img: &Mat,
    templates: &[(Mat, f64)],
    rayon_threads: usize,
    cv_threads: i32,
    iterations: usize,
) {
    // Set OpenCV internal threads
    opencv::core::set_num_threads(cv_threads).unwrap();

    // Set rayon threads
    // We can't change the global pool after init, so we use a custom pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(rayon_threads)
        .build()
        .unwrap();

    // Warmup
    pool.install(|| {
        match_templates_parallel(edge_img, templates);
    });

    // Benchmark parallel
    let start = Instant::now();
    for _ in 0..iterations {
        pool.install(|| {
            match_templates_parallel(edge_img, templates);
        });
    }
    let par_elapsed = start.elapsed();
    let par_avg_us = par_elapsed.as_micros() as f64 / iterations as f64;

    // Benchmark sequential (only once per config for comparison)
    let start = Instant::now();
    for _ in 0..iterations {
        match_templates_sequential(edge_img, templates);
    }
    let seq_elapsed = start.elapsed();
    let seq_avg_us = seq_elapsed.as_micros() as f64 / iterations as f64;

    let speedup = seq_avg_us / par_avg_us;

    println!(
        "{:<30} seq={:>8.0}us  par={:>8.0}us  speedup={:.2}x",
        label, seq_avg_us, par_avg_us, speedup
    );
}

fn main() {
    println!("=== S8 Template Matching Rayon Benchmark ===");
    println!("CPU: {} threads available", std::thread::available_parallelism().unwrap());
    println!();

    let (edge_img, templates) = create_test_data();
    println!(
        "Edge image: {}x{}, Templates: {} ({} rotations)",
        edge_img.cols(),
        edge_img.rows(),
        templates.len(),
        templates.len()
    );
    println!();

    let iterations = 50;

    println!("{:<30} {:>12}  {:>12}  {}", "Config", "Sequential", "Parallel", "Speedup");
    println!("{}", "-".repeat(75));

    // Test varying rayon thread counts with OpenCV threads=1
    for &threads in &[1, 2, 4, 6, 8, 12, 16, 24, 32] {
        let label = format!("rayon={:>2}, cv=1", threads);
        bench_config(&label, &edge_img, &templates, threads, 1, iterations);
    }

    println!();

    // Test with OpenCV internal threads enabled
    for &threads in &[1, 4, 8, 16] {
        let label = format!("rayon={:>2}, cv=auto", threads);
        bench_config(&label, &edge_img, &templates, threads, 0, iterations);
    }

    println!();

    // Test sequential-only with varying CV threads
    for &cv_t in &[1, 2, 4, 8] {
        opencv::core::set_num_threads(cv_t).unwrap();
        let start = Instant::now();
        for _ in 0..iterations {
            match_templates_sequential(&edge_img, &templates);
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / iterations as f64;
        println!("sequential, cv={:<13} avg={:>8.0}us", cv_t, avg_us);
    }
}
