//! CLI tool for running, benchmarking, and profiling individual pipeline stages.
//!
//! ```text
//! oven-vision-stages run <stage> --input <image.jpg> [--templates <dir>] [--output <dir>]
//! oven-vision-stages bench <stage> --input <image.jpg> [--iterations 1000]
//! oven-vision-stages profile <stage> --input <image.jpg> [--seconds 10] [--output flamegraph.svg]
//! oven-vision-stages list
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, Subcommand};
use opencv::core::Mat;
use opencv::prelude::*;

use oven_vision::pipeline::extract_band::ExtractBand;
use oven_vision::pipeline::final_check::FinalCheck;
use oven_vision::pipeline::final_detect::FinalDetect;
use oven_vision::pipeline::find_clock::FindClock;
use oven_vision::pipeline::find_features::FindFeatures;
use oven_vision::pipeline::find_lines::FindLines;
use oven_vision::pipeline::find_stove::FindStove;
use oven_vision::pipeline::find_verticals::FindVerticals;
use oven_vision::pipeline::perspective::Perspective;
use oven_vision::pipeline::refine_warp::RefineWarp;
use oven_vision::pipeline::sanity::SanityCheck;
use oven_vision::pipeline::stage::PipelineState;
use oven_vision::pipeline::testdata;
use oven_vision::pipeline::util::Templates;
use oven_vision::pipeline::warp_check::WarpCheck;
use oven_vision::pipeline::{Pipeline, PipelineConfig, Stage};

#[derive(Parser)]
#[command(name = "oven-vision-stages", about = "Run, benchmark, or profile individual pipeline stages")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a single stage and write output images.
    Run {
        /// Stage name (e.g. FindStove, FindLines, ..., FinalCheck) or "all".
        stage: String,
        /// Input image path.
        #[arg(short, long)]
        input: PathBuf,
        /// Path to templates directory (default: /templates).
        #[arg(short, long, default_value = "/templates")]
        templates: PathBuf,
        /// Output directory for result images and state JSON.
        #[arg(short, long, default_value = "./stage_output")]
        output: PathBuf,
        /// Pipeline state JSON to load instead of using defaults.
        #[arg(long)]
        state: Option<PathBuf>,
    },
    /// Benchmark a stage with statistical timing.
    Bench {
        /// Stage name or "all".
        stage: String,
        /// Input image path.
        #[arg(short, long)]
        input: PathBuf,
        /// Path to templates directory (default: /templates).
        #[arg(short, long, default_value = "/templates")]
        templates: PathBuf,
        /// Number of iterations.
        #[arg(long, default_value_t = 1000)]
        iterations: u32,
        /// Pipeline state JSON to load instead of using defaults.
        #[arg(long)]
        state: Option<PathBuf>,
    },
    /// Profile a stage and emit a flamegraph SVG.
    Profile {
        /// Stage name.
        stage: String,
        /// Input image path.
        #[arg(short, long)]
        input: PathBuf,
        /// Path to templates directory (default: /templates).
        #[arg(short, long, default_value = "/templates")]
        templates: PathBuf,
        /// Duration in seconds to run the profiling loop.
        #[arg(long, default_value_t = 10)]
        seconds: u64,
        /// Output path for the flamegraph SVG.
        #[arg(short, long, default_value = "flamegraph.svg")]
        output: PathBuf,
        /// Pipeline state JSON to load instead of using defaults.
        #[arg(long)]
        state: Option<PathBuf>,
    },
    /// List all pipeline stages.
    List,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("oven_vision=info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Run {
            stage,
            input,
            templates,
            output,
            state,
        } => cmd_run(&stage, &input, &templates, &output, state.as_deref()),
        Command::Bench {
            stage,
            input,
            templates,
            iterations,
            state,
        } => cmd_bench(&stage, &input, &templates, iterations, state.as_deref()),
        Command::Profile {
            stage,
            input,
            templates,
            seconds,
            output,
            state,
        } => cmd_profile(&stage, &input, &templates, seconds, &output, state.as_deref()),
        Command::List => cmd_list(),
    }
}

// ---------------------------------------------------------------------------
// Subcommand implementations
// ---------------------------------------------------------------------------

fn cmd_run(
    stage_name: &str,
    input: &std::path::Path,
    templates_dir: &std::path::Path,
    output_dir: &std::path::Path,
    state_path: Option<&std::path::Path>,
) {
    std::fs::create_dir_all(output_dir).expect("failed to create output directory");

    let image = testdata::load_image(input).expect("failed to load input image");
    let templates = Arc::new(Templates::load_from(templates_dir));

    if stage_name == "all" {
        run_full_pipeline(&image, templates, output_dir, state_path);
        return;
    }

    let stages = build_all_stages(templates);
    let (stage, _idx) = find_stage(&stages, stage_name);

    let mut state = load_or_default_state(stage_name, state_path);
    let raw = image.try_clone().expect("clone raw");
    let max_retries = stage.max_retries();

    let start = Instant::now();
    let mut first_success: Option<u32> = None;

    // Run ALL iterations so every parameter variation is visible.
    for iteration in 0..=max_retries {
        // Reset state each iteration so earlier mutations don't accumulate.
        let mut iter_state = load_or_default_state(stage_name, state_path);
        let mut dst = Mat::default();
        let (outcome, img_out) = stage
            .run(&mut iter_state, &image, &mut dst, &raw, iteration)
            .expect("stage run failed");

        // Save output image for this iteration
        let working = match img_out {
            oven_vision::pipeline::ImageOutput::Transformed => &dst,
            oven_vision::pipeline::ImageOutput::Passthrough => &image,
        };
        let img_path = output_dir.join(format!("{stage_name}_{iteration:03}_output.jpg"));
        opencv::imgcodecs::imwrite_def(&img_path.to_string_lossy(), working)
            .expect("failed to write output image");

        // Save debug image for this iteration
        if let Ok(Some((label, jpeg))) = stage.debug_image(&iter_state, working, &raw) {
            let debug_path = output_dir.join(format!("{stage_name}_{iteration:03}_debug.jpg"));
            std::fs::write(&debug_path, &jpeg).expect("failed to write debug image");
            println!(
                "  iter {iteration}: {} ({})",
                format_outcome(&outcome),
                label
            );
        } else {
            println!("  iter {iteration}: {}", format_outcome(&outcome));
        }

        if first_success.is_none() {
            if matches!(outcome, oven_vision::pipeline::stage::StageOutcome::Success) {
                first_success = Some(iteration);
                // Write the first-success state as the canonical state for chaining
                state = iter_state;
            }
        }
    }
    let elapsed = start.elapsed();

    let total_iters = max_retries + 1;
    println!("stage: {}", stage.descriptor().label);
    println!("elapsed: {elapsed:?}");
    match first_success {
        Some(iter) => println!("first success: iter {iter} (of {total_iters} total)"),
        None => println!("no successful iteration (ran {total_iters} total)"),
    }

    // Write canonical output/debug from the first successful iteration (or iter 0 fallback)
    let canon_iter = first_success.unwrap_or(0);
    let canon_out = output_dir.join(format!("{stage_name}_{canon_iter:03}_output.jpg"));
    let img_path = output_dir.join(format!("{stage_name}_output.jpg"));
    std::fs::copy(&canon_out, &img_path).expect("failed to copy canonical output");
    println!("output image: {}", img_path.display());

    let canon_dbg = output_dir.join(format!("{stage_name}_{canon_iter:03}_debug.jpg"));
    if canon_dbg.exists() {
        let debug_path = output_dir.join(format!("{stage_name}_debug.jpg"));
        std::fs::copy(&canon_dbg, &debug_path).expect("failed to copy canonical debug");
        println!("debug image: {}", debug_path.display());
    }

    // Write state JSON
    let state_path = output_dir.join(format!("{stage_name}_state.json"));
    let json = serde_json::to_string_pretty(&state).expect("failed to serialize state");
    std::fs::write(&state_path, json).expect("failed to write state");
    println!("state: {}", state_path.display());
}

fn cmd_bench(
    stage_name: &str,
    input: &std::path::Path,
    templates_dir: &std::path::Path,
    iterations: u32,
    state_path: Option<&std::path::Path>,
) {
    let image = testdata::load_image(input).expect("failed to load input image");
    let templates = Arc::new(Templates::load_from(templates_dir));
    let stages = build_all_stages(templates);

    let stage_names: Vec<&str> = if stage_name == "all" {
        stages.iter().map(|s| s.descriptor().name).collect()
    } else {
        vec![stage_name]
    };

    for name in &stage_names {
        let (stage, _idx) = find_stage(&stages, name);
        let base_state = load_or_default_state(name, state_path);
        let raw = image.try_clone().expect("clone raw");

        let mut timings = Vec::with_capacity(iterations as usize);

        // Warmup: 10% of iterations, minimum 5
        let warmup = (iterations / 10).max(5);
        for _ in 0..warmup {
            let mut state = base_state.clone();
            let mut dst = Mat::default();
            let _ = stage.run(&mut state, &image, &mut dst, &raw, 0);
        }

        // Measured runs
        for _ in 0..iterations {
            let mut state = base_state.clone();
            let mut dst = Mat::default();
            let start = Instant::now();
            let _ = stage.run(&mut state, &image, &mut dst, &raw, 0);
            timings.push(start.elapsed());
        }

        timings.sort();
        let total: std::time::Duration = timings.iter().sum();
        let mean = total / iterations;
        let p50 = timings[timings.len() / 2];
        let p95 = timings[(timings.len() as f64 * 0.95) as usize];
        let p99 = timings[(timings.len() as f64 * 0.99) as usize];
        let min = timings[0];
        let max = timings[timings.len() - 1];

        println!("--- {} ({} iterations) ---", stage.descriptor().label, iterations);
        println!("  mean:  {:>10.3?}", mean);
        println!("  p50:   {:>10.3?}", p50);
        println!("  p95:   {:>10.3?}", p95);
        println!("  p99:   {:>10.3?}", p99);
        println!("  min:   {:>10.3?}", min);
        println!("  max:   {:>10.3?}", max);
        println!(
            "  throughput: {:.1} ops/sec",
            iterations as f64 / total.as_secs_f64()
        );
        println!();
    }
}

fn cmd_profile(
    stage_name: &str,
    input: &std::path::Path,
    templates_dir: &std::path::Path,
    seconds: u64,
    output_path: &std::path::Path,
    state_path: Option<&std::path::Path>,
) {
    let image = testdata::load_image(input).expect("failed to load input image");
    let templates = Arc::new(Templates::load_from(templates_dir));
    let stages = build_all_stages(templates);
    let (stage, _idx) = find_stage(&stages, stage_name);
    let base_state = load_or_default_state(stage_name, state_path);
    let raw = image.try_clone().expect("clone raw");

    let duration = std::time::Duration::from_secs(seconds);

    println!(
        "profiling {} for {} seconds...",
        stage.descriptor().label,
        seconds
    );

    // Hot loop with timing. For flamegraph output, use `cargo bench` with
    // pprof integration (see benches/stages.rs).
    let start = Instant::now();
    let mut count: u64 = 0;
    while start.elapsed() < duration {
        let mut state = base_state.clone();
        let mut dst = Mat::default();
        let _ = stage.run(&mut state, &image, &mut dst, &raw, 0);
        count += 1;
    }
    let elapsed = start.elapsed();

    println!("completed {count} iterations in {elapsed:.2?}");
    println!(
        "average: {:.3?} per iteration",
        elapsed / count.max(1) as u32
    );
    println!(
        "throughput: {:.1} ops/sec",
        count as f64 / elapsed.as_secs_f64()
    );
    println!(
        "note: for flamegraph output, use `cargo bench` with pprof integration (see benches/stages.rs)"
    );

    // Write a simple JSON report
    let report = serde_json::json!({
        "stage": stage_name,
        "seconds": seconds,
        "iterations": count,
        "elapsed_ms": elapsed.as_millis(),
        "avg_us": elapsed.as_micros() as f64 / count.max(1) as f64,
        "ops_per_sec": count as f64 / elapsed.as_secs_f64(),
    });
    let report_path = output_path.with_extension("json");
    std::fs::write(&report_path, serde_json::to_string_pretty(&report).unwrap())
        .expect("failed to write profile report");
    println!("report: {}", report_path.display());
}

fn cmd_list() {
    let templates = Arc::new(Templates::load_from(std::path::Path::new("/templates")));
    let stages = build_all_stages(templates);
    println!("{:<4} {:<20} {:<20} {}", "#", "Name", "Label", "Fallback");
    println!("{}", "-".repeat(70));
    for (i, stage) in stages.iter().enumerate() {
        let desc = stage.descriptor();
        println!(
            "{:<4} {:<20} {:<20} {}",
            i + 1,
            desc.name,
            desc.label,
            desc.fallback.unwrap_or("(none)")
        );
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build all 12 stages in pipeline order with the given templates.
fn build_all_stages(templates: Arc<Templates>) -> Vec<Box<dyn Stage>> {
    let find_features = Arc::new(FindFeatures::new(templates.clone()));
    vec![
        Box::new(FindStove::new()),
        Box::new(FindLines::new()),
        Box::new(FindVerticals::new()),
        Box::new(Perspective::new()),
        Box::new(WarpCheck::new()),
        Box::new(ExtractBand::new()),
        Box::new(FindClock::new(templates)),
        Box::new(find_features.clone()),
        Box::new(SanityCheck::new()),
        Box::new(RefineWarp::new()),
        Box::new(FinalDetect::new(find_features)),
        Box::new(FinalCheck::new()),
    ]
}

/// Find a stage by name in the stage list. Panics with a helpful message if not found.
fn find_stage<'a>(stages: &'a [Box<dyn Stage>], name: &str) -> (&'a dyn Stage, usize) {
    for (i, s) in stages.iter().enumerate() {
        if s.descriptor().name.eq_ignore_ascii_case(name) {
            return (s.as_ref(), i);
        }
    }
    eprintln!("unknown stage: {name}");
    eprintln!("available stages:");
    for s in stages {
        eprintln!("  {}", s.descriptor().name);
    }
    std::process::exit(1);
}

/// Load pipeline state from a JSON file, or use the default factory for the stage.
fn load_or_default_state(stage_name: &str, state_path: Option<&std::path::Path>) -> PipelineState {
    if let Some(path) = state_path {
        let data = std::fs::read_to_string(path).expect("failed to read state JSON");
        serde_json::from_str(&data).expect("failed to parse state JSON")
    } else {
        testdata::state_for_stage(stage_name).unwrap_or_else(|| {
            eprintln!("no default state factory for stage: {stage_name}");
            std::process::exit(1);
        })
    }
}

/// Format a stage outcome for display.
fn format_outcome(outcome: &oven_vision::pipeline::stage::StageOutcome) -> String {
    match outcome {
        oven_vision::pipeline::stage::StageOutcome::Success => "Success".to_owned(),
        oven_vision::pipeline::stage::StageOutcome::Retry(reason) => format!("Retry: {reason}"),
        oven_vision::pipeline::stage::StageOutcome::Exhausted(reason) => {
            format!("Exhausted: {reason}")
        }
    }
}

/// Run the full pipeline on an image and write all debug images.
fn run_full_pipeline(
    image: &Mat,
    templates: Arc<Templates>,
    output_dir: &std::path::Path,
    state_path: Option<&std::path::Path>,
) {
    let find_features = Arc::new(FindFeatures::new(templates.clone()));
    let stages: Vec<Box<dyn Stage>> = vec![
        Box::new(FindStove::new()),
        Box::new(FindLines::new()),
        Box::new(FindVerticals::new()),
        Box::new(Perspective::new()),
        Box::new(WarpCheck::new()),
        Box::new(ExtractBand::new()),
        Box::new(FindClock::new(templates)),
        Box::new(find_features.clone()),
        Box::new(SanityCheck::new()),
        Box::new(RefineWarp::new()),
        Box::new(FinalDetect::new(find_features)),
        Box::new(FinalCheck::new()),
    ];

    let mut pipeline = Pipeline::new(
        stages,
        PipelineConfig {
            cache_path: output_dir.join("pipeline_cache.json"),
            max_frame_attempts: 1,
        },
    );

    if let Some(path) = state_path {
        let data = std::fs::read_to_string(path).expect("failed to read state JSON");
        pipeline.state = serde_json::from_str(&data).expect("failed to parse state JSON");
    }

    let start = Instant::now();
    match pipeline.calibrate(image) {
        Ok(()) => {
            let elapsed = start.elapsed();
            println!("full pipeline completed in {elapsed:?}");
        }
        Err(err) => {
            let elapsed = start.elapsed();
            println!("pipeline failed after {elapsed:?}: {err}");
        }
    }

    // Write debug images
    for (idx, (label, jpeg)) in pipeline.debug_images().iter().enumerate() {
        let filename = format!("{:03}_{}.jpg", idx, label.replace(':', "_"));
        let path = output_dir.join(&filename);
        std::fs::write(&path, jpeg).expect("failed to write debug image");
        println!("  {}", path.display());
    }

    // Write final state
    let state_out = output_dir.join("pipeline_state.json");
    let json = serde_json::to_string_pretty(pipeline.state()).expect("serialize state");
    std::fs::write(&state_out, json).expect("failed to write state");
    println!("state: {}", state_out.display());
}
