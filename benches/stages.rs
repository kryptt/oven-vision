//! Criterion benchmarks for individual pipeline stages.
//!
//! Run with: `cargo bench --bench stages`
//!
//! Each stage benchmark creates the required input state and a synthetic image,
//! then measures the `stage.run()` call. For meaningful numbers, place a real
//! oven image at `tests/fixtures/oven.jpg`; otherwise a synthetic grey frame is
//! used (useful for profiling overhead but not representative of real CV work).

use std::path::Path;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use opencv::core::Mat;
use opencv::prelude::*;

use oven_vision::pipeline::Stage;
use oven_vision::pipeline::extract_band::ExtractBand;
use oven_vision::pipeline::final_check::FinalCheck;
use oven_vision::pipeline::final_detect::FinalDetect;
use oven_vision::pipeline::find_clock::FindClock;
use oven_vision::pipeline::find_corner::FindCorner;
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

/// Try to load a real fixture image; fall back to a synthetic frame.
fn load_test_image() -> Mat {
    let fixture = Path::new("tests/fixtures/oven.jpg");
    if fixture.exists() {
        testdata::load_image(fixture).expect("failed to load fixture image")
    } else {
        testdata::synthetic_frame(2560, 1440).expect("failed to create synthetic frame")
    }
}

/// Benchmark a single stage given its boxed trait object, input state, and image.
fn bench_stage(
    c: &mut Criterion,
    stage: &dyn oven_vision::pipeline::Stage,
    base_state: &PipelineState,
    image: &Mat,
) {
    let raw = image.try_clone().unwrap();
    c.bench_function(stage.descriptor().label, |b| {
        b.iter(|| {
            let mut state = base_state.clone();
            let mut dst = Mat::default();
            let _ = stage.run(&mut state, image, &mut dst, &raw, 0);
        });
    });
}

fn bench_all_stages(c: &mut Criterion) {
    let image = load_test_image();
    let templates = Arc::new(Templates::load_from(Path::new("templates")));
    let find_features = Arc::new(FindFeatures::new(templates.clone()));

    let stages: Vec<(Box<dyn oven_vision::pipeline::Stage>, PipelineState)> = vec![
        (Box::new(FindStove::new()), testdata::state_for_find_stove()),
        (Box::new(FindLines::new()), testdata::state_for_find_lines()),
        (
            Box::new(FindVerticals::new()),
            testdata::state_for_find_verticals(),
        ),
        (
            Box::new(Perspective::new()),
            testdata::state_for_perspective(),
        ),
        (Box::new(WarpCheck::new()), testdata::state_for_warp_check()),
        (
            Box::new(ExtractBand::new()),
            testdata::state_for_extract_band(),
        ),
        (
            Box::new(FindClock::new(templates)),
            testdata::state_for_find_clock(),
        ),
        (
            Box::new(find_features.clone()),
            testdata::state_for_find_features(),
        ),
        (
            Box::new(SanityCheck::new()),
            testdata::state_for_sanity_check(),
        ),
        (
            Box::new(FindCorner::new()),
            testdata::state_for_find_corner(),
        ),
        (
            Box::new(RefineWarp::new()),
            testdata::state_for_refine_warp(),
        ),
        (
            Box::new(FinalDetect::new(find_features)),
            testdata::state_for_final_detect(),
        ),
        (
            Box::new(FinalCheck::new()),
            testdata::state_for_final_check(),
        ),
    ];

    for (stage, state) in &stages {
        bench_stage(c, stage.as_ref(), state, &image);
    }
}

criterion_group! {
    name = pipeline_stages;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(std::time::Duration::from_secs(5));
    targets = bench_all_stages
}
criterion_main!(pipeline_stages);
