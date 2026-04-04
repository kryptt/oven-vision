//! Factories for constructing valid `PipelineState` snapshots that allow
//! individual stages to run in isolation.
//!
//! Each function returns the *minimum* state a stage needs as input.
//! Real test images should be placed in `tests/fixtures/`.

use opencv::core::{Mat, Scalar};
use opencv::imgproc;
use opencv::prelude::*;

use super::stage::{
    CircleFeature, CropRegion, DetectedFeatures, KnobSearchArea, Line, LinePair,
    PerspectiveCorrection, PipelineState, TransformMatrix, VerticalPair,
};

/// A 2560x1440 BGR frame filled with a solid colour.
/// Good enough for stages that only read dimensions, not pixel content.
pub fn synthetic_frame(width: i32, height: i32) -> Result<Mat, opencv::Error> {
    let mat = Mat::new_rows_cols_with_default(height, width, opencv::core::CV_8UC3, Scalar::all(128.0))?;
    Ok(mat)
}

/// A synthetic image with horizontal lines drawn at the expected positions.
/// Useful for S2 (FindLines) and S5 (WarpCheck).
pub fn synthetic_with_lines(width: i32, height: i32) -> Result<Mat, opencv::Error> {
    let mut mat = Mat::new_rows_cols_with_default(height, width, opencv::core::CV_8UC3, Scalar::all(40.0))?;
    let white = Scalar::new(220.0, 220.0, 220.0, 0.0);

    // Two horizontal chrome-like lines
    let y_top = height / 3;
    let y_bot = y_top + height / 6;
    imgproc::line(
        &mut mat,
        opencv::core::Point::new(0, y_top),
        opencv::core::Point::new(width, y_top),
        white,
        3,
        imgproc::LINE_8,
        0,
    )?;
    imgproc::line(
        &mut mat,
        opencv::core::Point::new(0, y_bot),
        opencv::core::Point::new(width, y_bot),
        white,
        3,
        imgproc::LINE_8,
        0,
    )?;
    Ok(mat)
}

/// Load a real image from disk. Returns an error if the path is invalid.
pub fn load_image(path: &std::path::Path) -> Result<Mat, opencv::Error> {
    let img = opencv::imgcodecs::imread(
        &path.to_string_lossy(),
        opencv::imgcodecs::IMREAD_COLOR,
    )?;
    if img.empty() {
        return Err(opencv::Error::new(
            opencv::core::StsError,
            format!("failed to load image: {}", path.display()),
        ));
    }
    Ok(img)
}

// ---------------------------------------------------------------------------
// Per-stage state factories
// ---------------------------------------------------------------------------

/// State required to run S1 (FindStove): none -- it reads only the raw frame.
pub fn state_for_find_stove() -> PipelineState {
    PipelineState::default()
}

/// State required to run S2 (FindLines): crop must be set.
pub fn state_for_find_lines() -> PipelineState {
    PipelineState {
        crop: Some(CropRegion {
            x: 100,
            y: 200,
            width: 800,
            height: 300,
        }),
        ..PipelineState::default()
    }
}

/// State required to run S3 (FindVerticals): crop + lines.
pub fn state_for_find_verticals() -> PipelineState {
    PipelineState {
        crop: Some(CropRegion {
            x: 100,
            y: 200,
            width: 800,
            height: 300,
        }),
        lines: Some(LinePair {
            top: Line { x1: 0.0, y1: 50.0, x2: 800.0, y2: 48.0 },
            bottom: Line { x1: 0.0, y1: 100.0, x2: 800.0, y2: 98.0 },
            avg_theta: std::f64::consts::FRAC_PI_2,
        }),
        ..PipelineState::default()
    }
}

/// State required to run S4 (Perspective): crop + lines + verticals.
pub fn state_for_perspective() -> PipelineState {
    PipelineState {
        verticals: Some(VerticalPair {
            left: Line { x1: 10.0, y1: 0.0, x2: 10.0, y2: 300.0 },
            right: Line { x1: 790.0, y1: 0.0, x2: 790.0, y2: 300.0 },
        }),
        ..state_for_find_verticals()
    }
}

/// State required to run S5 (WarpCheck): perspective must be set.
pub fn state_for_warp_check() -> PipelineState {
    PipelineState {
        perspective: Some(PerspectiveCorrection {
            matrix: TransformMatrix([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
            output_width: 800,
            output_height: 200,
        }),
        ..state_for_perspective()
    }
}

/// State required to run S6 (ExtractBand): perspective + lines.
pub fn state_for_extract_band() -> PipelineState {
    state_for_warp_check()
}

/// State required to run S7 (FindClock): knob_search must be set.
pub fn state_for_find_clock() -> PipelineState {
    PipelineState {
        knob_search: Some(KnobSearchArea {
            y_min: 10.0,
            y_max: 190.0,
            x_min: 0.0,
            clock_center_x: 0.0,
            clock_center_y: 0.0,
            clock_radius: 0.0,
            corner_x: None,
            corner_y: None,
        }),
        ..state_for_extract_band()
    }
}

/// State required to run S8 (FindFeatures): knob_search with x_min set.
pub fn state_for_find_features() -> PipelineState {
    PipelineState {
        knob_search: Some(KnobSearchArea {
            y_min: 10.0,
            y_max: 190.0,
            x_min: 80.0,
            clock_center_x: 50.0,
            clock_center_y: 100.0,
            clock_radius: 25.0,
            corner_x: None,
            corner_y: None,
        }),
        ..state_for_find_clock()
    }
}

/// State required to run S9 (SanityCheck): features must be present.
pub fn state_for_sanity_check() -> PipelineState {
    PipelineState {
        features: Some(mock_features()),
        ..state_for_find_features()
    }
}

/// State required to run S10 (FindCorner): features must be present.
pub fn state_for_find_corner() -> PipelineState {
    state_for_sanity_check()
}

/// State required to run S11 (RefineWarp): features + corner.
pub fn state_for_refine_warp() -> PipelineState {
    let mut s = state_for_find_corner();
    if let Some(ks) = s.knob_search.as_mut() {
        ks.corner_x = Some(700.0);
        ks.corner_y = Some(10.0);
    }
    s
}

/// State required to run S12 (FinalDetect): same as S8 input but after refinement.
pub fn state_for_final_detect() -> PipelineState {
    let mut s = state_for_refine_warp();
    // Clear features so S12 runs fresh detection
    s.features = None;
    s
}

/// State required to run S13 (FinalCheck): features from S12 must be present.
pub fn state_for_final_check() -> PipelineState {
    PipelineState {
        features: Some(mock_features()),
        ..state_for_refine_warp()
    }
}

/// Construct a plausible `DetectedFeatures` with 10 evenly-spaced knobs.
fn mock_features() -> DetectedFeatures {
    DetectedFeatures {
        clock: CircleFeature {
            center_x: 50.0,
            center_y: 100.0,
            radius: 25.0,
        },
        knobs: (0..10)
            .map(|i| CircleFeature {
                center_x: 120.0 + i as f64 * 65.0,
                center_y: 100.0,
                radius: 12.0,
            })
            .collect(),
        off_angles: vec![90.0; 10],
    }
}

/// Map a stage name to the correct state factory.
pub fn state_for_stage(name: &str) -> Option<PipelineState> {
    match name {
        "FindStove" => Some(state_for_find_stove()),
        "FindLines" => Some(state_for_find_lines()),
        "FindVerticals" => Some(state_for_find_verticals()),
        "Perspective" => Some(state_for_perspective()),
        "WarpCheck" => Some(state_for_warp_check()),
        "ExtractBand" => Some(state_for_extract_band()),
        "FindClock" => Some(state_for_find_clock()),
        "FindFeatures" => Some(state_for_find_features()),
        "SanityCheck" => Some(state_for_sanity_check()),
        "FindCorner" => Some(state_for_find_corner()),
        "RefineWarp" => Some(state_for_refine_warp()),
        "FinalDetect" => Some(state_for_final_detect()),
        "FinalCheck" => Some(state_for_final_check()),
        _ => None,
    }
}
