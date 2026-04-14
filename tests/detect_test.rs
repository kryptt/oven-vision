use oven_vision::detect::{DialState, HeatLevel, classify_dial};

#[test]
fn classify_at_off_position() {
    let state = classify_dial(90.0, 90.0, 15.0);
    assert_eq!(state, DialState::Off);
}

#[test]
fn classify_within_tolerance_is_off() {
    let state = classify_dial(100.0, 90.0, 15.0);
    assert_eq!(state, DialState::Off);
}

#[test]
fn classify_at_tolerance_boundary_is_off() {
    let state = classify_dial(105.0, 90.0, 15.0);
    assert_eq!(state, DialState::Off);
}

#[test]
fn classify_90_deg_from_off_is_medium() {
    let state = classify_dial(180.0, 90.0, 15.0);
    assert_eq!(state, DialState::On(HeatLevel::Medium));
}

#[test]
fn classify_at_max_rotation_is_max() {
    let state = classify_dial(180.0, 0.0, 0.0);
    assert_eq!(state, DialState::On(HeatLevel::Max));
}

#[test]
fn classify_high_rotation() {
    let state = classify_dial(150.0, 0.0, 15.0);
    assert_eq!(state, DialState::On(HeatLevel::Max));
}

#[test]
fn classify_small_rotation_is_low() {
    let state = classify_dial(120.0, 90.0, 15.0);
    assert_eq!(state, DialState::On(HeatLevel::Low));
}

#[test]
fn classify_wraparound_off_at_350_detected_at_5() {
    let state = classify_dial(5.0, 350.0, 15.0);
    assert_eq!(state, DialState::Off);
}

#[test]
fn classify_wraparound_off_at_350_detected_at_10() {
    let state = classify_dial(10.0, 350.0, 15.0);
    assert_eq!(state, DialState::On(HeatLevel::Low));
}

#[test]
fn wide_tolerance_covers_noisy_chrome_dials() {
    // With 40° tolerance, a 35° deviation from off should still be Off
    let state = classify_dial(125.0, 90.0, 40.0);
    assert_eq!(state, DialState::Off);
}

#[test]
fn wide_tolerance_still_detects_clearly_on() {
    // 90° from off with 40° tolerance: effective=50, range=140, frac=0.357 → Medium
    let state = classify_dial(180.0, 90.0, 40.0);
    assert_eq!(state, DialState::On(HeatLevel::Medium));
}
