use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::stage::PipelineState;

/// Schema version -- bump when PipelineState fields change.
const STATE_SCHEMA_VERSION: u32 = 1;

/// Compute a cache version from the ordered stage names and schema version.
pub fn compute_cache_version(stages: &[Box<dyn super::Stage>]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for s in stages {
        s.descriptor().name.hash(&mut hasher);
    }
    STATE_SCHEMA_VERSION.hash(&mut hasher);
    hasher.finish()
}

/// Wrapper that adds versioning around the serialized pipeline state.
#[derive(Debug, Serialize, Deserialize)]
pub struct PipelineCache {
    pub version: u64,
    /// Frame dimensions at calibration time — mismatch invalidates the cache.
    pub frame_width: u32,
    pub frame_height: u32,
    pub state: PipelineState,
}

#[derive(Debug)]
pub enum CacheError {
    Io(io::Error),
    Json(serde_json::Error),
    VersionMismatch {
        on_disk: u64,
        expected: u64,
    },
    FrameSizeMismatch {
        cached: (u32, u32),
        current: (u32, u32),
    },
    Incomplete,
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheError::Io(err) => write!(f, "cache I/O error: {err}"),
            CacheError::Json(err) => write!(f, "cache JSON error: {err}"),
            CacheError::VersionMismatch { on_disk, expected } => {
                write!(
                    f,
                    "cache version mismatch: on-disk={on_disk}, expected={expected}"
                )
            }
            CacheError::FrameSizeMismatch { cached, current } => {
                write!(
                    f,
                    "frame size mismatch: cached={}x{}, current={}x{}",
                    cached.0, cached.1, current.0, current.1
                )
            }
            CacheError::Incomplete => write!(f, "cached pipeline state is not fully validated"),
        }
    }
}

impl std::error::Error for CacheError {}

impl PipelineCache {
    /// Create a new cache entry from a validated pipeline state.
    pub fn new(state: PipelineState, frame_width: u32, frame_height: u32, version: u64) -> Self {
        Self {
            version,
            frame_width,
            frame_height,
            state,
        }
    }

    /// Save the cache to a JSON file.
    pub fn save(&self, path: &Path) -> Result<(), CacheError> {
        let json = serde_json::to_string_pretty(self).map_err(CacheError::Json)?;
        std::fs::write(path, json).map_err(CacheError::Io)
    }

    /// Load and validate a cache file. Returns `Err` if the file is missing,
    /// the version doesn't match, the frame dimensions changed, or the
    /// pipeline state isn't fully validated.
    pub fn load(
        path: &Path,
        current_width: u32,
        current_height: u32,
        expected_version: u64,
    ) -> Result<PipelineState, CacheError> {
        let json = std::fs::read_to_string(path).map_err(CacheError::Io)?;
        let cache: PipelineCache = serde_json::from_str(&json).map_err(CacheError::Json)?;

        if cache.version != expected_version {
            return Err(CacheError::VersionMismatch {
                on_disk: cache.version,
                expected: expected_version,
            });
        }

        if cache.frame_width != current_width || cache.frame_height != current_height {
            return Err(CacheError::FrameSizeMismatch {
                cached: (cache.frame_width, cache.frame_height),
                current: (current_width, current_height),
            });
        }

        if !cache.state.validated {
            return Err(CacheError::Incomplete);
        }

        Ok(cache.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::stage::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    const TEST_VERSION: u64 = 42;

    fn sample_state() -> PipelineState {
        PipelineState {
            crop: Some(CropRegion {
                x: 100,
                y: 200,
                width: 800,
                height: 300,
            }),
            lines: Some(LinePair {
                top: Line {
                    x1: 0.0,
                    y1: 10.0,
                    x2: 800.0,
                    y2: 5.0,
                },
                bottom: Line {
                    x1: 0.0,
                    y1: 50.0,
                    x2: 800.0,
                    y2: 45.0,
                },
            }),
            verticals: Some(VerticalPair {
                left: Line {
                    x1: 10.0,
                    y1: 0.0,
                    x2: 10.0,
                    y2: 300.0,
                },
                right: Line {
                    x1: 790.0,
                    y1: 0.0,
                    x2: 790.0,
                    y2: 300.0,
                },
            }),
            perspective: Some(PerspectiveCorrection {
                matrix: TransformMatrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                output_width: 800,
                output_height: 200,
            }),
            knob_search: Some(KnobSearchArea {
                y_min: 10.0,
                y_max: 190.0,
                x_min: 80.0,
                clock_center_x: 50.0,
                clock_center_y: 100.0,
                clock_radius: 25.0,
            }),
            features: Some(DetectedFeatures {
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
            }),
            validated: true,
        }
    }

    #[test]
    fn round_trip_save_load() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_owned();

        let state = sample_state();
        let cache = PipelineCache::new(state.clone(), 2560, 1440, TEST_VERSION);
        cache.save(&path).unwrap();

        let loaded = PipelineCache::load(&path, 2560, 1440, TEST_VERSION).unwrap();
        assert_eq!(loaded, state);
    }

    #[test]
    fn version_mismatch_rejects() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_owned();

        let state = sample_state();
        // Write with a wrong version
        let mut cache = PipelineCache::new(state, 2560, 1440, TEST_VERSION);
        cache.version = 999;
        cache.save(&path).unwrap();

        let err = PipelineCache::load(&path, 2560, 1440, TEST_VERSION).unwrap_err();
        assert!(matches!(err, CacheError::VersionMismatch { .. }));
    }

    #[test]
    fn frame_size_mismatch_rejects() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_owned();

        let state = sample_state();
        let cache = PipelineCache::new(state, 2560, 1440, TEST_VERSION);
        cache.save(&path).unwrap();

        // Load with different dimensions
        let err = PipelineCache::load(&path, 1920, 1080, TEST_VERSION).unwrap_err();
        assert!(matches!(err, CacheError::FrameSizeMismatch { .. }));
    }

    #[test]
    fn incomplete_state_rejects() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_owned();

        let mut state = sample_state();
        state.validated = false;
        let cache = PipelineCache::new(state, 2560, 1440, TEST_VERSION);
        cache.save(&path).unwrap();

        let err = PipelineCache::load(&path, 2560, 1440, TEST_VERSION).unwrap_err();
        assert!(matches!(err, CacheError::Incomplete));
    }

    #[test]
    fn missing_file_returns_error() {
        let err = PipelineCache::load(
            Path::new("/nonexistent/cache.json"),
            2560,
            1440,
            TEST_VERSION,
        )
        .unwrap_err();
        assert!(matches!(err, CacheError::Io(_)));
    }

    #[test]
    fn corrupt_json_returns_error() {
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{{not valid json").unwrap();
        let err = PipelineCache::load(tmp.path(), 2560, 1440, TEST_VERSION).unwrap_err();
        assert!(matches!(err, CacheError::Json(_)));
    }
}
