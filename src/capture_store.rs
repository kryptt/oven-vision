use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use image::DynamicImage;
use tracing::{debug, info, warn};

use crate::annotate::encode_jpeg;
use crate::detect::DialReading;

/// Saves annotated frames to disk when dial confidence is below a threshold.
///
/// Rate-limits captures and rotates old files to prevent unbounded disk usage.
pub struct CaptureStore {
    dir: PathBuf,
    max_files: usize,
    min_interval: Duration,
    confidence_threshold: f64,
    last_capture: Option<Instant>,
}

impl CaptureStore {
    pub fn new(dir: PathBuf, max_files: usize, min_interval_secs: u64, threshold: f64) -> Self {
        Self {
            dir,
            max_files,
            min_interval: Duration::from_secs(min_interval_secs),
            confidence_threshold: threshold,
            last_capture: None,
        }
    }

    /// Conditionally save an annotated frame when any dial reading has low
    /// confidence and enough time has elapsed since the last capture.
    ///
    /// Returns the path to the saved file, or `None` if the capture was
    /// skipped (no low-confidence reading or rate-limited).
    pub fn maybe_capture(
        &mut self,
        annotated_frame: &DynamicImage,
        readings: &[DialReading],
    ) -> Result<Option<PathBuf>, io::Error> {
        // Check if any reading is below the confidence threshold
        let has_low_confidence = readings
            .iter()
            .any(|r| r.confidence < self.confidence_threshold);

        if !has_low_confidence {
            return Ok(None);
        }

        // Rate-limit captures
        if let Some(last) = self.last_capture {
            if last.elapsed() < self.min_interval {
                debug!("low-confidence frame skipped (rate-limited)");
                return Ok(None);
            }
        }

        // Ensure output directory exists
        fs::create_dir_all(&self.dir)?;

        // Generate timestamped filename
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let secs = now.as_secs();

        // Format as a rough ISO-ish timestamp using integer arithmetic
        // (avoids pulling in chrono just for this)
        let filename = format_timestamp_filename(secs);
        let path = self.dir.join(filename);

        let jpeg_bytes = encode_jpeg(annotated_frame, 85);
        fs::write(&path, &jpeg_bytes)?;

        info!(path = %path.display(), bytes = jpeg_bytes.len(), "saved low-confidence capture");

        self.last_capture = Some(Instant::now());

        // Rotate old files
        if let Err(err) = rotate_files(&self.dir, self.max_files) {
            warn!(%err, "failed to rotate capture files");
        }

        Ok(Some(path))
    }
}

/// Build a filename like `2026-04-02T12-30-45_low_confidence.jpg` from a Unix
/// timestamp. Uses simple arithmetic to avoid a datetime library dependency.
fn format_timestamp_filename(epoch_secs: u64) -> String {
    // Days from Unix epoch, accounting for leap years
    let secs_per_day: u64 = 86400;
    let mut remaining = epoch_secs;

    let hours = (remaining % secs_per_day) / 3600;
    let minutes = (remaining % 3600) / 60;
    let seconds = remaining % 60;

    remaining /= secs_per_day;

    // Calculate year/month/day from days since epoch (1970-01-01)
    let mut year: u64 = 1970;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        year += 1;
    }

    let days_in_months: [u64; 12] = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month: u64 = 1;
    for &days in &days_in_months {
        if remaining < days {
            break;
        }
        remaining -= days;
        month += 1;
    }
    let day = remaining + 1;

    format!(
        "{year:04}-{month:02}-{day:02}T{hours:02}-{minutes:02}-{seconds:02}_low_confidence.jpg"
    )
}

fn is_leap(year: u64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Remove the oldest `.jpg` files in `dir` until the count is at most
/// `max_files`. Files are sorted by name (oldest first, since names are
/// timestamped).
fn rotate_files(dir: &Path, max_files: usize) -> Result<(), io::Error> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("jpg") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if files.len() <= max_files {
        return Ok(());
    }

    // Sort by filename (oldest timestamps first)
    files.sort();

    let to_remove = files.len() - max_files;
    for path in files.iter().take(to_remove) {
        debug!(path = %path.display(), "removing old capture");
        fs::remove_file(path)?;
    }

    Ok(())
}
