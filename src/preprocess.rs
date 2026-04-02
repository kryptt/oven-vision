use opencv::core::{Mat, Size};
use opencv::imgproc;
use opencv::prelude::*;

/// Preprocess a BGR color frame into an enhanced grayscale image.
///
/// Steps: BGR-to-gray, CLAHE contrast enhancement, median blur.
pub fn preprocess(frame: &Mat) -> Result<Mat, opencv::Error> {
    let mut gray = Mat::default();
    imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    let mut enhanced = Mat::default();
    let mut clahe = imgproc::create_clahe(2.0, Size::new(8, 8))?;
    clahe.apply(&gray, &mut enhanced)?;

    let mut blurred = Mat::default();
    imgproc::median_blur(&enhanced, &mut blurred, 5)?;

    Ok(blurred)
}
