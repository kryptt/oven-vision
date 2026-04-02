use image::{DynamicImage, GrayImage};
use imageproc::contrast::equalize_histogram;
use imageproc::filter::median_filter;

pub fn preprocess(image: &DynamicImage) -> GrayImage {
    let gray = image.to_luma8();
    let equalized = equalize_histogram(&gray);
    median_filter(&equalized, 2, 2)
}
