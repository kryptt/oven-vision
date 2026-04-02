use image::DynamicImage;
use std::fmt;

#[derive(Debug)]
pub enum CaptureError {
    Http(reqwest::Error),
    Decode(image::ImageError),
}

impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(err) => write!(f, "HTTP request failed: {err}"),
            Self::Decode(err) => write!(f, "image decode failed: {err}"),
        }
    }
}

impl std::error::Error for CaptureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Http(err) => Some(err),
            Self::Decode(err) => Some(err),
        }
    }
}

pub async fn fetch_frame(
    client: &reqwest::Client,
    url: &str,
) -> Result<DynamicImage, CaptureError> {
    let bytes = client
        .get(url)
        .send()
        .await
        .map_err(CaptureError::Http)?
        .bytes()
        .await
        .map_err(CaptureError::Http)?;

    image::load_from_memory(&bytes).map_err(CaptureError::Decode)
}
