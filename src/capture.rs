use opencv::core::{Mat, Vector};
use opencv::imgcodecs;
use opencv::prelude::*;
use std::fmt;

#[derive(Debug)]
pub enum CaptureError {
    Http(reqwest::Error),
    Decode(opencv::Error),
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
) -> Result<Mat, CaptureError> {
    let bytes = client
        .get(url)
        .send()
        .await
        .map_err(CaptureError::Http)?
        .bytes()
        .await
        .map_err(CaptureError::Http)?;

    let buf = Vector::from_slice(&bytes);
    imgcodecs::imdecode(&buf, imgcodecs::IMREAD_COLOR).map_err(CaptureError::Decode)
}
