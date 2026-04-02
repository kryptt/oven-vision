use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LedState {
    Off,
    On,
    Heating,
}

impl fmt::Display for LedState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Off => write!(f, "off"),
            Self::On => write!(f, "on"),
            Self::Heating => write!(f, "heating"),
        }
    }
}

pub struct LedReading {
    pub label: String,
    pub state: LedState,
}
