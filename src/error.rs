//! Error types for landfold.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum LandfoldError {
    #[error("invalid transfer-function parameters: {0}")]
    TransferParams(&'static str),
    #[error("unsupported transfer function")]
    UnsupportedTransfer,
    #[error("metric size mismatch: left={left} right={right}")]
    MetricSize { left: usize, right: usize },
    #[error("period array has length {got}, expected {expected}")]
    PeriodSize { got: usize, expected: usize },
    #[error("empty point set")]
    Empty,
    #[error("inconsistent shapes: {0}")]
    Shape(&'static str),
    #[error("low-dim {low} must be in 1..={high}")]
    LowDim { low: usize, high: usize },
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse: {0}")]
    Parse(String),
    #[error("dlpack: {0}")]
    Dlpack(String),
    #[error("optimizer failed: {0}")]
    Optimize(String),
    #[error("{0}")]
    Msg(String),
}

pub type Result<T> = std::result::Result<T, LandfoldError>;
