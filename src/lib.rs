//! Sigmoid-distance nonlinear embedding for high-dimensional molecular data.
//!
//! Implements the transfer-function MDS of Ceriotti, Tribello and Parrinello
//! (PNAS 2011, DOI 10.1073/pnas.1108486108). Pairwise distances are mapped
//! through a generalised sigmoid, then a low-dimensional layout is fitted by
//! conjugate-gradient on the mismatch of transformed distances.
//!
//! Hot path: Rayon pairwise distances, GEMM Euclidean Gram matrix, analytic
//! χ gradient, Polak-Ribiere CG. Embeddings export as DLPack via `dlpk`.

pub mod array;
pub mod cg;
pub mod error;
pub mod hist;
pub mod io;
pub mod iter;
pub mod landmark;
pub mod mds;
pub mod metric;
pub mod pairwise;
pub mod project;
pub mod search;
pub mod stress;
pub mod transfer;

pub use array::to_dlpack;
pub use error::{LandfoldError, Result};
pub use hist::{coordination_histogram, coordination_numbers, FreeEnergy, Histogram1d, Histogram2d};
pub use io::{read_points, read_points_path, write_points, PointSet};
pub use iter::{embed, embed_points, Embedding, IterOpts, Solver};
pub use search::StochOpts;
pub use landmark::{farthest_point, Landmarks};
pub use mds::{classical_mds, mds_from_points, randomized_mds, MdsMode, MdsReport};
pub use metric::{Dot, Euclid, Metric, Periodic, Sphere};
pub use pairwise::{apply_transfer, pairwise, pairwise_euclid};
pub use project::{project_many, project_one, ProjOpts};
pub use stress::{Stress, StressEval};
pub use transfer::{Transfer, TransferMode};

#[cfg(feature = "python")]
mod python;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
