//! landfold: Landscape And Nonlinear Distance Folding Onto Low Dimensions.
//!
//! Implements the transfer-function MDS of Ceriotti, Tribello and Parrinello
//! (PNAS 2011, DOI 10.1073/pnas.1108486108). Pairwise distances are mapped
//! through a generalised sigmoid, then a low-dimensional layout is fitted by
//! conjugate-gradient on the mismatch of transformed distances.
//!
//! Hot path: Rayon pairwise distances, GEMM Euclidean Gram matrix, analytic
//! χ gradient, Polak-Ribiere CG. Embeddings export as DLPack via `dlpk`.
//! Optional `--features python` binds CPython on pyo3/numpy 0.29 so the
//! graph shares one major with dlpk 0.4.1; dlpk's `pyo3` feature stays off.

pub mod anneal;
pub mod array;
pub mod artifact;
pub mod cg;
pub mod chi_obj;
pub mod error;
pub mod floor;
#[cfg(feature = "highs")]
pub mod highs_slp;
pub mod hist;
pub mod io;
pub mod iter;
pub mod landmark;
pub mod mds;
pub mod metric;
pub mod pairwise;
pub mod project;
pub mod provenance;
#[cfg(feature = "readcon")]
pub mod readcon;
pub mod replica;
pub mod search;
pub mod stress;
pub mod transfer;

pub use anneal::AnnealOpts;
pub use array::to_dlpack;
pub use artifact::{EMBEDDING_SCHEMA, FES_SCHEMA, PROJECTION_SCHEMA};
pub use chi_obj::{ChiObjective, UNBOUNDED as CHI_UNBOUNDED};
pub use error::{LandfoldError, Result};
pub use floor::occupancy_map_floor;
#[cfg(feature = "highs")]
pub use highs_slp::HighsOpts;
pub use hist::{
    FreeEnergy, Histogram1d, Histogram2d, coordination_histogram, coordination_numbers,
    fes_from_points,
};
pub use io::{PointSet, read_points, read_points_path, write_points};
pub use iter::{Embedding, IterOpts, Solver, embed, embed_points};
pub use landmark::{Landmarks, farthest_point, farthest_point_ifirst, voronoi_weights};
pub use mds::{MdsMode, MdsReport, classical_mds, mds_from_points, randomized_mds};
pub use metric::{Dot, Euclid, L1, Metric, Periodic, Sphere};
pub use pairwise::{apply_transfer, pairwise, pairwise_euclid};
pub use project::{
    ProjOpts, ProjReport, project_many, project_many_report, project_one, project_report,
};
pub use replica::ReplicaOpts;
pub use search::StochOpts;
pub use stress::{Stress, StressEval, query_chi};
pub use transfer::{Transfer, TransferMode};

#[cfg(feature = "python")]
mod python;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
