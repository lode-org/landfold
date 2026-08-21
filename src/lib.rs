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
pub mod axis;
pub mod bands;
pub mod cg;
pub mod chi_obj;
pub mod error;
pub mod fieldgp;
pub mod floor;
pub mod gapsplit;
#[cfg(feature = "hdf5")]
pub mod hdf5;
#[cfg(feature = "highs")]
pub mod highs_slp;
pub mod hist;
pub mod io;
pub mod iter;
pub mod landmark;
pub mod mds;
pub mod nearfar;
pub mod metric;
pub mod pacmap;
pub mod pairwise;
pub mod phate;
pub mod project;
pub mod provenance;
#[cfg(feature = "readcon")]
pub mod readcon;
pub mod replica;
pub mod scale;
pub mod search;
pub mod stress;
pub mod trajectory;
pub mod transfer;

pub use anneal::AnnealOpts;
pub use array::to_dlpack;
pub use artifact::{EMBEDDING_SCHEMA, FES_SCHEMA, PROJECTION_SCHEMA};
pub use axis::{AxisModel, axis_embed, axis_fit, axis_project, double_well};
pub use bands::{BandOpts, BandReport, bands_embed};
pub use chi_obj::{ChiObjective, UNBOUNDED as CHI_UNBOUNDED};
pub use error::{LandfoldError, Result};
pub use fieldgp::{FieldGp, FieldPredict, basin_coordinate, fit_imq_map, imq, predict_imq};
pub use floor::occupancy_map_floor;
pub use gapsplit::{GapReport, gap_pair_weights, gap_split_embed, suggest_tau};
#[cfg(feature = "hdf5")]
pub use hdf5::read_hdf5_batch;
#[cfg(feature = "highs")]
pub use highs_slp::HighsOpts;
pub use hist::{
    FreeEnergy, Histogram1d, Histogram2d, coordination_histogram, coordination_numbers,
    fes_from_points, joint_pairwise_hist,
};
pub use io::{PointSet, read_points, read_points_path, write_plumed, write_points};
pub use iter::{Embedding, IterOpts, Solver, embed, embed_points, embed_sigma_schedule};
pub use landmark::{
    Landmarks, LandmarkMode, farthest_point, farthest_point_ifirst, select_landmarks,
    voronoi_weights,
};
pub use mds::{MdsMode, MdsReport, classical_mds, mds_from_points, randomized_mds};
pub use nearfar::{NearFarOpts, nearfar_embed};
pub use metric::{Dot, Euclid, Fisher, L1, Metric, Periodic, Sphere, Stretch, Wasserstein1};
pub use pacmap::{PacmapOpts, knn_project, pacmap_embed, rank_uniform};
pub use pairwise::{apply_transfer, pairwise, pairwise_euclid};
pub use phate::{PhateModel, PhateOpts, phate_embed, phate_project, slow_mode};
pub use project::{
    ProjOpts, ProjReport, project_many, project_many_report, project_one, project_report,
};
#[cfg(feature = "readcon")]
pub use readcon::{
    frame_positions, frames_batch, frames_positions, read_con_batch, read_con_frames,
    read_con_positions,
};
#[cfg(feature = "readcon-chemfiles")]
pub use readcon::{
    read_trajectory_batch, read_trajectory_frames, read_trajectory_positions,
};
pub use provenance::{EON_COMPATIBILITY_SCHEMA, EngineCompatibility, PROVENANCE_SCHEMA, Provenance};
pub use replica::ReplicaOpts;
pub use scale::{ScaleReport, StretchReport, suggest_alpha, suggest_scale};
pub use search::StochOpts;
pub use stress::{Stress, StressEval, query_chi};
pub use trajectory::FrameBatch;
pub use transfer::{FUN_SPEC_HELP, Transfer, TransferMode};

#[cfg(feature = "python")]
mod python;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Exact `eindir` revision discovered at build time, or `unknown` when the
/// dependency source is not a Git checkout and no override is provided.
pub const EINDIR_REVISION: &str = env!("LANDFOLD_EINDIR_REVISION");
