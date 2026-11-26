# Changelog

## Unreleased

- Add descriptor-agnostic DECAF with projectable classical MDS,
  broken-stick dimension selection, the paper bandwidth heuristic,
  mean-shift classes, fuzzy memberships, and medoid representatives.
- Keep a far-tail wiggle or a shoulder ripple from becoming the DECAF paper
  bandwidth. A gapless cloud uses its median nearest-neighbor distance, so
  a disordered embedding is not reported as one class.

## 0.1.0

MIT crate **landfold** (Landscape And Nonlinear Distance Folding Onto
Low Dimensions).

- Transfer functions, Euclid GEMM pairwise, χ with analytic gradient
- Default solver: Polak-Ribiere CG. Extra arms: stochastic pairs,
  annealing, replica exchange
- Torgerson and Halko MDS init, out-of-sample project, Gonzalez
  landmarks, FES invert
- C++ oracle goldens (`cpp_parity`), CPython `python` feature on
  pyo3 0.29
- Format-neutral `FrameBatch` ingestion through optional readcon,
  readcon-chemfiles, and Chemparseplot-compatible HDF5 adapters, with
  stable identities, units, cells, metadata, path observables, and gradients
- Runnable Python Chemparseplot HDF5 bridge with versioned Landfold results
