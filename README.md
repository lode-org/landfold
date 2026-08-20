# landfold

**landfold**: Landscape And Nonlinear Distance Folding Onto Low
Dimensions. MIT.

The transfer function and stress follow Ceriotti, Tribello and
Parrinello, *PNAS* **2011**,
[10.1073/pnas.1108486108](https://doi.org/10.1073/pnas.1108486108).
Each published kernel cites its paper in rustdoc.

<p align="center">
  <img src="branding/logo/landfold-logo-light.svg" width="360" alt="landfold">
</p>

## CLI

```
landfold embed -D 10 -d 2 --fun-hd ceriotti,5,8,1 --fun-ld ceriotti,5,2,2 < hd.dat
landfold embed -D 10 -d 2 --fun-hd ceriotti,5,8,1 --fun-ld imq,5 < hd.dat
landfold landmarks -D 30 -n 200 --seed 1 --indices < hd.dat
landfold project -D 30 -d 2 --high-file lm.hd --low-file lm.ld \
    --grid 1.0,21,201 --refine 3 < frames.dat
landfold dist -D 30 < hd.dat
landfold mds -D 30 -d 2 --distances < hd.dat
landfold fes --input ld.dat --svg fes.svg --csv fes.csv \
    --frames frames.xyz --cn-csv cn.csv
rgpycrumbs landfold plot-energy --input cloud.energy.csv \
    --output energy.png --frame rmsd --method grad_imq
rgpycrumbs landfold plot-fes --input fes.csv --output fes.png \
    --kt 0.168 --fmax 2 --method grad_imq
```

Energy on the 2D plane is the MethodsX representation
(`plot.representation.plot_energy`, `rgpycrumbs landfold plot-energy`).
`landfold fes --svg` is the occupancy heatmap. Occupancy publication
figures use `plot.landfold.plot_fes` via `rgpycrumbs landfold plot-fes`.

`--stoch`, `--anneal`, `--replica`, and `--highs` (needs
`--features highs`) are extra solver arms. `--highs` is
bound-constrained L-BFGS-QP via HiGHS.
For the HiGHS arm, the `--trust` option sets the per-coordinate trust radius
and `--box lo,hi` applies uniform coordinate bounds. Without `--trust`, the
CLI scales the initial radius from the low-D coordinate span; an explicit
value remains absolute.
`Solver::Standard` (full-pair Polak-Ribiere CG) stays the default.
`landfold mds` is the Torgerson (1952)
initialiser; `--distances` prints pairwise distances of the embedding
(the C++ Torgerson pairwise invariant).

`landfold project` places new high-D rows by a coarse then fine χ
grid and optional `--refine` Polak-Ribiere steps; `--print-error`
appends χ and the nearest-landmark distance. `landfold landmarks` is
Gonzalez farthest-point sampling (`--voronoi` for Voronoi masses).
`landfold fes` writes `F = -kT ln(rho/rhomax)` as CSV/SVG; `--frames`
plus `--cn-csv` is the coordination histogram of the cluster-FES
figure class. `--blur`, `--floor`, and `--fmax` match that panel.

The public Ceriotti comparison script runs the bundled LJ38 teaching set
and the beta-hairpin landmark set when the external sketch-map examples are
available:

```
LANDFOLD=target/release/landfold \
SKETCHMAP_PROTEIN=/path/to/sketchmap/examples/protein \
scripts/ceriotti_public_bench.sh
```

It reports stress and wall time for the standard, xtsci L-BFGS, and HiGHS
arms, including a published-map polish with an explicit trust radius.

C++ goldens are `tests/goldens/cpp_oracle.txt`, rebuilt on the remote
builder by `scripts/gen_cpp_goldens.sh` against the HaoZeke `addLocks`
tree (`SKMAP_SRC`). `cargo test --release --test cpp_parity` loads that file.

## Python

`--features python` binds `embed_euclid`, `project_euclid`,
`farthest_euclid`, and `fes_xy`. The corresponding `*_result` functions
return dictionaries tagged with `landfold.*.v1` schemas and retain stress,
density, χ, nearest-landmark diagnostics, and caller-supplied source metadata
for plotting adapters. pyo3/numpy stay on 0.29 so the graph
shares one major with dlpk 0.4.1. dlpk's `pyo3` feature stays off
(`pyo3-ffi` is `links = "python"`). Check on the remote builder:
Install the optional Chemparseplot bridge dependencies with
`pip install 'landfold[chemparseplot]'`.

The structured dictionaries use `coordinates` plus `stress` for embeddings,
`coordinates` plus `chi`, `nearest_distance`, and `nearest_index` for
projections, and `x`, `y`, `free_energy`, and `density` for FES results.
Each includes a `schema` key and a `metadata` dictionary. The `*_result`
functions require `metadata["provenance"]` with schema `landfold.provenance.v1`,
`run_id`, a `sha256:`-prefixed
input digest, `engine_id`, `protocol_family`, protocol major/minor, eindir ABI
layout, and DLPack major/minor fields. `rgpot` records additionally require
the exact 40-digit `eindir_revision`. This keeps a low-dimensional plot
joinable to the exact anneal/rgpot source without making Landfold evaluate the
objective engine.

```
bash scripts/check_pyo3_pin.sh
cargo tree -e features --features python
```

`landfold --help` records the pin. The CLI itself does not link pyo3.
Python builds expose `landfold.eindir_revision`, discovered from the sibling
Git checkout or overridden with `LANDFOLD_EINDIR_REVISION`; the HDF5 bridge
copies it into provenance when the value is an exact commit revision.

## readcon and chemparseplot

The optional `readcon` feature reads canonical CON/CONVEL files through
`readcon-core` while keeping complete `ConFrame` values available for
headers, units, atom IDs, and optional sections:

```
cargo add landfold --features readcon
cargo add landfold --features readcon-chemfiles
cargo add landfold --features hdf5
```

`readcon-chemfiles` adds foreign trajectory ingress through readcon's
chemfiles conversion layer. Use `read_con_frames` or
`read_trajectory_frames` when frame metadata is needed, and the corresponding
`*_positions` helpers when a validated `(n_atoms, 3)` array is the algorithm
input. `read_con_batch` and `read_trajectory_batch` expose the same inputs as a
format-neutral `FrameBatch`, retaining frame IDs, atom IDs, the length unit, and
typed per-frame JSON metadata in `FrameBatch::metadata`. CON and Chemfiles
adapters also preserve stable per-atom chemical symbols in
`FrameBatch::atom_symbols`.
All frames in a trajectory must retain the same atom IDs, symbols, and count.

This feature links the native Chemfiles library. On systems whose CMake
version rejects Chemfiles' legacy minimum-version declaration, configure the
native build with CMAKE_POLICY_VERSION_MINIMUM=3.5; the lean readcon
feature does not require Chemfiles or CMake.

The adapter functions are also re-exported at the crate root when their
feature is enabled, so `landfold::read_con_batch` and
`landfold::read_trajectory_batch` are available without importing the module.

`FrameBatch::flattened_points` converts that contract to one
`(n_atoms * 3)` row per frame for distance embedding. The inverse
`FrameBatch::from_flattened_points` validates HDF5/NumPy-style arrays before
restoring frame and atom identity:

```rust
let batch = landfold::FrameBatch::from_flattened_points(
    points,
    atom_ids,
    frame_ids,
    Some("angstrom".into()),
)?;
let embedding = landfold::embed_points(batch.flattened_points()?.view(), &metric, &opts)?;
```

The optional `hdf5` feature reads the chemparseplot interchange layout directly
in Rust. It consumes `/path/images`, and optionally `/path/frame_ids`,
`/path/energies`, `/path/f_para`, `/path/rxn_coord`, `/path/gradients`, `/metadata/atom_ids`,
`/metadata/atomic_numbers`, and `/metadata/cell`, returning the same validated
`FrameBatch`. `/metadata/atom_symbols` is also accepted for stable chemical
identity. Atomic numbers, symbols, and a finite 3x3 cell are retained as typed
fields; Chemparseplot’s flattened nine-element cell representation is normalized
to that same 3x3 field. Per-image gradients are retained as
`FrameBatch::gradients` with shape `(n_frames, 3 * n_atoms)`.
`/metadata/length_unit` is retained as `FrameBatch::length_unit` when present;
string metadata accepts variable-length Unicode/ASCII and fixed-width ASCII
datasets up to 1024 bytes per value.
per-image path observables are available through
`FrameBatch::frame_metadata` when present:

```rust
let batch = landfold::read_hdf5_batch(path)?;
```

HDF5 is an optional native dependency. `readcon` remains the canonical CON
reader, `readcon-chemfiles` remains the broad foreign-format ingress, and all
three adapters converge on `FrameBatch` before embedding. The Rust HDF5 adapter
retains coordinate identity and the scalar path observables needed for profile
analysis; the Python ChemGP bridge also forwards richer source metadata around
the same coordinate contract.

For ChemGP NEB HDF5 files, the runnable
[`examples/chemparseplot_hdf5.py`](examples/chemparseplot_hdf5.py) bridge uses
`chemparseplot.parse.trajectory.hdf5.load_neb_result`, reshapes its image path,
and forwards source metadata to `embed_euclid_result`. The bridge records
explicit frame and atom IDs, preserves optional ChemGP metadata such as atomic
numbers and cells, retains finite per-image `energies`, `f_para`, and
`rxn_coord` under `metadata["path_observables"]`, retains the validated
flattened force-gradient array under `metadata["path_gradients"]`, and uses
`None` when the source does not declare a length unit rather than guessing one.

## License

MIT. See `CITATION.cff` for the papers to cite with the method.
