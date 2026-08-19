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
landfold embed -D 30 -d 2 --fun-hd 6,8,8 --fun-ld 6,2,8 --steps 50 < hd.dat
landfold landmarks -D 30 -n 200 --seed 1 --indices < hd.dat
landfold project -D 30 -d 2 --high-file lm.hd --low-file lm.ld \
    --grid 1.0,21,201 --refine 3 < frames.dat
landfold dist -D 30 < hd.dat
landfold mds -D 30 -d 2 --distances < hd.dat
landfold fes --input ld.dat --svg fes.svg --csv fes.csv \
    --frames frames.xyz --cn-csv cn.csv
```

`--stoch`, `--anneal`, `--replica`, and `--highs` (needs
`--features highs`) are extra solver arms. `--highs` is
bound-constrained sequential LP via HiGHS.
`Solver::Standard` (full-pair Polak-Ribiere CG) stays the default.
`landfold mds` is the Torgerson (1952)
initialiser; `--distances` prints pairwise distances of the embedding
(the C++ Torgerson pairwise invariant).

`landfold project` places new high-D rows by a coarse then fine χ
grid and optional `--refine` Polak-Ribiere steps; `--print-error`
appends χ and the nearest-landmark distance. `landfold landmarks` is
Gonzalez farthest-point sampling (`--voronoi` for Voronoi masses).
`landfold fes` writes `F = -ln(rho/rhomax)` as CSV/SVG; `--frames`
plus `--cn-csv` is the coordination histogram of the cluster-FES
figure class. `--blur`, `--floor`, and `--fmax` match that panel.

C++ goldens are `tests/goldens/cpp_oracle.txt`, rebuilt on the remote
builder by `scripts/gen_cpp_goldens.sh` against the HaoZeke `addLocks`
tree (`SKMAP_SRC`). `cargo test --release --test cpp_parity` loads that file.

## Python

`--features python` binds `embed_euclid`, `project_euclid`,
`farthest_euclid`, and `fes_xy`. pyo3/numpy stay on 0.29 so the graph
shares one major with dlpk 0.4.1. dlpk's `pyo3` feature stays off
(`pyo3-ffi` is `links = "python"`). Check on the remote builder:

```
bash scripts/check_pyo3_pin.sh
cargo tree -e features --features python
```

`landfold --help` records the pin. The CLI itself does not link pyo3.

## readcon and chemparseplot

The optional `readcon` feature reads canonical CON/CONVEL files through
`readcon-core` while keeping complete `ConFrame` values available for
headers, units, atom IDs, and optional sections:

```
cargo add landfold --features readcon
cargo add landfold --features readcon-chemfiles
```

`readcon-chemfiles` adds foreign trajectory ingress through readcon's
chemfiles conversion layer. Use `read_con_frames` or
`read_trajectory_frames` when frame metadata is needed, and the corresponding
`*_positions` helpers when a validated `(n_atoms, 3)` array is the algorithm
input. All frames in a trajectory must retain the same atom IDs and count.

HDF5 trajectory loading belongs at the Python integration boundary: use
`readcon-chemfiles` or `chemparseplot` to ingest the trajectory, pass NumPy
coordinate arrays to landfold's existing Python functions, and retain source
frame IDs and units in the surrounding result object. Landfold does not make
HDF5 a required Rust dependency or replace CON as the canonical frame format.

## License

MIT. See `CITATION.cff` for the papers to cite with the method.
