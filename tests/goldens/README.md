# DECAF golden fixtures

These fixtures were generated with King Chun Lai's reference implementation,
`fhi-decaf` 1.0.6, at commit
`dd8b2c5bdf224cc583855133d4c63a985763294f`.

From the landfold repository root, reproduce them with:

```sh
git clone https://gitlab.mpcdf.mpg.de/klai/decaf.git ../reference-decaf
git -C ../reference-decaf checkout dd8b2c5bdf224cc583855133d4c63a985763294f
python3 -m venv ../reference-decaf/.venv
../reference-decaf/.venv/bin/python -m pip install -e ../reference-decaf
PYTHONPATH=../reference-decaf/src \
  ../reference-decaf/.venv/bin/python \
  tests/goldens/generate_decaf_goldens.py \
  --reference-root ../reference-decaf \
  --output-dir target/decaf-goldens
python3 - <<'PY'
from pathlib import Path
import numpy as np

expected = Path("tests/goldens")
actual = Path("target/decaf-goldens")
for name in [
    "decaf_pah_distances.txt",
    "decaf_pd211_distances.txt",
    "decaf_pd211_embedded.txt",
    "decaf_oos_reference_distances.txt",
]:
    assert np.allclose(
        np.loadtxt(expected / name),
        np.loadtxt(actual / name),
        rtol=0,
        atol=2e-11,
    ), name
for name in ["decaf_pah_labels.txt", "decaf_pd211_labels.txt"]:
    assert np.array_equal(
        np.loadtxt(expected / name, dtype=int),
        np.loadtxt(actual / name, dtype=int),
    ), name
PY
```

The PAH fixture combines the published benzene, naphthalene, anthracene,
tetracene, phenanthrene, and graphene structures. The Pd fixture uses the
published `examples/Structures/Pd_211.con` structure from `fhi-decaf` 1.0.6.
Its SOAP call uses nearest-neighbour distance `2.75`,
`SOAP_L_nmax=4`, `SOAP_L_lmax=3`, `SOAP_S_nmax=8`,
`SOAP_S_lmax=4`, `SOAP_S_cut=sqrt(1/2)+0.5`,
`SOAP_S_sigma=SOAP_S_cut/8`, `SOAP_L_cut=sqrt(3)/2+1`,
`SOAP_L_sigma=SOAP_L_cut/8`, and `periodic=True`. The workflow uses classical
MDS, mean-shift clustering, the Gaussian bandwidth estimator with
`sigma=0.0092`, automatic dimension selection (`dim=0`), and
`reorder_mode_x=True`.

The out-of-sample fixture uses normalized synthetic vectors and two
non-centroid queries; it stores distances from the reference training
embedding to the reference projections so eigensolver sign choices do not
affect the comparison. The tolerance in the check allows for SOAP
floating-point noise between BLAS implementations; the observed maximum PAH
distance difference in the recorded regeneration was below `4e-15`.
