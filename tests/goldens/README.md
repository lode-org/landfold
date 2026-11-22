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
  --output-dir tests/goldens
git diff --exit-code -- \
  tests/goldens/decaf_pah_distances.txt \
  tests/goldens/decaf_pah_labels.txt \
  tests/goldens/decaf_pd211_distances.txt \
  tests/goldens/decaf_pd211_embedded.txt \
  tests/goldens/decaf_pd211_labels.txt \
  tests/goldens/decaf_oos_reference_distances.txt
```

The PAH fixture combines the published benzene, naphthalene, anthracene,
tetracene, phenanthrene, and graphene structures. The Pd fixture uses the
published Pd(211) surface. The out-of-sample fixture uses normalized synthetic
vectors and two non-centroid queries; it stores distances from the reference
training embedding to the reference projections so eigensolver sign choices do
not affect the comparison.
