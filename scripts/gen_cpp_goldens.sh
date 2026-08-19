#!/usr/bin/env bash
# Build HaoZeke/sketchmap addLocks (valarray ostream) and dump NLDR* goldens.
# Blocks: xfer f/df, pair Euclid/PBC, mds Torgerson pairwise, chi/grad,
# chi1 query, Gonzalez farthest-point indices.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${SKMAP_SRC:-$HOME/Git/Github/HaoZeke/sketchmap}"
OUT="${1:-$ROOT/tests/goldens/cpp_oracle.txt}"

if [ ! -f "$SRC/tools/libdimred.cpp" ]; then
  echo "missing $SRC/tools/libdimred.cpp (set SKMAP_SRC)" >&2
  exit 1
fi
if ! grep -q 'explicit valarray ostream\|operator<<(std::ostream& os, const std::valarray' "$SRC/libs/tbdefs.hpp"; then
  echo "SKMAP_SRC=$SRC is missing the valarray ostream fix (HaoZeke addLocks / ae46f30)" >&2
  exit 1
fi

export CXX="${CXX:-g++}"
# addLocks make.in uses OpenBLAS. Probe default path, then $HOME/.local/lib
# (rg.terra keeps the soname there), then netlib.
if [ -z "${LLAPACK:-}" ]; then
  if echo 'int main(){return 0;}' | $CXX -x c++ - -lopenblas -o /tmp/landfold-blas-probe 2>/dev/null; then
    LLAPACK=-lopenblas
  elif [ -e "${HOME}/.local/lib/libopenblas.so" ] || [ -e "${HOME}/.local/lib/libopenblas.so.0" ]; then
    LLAPACK="-L${HOME}/.local/lib -Wl,-rpath,${HOME}/.local/lib -lopenblas"
  else
    LLAPACK="-llapack -lblas"
  fi
fi
export CXXFLAGS="-g -O2 -std=c++14 -I${SRC}/libs"
make -C "$SRC/libs" -j"$(nproc)"
$CXX $CXXFLAGS -I"$SRC/tools" -c -o "$SRC/tools/libdimred.o" "$SRC/tools/libdimred.cpp"

$CXX -O2 -std=c++14 \
  -I"$SRC/tools" -I"$SRC/libs" \
  -o /tmp/landfold-cpp-oracle \
  "$ROOT/oracle/oracle.cpp" \
  "$SRC/tools/libdimred.o" \
  -L"$SRC/libs" -ltoolbox $LLAPACK

mkdir -p "$(dirname "$OUT")"
/tmp/landfold-cpp-oracle > "$OUT"
echo "wrote $OUT ($(wc -l < "$OUT") lines) from $SRC ($LLAPACK)"
