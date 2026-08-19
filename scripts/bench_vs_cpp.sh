#!/usr/bin/env bash
# Time landfold chi+grad against HaoZeke addLocks NLDRITERChi on the same LCG set.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${SKMAP_SRC:-$HOME/Git/Github/HaoZeke/sketchmap}"
N="${1:-400}"
REPS="${2:-40}"

if [ -z "${LLAPACK:-}" ]; then
  if [ -e "${HOME}/.local/lib/libopenblas.so" ] || [ -e "${HOME}/.local/lib/libopenblas.so.0" ]; then
    LLAPACK="-L${HOME}/.local/lib -Wl,-rpath,${HOME}/.local/lib -lopenblas"
  else
    LLAPACK="-lopenblas"
  fi
fi

export CXX="${CXX:-g++}"
export CXXFLAGS="-g -O3 -std=c++14 -I${SRC}/libs"
make -C "$SRC/libs" -j"$(nproc)"
if [ ! -f "$SRC/tools/libdimred.o" ]; then
  $CXX $CXXFLAGS -I"$SRC/tools" -c -o "$SRC/tools/libdimred.o" "$SRC/tools/libdimred.cpp"
fi
$CXX -O3 -std=c++14 -I"$SRC/tools" -I"$SRC/libs" \
  -o /tmp/landfold-cpp-bench \
  "$ROOT/oracle/bench_chi.cpp" "$SRC/tools/libdimred.o" \
  -L"$SRC/libs" -ltoolbox $LLAPACK

echo "=== C++ ==="
/tmp/landfold-cpp-bench "$N" "$REPS"
echo "=== landfold ==="
cargo run --release --example bench_chi -- "$N" "$REPS"
echo "compare ns_per_eval: landfold must be smaller than cpp"
