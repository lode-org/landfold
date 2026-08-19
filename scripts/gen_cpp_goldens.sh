#!/usr/bin/env bash
# Compile the C++ kernel excerpt and dump goldens. Remote builder only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/tests/goldens/cpp_oracle.txt}"
g++ -O2 -std=c++14 -o /tmp/landfold-cpp-oracle "$ROOT/oracle/cosmo_kernels.cpp"
mkdir -p "$(dirname "$OUT")"
/tmp/landfold-cpp-oracle > "$OUT"
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
