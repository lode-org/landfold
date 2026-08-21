#!/usr/bin/env bash
# Independent asinh sketch-map vs published Ceriotti path.
# LJ38 teaching TSE landmarks + beta-hairpin 30-D landmarks.
# Requires a release landfold (remote builder). Does not compile here.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LANDFOLD="${LANDFOLD:?set LANDFOLD to the release binary}"
LJ38="$ROOT/examples/cosmo-lj38"
SKMAP="${SKETCHMAP_PROTEIN:-$HOME/Git/Github/HaoZeke/sketchmap/examples/protein}"
OUT="${OUT:-/tmp/landfold-asinh}"
mkdir -p "$OUT"

if [ ! -s "$LJ38/ts.cv" ]; then
  awk '{for(i=3;i<=12;i++) printf "%s ", $i; print ""}' "$LJ38/ts.all" >"$LJ38/ts.cv"
fi

echo "# LJ38 asinh on the published 200-landmark path"
"$LANDFOLD" embed -D 10 -d 2 -w --fun-hd asinh,5 --fun-ld asinh,5 \
  --init-f --preopt 40 --steps 40 \
  <"$LJ38/out/lj38.lm" >"$OUT/lj38_asinh.ld"
"$LANDFOLD" project -D 10 -d 2 -w \
  --high-file "$LJ38/out/lj38.lm" --low-file "$OUT/lj38_asinh.ld" \
  --fun-hd asinh,5 --fun-ld asinh,5 --grid 12,11,41 --refine 3 \
  <"$LJ38/ts.cv" >"$OUT/ts_asinh.proj"
cp "$OUT/ts_asinh.proj" /tmp/ts_asinh.proj

if [ -s "$SKMAP/lm4.30cv.w01.1" ]; then
  echo "# protein asinh on 1000 30-D landmarks (period 2 pi)"
  "$LANDFOLD" embed -D 30 -d 2 -w --pi 6.283185307179586 \
    --fun-hd asinh,6 --fun-ld asinh,6 --init-f --steps 40 \
    <"$SKMAP/lm4.30cv.w01.1" >"$OUT/protein_asinh.ld"
fi

echo "# wrote $OUT"
echo "# plot: python3 $ROOT/scripts/compare_asinh_fig.py"
