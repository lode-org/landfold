#!/usr/bin/env bash
# Out-of-sample project the public Ceriotti trajectories, then draw FES figures.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LANDFOLD="${LANDFOLD:-$ROOT/target/release/landfold}"
LJ38="$ROOT/examples/cosmo-lj38"
SKMAP="${SKETCHMAP_PROTEIN:-$HOME/Git/Github/HaoZeke/sketchmap/examples/protein}"
OUT="${OUT:-/tmp/landfold-ceriotti}"
FIGS="${FIGS:-$OUT/figs}"
PROJECT_GRID="${PROJECT_GRID:-20,11,15}"
PROJECT_REFINE="${PROJECT_REFINE:-1}"
MAX_PROTEIN_FRAMES="${MAX_PROTEIN_FRAMES:-2000}"
mkdir -p "$OUT" "$FIGS"

if [ ! -x "$LANDFOLD" ]; then
  echo "set LANDFOLD" >&2
  exit 1
fi

echo "# project LJ38 TSE (3451 frames) from highs landmarks"
if [ ! -s "$OUT/lj38.lm" ]; then
  awk '{for(i=3;i<=12;i++) printf "%s ", $i; print ""}' "$LJ38/ts.all" >"$LJ38/ts.cv"
  "$LANDFOLD" landmarks -D 10 -n 200 --seed 1 <"$LJ38/ts.cv" >"$OUT/lj38.lm"
fi
if [ ! -s "$OUT/lj38-highs.ld" ]; then
  "$LANDFOLD" embed -D 10 -d 2 -w --fun-hd 5,8,1 --fun-ld 5,2,2 --steps 40 --highs \
    <"$OUT/lj38.lm" >"$OUT/lj38-highs.ld"
fi
if [ ! -s "$LJ38/ts.cv" ]; then
  awk '{for(i=3;i<=12;i++) printf "%s ", $i; print ""}' "$LJ38/ts.all" >"$LJ38/ts.cv"
fi
echo start lj38-project
"$LANDFOLD" project -D 10 -d 2 -w \
  --high-file "$OUT/lj38.lm" --low-file "$OUT/lj38-highs.ld" \
  --fun-hd 5,8,1 --fun-ld 5,2,2 \
  --grid 12,11,41 --refine 3 \
  <"$LJ38/ts.cv" >"$OUT/lj38-oos.proj"

echo "# protein landmarks: polish published map with highs --trust 20"
if [ ! -s "$OUT/protein-polish-highs.ld" ]; then
  "$LANDFOLD" embed -D 30 -d 2 -w --pi 6.283185307179586 \
    --fun-hd 6,8,8 --fun-ld 6,2,8 --steps 40 --highs --trust 20 \
    --init "$SKMAP/lm4.30cv.w01.1.proj" \
    <"$SKMAP/lm4.30cv.w01.1" >"$OUT/protein-polish-highs.ld"
fi

echo "# project hairpin trajectory (30-D Ramachandran, all frames)"
if [ ! -s "$SKMAP/colvar.wt.30cv.4" ]; then
  echo "missing $SKMAP/colvar.wt.30cv.4" >&2
  exit 1
fi
awk '{for(i=1;i<=30;i++) printf "%s ", $i; print ""}' "$SKMAP/colvar.wt.30cv.4" \
  >"$OUT/protein.cv"
echo start protein-project
if [ "$MAX_PROTEIN_FRAMES" -gt 0 ]; then
  awk -v max_frames="$MAX_PROTEIN_FRAMES" 'NR <= max_frames { print }' "$OUT/protein.cv" >"$OUT/protein-project.cv"
else
  cp "$OUT/protein.cv" "$OUT/protein-project.cv"
fi
"$LANDFOLD" project -D 30 -d 2 -w \
  --pi 6.283185307179586 \
  --high-file "$SKMAP/lm4.30cv.w01.1" --low-file "$OUT/protein-polish-highs.ld" \
  --fun-hd 6,8,8 --fun-ld 6,2,8 \
  --grid "$PROJECT_GRID" --refine "$PROJECT_REFINE" \
  <"$OUT/protein-project.cv" >"$OUT/protein-oos.proj"

python3 "$ROOT/scripts/ceriotti_figures.py" \
  --out "$FIGS" \
  --lj38-proj "$OUT/lj38-oos.proj" \
  --protein-lm-pub "$SKMAP/lm4.30cv.w01.1.proj" \
  --protein-lm-ours "$OUT/protein-polish-highs.ld" \
  --protein-smap-pub "$SKMAP/colvar.wt.30cv.4.smap" \
  --protein-smap-ours "$OUT/protein-oos.proj"
echo "# figs in $FIGS"
