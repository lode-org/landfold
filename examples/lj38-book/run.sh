#!/usr/bin/env bash
# Quenched minima book -> SOAP or sorted-pair HD -> landfold asinh -> energy IDW.
# Requires a release landfold. Does not compile here.
# Required env:
#   LANDFOLD  release landfold binary
#   MIN       .min book, one row per structure: E then 3N coordinates
# Optional:
#   KIND      soap (default) or pair
#   OUT       working directory (default ./out)
#   NATOMS    atoms per structure (default 38)
#   NLAND     Gonzalez landmarks (default 200)
#   WORKERS   SOAP fingerprint processes (default 1)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LANDFOLD="${LANDFOLD:?set LANDFOLD to the release binary}"
MIN="${MIN:?set MIN to the .min book (E + 3N per row)}"
KIND="${KIND:-soap}"
OUT="${OUT:-$HERE/out}"
NATOMS="${NATOMS:-38}"
NLAND="${NLAND:-200}"
WORKERS="${WORKERS:-1}"
test -x "$LANDFOLD" || { echo "LANDFOLD=$LANDFOLD is not executable" >&2; exit 2; }
test -s "$MIN" || { echo "missing MIN=$MIN" >&2; exit 2; }
mkdir -p "$OUT"

python3 "$HERE/min_to_hist.py" \
  --min "$MIN" --kind "$KIND" --n-atoms "$NATOMS" \
  --out "$OUT/${KIND}.hist" \
  --energy "$OUT/book.energy" \
  --sigma "$OUT/sigma.txt" \
  --landmarks "$OUT/${KIND}.lm" \
  --land-idx "$OUT/${KIND}.idx" \
  --n-land "$NLAND" \
  --dest "$OUT" \
  --workers "$WORKERS"

D=$(awk 'NF && $1 !~ /^#/ {print NF; exit}' "$OUT/${KIND}.hist")
SIGMA=$(tr -d '[:space:]' < "$OUT/sigma.txt")
echo "# ${KIND} D=$D nland=$NLAND sigma=$SIGMA"

"$LANDFOLD" embed -D "$D" -d 2 \
  --fun-hd "asinh,$SIGMA" --fun-ld "asinh,$SIGMA" \
  --init-f --preopt 40 --steps 40 --center \
  < "$OUT/${KIND}.lm" > "$OUT/${KIND}.ld" 2> "$OUT/${KIND}.embed.err"
echo "# embed $(tr '\n' ' ' < "$OUT/${KIND}.embed.err")"

"$LANDFOLD" project -D "$D" -d 2 \
  --fun-hd "asinh,$SIGMA" --fun-ld "asinh,$SIGMA" \
  --high-file "$OUT/${KIND}.lm" --low-file "$OUT/${KIND}.ld" \
  --grid 12,11,41 --refine 3 \
  < "$OUT/${KIND}.hist" > "$OUT/${KIND}.proj" 2> "$OUT/${KIND}.proj.err"
echo "# project lines=$(awk 'NF && $1 !~ /^#/ {c++} END{print c+0}' "$OUT/${KIND}.proj")"

python3 "$HERE/energy_idw.py" \
  --xy "$OUT/${KIND}.proj" \
  --energy "$OUT/book.energy" \
  --out "$OUT/${KIND}_energy_idw.png"

echo "wrote $OUT"
echo "energy field is IDW of E-E_GM; do not use landfold fes (occupancy invert)"
