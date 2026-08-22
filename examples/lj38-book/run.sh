#!/usr/bin/env bash
# Quenched minima book -> SOAP or sorted-pair HD -> landfold asinh -> energy IDW.
# KIND=sheap is pair HD then the SHEAP dual-funnel energy IDW (no landfold).
# KIND=pair also runs SHEAP after the pair plane exists.
# Requires a release landfold for soap/pair. Does not compile here.
# Required env:
#   MIN       .min book, one row per structure: E then 3N coordinates
#   LANDFOLD  release landfold binary (soap/pair only)
# Optional:
#   KIND      soap (default), pair, or sheap
#   ENERGY    one energy per row (default $OUT/book.energy)
#   OUT       working directory (default ./out)
#   NATOMS    atoms per structure (default 38)
#   NLAND     Gonzalez landmarks (default 200)
#   WORKERS   SOAP fingerprint processes (default 1)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MIN="${MIN:?set MIN to the .min book (E + 3N per row)}"
KIND="${KIND:-soap}"
OUT="${OUT:-$HERE/out}"
NATOMS="${NATOMS:-38}"
NLAND="${NLAND:-200}"
WORKERS="${WORKERS:-1}"
test -s "$MIN" || { echo "missing MIN=$MIN" >&2; exit 2; }
case "$KIND" in
  soap|pair|sheap) ;;
  *) echo "KIND=$KIND must be soap, pair, or sheap" >&2; exit 2 ;;
esac
mkdir -p "$OUT"

run_pair_hd() {
  python3 "$HERE/min_to_hist.py" \
    --min "$MIN" --kind pair --n-atoms "$NATOMS" \
    --out "$OUT/pair.hist" \
    --energy "$OUT/book.energy" \
    --sigma "$OUT/sigma.txt" \
    --dest "$OUT" \
    --workers "$WORKERS"
}

run_sheap() {
  local energy="${ENERGY:-$OUT/book.energy}"
  local args=(
    --min "$MIN"
    --energy "$energy"
    --dest "$OUT"
    --out "$OUT/sheap_idw.png"
    --n-atoms "$NATOMS"
  )
  if [[ -s "$OUT/pair.hist" ]]; then
    args+=(--pair-hist "$OUT/pair.hist")
  fi
  MIN="$MIN" ENERGY="$energy" NATOMS="$NATOMS" SHEAP_DEST="$OUT" SHEAP_OUT="$OUT" \
    SHEAP_FIG="$OUT/sheap_idw.png" \
    python3 "$HERE/sheap.py" "${args[@]}"
}

if [[ "$KIND" == "sheap" ]]; then
  run_pair_hd
  run_sheap
  echo "wrote $OUT"
  echo "energy field is IDW of E-E_GM; do not use landfold fes (occupancy invert)"
  exit 0
fi

LANDFOLD="${LANDFOLD:?set LANDFOLD to the release binary}"
test -x "$LANDFOLD" || { echo "LANDFOLD=$LANDFOLD is not executable" >&2; exit 2; }

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

if [[ "$KIND" == "pair" ]]; then
  run_sheap
fi

echo "wrote $OUT"
echo "energy field is IDW of E-E_GM; do not use landfold fes (occupancy invert)"
