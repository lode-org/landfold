#!/usr/bin/env bash
# Run every landfold embedder on an occupancy-book n4..n13 table.
# Requires a release landfold (remote builder). Does not compile here.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LANDFOLD="${LANDFOLD:?set LANDFOLD to the release binary}"
CV="${1:?usage: occ_book_suite.sh HD.cv [label]}"
LABEL="${2:-occ}"
OUT="${OUT:-/tmp/landfold-occ-book}"
REFS="${REFS:-$ROOT/examples/cosmo-lj38/out/lj38_refs.cv}"
mkdir -p "$OUT"

n=$(awk 'NF{c++} END{print c+0}' "$CV")
# Landmarks write a trailing weight. The occupancy .cv tables do not.
# Keep one weighted table so embed/project -w matches the Ceriotti path.
WCV="$OUT/${LABEL}.wcv"
awk '{print $0, 1}' "$CV" >"$WCV"
echo "# $LABEL n=$n $(date -u +%Y-%m-%dT%H:%M:%SZ)"

run() {
  local name="$1"
  shift
  echo "# $name"
  if ! "$@" >"$OUT/${LABEL}_${name}.ld" 2>"$OUT/${LABEL}_${name}.err"; then
    echo "FAIL $name" >&2
    tail -20 "$OUT/${LABEL}_${name}.err" >&2
    return 1
  fi
  awk '/^# stress/{print}' "$OUT/${LABEL}_${name}.err" || true
}

"$LANDFOLD" landmarks -D 10 -n 200 --seed 1 -w <"$WCV" >"$OUT/${LABEL}.lm"

run ceriotti "$LANDFOLD" embed -D 10 -d 2 -w \
  --fun-hd 5,8,1 --fun-ld 5,2,2 --init-f --preopt 40 --steps 40 \
  <"$OUT/${LABEL}.lm"
"$LANDFOLD" project -D 10 -d 2 -w \
  --high-file "$OUT/${LABEL}.lm" --low-file "$OUT/${LABEL}_ceriotti.ld" \
  --fun-hd 5,8,1 --fun-ld 5,2,2 --grid 12,11,41 --refine 3 \
  <"$CV" >"$OUT/${LABEL}_ceriotti.proj"

run asinh "$LANDFOLD" embed -D 10 -d 2 -w \
  --fun-hd asinh,5 --fun-ld asinh,5 --init-f --preopt 40 --steps 40 \
  <"$OUT/${LABEL}.lm"
"$LANDFOLD" project -D 10 -d 2 -w \
  --high-file "$OUT/${LABEL}.lm" --low-file "$OUT/${LABEL}_asinh.ld" \
  --fun-hd asinh,5 --fun-ld asinh,5 --grid 12,11,41 --refine 3 \
  <"$CV" >"$OUT/${LABEL}_asinh.proj"

run phate "$LANDFOLD" embed -D 10 -d 2 -w --phate <"$WCV"
run pacmap "$LANDFOLD" embed -D 10 -d 2 -w --pacmap <"$WCV"
run nearfar "$LANDFOLD" embed -D 10 -d 2 -w --nearfar --init-f --steps 80 \
  <"$OUT/${LABEL}.lm"
"$LANDFOLD" project -D 10 -d 2 -w --nearfar \
  --high-file "$OUT/${LABEL}.lm" --low-file "$OUT/${LABEL}_nearfar.ld" \
  --knn 10 <"$CV" >"$OUT/${LABEL}_nearfar.proj"

run bands "$LANDFOLD" embed -D 10 -d 2 -w --bands --init-f --warm 0 \
  --far-weight 0.5 --steps 80 <"$OUT/${LABEL}.lm"
"$LANDFOLD" project -D 10 -d 2 -w --bands \
  --high-file "$OUT/${LABEL}.lm" --low-file "$OUT/${LABEL}_bands.ld" \
  --knn 10 <"$CV" >"$OUT/${LABEL}_bands.proj"

run gapsplit "$LANDFOLD" embed -D 10 -d 2 -w --gapsplit --init-f --steps 40 \
  <"$OUT/${LABEL}.lm"
"$LANDFOLD" project -D 10 -d 2 -w \
  --high-file "$OUT/${LABEL}.lm" --low-file "$OUT/${LABEL}_gapsplit.ld" \
  --fun-hd 5,8,1 --fun-ld 5,2,2 --grid 12,11,41 --refine 3 \
  <"$CV" >"$OUT/${LABEL}_gapsplit.proj"

if [ -s "$REFS" ]; then
  run axis "$LANDFOLD" embed -D 10 -d 2 -w --axis "$REFS" <"$WCV"
  if [ -s "$OUT/${LABEL}_ceriotti.proj" ]; then
    "$LANDFOLD" field -D 10 --low-file "$OUT/${LABEL}_ceriotti.proj" --refs "$REFS" \
      <"$CV" >"$OUT/${LABEL}_field.csv" 2>"$OUT/${LABEL}_field.err" || true
  fi
fi

echo "# wrote $OUT"
