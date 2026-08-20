#!/usr/bin/env bash
# Public Ceriotti sketch-map sets: LJ38 teaching TSE and the JCTC 2015
# beta-hairpin landmarks shipped with lab-cosmo/sketchmap.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LANDFOLD="${LANDFOLD:-$ROOT/target/release/landfold}"
if [ ! -x "$LANDFOLD" ]; then
  echo "set LANDFOLD to a release binary built with --features highs" >&2
  exit 1
fi
LJ38="$ROOT/examples/cosmo-lj38"
SKMAP="${SKETCHMAP_PROTEIN:-$HOME/Git/Github/HaoZeke/sketchmap/examples/protein}"
OUT="${OUT:-/tmp/landfold-ceriotti}"
mkdir -p "$OUT"

run_embed() {
  local label="$1"
  shift
  local t0 t1 ms
  t0=$(date +%s.%N)
  if ! "$@" >"$OUT/${label}.ld" 2>"$OUT/${label}.err"; then
    echo "FAIL $label"
    cat "$OUT/${label}.err" >&2
    return 1
  fi
  t1=$(date +%s.%N)
  ms=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", (b-a)*1000}')
  local stress
  stress=$(awk '/^# stress/{print $3}' "$OUT/${label}.err")
  printf '%-28s stress=%-14s ms=%s\n' "$label" "$stress" "$ms"
}

echo "# $(basename "$LANDFOLD")  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "# LJ38 10-D TSE landmarks (Ceriotti course / JCTC 10.1021/ct3010563 params)"
if [ ! -s "$LJ38/ts.cv" ]; then
  awk '{for(i=3;i<=12;i++) printf "%s ", $i; print ""}' "$LJ38/ts.all" >"$LJ38/ts.cv"
fi
"$LANDFOLD" landmarks -D 10 -n 200 --seed 1 <"$LJ38/ts.cv" >"$OUT/lj38.lm"
for arm in standard lbfgs highs; do
  extra=()
  [ "$arm" = lbfgs ] && extra=(--lbfgs)
  [ "$arm" = highs ] && extra=(--highs)
  run_embed "lj38-$arm" "$LANDFOLD" embed -D 10 -d 2 -w \
    --fun-hd 5,8,1 --fun-ld 5,2,2 --steps 40 "${extra[@]}" \
    <"$OUT/lj38.lm"
done

echo "# beta-hairpin 30-D landmarks (Ardevol et al. JCTC 2015 10.1021/ct500950z)"
if [ ! -s "$SKMAP/lm4.30cv.w01.1" ]; then
  echo "missing $SKMAP/lm4.30cv.w01.1 (set SKETCHMAP_PROTEIN)" >&2
  exit 1
fi
# published map as a zero-step baseline (chi of the C++ sketch-map)
run_embed "protein-published" "$LANDFOLD" embed -D 30 -d 2 -w --pi 6.283185307179586 \
  --fun-hd 6,8,8 --fun-ld 6,2,8 --steps 0 \
  --init "$SKMAP/lm4.30cv.w01.1.proj" \
  <"$SKMAP/lm4.30cv.w01.1"
for arm in standard lbfgs highs; do
  extra=()
  [ "$arm" = lbfgs ] && extra=(--lbfgs)
  [ "$arm" = highs ] && extra=(--highs)
  run_embed "protein-$arm" "$LANDFOLD" embed -D 30 -d 2 -w --pi 6.283185307179586 \
    --fun-hd 6,8,8 --fun-ld 6,2,8 --steps 40 "${extra[@]}" \
    <"$SKMAP/lm4.30cv.w01.1"
done
# Same landmarks, polish the published C++ map. Highs trust matches the
# published span (~50), not the 0.5 default from the two-well toy.
for arm in standard lbfgs highs; do
  extra=()
  [ "$arm" = lbfgs ] && extra=(--lbfgs)
  [ "$arm" = highs ] && extra=(--highs --trust 20)
  run_embed "protein-polish-$arm" "$LANDFOLD" embed -D 30 -d 2 -w --pi 6.283185307179586 \
    --fun-hd 6,8,8 --fun-ld 6,2,8 --steps 40 --init "$SKMAP/lm4.30cv.w01.1.proj" \
    "${extra[@]}" \
    <"$SKMAP/lm4.30cv.w01.1"
done
echo "# wrote $OUT"
