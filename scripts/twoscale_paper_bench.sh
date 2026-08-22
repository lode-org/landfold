#!/usr/bin/env bash
# Two-scale (Ceriotti near, asinh far) plus midweight on hairpin and LJ38.
# Requires a release landfold (remote builder). Does not compile here.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LANDFOLD="${LANDFOLD:?set LANDFOLD to the release binary}"
LJ38="$ROOT/examples/cosmo-lj38"
SKMAP="${SKETCHMAP_PROTEIN:-$HOME/Git/Github/HaoZeke/sketchmap/examples/protein}"
OUT="${OUT:-/tmp/landfold-twoscale}"
mkdir -p "$OUT"
PI=6.283185307179586

embed_prot() {
  local name="$1"
  shift
  echo "# protein $name"
  "$@" <"$SKMAP/lm4.30cv.w01.1" >"$OUT/protein_${name}.ld"
}

echo "# hairpin P(D,d) two-scale vs published 6,8,8"
if [ -s "$SKMAP/lm4.30cv.w01.1" ]; then
  embed_prot asinh_mw "$LANDFOLD" embed -D 30 -d 2 -w --pi "$PI" \
    --fun-hd asinh,6 --fun-ld asinh,6 --midweight --init-f --steps 40
  embed_prot ms "$LANDFOLD" embed -D 30 -d 2 -w --pi "$PI" \
    --fun-hd ms,2,5,8 --fun-ld ms,2,5,8 --init-f --steps 40
  embed_prot cer_asinh "$LANDFOLD" embed -D 30 -d 2 -w --pi "$PI" \
    --fun-hd 6,8,8 --fun-ld asinh,6 --init-f --steps 40
  embed_prot cer_asinh_mw "$LANDFOLD" embed -D 30 -d 2 -w --pi "$PI" \
    --fun-hd 6,8,8 --fun-ld asinh,6 --midweight --init-f --steps 40
  embed_prot asinh_cer "$LANDFOLD" embed -D 30 -d 2 -w --pi "$PI" \
    --fun-hd asinh,6 --fun-ld 6,2,8 --init-f --steps 40
  embed_prot asinh_cer_mw "$LANDFOLD" embed -D 30 -d 2 -w --pi "$PI" \
    --fun-hd asinh,6 --fun-ld 6,2,8 --midweight --init-f --steps 40
  embed_prot ts "$LANDFOLD" embed -D 30 -d 2 -w --pi "$PI" \
    --fun-hd ts,6,8,8 --fun-ld ts,6,8,8 --init-f --steps 40
  embed_prot ts_mw "$LANDFOLD" embed -D 30 -d 2 -w --pi "$PI" \
    --fun-hd ts,6,8,8 --fun-ld ts,6,8,8 --midweight --init-f --steps 40
fi

echo "# LJ38 published 200-landmark path"
"$LANDFOLD" embed -D 10 -d 2 -w --fun-hd ts,5,8,1 --fun-ld ts,5,8,1 \
  --midweight --init-f --preopt 40 --steps 40 \
  <"$LJ38/out/lj38.lm" >"$OUT/lj38_ts_mw.ld"

echo "# wrote $OUT"
echo "# plot: python3 $ROOT/scripts/compare_protein_twoscale_fig.py"
