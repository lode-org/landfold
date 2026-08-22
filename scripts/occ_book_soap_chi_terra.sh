#!/usr/bin/env bash
# Embed 24-D SOAP landmarks and project the occupancy book.
# Required env:
#   LANDFOLD  release landfold binary
#   WORK      directory with soap.hist, soap.lm, sigma.txt
# Optional:
#   INIT      existing LD table for --init (same n as soap.lm)
#   STEPS PREOPT FUN_HD FUN_LD GRID REFINE MIDWEIGHT CENTER
set -euo pipefail
: "${LANDFOLD:?set LANDFOLD to the release binary}"
: "${WORK:?set WORK to the working directory}"
test -x "$LANDFOLD" || { echo "missing LANDFOLD=$LANDFOLD" >&2; exit 2; }
test -s "$WORK/soap.hist" || { echo "missing $WORK/soap.hist" >&2; exit 2; }
test -s "$WORK/soap.lm" || { echo "missing $WORK/soap.lm" >&2; exit 2; }
test -s "$WORK/sigma.txt" || { echo "missing $WORK/sigma.txt" >&2; exit 2; }

SIGMA=$(tr -d '[:space:]' < "$WORK/sigma.txt")
D=$(awk 'NF && $1 !~ /^#/ {print NF; exit}' "$WORK/soap.hist")
NLM=$(awk 'NF && $1 !~ /^#/ {c++} END{print c+0}' "$WORK/soap.lm")
NALL=$(awk 'NF && $1 !~ /^#/ {c++} END{print c+0}' "$WORK/soap.hist")
STEPS="${STEPS:-80}"
PREOPT="${PREOPT:-40}"
FUN_HD="${FUN_HD:-asinh,$SIGMA}"
FUN_LD="${FUN_LD:-identity}"
GRID="${GRID:-12,11,41}"
REFINE="${REFINE:-3}"
echo "# soap_chi D=$D nland=$NLM nall=$NALL sigma=$SIGMA hd=$FUN_HD ld=$FUN_LD steps=$STEPS preopt=$PREOPT"

EMBED=(
  "$LANDFOLD" embed -D "$D" -d 2
  --fun-hd "$FUN_HD" --fun-ld "$FUN_LD"
  --init-f --preopt "$PREOPT" --steps "$STEPS"
)
if [ "${CENTER:-1}" != "0" ]; then
  EMBED+=(--center)
fi
if [ "${MIDWEIGHT:-0}" = "1" ]; then
  EMBED+=(--midweight)
fi
if [ -n "${INIT:-}" ] && [ -s "$INIT" ]; then
  EMBED+=(--init "$INIT")
  echo "# init $INIT"
fi
if [ -n "${STRETCH:-}" ] && [ -s "$STRETCH" ]; then
  EMBED+=(--stretch "$STRETCH")
  echo "# stretch $STRETCH"
fi
if [ -n "${ALPHA:-}" ]; then
  EMBED+=(--alpha "$ALPHA")
fi
"${EMBED[@]}" < "$WORK/soap.lm" > "$WORK/soap.ld" 2> "$WORK/soap.err"
echo "# embed $(tr '\n' ' ' < "$WORK/soap.err")"

if [ "$NLM" -ge "$NALL" ]; then
  cp -f "$WORK/soap.ld" "$WORK/soap.proj"
  echo "# project skipped (all points embedded)" > "$WORK/soap.proj.err"
  echo "# project lines=$NLM (embed=all)"
  exit 0
fi

"$LANDFOLD" project -D "$D" -d 2 \
  --fun-hd "$FUN_HD" --fun-ld "$FUN_LD" \
  --high-file "$WORK/soap.lm" --low-file "$WORK/soap.ld" \
  --grid "$GRID" --refine "$REFINE" \
  < "$WORK/soap.hist" > "$WORK/soap.proj" 2> "$WORK/soap.proj.err"
echo "# project lines=$(awk 'NF && $1 !~ /^#/ {c++} END{print c+0}' "$WORK/soap.proj")"
