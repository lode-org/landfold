#!/usr/bin/env bash
# Embed 24-D SOAP landmarks and project the occupancy book.
# Required env:
#   LANDFOLD  release landfold binary
#   WORK      directory with soap.hist, soap.lm, sigma.txt
# Optional:
#   INIT      existing LD table for --init (same n as soap.lm)
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
echo "# soap_chi D=$D nland=$NLM sigma=$SIGMA"

EMBED=(
  "$LANDFOLD" embed -D "$D" -d 2
  --fun-hd "asinh,$SIGMA" --fun-ld "asinh,$SIGMA"
  --init-f --preopt 40 --steps 40 --center
)
if [ -n "${INIT:-}" ] && [ -s "$INIT" ]; then
  EMBED+=(--init "$INIT")
fi
"${EMBED[@]}" < "$WORK/soap.lm" > "$WORK/soap.ld" 2> "$WORK/soap.err"
echo "# embed $(tr '\n' ' ' < "$WORK/soap.err")"

"$LANDFOLD" project -D "$D" -d 2 \
  --fun-hd "asinh,$SIGMA" --fun-ld "asinh,$SIGMA" \
  --high-file "$WORK/soap.lm" --low-file "$WORK/soap.ld" \
  --grid 12,11,41 --refine 3 \
  < "$WORK/soap.hist" > "$WORK/soap.proj" 2> "$WORK/soap.proj.err"
echo "# project lines=$(awk 'NF && $1 !~ /^#/ {c++} END{print c+0}' "$WORK/soap.proj")"
