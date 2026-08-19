#!/usr/bin/env bash
# LJ38 teaching map: landmarks -> embed -> project -> FES -> xyzrender.
# Exercise 5 parameters: fun-hd 5,8,1  fun-ld 5,2,2  kT=0.168
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
LANDFOLD="${LANDFOLD:-}"
if [ -z "$LANDFOLD" ]; then
  if [ -x "$ROOT/../../target/release/landfold" ]; then
    LANDFOLD="$ROOT/../../target/release/landfold"
  elif command -v landfold >/dev/null; then
    LANDFOLD="$(command -v landfold)"
  else
    echo "landfold binary not found; set LANDFOLD=" >&2
    exit 1
  fi
elif [ ! -x "$LANDFOLD" ]; then
  # relative to the caller's cwd
  if [ -x "$OLDPWD/$LANDFOLD" ]; then
    LANDFOLD="$OLDPWD/$LANDFOLD"
  else
    echo "LANDFOLD=$LANDFOLD is not executable" >&2
    exit 1
  fi
fi
echo "using $LANDFOLD"

mkdir -p out
# 10-D CVs if missing
if [ ! -s ts.cv ]; then
  awk '{for(i=3;i<=12;i++) printf "%s ", $i; print ""}' ts.all > ts.cv
fi

# 200 landmarks (exercise used 500 on a longer out.all)
"$LANDFOLD" landmarks -D 10 -n 200 --seed 1 < ts.cv > out/lj38.lm

"$LANDFOLD" embed -D 10 -d 2 --fun-hd 5,8,1 --fun-ld 5,2,2 --steps 40 \
  < out/lj38.lm > out/lj38.ld

"$LANDFOLD" project -D 10 -d 2 \
  --high-file out/lj38.lm --low-file out/lj38.ld \
  --fun-hd 5,8,1 --fun-ld 5,2,2 \
  --grid 12,11,41 --refine 3 \
  < ts.cv > out/ts.proj

"$LANDFOLD" fes --input out/ts.proj --nx 80 --ny 80 --kt 0.168 \
  --csv out/fes.csv --svg out/fes.svg

# xyzrender wants real element symbols
for f in lj38_fcc.xyz lj38_ico.xyz; do
  [ -f "$f" ] && sed -i 's/^X /Ar /' "$f"
done

if command -v xyzrender >/dev/null; then
  for f in lj38_fcc.xyz lj38_ico.xyz lj38.17.xyz lj38.19.xyz lj38.xyz; do
    [ -f "$f" ] || continue
    xyzrender -t --orient -S 480 -a 0.95 --mol-color '#2a6f97' \
      -o "out/${f%.xyz}.png" "$f"
  done
fi

echo "wrote $ROOT/out"
