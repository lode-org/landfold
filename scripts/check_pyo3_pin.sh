#!/usr/bin/env bash
# Graph check: one pyo3 major with --features python.
# Run on the remote builder. cargo tree resolves; it does not compile
# the extension. Do not invent a second pyo3.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GOLDEN="${1:-$ROOT/tests/goldens/pyo3_majors.txt}"
want_major="$(awk '$1=="pyo3"{print $2; exit}' "$GOLDEN")"
if [[ -z "$want_major" ]]; then
  echo "missing pyo3 major in $GOLDEN" >&2
  exit 1
fi

tree="$(cargo tree -e features --features python)"
mapfile -t majors < <(printf '%s\n' "$tree" | awk '
  {
    for (i = 1; i <= NF; i++) {
      if ($i == "pyo3" && $(i + 1) ~ /^v[0-9]+\.[0-9]+/) {
        v = $(i + 1)
        sub(/^v/, "", v)
        split(v, a, ".")
        print a[1] "." a[2]
      }
    }
  }' | sort -u)

if [[ ${#majors[@]} -ne 1 ]]; then
  echo "expected one pyo3 major, got: ${majors[*]:-none}" >&2
  echo "$tree" | grep -E '(^|[^[:alnum:]-])pyo3 v' || true
  exit 1
fi
if [[ "${majors[0]}" != "$want_major" ]]; then
  echo "pyo3 major ${majors[0]} != golden $want_major" >&2
  exit 1
fi

dlpk_ver="$(awk '$1=="dlpk"{print $2; exit}' "$GOLDEN")"
if ! printf '%s\n' "$tree" | grep -F "dlpk v${dlpk_ver}" >/dev/null; then
  echo "cargo tree is missing dlpk ${dlpk_ver}" >&2
  exit 1
fi
if printf '%s\n' "$tree" | grep -F 'dlpk feature "pyo3"' >/dev/null; then
  echo "dlpk ${dlpk_ver} must not enable its pyo3 feature" >&2
  exit 1
fi

echo "ok: cargo tree -e features --features python has pyo3 $want_major only"
