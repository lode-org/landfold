#!/usr/bin/env python3
"""Do not draw stretch occupancy with OOS crystal stars.

ts.all is a TSE. fcc and ico are not in it. Projecting those two
xyz frames onto a TSE KDE puts the stars in empty space. That panel
is not a two-basin landscape. Use the PHATE figures instead
(docs/ceriotti-figs/lj38_phate_vs_ceriotti.png,
docs/ceriotti-figs/lj38_phate_committor.png).
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "refused: OOS fcc/ico on a TSE occupancy is not a basin figure "
        "(see scripts/compare_phate_fig.py)"
    )


if __name__ == "__main__":
    main()
