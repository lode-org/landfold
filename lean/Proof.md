# Proof

## Statement

Four identities over the rationals licence the gap-split embedding.

1. If pair weights vanish across a Boolean cut, Ceriotti χ is the sum
   of the two block stresses.
2. The displayed map `(ψ, s)` satisfies
   `‖Φ(i)-Φ(j)‖ ≥ |ψ(i)-ψ(j)|`.
3. For a unit vector `u` in the plane,
   `‖v‖² - (u·v)² = (u₁v₂ - u₂v₁)² ≥ 0`.
4. A convex combination of landmark coordinates lies above the
   smallest landmark coordinate.

## Overview

The proofs are finite algebra. No spectral theorem is used. The
slow mode `ψ` is supplied by the locally scaled walk
(Coifman-Lafon; Rohrdanz, Zheng, Maggioni, Clementi) and is an
input to the identities, not a derived object inside them.

Lean: `lean/LandfoldAlg/`. No Mathlib. Lean 4.29.

## Proof

### Decoupling

Write `t(i,j) = 1_{i<j} w(i,j) δ(i,j)²` and `χ = Σ t`.
On a pair with `p(i) ≠ p(j)` the hypothesis forces `w(i,j) = 0`,
so `t = 0` and both block terms vanish. On a pair with
`p(i) = p(j) = true` the A-block term equals `t` and the B-block
term is zero. The false-false case is symmetric. Summing over the
list of ordered pairs, and using that foldl of `f+g` is the sum of
the foldls, gives `χ(w) = χ(w_A) + χ(w_B)`.

### Display separation

`(Δψ)² + (Δs)² ≥ (Δψ)²` because `(Δs)² ≥ 0`.

### Residual seminorm

Expand `‖v‖²‖u‖² - (u·v)²` in two coordinates. The cross terms
cancel and the remainder is `(u₁v₂ - u₂v₁)²`. Set `‖u‖² = 1`.
A square over `ℚ` is nonnegative.

### Nyström hull

`bary - m · mass = Σ λ_j (y_j - m)`. Each `λ_j ≥ 0` and each
`y_j ≥ m`, so the sum is nonnegative. If `mass = 1` then
`bary ≥ m`.

## Discussion

Ceriotti χ on raw Euclidean distances mixes every contrast into one
stress (PNAS 2011). Stretch *inserts* a named axis into that metric.
The identities say the dual operation is legal: take the slow
coordinate the sample already has, kill the cross weights, and let
χ run only inside the blocks. The public LJ38 `ts.all` is a
solid-liquid TSE; the named fcc-ico axis is the wrong `u`
(identity 3). Gap-split uses the data-driven `ψ` instead.

Coifman, R. R.; Lafon, S. *Appl. Comput. Harmon. Anal.* **21**, 5
(2006), 10.1016/j.acha.2006.04.006.

Rohrdanz, M. A.; Zheng, W.; Maggioni, M.; Clementi, C. *J. Chem.
Phys.* **134**, 124116 (2011), 10.1063/1.3569857.

Ceriotti, M.; Tribello, G. A.; Parrinello, M. *Proc. Natl. Acad.
Sci. U.S.A.* **108**, 13023 (2011), 10.1073/pnas.1108486108.
