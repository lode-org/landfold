# Hand proofs

## Approach registry

- A1. Direct pair split of the χ summand. Used.
- A2. Spectral Cheeger bound for ψ₂. Blocked: needs the Dirichlet form
  and a continuous manifold. Not required for the algorithm licence.
- A3. Wasserstein-1 on the CN histogram. Discrete W1 on an ordered
  line is the weighted L1 of prefix masses; unit bins reduce that to
  L1 of the CDFs. Lean `w1_of_unit` / hand proof `lean/NearFar.md`.

## 1. Decoupling

Write `t(i,j) = 1_{i<j} w(i,j) δ(i,j)²`, so `χ = Σ_{i,j} t(i,j)`.
For each ordered pair, either `p(i)=p(j)` or not.

- If `p(i) ≠ p(j)`, the hypothesis gives `w(i,j)=0`, hence `t(i,j)=0`.
  The restricted weights `w_A` and `w_B` also vanish on that pair, so
  both restricted terms are 0.
- If `p(i)=p(j)=true`, then `w_A(i,j)=w(i,j)` and `w_B(i,j)=0`, so
  `t = t_A + t_B`.
- If `p(i)=p(j)=false`, then `t = t_B` and `t_A = 0`.

Summing over all pairs gives the identity. No analysis is used.

## 2. Display separation

`(ψ_i-ψ_j)² + (s_i-s_j)² ≥ (ψ_i-ψ_j)²` because the second square is
nonnegative. Take square roots (both sides nonnegative).

## 3. Residual seminorm

Expand
`‖v‖² ‖u‖² - (u·v)² = (u₁ v₂ - u₂ v₁)²`
in two dimensions (the Cauchy determinant identity). Set `‖u‖² = 1`.

Equality: `u₁ v₂ = u₂ v₁`. If `u ≠ 0` this is `v ∥ u` over `ℚ`
whenever `u` has a nonzero coordinate (clear the other by the product).

## 4. Nyström hull

`y - min y_k = Σ λ_j (y_j - min y_k) ≥ 0` because each factor is
nonnegative. The max bound is the same with signs flipped.

## Algorithm

Rohrdanz / Coifman supply `ψ` as the Fiedler function of the locally
scaled walk. Theorem 1 says Ceriotti `χ` with `w = 1_{|Δψ|≤τ}` is
exactly χ on the level sets of `ψ`. Theorem 2 says the displayed
`(ψ, s)` still separates those level sets by at least the `ψ` gap.
Theorem 3 says this is the dual of stretch: stretch *inserts* a named
axis, residual / gap-split *removes* the axis the data already have.
Theorem 4 is the OOS placement used on new frames.
