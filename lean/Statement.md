# Statement

Objects.

- A finite index set `I = {0, ..., n-1}`.
- Pair weights `w : I × I → ℚ`, `w ≥ 0`, `w(i,j) = w(j,i)`, `w(i,i) = 0`.
- Pair residuals `δ : I × I → ℚ` (in the Ceriotti stress these are `F(D_ij) - f(d_ij)`).
- The pairwise stress
  `χ(w,δ) = Σ_{i < j} w(i,j) δ(i,j)²`.
- A cut `p : I → Bool`. Write `A = p⁻¹(true)`, `B = p⁻¹(false)`.
- A slow coordinate `ψ : I → ℚ`.
- A displayed map `Φ(i) = (ψ(i), s(i))` into `ℚ × ℚ`.

Theorems.

1. Decoupling. If `w(i,j) = 0` whenever `p(i) ≠ p(j)`, then
   `χ(w,δ) = χ(w|_A, δ) + χ(w|_B, δ)`.
   Cross pairs do not enter the Ceriotti objective.

2. Display decomposition. For the Euclidean product
   `‖Φ(i) - Φ(j)‖² = (ψ(i)-ψ(j))² + (s(i)-s(j))²`.
   The second term is a square in `s`. The slow coordinate is therefore
   a lower bound on displayed distance, independently of `s`.

3. Residual identity (two dimensions).
   `‖v‖² ‖u‖² - (u·v)² = (u₁ v₂ - u₂ v₁)²`.
   If `‖u‖² = 1` the residual is that square. Stretch *adds*
   `(u·v)²`; the residual *removes* it. The named fcc-ico axis is a
   choice of `u`. The slow mode `ψ` is the data-driven choice.

These four statements are the licence for the gap-split embedding:
take `ψ` to be the second eigenfunction of a locally scaled diffusion
operator (Coifman-Lafon, Rohrdanz), set `w(i,j) = 0` when
`|ψ(i)-ψ(j)|` exceeds a cut, minimise Ceriotti `χ` on the remaining
pairs to obtain `s`, and display `Φ = (ψ, s)`.
