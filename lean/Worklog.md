# Worklog

- Statement: four elementary identities that licence gap-split χ.
- Hand proofs: A1 (pair split) complete. Cheeger blocked on purpose.
- Lean: `lake build` on Lean 4.29, no Mathlib, no sorry.
  `chi_decouples`, `cauchy_det`, `residual_eq_square`, `display_minus_s`.
- Near/far embedder licence: `LandfoldAlg/NearFar.lean` + `NearFar.md`.
  `w1_of_unit`, `countLE_strict`, `neighbor_preserved`, `split_term`,
  `farPreserved_ignores_unmasked`.
- Rust: `landfold embed --gapsplit`. LJ38: fcc/ico share ψ
  (Δψ = 0.014) and split on s (Δs = 7.69). Stars sit in the density.
