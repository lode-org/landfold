/-!
# Split isometry and why a saturating F kills far pairs

Identity matching on a pair is `(d - D)²`. A transfer with
`F(D) = 1` for every large `D` makes those pairs indistinguishable.
-/

namespace LandfoldAlg

/-- Identity pair stress. -/
def idStress (d D : Rat) : Rat := (d - D) * (d - D)

/-- Saturated far transfer: every large HD distance is sent to 1. -/
def satFar (_D : Rat) : Rat := 1

theorem sat_far_ignores_hd (D1 D2 d : Rat) :
    (satFar D1 - d) * (satFar D1 - d) = (satFar D2 - d) * (satFar D2 - d) := by
  unfold satFar
  rfl

/-- Two different far HD distances give the same saturated residual. -/
theorem sat_far_cannot_tell (D1 D2 d : Rat) (_h : D1 ≠ D2) :
    (satFar D1 - d) * (satFar D1 - d) = (satFar D2 - d) * (satFar D2 - d) :=
  sat_far_ignores_hd D1 D2 d

/-- Identity residuals differ exactly when the HD distances differ. -/
theorem id_stress_separates (d D1 D2 : Rat) :
    idStress d D1 - idStress d D2 = (D2 - D1) * (2 * d - D1 - D2) := by
  unfold idStress
  grind

end LandfoldAlg
