import LandfoldAlg.NearFar

/-!
# Three-band loss

Near pairs use identity stress. Mid pairs use a transfer. Far pairs
use `1/(1+d²)` (depends on `d`) except the diameter pins, which keep
identity. A saturating mid transfer cannot tell two far distances
apart. `satFar` is constantly 1; `repFar` is not.
-/

namespace LandfoldAlg

/-- Mid-band residual through a transfer `F`. -/
def midStress (F d : Rat) : Rat := (F - d) * (F - d)

/-- Three-band pair term. `near` / `far` take identity; otherwise mid. -/
def bandTerm (near far : Bool) (d D F : Rat) : Rat :=
  if near then idStress d D
  else if far then idStress d D
  else midStress F d

theorem band_near_is_identity (d D F : Rat) :
    bandTerm true false d D F = idStress d D := by
  unfold bandTerm
  rfl

theorem band_far_is_identity (d D F : Rat) :
    bandTerm false true d D F = idStress d D := by
  unfold bandTerm
  rfl

/-- Saturated mid residual ignores the HD distance. -/
theorem sat_mid_cannot_tell (D1 D2 d : Rat) :
    midStress (satFar D1) d = midStress (satFar D2) d := by
  unfold midStress satFar
  rfl

/-- Far-band identity still separates two HD distances. -/
theorem band_far_separates (d D1 D2 F : Rat) :
    bandTerm false true d D1 F - bandTerm false true d D2 F =
      (D2 - D1) * (2 * d - D1 - D2) := by
  unfold bandTerm
  simpa using id_stress_separates d D1 D2

/-- PaCMAP-style far repulsion. -/
def repFar (d : Rat) : Rat := 1 / (1 + d * d)

theorem sat_far_is_one (D : Rat) : satFar D = 1 := by
  unfold satFar
  rfl

/-- Unlike `satFar`, the repulsive far term still depends on `d`. -/
theorem rep_far_depends_on_d : repFar 0 ≠ repFar 1 := by
  unfold repFar
  native_decide

end LandfoldAlg
