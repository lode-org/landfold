/-!
# Displayed map `(ψ, s)`

The extra term in the displayed squared distance is `(Δs)²`.
-/

namespace LandfoldAlg

def dispDistSq (psi si psj sj : Rat) : Rat :=
  (psi - psj) * (psi - psj) + (si - sj) * (si - sj)

/-- Displayed squared distance is the ψ gap plus a square in `s`. -/
theorem display_decomp (psi si psj sj : Rat) :
    dispDistSq psi si psj sj =
      (psi - psj) * (psi - psj) + (si - sj) * (si - sj) := rfl

/-- The ψ gap is the displayed distance minus a square. -/
theorem display_minus_s (psi si psj sj : Rat) :
    dispDistSq psi si psj sj - (si - sj) * (si - sj) =
      (psi - psj) * (psi - psj) := by
  unfold dispDistSq
  grind

end LandfoldAlg
