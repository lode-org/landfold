/-!
# χ decoupling across a cut

A mixed pair of a cut that kills cross weights contributes nothing
to Ceriotti χ. The gap-split algorithm uses that to drop those pairs.
-/

namespace LandfoldAlg

/-- One pair's contribution to χ. -/
def term {n : Nat} (w δ : Fin n → Fin n → Rat) (i j : Fin n) : Rat :=
  if i.val < j.val then w i j * δ i j * δ i j else 0

/-- Restrict weights to a block. -/
def onBlock {n : Nat} (p : Fin n → Bool) (w : Fin n → Fin n → Rat) :
    Fin n → Fin n → Rat :=
  fun i j => if p i && p j then w i j else 0

/-- A mixed pair with a vanishing weight contributes nothing. -/
theorem chi_decouples {n : Nat} (w δ : Fin n → Fin n → Rat)
    (p : Fin n → Bool)
    (hcut : ∀ i j : Fin n, p i ≠ p j → w i j = 0)
    (i j : Fin n) (hmix : p i ≠ p j) :
    term w δ i j = 0 := by
  unfold term
  split
  · rw [hcut i j hmix]
    simp
  · rfl

end LandfoldAlg
