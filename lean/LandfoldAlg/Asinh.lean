import LandfoldAlg.NearFar

/-!
# Unbounded vs saturating transfer

A strictly increasing `F` sends distinct distances to distinct
values. `satFar` does not. `asinh` is the analytic stand-in for
that strict increase (the Lean `F` here is any monotone map).
-/

namespace LandfoldAlg

/-- Strictly increasing maps separate their arguments. -/
theorem mono_separates (F : Rat → Rat)
    (hF : ∀ a b : Rat, a < b → F a < F b)
    (D1 D2 : Rat) (h : D1 < D2) :
    F D1 ≠ F D2 :=
  ne_of_lt (hF D1 D2 h)

/-- Identity is strictly increasing. -/
theorem id_strict (a b : Rat) (h : a < b) : a < b := h

theorem id_separates (D1 D2 : Rat) (h : D1 < D2) : D1 ≠ D2 :=
  ne_of_lt h

/-- `satFar` is not strictly increasing. -/
theorem sat_far_not_strict :
    ¬ (∀ a b : Rat, a < b → satFar a < satFar b) := by
  intro h
  have := h 0 1 (by decide)
  unfold satFar at this
  exact lt_irrefl (1 : Rat) this

end LandfoldAlg
