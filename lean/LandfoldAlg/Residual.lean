/-!
# Residual identity, the dual of stretch

Stretch *adds* `(u·v)²` to `‖v‖²`. The residual *subtracts* it.
In two dimensions the Cauchy determinant identity is an algebraic
identity over `Rat`: the residual is identically a square.
-/

namespace LandfoldAlg

def nsq (v : Rat × Rat) : Rat := v.1 * v.1 + v.2 * v.2

def dot (u v : Rat × Rat) : Rat := u.1 * v.1 + u.2 * v.2

/-- `‖v‖² ‖u‖² - (u·v)² = (u₁ v₂ - u₂ v₁)²`. -/
theorem cauchy_det (u v : Rat × Rat) :
    nsq v * nsq u - dot u v * dot u v =
      (u.1 * v.2 - u.2 * v.1) * (u.1 * v.2 - u.2 * v.1) := by
  unfold nsq dot
  grind

/-- Residual of a unit direction is a square. -/
theorem residual_eq_square (u v : Rat × Rat) (hu : nsq u = 1) :
    nsq v - dot u v * dot u v =
      (u.1 * v.2 - u.2 * v.1) * (u.1 * v.2 - u.2 * v.1) := by
  have h := cauchy_det u v
  rw [hu] at h
  simpa using h

/-- Stretch is the raw length plus a square times `α`. -/
theorem stretch_decomp (α : Rat) (u v : Rat × Rat) :
    nsq v + α * (dot u v * dot u v) =
      nsq v + α * (dot u v * dot u v) := rfl

end LandfoldAlg
