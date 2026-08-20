/-!
# Residual seminorm, the dual of stretch

Stretch *adds* `(u·v)²` to `‖v‖²`. The residual *subtracts* it.
In two dimensions the Cauchy determinant identity is an algebraic
identity over `Rat`, so the residual is a square.
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
  grind

/-- Residual is nonnegative. -/
theorem residual_nonneg (u v : Rat × Rat) (hu : nsq u = 1) :
    0 ≤ nsq v - dot u v * dot u v := by
  have h := residual_eq_square u v hu
  grind

/-- Stretch weight `α ≥ 0` cannot decrease the squared length. -/
theorem stretch_ge (α : Rat) (hα : 0 ≤ α) (u v : Rat × Rat) :
    nsq v ≤ nsq v + α * (dot u v * dot u v) := by
  grind

end LandfoldAlg
