/-!
# Exact contrast axis

The signed projection onto `b - a` is linear. The difference of
projections is the high-D contrast itself, so two distinct refs
cannot collapse. A saturating `F` (see `NearFar`) cannot say this.
-/

import LandfoldAlg.Residual

namespace LandfoldAlg

def sub (u v : Rat × Rat) : Rat × Rat := (u.1 - v.1, u.2 - v.2)

/-- Unnormalised projection of `x` onto the `a → b` contrast. -/
def axisProj (a b x : Rat × Rat) : Rat :=
  dot (sub x a) (sub b a)

theorem axis_at_a (a b : Rat × Rat) : axisProj a b a = 0 := by
  unfold axisProj sub dot
  grind

theorem axis_at_b (a b : Rat × Rat) :
    axisProj a b b = nsq (sub b a) := by
  unfold axisProj sub nsq dot
  grind

/-- `p(x) - p(y) = (x - y) · (b - a)`. -/
theorem axis_diff (a b x y : Rat × Rat) :
    axisProj a b x - axisProj a b y = dot (sub x y) (sub b a) := by
  unfold axisProj sub dot
  grind

/-- Distinct refs give a nonzero gap on the axis. -/
theorem axis_separates (a b : Rat × Rat) (h : nsq (sub b a) ≠ 0) :
    axisProj a b b - axisProj a b a ≠ 0 := by
  have hb := axis_at_b a b
  have ha := axis_at_a a b
  simpa [hb, ha] using h

end LandfoldAlg
