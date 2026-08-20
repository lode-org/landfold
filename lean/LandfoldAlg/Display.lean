/-!
# Displayed map `(ψ, s)` and Nyström hull

The slow coordinate is a lower bound on displayed Euclidean distance.
A convex combination of landmark coordinates stays inside their range.
-/

namespace LandfoldAlg

def dispDistSq (ψi si ψj sj : Rat) : Rat :=
  (ψi - ψj) * (ψi - ψj) + (si - sj) * (si - sj)

theorem display_separates (ψi si ψj sj : Rat) :
    (ψi - ψj) * (ψi - ψj) ≤ dispDistSq ψi si ψj sj := by
  unfold dispDistSq
  grind

/-- Barycentric combination along a finite list of `(weight, coord)`. -/
def bary : List (Rat × Rat) → Rat
  | [] => 0
  | (λ, y) :: rest => λ * y + bary rest

def mass : List (Rat × Rat) → Rat
  | [] => 0
  | (λ, _) :: rest => λ + mass rest

lemma bary_sub_min (pairs : List (Rat × Rat)) (m : Rat)
    (hλ : ∀ p, p ∈ pairs → 0 ≤ p.1)
    (hy : ∀ p, p ∈ pairs → m ≤ p.2) :
    0 ≤ bary pairs - m * mass pairs := by
  induction pairs with
  | nil => simp [bary, mass]
  | cons p ps ih =>
    have hλp : 0 ≤ p.1 := hλ p (List.mem_cons_self _ _)
    have hyp : m ≤ p.2 := hy p (List.mem_cons_self _ _)
    have ih' : 0 ≤ bary ps - m * mass ps := by
      refine ih ?_ ?_
      · intro q hq; exact hλ q (List.mem_cons_of_mem _ hq)
      · intro q hq; exact hy q (List.mem_cons_of_mem _ hq)
    simp [bary, mass]
    grind

/-- Nyström / barycentric placement stays above the smallest landmark. -/
theorem nystrom_ge_min (pairs : List (Rat × Rat)) (m : Rat)
    (hλ : ∀ p, p ∈ pairs → 0 ≤ p.1)
    (hone : mass pairs = 1)
    (hy : ∀ p, p ∈ pairs → m ≤ p.2) :
    m ≤ bary pairs := by
  have h := bary_sub_min pairs m hλ hy
  grind

end LandfoldAlg
