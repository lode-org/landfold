/-!
# χ decoupling across a cut

If pair weights vanish whenever a Boolean cut `p` disagrees, Ceriotti
χ is the sum of the two block stresses. The gap-split algorithm sets
`p` from a threshold on the slow mode `ψ`.
-/

namespace LandfoldAlg

/-- One pair's contribution to χ. -/
def term {n : Nat} (w δ : Fin n → Fin n → Rat) (i j : Fin n) : Rat :=
  if i.val < j.val then w i j * δ i j * δ i j else 0

/-- All ordered pairs. -/
def pairs (n : Nat) : List (Fin n × Fin n) :=
  (List.finRange n).flatMap fun i => (List.finRange n).map fun j => (i, j)

/-- χ = Σ_{i<j} w_ij δ_ij². -/
def chi {n : Nat} (w δ : Fin n → Fin n → Rat) : Rat :=
  (pairs n).foldl (fun acc p => acc + term w δ p.1 p.2) 0

/-- Restrict weights to a block. -/
def onBlock {n : Nat} (p : Fin n → Bool) (w : Fin n → Fin n → Rat) :
    Fin n → Fin n → Rat :=
  fun i j => if p i && p j then w i j else 0

lemma foldl_add (l : List α) (f g : α → Rat) :
    l.foldl (fun acc x => acc + f x + g x) 0 =
      l.foldl (fun acc x => acc + f x) 0 +
        l.foldl (fun acc x => acc + g x) 0 := by
  suffices ∀ a b,
      l.foldl (fun acc x => acc + f x + g x) (a + b) =
        l.foldl (fun acc x => acc + f x) a +
          l.foldl (fun acc x => acc + g x) b by
    simpa using this 0 0
  intro a b
  induction l generalizing a b with
  | nil => simp
  | cons x xs ih =>
    simp
    have h : a + b + f x + g x = (a + f x) + (b + g x) := by grind
    rw [h]
    exact ih (a + f x) (b + g x)

lemma term_split {n : Nat} (w δ : Fin n → Fin n → Rat)
    (p : Fin n → Bool)
    (hcut : ∀ i j : Fin n, p i ≠ p j → w i j = 0)
    (i j : Fin n) :
    term w δ i j =
      term (onBlock p w) δ i j +
        term (onBlock (fun k => !p k) w) δ i j := by
  unfold term onBlock
  by_cases hij : i.val < j.val
  · simp [hij]
    match hpi : p i, hpj : p j with
    | true, true => simp
    | false, false => simp
    | true, false =>
      have hw := hcut i j (by simp [hpi, hpj])
      simp [hw]
    | false, true =>
      have hw := hcut i j (by simp [hpi, hpj])
      simp [hw]
  · simp [hij]

/-- A mixed pair with a vanishing weight contributes nothing. -/
theorem mixed_term_zero {n : Nat} (w δ : Fin n → Fin n → Rat)
    (p : Fin n → Bool)
    (hcut : ∀ i j : Fin n, p i ≠ p j → w i j = 0)
    (i j : Fin n) (hmix : p i ≠ p j) :
    term w δ i j = 0 := by
  unfold term
  split_ifs
  · simp [hcut i j hmix]
  · rfl

/-- χ splits into the two blocks of a cut that kills cross weights. -/
theorem chi_decouples {n : Nat} (w δ : Fin n → Fin n → Rat)
    (p : Fin n → Bool)
    (hcut : ∀ i j : Fin n, p i ≠ p j → w i j = 0) :
    chi w δ =
      chi (onBlock p w) δ +
        chi (onBlock (fun k => !p k) w) δ := by
  unfold chi
  let f : Fin n × Fin n → Rat := fun q => term (onBlock p w) δ q.1 q.2
  let g : Fin n × Fin n → Rat :=
    fun q => term (onBlock (fun k => !p k) w) δ q.1 q.2
  have step : ∀ q : Fin n × Fin n, term w δ q.1 q.2 = f q + g q :=
    fun q => term_split w δ p hcut q.1 q.2
  have hs :
      (pairs n).foldl (fun acc q => acc + term w δ q.1 q.2) 0 =
        (pairs n).foldl (fun acc q => acc + f q + g q) 0 := by
    suffices ∀ (l : List (Fin n × Fin n)) a,
        l.foldl (fun acc q => acc + term w δ q.1 q.2) a =
          l.foldl (fun acc q => acc + f q + g q) a by
      simpa using this (pairs n) 0
    intro l a
    induction l generalizing a with
    | nil => simp
    | cons q qs ih =>
      simp [step q]
      exact ih _
  rw [hs]
  exact foldl_add (pairs n) f g

end LandfoldAlg
