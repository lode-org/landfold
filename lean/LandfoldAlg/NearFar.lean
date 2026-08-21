/-!
# Near / far identities for a landfold embedder

Three facts licence a split-loss embedder.

1. Discrete \(W_1\) on an ordered line is the weighted \(\ell^1\) of
   prefix masses. Unit-spaced bins reduce that to \(\ell^1\) of the CDFs.
2. Independent empirical-CDF transforms preserve each axis rank order,
   so one-dimensional neighbour ranks are invariant.
3. Split loss \(L = L_{\mathrm{near}}(\mathrm{kNN}) + L_{\mathrm{far}}(S)\).
   A \((1+\varepsilon)\)-JL map on \(S\) need not preserve near pairs;
   those sit in the \(k\)-NN term.

No Mathlib. Lean 4.29 Init + grind. The Kantorovich and JL existence
arguments are the hand proofs in `lean/NearFar.md`.
-/

namespace LandfoldAlg

/-! ## Discrete \(W_1\) on an ordered line -/

/-- Prefix mass `pref a k = a_0 + ... + a_{k-1}`. `pref a 0 = 0`. -/
def pref {m : Nat} (a : Fin m → Rat) : Nat → Rat
  | 0 => 0
  | k + 1 => pref a k + if h : k < m then a ⟨k, h⟩ else 0

/-- `f 0 + ... + f (N-1)`. -/
def sumTo (N : Nat) (f : Nat → Rat) : Rat :=
  match N with
  | 0 => 0
  | N + 1 => sumTo N f + f N

/-- Discrete 1-Wasserstein on support `x`:
    \(W_1(a,b)=\sum_{k=0}^{m-2}|A_{k+1}-B_{k+1}|\,(x_{k+1}-x_k)\),
    where \(A_{k+1}=\mathrm{pref}\,a\,(k+1)\). -/
def w1Line {m : Nat} (x a b : Fin m → Rat) : Rat :=
  sumTo m.pred fun k =>
    if h0 : k < m then
      if h1 : k + 1 < m then
        (pref a (k + 1) - pref b (k + 1)).abs * (x ⟨k + 1, h1⟩ - x ⟨k, h0⟩)
      else 0
    else 0

/-- \(\ell^1\) of the first `m-1` CDF (prefix) values. -/
def l1Cdf {m : Nat} (a b : Fin m → Rat) : Rat :=
  sumTo m.pred fun k => (pref a (k + 1) - pref b (k + 1)).abs

def unitSpaced {m : Nat} (x : Fin m → Rat) : Prop :=
  ∀ (k : Nat) (h0 : k < m) (h1 : k + 1 < m),
    x ⟨k + 1, h1⟩ - x ⟨k, h0⟩ = 1

theorem sumTo_succ (N : Nat) (f : Nat → Rat) :
    sumTo (N + 1) f = sumTo N f + f N := rfl

theorem sumTo_congr (N : Nat) (f g : Nat → Rat)
    (h : ∀ k, k < N → f k = g k) :
    sumTo N f = sumTo N g := by
  induction N with
  | zero => rfl
  | succ N ih =>
    rw [sumTo_succ, sumTo_succ, ih (fun k hk => h k (Nat.lt_succ_of_lt hk)),
        h N (Nat.lt_succ_self N)]

theorem sumTo_zero_of (N : Nat) (f : Nat → Rat) (h : ∀ k, k < N → f k = 0) :
    sumTo N f = 0 := by
  induction N with
  | zero => rfl
  | succ N ih =>
    rw [sumTo_succ, ih (fun k hk => h k (Nat.lt_succ_of_lt hk)),
        h N (Nat.lt_succ_self N)]
    grind

/-- Unit-spaced bins: \(W_1\) is \(\ell^1\) of the CDFs. -/
theorem w1_of_unit {m : Nat} (x a b : Fin m → Rat) (hu : unitSpaced x) :
    w1Line x a b = l1Cdf a b := by
  unfold w1Line l1Cdf
  apply sumTo_congr
  intro k hk
  have h1 : k + 1 < m := Nat.succ_lt_of_lt_pred hk
  have h0 : k < m := Nat.lt_of_succ_lt h1
  simp [h0, h1, hu k h0 h1]

theorem w1_self {m : Nat} (x a : Fin m → Rat) : w1Line x a a = 0 := by
  unfold w1Line
  apply sumTo_zero_of
  intro k _hk
  split
  · split
    · rw [Rat.sub_self, Rat.abs_of_nonneg Rat.le_refl]
      grind
    · rfl
  · rfl

theorem w1_comm {m : Nat} (x a b : Fin m → Rat) :
    w1Line x a b = w1Line x b a := by
  unfold w1Line
  apply sumTo_congr
  intro k _hk
  split
  · split
    · rw [Rat.abs_sub_comm]
    · rfl
  · rfl

theorem pref_one {m : Nat} (a : Fin m → Rat) (h : 0 < m) :
    pref a 1 = a ⟨0, h⟩ := by
  simp [pref, h, Rat.zero_add]

/-- Two-point line: \(W_1=|a_0-b_0|\,(x_1-x_0)\). -/
theorem w1_two (x a b : Fin 2 → Rat) :
    w1Line x a b =
      (a ⟨0, by decide⟩ - b ⟨0, by decide⟩).abs *
        (x ⟨1, by decide⟩ - x ⟨0, by decide⟩) := by
  unfold w1Line
  simp [sumTo, pref, Rat.zero_add]

/-- Equal total mass on two bins: the two endpoint gaps match. -/
theorem two_mass_swap (a0 a1 b0 b1 : Rat) (h : a0 + a1 = b0 + b1) :
    (a0 - b0).abs = (a1 - b1).abs := by
  have : a0 - b0 = b1 - a1 := by
    grind
  rw [this, Rat.abs_sub_comm]

/-! ## Independent empirical-CDF rank maps -/

/-- Rank of value `t` among `y`: \(\#\{i:y_i\le t\}\). -/
def countLE {n : Nat} (y : Fin n → Rat) (t : Rat) : Nat :=
  (List.ofFn y).countP (fun z => decide (z ≤ t))

/-- Empirical CDF at a sample: \(u_i=F(y_i)=\mathrm{rank}(y_i)/n\). -/
def ecdf {n : Nat} (y : Fin n → Rat) (i : Fin n) : Rat :=
  (countLE y (y i) : Rat) / (n : Rat)

/-- Column `h` of an \(n\times d\) sample. -/
def col {n d : Nat} (Y : Fin n → Fin d → Rat) (h : Fin d) : Fin n → Rat :=
  fun i => Y i h

theorem countLE_mono {n : Nat} (y : Fin n → Rat) {s t : Rat} (hst : s ≤ t) :
    countLE y s ≤ countLE y t := by
  unfold countLE
  apply List.countP_mono_left
  intro z _ hz
  simp only [decide_eq_true_iff] at hz ⊢
  exact Rat.le_trans hz hst

theorem countP_lt_of_mono_and_witness {α : Type} (p q : α → Bool) :
    ∀ (l : List α),
      (∀ x ∈ l, p x = true → q x = true) →
      ∀ {w : α}, w ∈ l → p w = false → q w = true →
      l.countP p < l.countP q
  | [], _, w, hw, _, _ => nomatch hw
  | a :: l, himp, w, hw, hp, hq => by
    have ⟨ha, hl⟩ := List.forall_mem_cons.mp himp
    simp only [List.countP_cons]
    cases hw with
    | head =>
      rw [show p a = false from hp, show q a = true from hq]
      simp only [Bool.false_eq_true, ↓reduceIte, Nat.add_zero]
      exact Nat.lt_succ_of_le (List.countP_mono_left hl)
    | tail _ hmem =>
      cases hpA : p a
      · cases hqA : q a
        · simp only [Bool.false_eq_true, ↓reduceIte, Nat.add_zero]
          exact countP_lt_of_mono_and_witness p q l hl hmem hp hq
        · simp only [↓reduceIte]
          exact Nat.lt_succ_of_le (List.countP_mono_left hl)
      · have hqA : q a = true := ha hpA
        simp only [hqA, ↓reduceIte]
        exact Nat.succ_lt_succ
          (countP_lt_of_mono_and_witness p q l hl hmem hp hq)

/-- Strict rank: \(y_i<y_j\) forces \(\mathrm{rank}(y_i)<\mathrm{rank}(y_j)\). -/
theorem countLE_strict {n : Nat} (y : Fin n → Rat) (i j : Fin n)
    (hlt : y i < y j) :
    countLE y (y i) < countLE y (y j) := by
  unfold countLE
  refine countP_lt_of_mono_and_witness
    (fun z => decide (z ≤ y i))
    (fun z => decide (z ≤ y j))
    (List.ofFn y) ?himp (List.mem_ofFn.mpr ⟨j, rfl⟩) ?hp ?hq
  · intro z _ hz
    simp only [decide_eq_true_iff] at hz ⊢
    exact Rat.le_trans hz (Rat.le_of_lt hlt)
  · simp only [decide_eq_false_iff_not]
    exact Rat.not_le.mpr hlt
  · simp only [decide_eq_true_iff]
    exact Rat.le_refl

/-- The empirical CDF is a strictly increasing function of each sample. -/
theorem ecdf_strict {n : Nat} (y : Fin n → Rat) (i j : Fin n)
    (hn : 0 < n) (hlt : y i < y j) :
    ecdf y i < ecdf y j := by
  unfold ecdf
  have hr := countLE_strict y i j hlt
  have hnR : (0 : Rat) < (n : Rat) := Rat.natCast_pos.mpr hn
  have hne : (n : Rat) ≠ 0 := Rat.ne_of_gt hnR
  rw [Rat.div_lt_iff hnR, Rat.div_mul_cancel hne]
  exact Rat.natCast_lt_natCast.mpr hr

theorem ecdf_eq_of_eq {n : Nat} (y : Fin n → Rat) (i j : Fin n)
    (he : y i = y j) :
    ecdf y i = ecdf y j := by
  unfold ecdf
  rw [he]

/-- Successor along a 1-D axis: nothing of `y` sits strictly between. -/
def axisNeighbor {n : Nat} (y : Fin n → Rat) (i j : Fin n) : Prop :=
  y i < y j ∧ ∀ k : Fin n, ¬ (y i < y k ∧ y k < y j)

/-- Neighbour ranks along an axis survive the rank map. -/
theorem neighbor_preserved {n : Nat} (y : Fin n → Rat) (i j : Fin n)
    (h : axisNeighbor y i j) :
    axisNeighbor (fun k => (countLE y (y k) : Rat)) i j := by
  rcases h with ⟨hij, hempty⟩
  refine ⟨Rat.natCast_lt_natCast.mpr (countLE_strict y i j hij), ?_⟩
  intro k ⟨hki, hkj⟩
  have rki : countLE y (y i) < countLE y (y k) :=
    Rat.natCast_lt_natCast.mp hki
  have rkj : countLE y (y k) < countLE y (y j) :=
    Rat.natCast_lt_natCast.mp hkj
  have hyik : y i < y k := by
    refine Rat.not_le.mp ?_
    intro hle
    exact Nat.not_lt.mpr (countLE_mono y hle) rki
  have hykj : y k < y j := by
    refine Rat.not_le.mp ?_
    intro hle
    exact Nat.not_lt.mpr (countLE_mono y hle) rkj
  exact hempty k ⟨hyik, hykj⟩

/-- Rank on axis `h` ignores every other coordinate. -/
theorem col_rank_independent {n d : Nat}
    (Y Z : Fin n → Fin d → Rat) (h : Fin d)
    (hsame : ∀ i, Y i h = Z i h) (i : Fin n) :
    countLE (col Y h) (Y i h) = countLE (col Z h) (Z i h) := by
  have hc : col Y h = col Z h := funext hsame
  simp [hc, hsame i]

/-! ## Split loss and far-pair JL -/

/-- One ordered pair's contribution to a squared residual stress. -/
def pairTerm {n : Nat} (w δ : Fin n → Fin n → Rat) (i j : Fin n) : Rat :=
  if i.val < j.val then w i j * δ i j * δ i j else 0

def onMask {n : Nat} (mask : Fin n → Fin n → Bool) (w : Fin n → Fin n → Rat) :
    Fin n → Fin n → Rat :=
  fun i j => if mask i j then w i j else 0

/-- Disjoint near / far masks: the pair term splits with no double count. -/
theorem split_term {n : Nat}
    (near far : Fin n → Fin n → Bool)
    (hdisj : ∀ i j, ¬ (near i j = true ∧ far i j = true))
    (w δ : Fin n → Fin n → Rat) (i j : Fin n) :
    pairTerm (onMask (fun i j => near i j || far i j) w) δ i j =
      pairTerm (onMask near w) δ i j + pairTerm (onMask far w) δ i j := by
  unfold pairTerm onMask
  split
  · have hd := hdisj i j
    cases hN : near i j <;> cases hF : far i j <;> simp [hN, hF] at hd ⊢
    · grind
    · grind
    · grind
  · grind

/-- \((1+\varepsilon)\)-multiplicative preservation of squared distances on `S`.
    The predicate quantifies only over `S`; near pairs are unconstrained. -/
def farPreserved {n : Nat} (eps : Rat)
    (d2 d2' : Fin n → Fin n → Rat) (S : Fin n → Fin n → Bool) : Prop :=
  ∀ i j, S i j = true →
    (1 - eps) * d2 i j ≤ d2' i j ∧ d2' i j ≤ (1 + eps) * d2 i j

/-- Exact far-pair isometry kills the far stress on that pair. -/
theorem far_exact_zero {n : Nat}
    (w δ : Fin n → Fin n → Rat) (S : Fin n → Fin n → Bool)
    (hδ : ∀ i j, S i j = true → δ i j = 0)
    (i j : Fin n) (hS : S i j = true) :
    pairTerm (onMask S w) δ i j = 0 := by
  unfold pairTerm onMask
  split
  · simp [hδ i j hS]
  · rfl

/-- Pairs outside `S` are invisible to `farPreserved`. -/
theorem farPreserved_ignores_unmasked {n : Nat} (eps : Rat)
    (d2 d2' d2'' : Fin n → Fin n → Rat) (S : Fin n → Fin n → Bool)
    (hsame : ∀ i j, S i j = true → d2' i j = d2'' i j) :
    farPreserved eps d2 d2' S ↔ farPreserved eps d2 d2'' S := by
  unfold farPreserved
  constructor
  · intro h i j hS
    rw [← hsame i j hS]
    exact h i j hS
  · intro h i j hS
    rw [hsame i j hS]
    exact h i j hS

end LandfoldAlg
