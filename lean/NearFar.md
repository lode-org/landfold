# Near / far identities

Three facts licence a landfold embedder that does not run full-pairwise
Ceriotti chi. Lean: `LandfoldAlg/NearFar.lean`. No Mathlib. Lean 4.29.

Notation. A sample of `n` points in `d` coordinates is written
`Y : [n] -> R^d`. Axis `h` is the scalar list `y^{(h)}_i = Y_{i h}`.
A pair mask is a Boolean array on ordered pairs `i < j`.

## 1. Discrete W1 on an ordered line

### Statement

Let `x_1 < ... < x_m` be a strictly increasing support and let
`a, b in R_{\ge 0}^m` satisfy `sum_i a_i = sum_i b_i`. Write
`A_k = sum_{i=1}^k a_i` and `B_k = sum_{i=1}^k b_i` for the prefix
masses (right-continuous CDFs on the atoms). The 1-Wasserstein
distance of the atomic measures `mu = sum_i a_i delta_{x_i}` and
`nu = sum_i b_i delta_{x_i}` is

    W_1(mu, nu) = sum_{k=1}^{m-1} |A_k - B_k| (x_{k+1} - x_k).     (1)

If the support is unit-spaced, `x_{k+1} - x_k = 1` for every `k`,
then (1) collapses to the L1 distance of the CDFs:

    W_1(mu, nu) = sum_{k=1}^{m-1} |A_k - B_k| = ||A - B||_1,       (2)

the last coordinate of `A` and of `B` being the common total mass,
hence dropping out.

### Proof

Two arguments. The first is the classical CDF identity; the second
is an elementary cut-mass lower bound that an implementer can check
without measure theory. Both use only that the cost is `|x-y|` on a
totally ordered line.

CDF integral. On `R` with cost `|x-y|`,

    W_1(mu, nu) = int_R |F_mu(t) - F_nu(t)| dt                     (3)

(Vallender, *Theory Probab. Appl.* **18**, 784 (1974); Villani,
*Topics in Optimal Transportation*, Thm. 2.18). For the atomic
measures above, `F_mu(t) = 0` on `(-infty, x_1)`, `F_mu(t) = A_k`
on `[x_k, x_{k+1})`, and `F_mu(t) = A_m` on `[x_m, +infty)`. The
same holds for `nu` with prefixes `B`. The hypothesis `A_m = B_m`
kills the integrand outside `[x_1, x_m]`. On each half-open interval
`[x_k, x_{k+1})` the integrand is the constant `|A_k - B_k|`, and
the length of that interval is `x_{k+1} - x_k`. Summing (3) over
those `m-1` intervals is (1). Unit spacing sets every length to 1
and yields (2).

Cut mass. Fix a cut between `x_k` and `x_{k+1}`. Any coupling of
`mu` and `nu` must send net mass `|A_k - B_k|` across that cut:
the mass of `mu` on `{x_1,...,x_k}` is `A_k` and the mass of `nu`
on the same set is `B_k`, so the surplus (or deficit) has nowhere
to go but across. Each unit of mass that crosses travels at least
`x_{k+1} - x_k`. Summing the lower bounds over cuts gives

    W_1(mu, nu) >= sum_{k=1}^{m-1} |A_k - B_k| (x_{k+1} - x_k).

The monotone (quantile) coupling achieves equality: match mass from
left to right and never reverse a pair. That coupling transports
exactly `|A_k - B_k|` across the `k`-th gap and nothing else, so
the lower bound is sharp and (1) holds.

Two-point check. For `m = 2` the identity is
`W_1 = |a_1 - b_1| (x_2 - x_1)`. Equal total mass forces
`|a_1 - b_1| = |a_2 - b_2|`, which is the unique mass that must
move across the single gap. Lean: `w1_two`, `two_mass_swap`.

Unit bins. Substitute `x_{k+1} - x_k = 1` into (1). Lean: `w1_of_unit`.

### Implementer

Unequal totals first become probability masses (otherwise `W_1` of
the un-normalised measures is infinite). Unit-spaced integer bins
(coordination counts `n4..n13`) then use (2). This is
`Wasserstein1::dist_unchecked` in `src/metric.rs`.

```
sa = sum(max(a[i], 0) for i in 0..m)
sb = sum(max(b[i], 0) for i in 0..m)
if sa <= 0 or sb <= 0:
    return +inf
ca = 0.0
cb = 0.0
w1 = 0.0
for i in 0..m:
    ca += max(a[i], 0) / sa
    cb += max(b[i], 0) / sb
    w1 += abs(ca - cb)
```

The last addend is `|1 - 1| = 0`, so the loop of `m` prefixes equals
the sum over `k = 0..m-2` in (2). Do not L1 the raw masses `a` and
`b`; that is total variation, not `W_1`. For a general increasing
support, multiply each addend by `(x[i+1] - x[i])` and stop at
`i = m-2` as in (1).

## 2. Independent monotone transforms preserve rank order

### Statement

Let `y in R^n` be one coordinate of the sample. The empirical CDF is

    F(t) = (1/n) #{ k : y_k <= t },                                (4)

and the rank map is `u_i = F(y_i)`. Then:

(i)  `y_i < y_j` implies `u_i < u_j`. Ties are preserved:
     `y_i = y_j` implies `u_i = u_j`.

(ii) `j` is the axis successor of `i` in `y` (nothing of `y` sits
     strictly between) if and only if `j` is the axis successor of
     `i` in `u`. Neighbour ranks along the axis are unchanged.

(iii) The map on axis `h` is a function of column `h` alone. A
      change to any other coordinate leaves `u^{(h)}` fixed.

The `d` transforms `u^{(h)} = F_h(y^{(h)})` therefore run independently
and each is a strictly increasing function of that axis.

### Proof

Write `r(t) = #{ k : y_k <= t }`, so `u_i = r(y_i)/n`. For (i), the
hypothesis `y_i < y_j` gives `{k : y_k <= y_i} subset {k : y_k <= y_j}`
because `<=` is transitive. The inclusion is strict: index `j` belongs
to the second set (`y_j <= y_j`) and not the first (`y_j <= y_i` would
contradict `y_i < y_j`). Hence `r(y_i) <= r(y_j) - 1`, so
`r(y_i) < r(y_j)` and, for `n > 0`, `u_i < u_j`. Equal values give
equal ranks by substitution. Lean: `countLE_strict`, `ecdf_strict`,
`ecdf_eq_of_eq`.

For (ii), suppose nothing of `y` sits in `(y_i, y_j)` and `y_i < y_j`.
Then `r(y_i) < r(y_j)` by (i). If some `k` satisfied
`r(y_i) < r(y_k) < r(y_j)`, monotonicity of `r` on the image of `y`
would force `y_i < y_k < y_j`, a contradiction. Conversely, a point
strictly between `y_i` and `y_j` would give a rank strictly between
by (i). So the open intervals are empty together. Lean:
`neighbor_preserved`.

For (iii), `r` on axis `h` is `countLE` of column `h`. Replacing every
other column does not change that list. Lean: `col_rank_independent`.

The mid-rank variant that splits ties
(`u_i = (#{k : y_k < y_i} + (#{k : y_k = y_i}+1)/2)/n`) is still a
strictly increasing function of the distinct values, so (ii) survives
on the quotient that identifies ties. Lean `countLE` / `ecdf` is (4).
The landfold map `rank_uniform` is the argsort rank divided by
`n-1`, which is a different monotone transform of the same order
and still satisfies (ii) on distinct values (ties are broken by
stable original order).

### Implementer

Independent per coordinate `h = 0..d-1`. This is `rank_uniform` in
`src/pacmap.rs`. Input `Y` is `n x d` with `n >= 2`.

```
for h in 0..d-1:
    idx = argsort_stable(Y[:, h])     # idx[p] = point of rank p
    for p in 0..n-1:
        Y[idx[p], h] = p / (n - 1)    # {0, 1/(n-1), ..., 1}
```

The right-continuous empirical CDF (4), which Lean `ecdf` uses, is
the alternative

```
U[i, h] = count(k, Y[k, h] <= Y[i, h]) / n
```

and keeps ties equal. Both maps are strictly increasing on distinct
axis values, so the successor of a point among the distinct values
is the same before and after. Neighbour queries along one axis after
the transform must use the same axis; a Euclidean k-NN in the joint
`U` is a different graph.

## 3. Far-pair JL and the split loss

### Statement

Partition the unordered pairs of `{1,...,n}` into a near set `N`
and a far set `S`, with `N cap S = empty`. Typical choice:
`N` is the undirected k-NN graph of the high-D sample, and `S` is
a subset of the complementary pairs (all of them, a random sample,
or a farthest-point skeleton). The split loss is

    L = L_near(N) + L_far(S),                                      (5)

    L_near(N) = sum_{{i,j} in N} w_{ij} (d_ld(i,j) - d_hd(i,j))^2, (6)

    L_far(S)  = sum_{{i,j} in S} v_{ij} (d_ld(i,j) - d_hd(i,j))^2. (7)

(The Ceriotti transfer `F_HD(D)-F_LD(d)` may replace the raw
distance residual; the algebra below does not use the form of
`delta_{ij}`.)

Johnson-Lindenstrauss, restricted to `S`. Let `V(S)` be the vertices
incident to a pair in `S`, so `|V(S)| <= min(n, 2|S|)`. For every
`eps in (0,1)` there exists `k = O(eps^{-2} log |V(S)|)` and a
linear map `R : R^D -> R^k` such that

    (1-eps) ||Y_i - Y_j||^2  <=  ||R Y_i - R Y_j||^2
                         <=  (1+eps) ||Y_i - Y_j||^2               (8)

for every `{i,j} in S`. The map is not required to satisfy (8) on
`N`. Near pairs are handled by (6), which is a graph stress on the
k-NN edges and is independent of whether a random projection
preserves them.

### Proof

Additivity. The summand of (5) on a pair is `w delta^2` if the pair
is in `N`, `v delta^2` if the pair is in `S`, and `0` otherwise.
Disjointness of `N` and `S` prevents a pair from being counted
twice. A pair in neither mask contributes nothing. Lean:
`split_term`.

Exact far isometry. If `delta_{ij} = 0` on every pair of `S`, then
every far summand vanishes and `L_far(S) = 0`. Lean: `far_exact_zero`.
Inequality (8) with `eps = 0` is that case for squared Euclidean
distances.

JL existence (not in Lean). The classical lemma (Johnson and
Lindenstrauss, *Contemp. Math.* **26**, 189 (1984); Dasgupta and
Gupta, *Random Structures Algorithms* **22**, 60 (2003)) produces a
Gaussian or Rademacher matrix `R in R^{k x D}`, scaled by `k^{-1/2}`,
that `(1+eps)`-preserves all pairwise distances among `N` points
as soon as `k >= C eps^{-2} log N`. Apply it to the subset
`{Y_i : i in V(S)}`. The resulting `k` depends on `|V(S)|`, not on
`n` and not on `|N|`. The concentration argument is a tail bound on
`||R v||^2` for each of the `|S|` difference vectors `v = Y_i - Y_j`;
near-pair difference vectors never enter the union bound.

Independence of the near term. The predicate `farPreserved` (Lean)
quantifies only over pairs with `S i j = true`. Replacing the
embedded distances of every unmasked pair leaves the predicate
unchanged. Lean: `farPreserved_ignores_unmasked`. That is the
licence for not asking a projection, or a stress, to hold on `N`:
`L_near` already penalises k-NN distortion directly.

### Implementer

PaCMAP (Wang, Huang, Rudin, Shaposhnik, *J. Mach. Learn. Res.* **22**,
1 (2021)) is the landfold instance of (5), with a mid-near attractor
in between. Write `dtilde = ||y_i - y_j||^2 + 1`. The three pair
kinds, disjoint by construction in `build_pairs`, are

```
L_near = sum_{{i,j} in N}   dtilde / (10 + dtilde)
L_mid  = sum_{{i,j} in M}   dtilde / (10000 + dtilde)
L_far  = sum_{{i,j} in S}   1 / (1 + dtilde)
L      = L_near + L_mid + L_far
```

`N` is the undirected k-NN graph of the high-D metric (Euclid or
`Wasserstein1`). `M` is a stride through the next `4k` neighbours.
`S` is a stride through the farther half of each row. A pair already
taken as near is not reused as mid or far (`seen` in `build_pairs`).
Gradients of those summands are what `pacmap_embed` steps:

```
grad_near =  2*10     * u / (10 + dtilde)^2      # u = y_i - y_j
grad_mid  =  2*10000  * u / (10000 + dtilde)^2
grad_far  = -2        * u / (1 + dtilde)^2
```

The far term is repulsive. It does not match high-D distances; JL
is the licence for leaving `N` out of that term, not a formula for
`L_far`. A Ceriotti residual `F_HD(D) - F_LD(d)` may replace the
raw `d_ld - d_hd` in a stress-style split; the pair-level additivity
is the same (`split_term`).

Optional JL seed: draw `R` of shape `k x D` with
`k = ceil(C * log(max(2, |V(S)|)) / eps**2)` i.i.d. `N(0, 1/k)`
entries and start from `Y @ R.T`. Do not use that seed as a
substitute for `L_near`. Target dimension from JL is a seed, not a
certificate that (8) holds for a fixed draw.

## Discussion

Ceriotti chi on all `n choose 2` pairs mixes local neighbourhoods
with long distances in one sum. Identity (1) is the high-D metric
on coordination histograms (`--w1`). Identity (2) of the rank map
says each low-D coordinate may be replaced by its empirical CDF
(`--uniform`) without changing who is next to whom on that axis.
Identity (5) is the PaCMAP graph: k-NN attract, mid-near attract,
far repel. The three facts together replace an all-pairs objective
by `O(n k)` pair terms.

Lean covers the algebraic identities: (2) from (1) under unit
spacing, rank strictness, neighbour invariance, column independence,
pair-level split of (5), and the fact that `farPreserved` does not
mention `N`. The integral representation (3) and the JL tail bound
are the hand proofs above.
