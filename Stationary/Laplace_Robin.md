# Flat-Laplacian validation of the higher-order Robin conditions

`tests/test_laplace_robin.py` — 9 tests, ~160 s alone; the whole suite is now **35 tests,
~7 min** (measured 6:54), where `HUB.md` used to promise 4-5.  The two costs are the three
solve tests (~137 s in-suite) and the ratio-4 test (~100 s in-suite, whose solves carry a
growing mode and whose order-3 residual nests three radial derivatives).  If that matters for
`run_hub.sh --check`, the ratio-4 test is the one to gate behind a marker: it is the only
end-to-end demonstration that the order matters, but the other tests already pin the same
facts exactly.

```bash
python -m pytest tests/test_laplace_robin.py -q          # just this file
python -m pytest tests/ -q                              # everything
```

The point of the file: the production runs exercise the Robin conditions only inside the full
Einstein solve, where a wrong boundary operator looks like "the optimiser did not converge".
Here the same boundary operator is imposed on a problem whose exact solution is known in
closed form, so the conditions can be checked exactly and then solved with a network.

---

## 1. The problem

Flat Laplacian between two concentric spheres in Cartesian coordinates, with the outer sphere
normalised to unit radius and the same shell ratio as production:

    Delta lam   = 0                                        rho_in = 0.01 < |x| < rho_final = 1
    lam         = lam0 + S1 z + S2 (z^2 - (x^2+y^2)/2)      |x| = rho_in
    B_n[lam - lam0] = B_n[lam_exact - lam0]                 |x| = rho_final

with `lam0 = 1/2`, `S1 = 0.1`, `S2 = 0.05`. The third line is the higher-order Robin
condition with the manufactured source: the right-hand side is `robin_operator` applied to
the exact solution, so the exact solution satisfies the condition by construction whatever
the order. `B_n` is `robin_operator(., base=1, order=n)` — the production code path, called
directly (there is one function with one argument order; the wrapper that used to permute its
last two arguments in `losses.outer_bc_terms` was removed).

Because `lam` enters the production loss through the same machinery (`robin_exps['lam'] = 1`),
this isolates exactly two things: the Euler product itself, and the way its source is
subtracted.

**The rescaling is a relabelling, not a different problem.** `rho -> rho/rho_final` leaves
`lam` unchanged at corresponding points, leaves `rho^2 Delta lam` unchanged (two derivatives
contribute `L^2`, the `rho^2` prefactor `L^-2`) and leaves `B_n` unchanged (it is
dimensionless, built from `theta = rho d_rho`). So every number in this file would be the
same for the production shell `[1, 100]`; `test_the_statement_is_invariant_under_rescaling`
checks that against the un-normalised twin. It is also the sharpest argument for why the
condition has to be built in `theta`: a condition written in `d_rho` does not survive a
rescaling.

## 2. The condition being tested

`theta = rho d_rho` is the radial scaling operator, and `theta rho^-k = -k rho^-k`, so
`(theta + k)` annihilates exactly `rho^-k`. Every factor is a polynomial in the *same*
operator, so the factors commute and

    B_n = prod_{i=0}^{n-1} (theta + 1 + i)      annihilates exactly   rho^-1 ... rho^-n

and nothing else in the power family: any other `rho^p` picks up `prod(p + 1 + i)`, which is
non-zero for every growing mode. Since the operator contains no angular derivatives, its
solution space is (that power window) x (arbitrary function of angle) — which is why it is
transparent to multipoles and indifferent to which `l` carries the decay.

One trap, the source of the "second-order Robin condition" that does not work:

    (d_rho + a/rho)(d_rho + b/rho) = rho^-2 (theta + a - 1)(theta + b)

The two writings agree at first order but not when multiplied, because `d_rho + a/rho` and
`d_rho + b/rho` do not commute. A condition written directly in `d_rho` with the
characteristic polynomial of `theta^2 + 3 theta + 2` — i.e. `psi'' + 3 psi' + psi/rho = 0` —
is not scale covariant and annihilates no power law at all: at `rho = 7` it leaves 8.2e-2 on
`rho^-1` and 2.7e-2 on `rho^-2`, so it fails even the power an order-2 condition exists to
admit. Multiply in `theta`.

**Status of that trap.** `losses.robin_operator` is built in `theta` and always was — the
`d_rho` form was never in the code, only in the original prompt (`prompts.txt` line 3, which
now carries an editorial correction). What could still drift was the *quoted expansion*: the
explicit ODEs in `losses.py`, `problem.py` and `HUB.md` are a separate statement from the
operator and nothing checked them, so a comment could have gone stale silently.
`tests/test_pipeline.py::test_robin_operator_expansions_match_the_documented_forms` now
verifies them against the implementation — `n = 1..4` at `base 1` (`rho psi' + psi`;
`rho^2 psi'' + 4 rho psi' + 2 psi`; `rho^3 psi''' + 9 rho^2 psi'' + 18 rho psi' + 6 psi`;
`rho^4 psi'''' + 16 rho^3 psi''' + 72 rho^2 psi'' + 96 rho psi' + 24 psi`) and `n = 2` at
`base 2` (`rho^2 psi'' + 6 rho psi' + 6 psi`) — on a field that is three decaying harmonics
*plus an exponential*, so no accident of the power-law basis can hide a wrong coefficient.
The other trap on this path, the background subtraction, is §6.4.

## 3. The exact solution

    lam_exact(x) = lam0 + (S1 rho_in^2) z/rho^3 + (S2 rho_in^3) (z^2-(x^2+y^2)/2)/rho^5

The amplitudes are written that way so that on `rho = rho_in` the decaying harmonics reduce
to the prescribed data `lam0 + S1 n_z + S2 (3 n_z^2 - 1)/2`; numerically the multipole
amplitudes are `S1 rho_in^2 = 1e-05` and `S2 rho_in^3 = 5e-08`. The field itself is the same
as in the un-normalised shell: it runs from 0.375 to 0.65 on the inner sphere (a 0.15
deviation from `lam0`) and is within 1e-05 of `lam0` at `rho_final`.

* **Harmonic.** `z/rho^3` and `(z^2-(x^2+y^2)/2)/rho^5` are derivative combinations of
  `1/rho` (the `l = 1` and `l = 2` decaying harmonics). Measured: `max|Delta lam_exact| <
  1e-10` over the shell.
* **Carries the prescribed data.** Agreement with `lam_inner_bc(cfg)` below 1e-15, so the
  data and the reference are the *same* solution — which is what makes this a manufactured
  test rather than two inconsistent datasets.
* **Sits in the `B_3` window.** It decays as `rho^-2, rho^-3` with no `rho^-1` part. So `B_3`
  annihilates it identically, while `B_1` still sees the dipole and `B_2` the quadrupole.

## 4. What the source has to carry, per order

`|B_n[lam_exact - lam0]|` at three radii spanning the shell:

| radius | order 1 | order 2 | order 3 | order 4 |
|---|---|---|---|---|
| 0.015 | 4.939e-02 | 1.376e-02 | 2.1e-15 | 2.6e-14 |
| 0.07 | 1.772e-03 | 1.354e-04 | 6.0e-16 | 2.5e-15 |
| 1.0 (`rho_final`) | 8.064e-06 | 4.643e-08 | 1.5e-16 | 6.0e-16 |

These are *identical* to the values in the un-normalised shell `[1, 100]` at the
corresponding radii (4.939e-02 at `rho = 1.5`, 8.064e-06 at `rho = 100`), which is the
rescaling invariance of §1 made numerical. `B_n` is homogeneous in `rho`, so the pattern —
orders 1 and 2 leave a residual, orders 3 and 4 are machine zero — does not depend on where
the condition is imposed; the tests assert it at all three radii.

With the source present every order admits the solution, so **the source magnitude, not the
solve, is what discriminates the orders** (§6.3).

## 5. The tests

| test (`pytest -k`) | what it pins down |
|---|---|
| `manufactured` | `Delta lam_exact = 0`; `lam_exact` and `lam_inner_bc` agree on `rho = rho_in` |
| `invariant_under_rescaling` | `lam`, `rho^2 Delta lam` and `B_n` are unchanged at corresponding points of the twin shell `[1, 100]`, for `n = 1..4` |
| `source_vanishes` (3 radii) | orders 1, 2 leave a residual > 1e-9; orders 3, 4 < 1e-13; `base = 2, order = 2` also annihilates it |
| `network_solve_at_order_one` | SSBroyden from a random init at order 1 reaches `lam_exact` within `TOL` |
| `network_solve_at_order_three` | the same at order 3, whose source is ~1e-16 (the condition admits the quadrupole unaided) |
| `network_solve_with_an_order_ramp` | curriculum 1 -> 2 -> 3, parameters warm-started, reaches the same field |
| `order_matters_without_a_source` | a ratio-4 shell with the condition imposed homogeneously: orders 1/2/3 land 3.45e-03 / 1.24e-04 / 1.40e-05 from `lam_exact` |

The solver is a network — **6 hidden layers of 20 neurons, tanh** — minimised by **SSBroyden**;
§7 has the full specification.

## 6. Findings worth keeping

### 6.1 The raw Laplacian cannot hold the interior down; it needs the `rho^2` scaling

An unscaled `mean((Delta lam)^2)` is blind to *broad* interior error: a slowly varying hump of
amplitude `A` and radial width `L` has `Delta ~ A/L^2`, so at `L ~ rho` it contributes
`~A/rho^2` and vanishes as `rho` shrinks — and in the normalised shell it is `rho^2` that is
small. Measured (with a network, before this rescaling): an unscaled run whose loss had
fallen to 5.6e-6 was still 0.151 away from the exact solution, with the error spread through
the whole shell. Scaling the residual by `rho^2` — production's `scale_exps['lam_eq'] = 2` —
took the same configuration from 1.06e-01 to 1.78e-02. All tests here use the scaled form,
which is also what makes the residual invariant under the rescaling of §1.

### 6.2 Two things had to be right for the solver to be worth anything

Max error against `lam_exact` over 4096 fresh points, SSBroyden run to convergence
(`|g|_inf < 1e-8`, cap 2500) on 1056 points, seeds 0/3/11/21:

| schedule | max error over the four seeds | iterations |
|---|---|---|
| order 1 alone | 6.5e-07 .. 2.33e-06 | 719 .. 1301 |
| order 3 alone | 6.6e-07 .. 1.42e-06 | 921 .. 1396 |
| ramp 1 -> 2 -> 3 | 5.2e-07 .. 1.07e-06 | 719+85+342 .. 1301+529+858 |

So `TOL = 1e-05` leaves >= 4.3x margin, and the worst case is 0.002% of the solution's
amplitude (0.15) rather than the ~7% an Adam-trained version of the same network managed.

**The optimiser is the main effect.** Same loss, same points, same float64 parameters, Adam
against SSBroyden:

| optimiser | order 1 | order 3 |
|---|---|---|
| Adam + cosine, 2000 steps | 2.71e-03 | 7.17e-03 |
| Adam + cosine, 6000 steps | 3.27e-03 | 7.88e-03 |
| SSBroyden, 600 iterations | 2.3e-04..3.1e-04 | 9.0e-05..2.1e-04 |
| SSBroyden, 1000 iterations | — | 1.06e-04 |

Adam plateaus — three times the steps do not help — while SSBroyden lands **20-40x** lower, so
the assertion here can say "the solve is the exact solution to 0.1%" instead of "the solve is
in the right basin".

**Float64 parameters are mandatory, and were silently missing.** flax defaults to **float32**
parameters even with `jax_enable_x64` set (checked on flax 0.12.9: every leaf came back
float32), so the earlier numbers in this file were float32 solves inside an otherwise float64
test. That cost Adam a factor of ~1.5-2 (order 3: 6.3e-03..1.2e-02 with float32 against
7.17e-03 with float64) and does not explain its plateau — but it decides whether Crunch's
optimiser runs at all, because with float32 parameters its two `lax.cond` branches
(line-search failure vs success) disagree on dtype and the trace fails outright:

    cond branches must have equal output types but they differ
    ... output of true_fun at path .x_k has type float32[2241] but false_fun has float64[2241]

**Convergence, and the earlier stop.** The three solves run until `|g|_inf < 1e-8`, and every
phase converges (`status 0`) in 218..1337 iterations against the 2500 cap — so the numbers
above are properties of the method, not of a budget. The earlier configuration of this file
stopped at 600 iterations, where the order-3 solve was still improving by ~4x per 500
iterations (8.5e-04 at 300, 1.8e-04 at 600, 1.06e-04 at 1000, 8.23e-06 at the 2224-iteration
convergence point with `gtol = 1e-9`): it was measuring its own budget. The 1.2e-02
"representation floor" quoted in that version was likewise an artefact of the float32/Adam
setup — with float64 parameters a *supervised* fit of `lam_exact` (no PDE, no boundary
conditions) converges at 933 iterations to a max error of 2.47e-04.

What that does **not** change is §6.3: no solve can discriminate the *orders* at a shell ratio
of 100, because the solution itself differs by only ~1e-6 between them. The sharp per-order
claims live in §4, where they are exact. An earlier version of this file solved the same loss
over a 30-function separable basis by linear least squares and reached 1e-15 at every order;
it was replaced by the network at the project's request, at the cost of those last orders of
magnitude.

### 6.3 At a shell ratio of 100 the order cannot be discriminated by solving

The out-of-window content at `rho_final` is `S1 rho_in^2 ~ 1e-5` and `S2 rho_in^3 ~ 5e-8`, so an
order-1 condition that kept them would still be satisfied to ~8e-6, and the *solution* of the
BVP differs from `lam_exact` by only ~1e-6 between orders. Any solve-based "order matters"
assertion would be measuring noise at this ratio; order discrimination therefore lives in §4,
where it is exact.

Shrinking the shell fixes that, and it is now measured rather than estimated
(`test_the_order_matters_without_a_source_at_a_small_shell_ratio`). With `rho_in = 0.25`
(ratio 4) and the condition imposed *homogeneously* -- the manufactured source has to come off,
since it is exactly what makes every order exact by construction -- the solved fields land at

| order | max error vs `lam_exact` | the coefficient that survives |
|---|---|---|
| 1 | 3.45e-03 | spurious growing dipole, `A = S1/(rho_in + 2 rho_out^3/rho_in^2) = 3.1e-03`, plus 5.2e-04 from l = 2 |
| 2 | 1.24e-04 | dipole admitted, quadrupole still forced to a spurious growing `A = -1.3e-04` |
| 3 | 1.40e-05 | nothing outside the window is excited; this is the solver's floor |

Ratios of 28x and 9x, i.e. the order is what decides the field once the source is gone. The
data's leverage at the outer sphere scales as `rho_in^2`, which is why ratio 100 hides this
(a 5e-06 effect, at the solver's floor) and ratio 4 exposes it. The test asserts 10x and 3x.

The same effect appears inside the ansatz space, and there it is exact rather than small. For
the separable basis used by the earlier solver, every basis function lay in the window
`rho^-1 ... rho^-3`, so `B_3` and `B_4` annihilated *all* of them: the outer block of the
order-3 and order-4 least-squares systems was identically zero (largest entry 6.2e-15),
against 9.9e-3 at order 1 and 2.5e-4 at order 2.

### 6.4 The background subtraction has to match on both sides

`B_n` sees the background: applied to a constant it gives `prod(1+i) = 6` at order 3, so a
residual built as `B[lam]` with `inf_val = 0` on one side and `B[lam_exact]` with
`inf_val = lam0` on the other carries a constant `6*lam0 = 3`. That is not a rounding-level
error: the first version of the gradient loss had exactly this, and its initial loss was 900
(`= 100 * 3^2`, at the then-current weight) instead of 0.42, with the optimiser spending its
budget cancelling a constant. Both sides must be `B[. - lam0]`; the tests assert the vanishing
of the residual rather than trusting the convention.

### 6.5 The outer weight: what it does and does not control

Production uses `w_outer = 100`. The order-n Robin residual grows steeply with `n`, so that
weight is a different weight at every order. Order 3 on this problem:

| `w_bc` | 300 iterations on 352 points, seed 0 | converged on 1056 points, seeds 0/3/11 |
|---|---|---|
| 1 | 8.5e-04 | 8.24e-07 / 6.64e-07 / 1.16e-06 |
| 3 | — | 5.13e-07 / 6.80e-07 / 6.13e-07 |
| 10 | 1.0e-03 | 1.27e-06 / 5.09e-07 / 4.62e-07 |
| 100 | 2.2e-03 | **not converged**: hit the 2500-iteration cap with loss 1.6e-13 but `|g|_inf` still above 1e-8 |

The two columns say different things, and the second is the one that matters. At the shipped
configuration the weight is **not** a lever: `w_bc` anywhere in 1..10 converges to the same
field within a factor 2 (6.8e-07..1.3e-06), all of it >= 8x inside `TOL`. The 2.6x ordering in
the first column was a budget artefact -- 300 iterations stops the scheme mid-descent, where a
stiffer term still costs something (the same table under Adam at 2000 steps was 4.4e-03 /
1.7e-02 / 4.1e-02, an order of magnitude, because a clipped gradient step is far more exposed
to a dominant term than a line search is).

`w_bc = 100` is still qualitatively different, just not in the field: it never reached the
gradient tolerance, because the stiff term inflates `|g|_inf` at a field accuracy where the
loss is already 1e-13. **The stop criterion is weight-dependent**, which is worth knowing
before reading "converged" as a property of the solve alone.

`HUB.md` measures the same stiffness on the production runs, where the order-n Robin residual
at initialisation is 8.8e-04 at `n = 1`, 2.1e-01 at `n = 2` and 4.4e+03 at `n = 4`, and where
the order-2 control run failed because the outer term dominated the (clipped, Adam) gradient
rather than because of the condition. That is a statement about a first-order optimiser's
*trajectory*; the table above is about the *converged* field. Both hold.

### 6.6 The order ramp is the cheapest route to the best field

With the ramp the phases cannot disagree about the target (the source makes every order admit
the same exact solution), so it is a conditioning device. SSBroyden at 600 iterations per
phase, seeds 0/3/11:

| schedule | max error |
|---|---|
| order 1 alone | 2.3e-04..3.1e-04 |
| order 3 alone | 9.0e-05..2.1e-04 |
| ramp 1 -> 2 -> 3 | **1.6e-05..3.0e-05** |

The ramp ends an order of magnitude below the fixed-order solves for the same per-phase
budget: each phase starts from the previous field with a fresh inverse Hessian, so the
expensive highest-order residual only has to polish. Under Adam the ramp was also best, but
only by a factor ~2, and then the budget split mattered (400/400/1600 against 800/800/800 at
a fixed 2400-step total). The weight caveat of §6.5 applies to the ramp too: at `w_bc = 100`
the Adam-era ramp was 5.4e-02..6.1e-02, an order of magnitude worse.

## 7. The solver, in detail

### 7.1 What each test runs

| test (`pytest -k`) | solver | points | iterations | seeds |
|---|---|---|---|---|
| `manufactured` | none, property checks only | 128 shell + 64 on `rho = rho_in` | — | 0, 1 |
| `invariant_under_rescaling` | none, property checks only | 64 shell (new) + the same points x100 (twin) | — | 4 |
| `source_vanishes` (3 radii) | none, operator only | 1 point per radius | — | — |
| `network_solve_at_order_one` | network + SSBroyden, to convergence | 768 + 144 + 144, one fixed draw | 719..1301 (status 0) | model 1, batch 777 |
| `network_solve_at_order_three` | network + SSBroyden, to convergence | same | 921..1396 (status 0) | model 1, batch 777 |
| `network_solve_with_an_order_ramp` | network + SSBroyden, 3 phases | re-drawn per phase | 719+85+342 .. 1301+529+858, all status 0 | model 1, batches 777/1554/2331 |
| `order_matters_without_a_source` | network + SSBroyden, 3 solves, no source | 256 + 48 + 48 (ratio-4 shell) | 728..1079, all status 0 | model 1, batches 777 |
| error metric (every solve) | — | 4096 fresh shell points | — | 9 |

### 7.2 The network

```
inputs   [n_x, n_y, n_z, t, rho_in/rho]              n = x/rho,  t = log(rho/rho_in)/log(ratio)
body     6 x Dense(20), each followed by tanh        (no other activation, no normalisation
                                                      layer, no skip connections)
head     Dense(1), kernel N(0, 1e-2), zero bias
output   lam = lam0 + u                              (linear head, no exp)
dtype    float64 for every parameter and activation  (flax would default to float32)
```

* **Parameters: 2241** — `(5+1)*20 + 5*(20*20 + 20) + (20+1)`. No other trainable state.
* **Initialisation:** flax `Dense` defaults (`lecun_normal`) for the body; the head is scaled
  to `1e-2` so the network starts at `lam0`. The initial loss is therefore purely the
  inner-data mismatch (`Delta lam0 = 0` and the outer residual is `-source ~ 1e-16`), i.e.
  `w_bc * mean(r_in^2)`, measured 0.424 for the first phase's 48-point draw against 0.383 from
  a 200k-point evaluation of the same mean — a 48-point boundary draw only estimates it to
  ~15-30%.
* **Why these inputs:** `t` normalises the radial range to `[0, 1]` and `rho_in/rho` hands the
  network the decay variable directly. Without the latter the two-decade radial range has to
  be synthesised from `t` alone.
* **Not used:** Fourier features in `t` (tried at order 3 with 6000 Adam steps at
  `w_bc = 100`: 4.7e-02 against 6.7e-02 — a small gain, not worth the extra stiffness the
  higher-order condition then sees).

### 7.3 The points

* **Radial distribution: log-uniform.** `rho = exp(U[log rho_in, log rho_final])`
  (`stationary/problem.py::sample_shell` with `Config.radial = "log"`, the repo default). The
  shell spans exactly two decades, so `n = 256` gives ~128 points per decade. Uniform in
  `rho` was not used; biasing the measure towards `rho ~ rho_in` (`u^2`, `u^3` instead of `u`)
  was tried and made the solve worse (1.8e-02, 2.4e-02 against 1.1e-02, Adam era).
* **Angular distribution: uniform in solid angle**, by normalising Gaussian 3-vectors
  (`sphere_directions`), from an independent key — so radius and direction are independent and
  the points are a random cloud, not a tensor-product grid.
* **Counts:** 768 interior collocation points, 144 on `rho = rho_in`, 144 on `rho = rho_final`
  (`N_COLL`, `N_BND`) — 3x the round numbers, because §6.7 measures that the field error
  saturates there. The boundary sets are exactly on their spheres; the interior set is
  strictly inside.
* **Ordering: none.** Nothing is sorted, stratified or shuffled, and there are no
  mini-batches: the arrays are used in the order JAX produced them.
* **One draw per phase, held fixed.** A line search needs a single objective, so the
  quasi-Newton phase does *not* resample; each phase of a schedule draws its own points
  (`seed + 777*(phase+1)`).
* **Evaluation is off-sample:** `max_error` uses 4096 fresh log-uniform shell points
  (`PRNGKey(9)`), never the fitted ones. It is a max-norm metric, so a single unlucky point
  dominates it, which is why §6.2 quotes a range over seeds.

### 7.4 The SSBroyden phase

`solve_network` flattens the parameters and hands Crunch's SciPy-style wrapper a scalar
objective:

```python
flat0, unflatten = jax.flatten_util.ravel_pytree(params)
fun = lambda flat: loss(unflatten(flat), model, batch)      # one fixed batch
res = crunch_minimize(fun, flat0, args=(), method="BFGS",
                      options={"maxiter": 600, "gtol": 1e-9,
                               "initial_H": jnp.eye(flat0.size),      # dense, n^2
                               "update_method": "ssbroyden2", "initial_scale": False})
```

| ingredient | setting |
|---|---|
| objective | `mean((rho^2 Delta lam)^2) + w_bc*mean(r_in^2) + w_bc*mean(r_out^2)` over the 768 interior, 144 inner and 144 outer points, with `rho^2` as in production (`scale_exps['lam_eq'] = 2`) |
| `w_bc` | 1 (measured; §6.5) |
| recurrence | `update_method="ssbroyden2"` — Crunch's self-scaling Broyden (tau_k, rho_k^m, v_k, phi_k) |
| inverse Hessian | dense, initialised to the identity, carried across iterations (`result.hess_inv`) |
| line search | two-stage: primary Armijo-Wolfe c1 = 1e-4, c2 = 0.9, <= 15 evaluations; on failure a fallback with c2 = 0.8 then 0.5, <= 10 each |
| `initial_scale` | False here: the network starts at `lam0` and the loss is already small. Switching it on is slightly better at order 1 (2.58e-04 against 3.15e-04) and worse at order 3 (4.63e-04 against 1.80e-04) |
| iterations | stops on `\|g\|_inf < 1e-8`; `QN_MAXITER = 2500` is a cap that is never reached here (worst phase 1337). Order 3 alone: 1155 at order 1, 1337 at order 3, and 608 for order 3 when it warm-starts from the ramp |
| learning rate | none — a line search replaces it |
| batching | none: one full batch (1056 points) for the whole phase |
| cost | 17 s (order 1), 33 s (order 3), 42 s (the 1 -> 2 -> 3 ramp) on 8 CPU cores alone; inside the full suite the same three take 21 / 49 / 67 s, i.e. the suite is ~50% slower than the sum of its files |

**Import.** Crunch is a sibling checkout (`PINN/Jax`), not a dependency of this repo, so the
import is guarded: `CRUNCH_ROOT` overrides the location and the three solve tests *skip* with
a reason naming the path when it is unavailable (that is what keeps the suite green on the
hub, where only `Stationary/` is cloned).

### 7.5 The same optimiser in production (`stationary/train.py`)

The Adam phase is unchanged; the quasi-Newton phase now runs SSBroyden instead of
`optax.lbfgs`, with the same `--lbfgs-steps` budget (`cfg.qn_method = "ssbroyden"` by
default).

* **Soft import** (`_crunch_minimize`): falls back to `optax.lbfgs` with a printed reason when
  the checkout is missing or the import fails; `--qn-method lbfgs` forces the old path.
* **Memory guard.** The inverse Hessian is dense, `n_params^2`: the production network
  (13 828 parameters) needs **1.53 GB in float64 / 0.76 GB in float32**, and the rank-two
  update holds a few such arrays at once, so the true peak is roughly 4x that. Above
  `--qn-max-H-gb` (default 2.0) the phase declines with a printed explanation and falls back
  to L-BFGS rather than attempting the allocation.
* **`initial_scale=True` in production, and it is not optional there.** With `H = I` the first
  step is `-grad`, and after a short Adam warm-up that step is large enough that the Wolfe
  line search cannot bracket it: measured `0 iterations, status 3 (zoom failed)` with the flag
  off, versus real progress with it on. In the test the loss is already small at the random
  init, which is why the two settings differ.
* **Smoke comparison** (same seed, 611-parameter net, 200 Adam steps then 30 quasi-Newton
  iterations, identical batches):

  | phase | final loss | `lam_eq` rms | `ricci` rms | wall |
  |---|---|---|---|---|
  | SSBroyden | **2.35e-02** | 6.19e-02 | 1.80e-02 | 18.7 s |
  | optax.lbfgs | 4.04e-02 | 7.42e-02 | 2.26e-02 | 11.2 s |

  Better on every residual group at ~1.7x the wall time, in a 30-iteration toy setting — not a
  production verdict, but the plumbing is verified end to end (`--qn-method ssbroyden` prints
  `[qn] SSBroyden (ssbroyden2) from <root>: n parameters, ... GB inverse Hessian`).

### 6.7 How many points: the field error saturates at 3x

The converged field error is set by how many points the residual is enforced on, and it stops
improving well before the cost does. Worst over the three solves, seed 0, all run to
convergence:

| factor | points (interior + 2 boundary sets) | worst max error | time for the three solves |
|---|---|---|---|
| 1x | 256 + 48 + 48 = 352 | 1.19e-04 | 62 s |
| 2x | 512 + 96 + 96 = 704 | 1.01e-05 | 67 s |
| **3x** | 768 + 144 + 144 = 1056 | **2.33e-06** | 88 s |
| 5x | 1280 + 240 + 240 = 1760 | 1.97e-06 | 97 s |
| 10x | 2560 + 480 + 480 = 3520 | 2.27e-06 | 152 s |

The last two rows are within the noise of the max-norm metric (the 3x, 5x and 10x values are
all ~1-2e-06), so **10x buys nothing over 3x and costs 1.7x the time** — the file uses 3x.
What that also shows is that the 1x numbers quoted in the earlier version of this report
(1.2e-04 worst) were *sampling*-limited rather than representation-limited: the same 2241
parameters reach ~1e-06 once the residual is enforced on enough points.

Two checks that the extra points are not just being memorised: at 3x the max error over 40960
fresh points is within ~5% of the 4096-point value (e.g. 8.24e-07 against 8.54e-07), and the
residual is enforced on a single fixed draw per phase, never resampled. More points also make
the quasi-Newton phase *shorter* in iterations (order 3: 1337 at 1x against 921-1396 at 3x for
a much smaller error, and the ramp's middle phase drops to 47-85 iterations), because the
objective stops being dominated by sampling noise.

## 8. Not covered here

* The metric and connection Robin conditions (`h` with `base = 2`, `Gamma` with `base = 3`).
  Their operator behaviour is covered by `test_robin_operator_annihilates_powers` and
  `test_robin_fourth_order_kills_only_first_four_powers` in `tests/test_pipeline.py`; the
  `h -> I` background and the reason `--no-robin-G` exists are a separate matter.
* The full Einstein coupling, the gauge condition, and anything about the inner metric data.
* Whether SSBroyden helps at *production* scale. The smoke comparison in §7.5 is 30 iterations
  on a 611-parameter network; the production phase is 300+ iterations on 13 828 parameters,
  where the dense inverse Hessian is 1.5 GB and the per-iteration cost is `O(n^2)`. That run
  has not been made.
* The float32 round-off floor per order, which `HUB.md` measures for the production runs
  (orders 3 and 4 on `lambda` are not representable in float32 at `rho = 100`). This file is
  float64 throughout and sets `JAX_ENABLE_X64` at import.

## 9. Files

* `tests/test_laplace_robin.py` — the tests, the network and the SSBroyden phases.
* `tests/test_pipeline.py` — `test_robin_operator_expansions_match_the_documented_forms` and
  the strengthened `test_robin_second_order_lambda_form` (the operator's explicit ODE forms).
* `stationary/train.py` — the quasi-Newton phase now prefers SSBroyden
  (`_crunch_minimize`, `ssbroyden_phase`, `--qn-method`, `--qn-max-H-gb`).
* `stationary/problem.py` — `qn_method` and `qn_max_H_gb` on `Config`.
* `stationary/losses.py` — `robin_operator` is the operator under test; its call sites in
  `outer_bc_terms` now use its single argument order `(field_fun, x, base, order, inf_val)`.
* `Laplace_Robin.md` — this report.
