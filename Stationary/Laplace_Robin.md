# Flat-Laplacian validation of the higher-order Robin conditions

`tests/test_laplace_robin.py` — 10 tests, ~40 s, part of the normal suite (35 tests, ~2:35).

```bash
python -m pytest tests/test_laplace_robin.py -q          # just this file
python -m pytest tests/ -q                              # everything
```

The point of the file: the production runs exercise the Robin conditions only inside the
full Einstein solve, where a wrong boundary operator looks like "the optimiser did not
converge". Here the same boundary operator is imposed on a problem whose exact solution is
known in closed form, so the conditions can be checked to machine precision and then solved.

---

## 1. The problem

Same shell and chart as production — two concentric spheres, Cartesian coordinates,
`rho_in = 1`, `rho_out = 100` — but with the Einstein system replaced by the flat Laplacian:

    Delta lam   = 0                                        rho_in < |x| < rho_out
    lam         = lam0 + S1 z + S2 (z^2 - (x^2+y^2)/2)      |x| = rho_in  = 1
    B_n[lam - lam0] = B_n[lam_exact - lam0]                 |x| = rho_out = 100

with `lam0 = 1/2`, `S1 = 0.1`, `S2 = 0.05`. The third line is the higher-order Robin
condition with the manufactured source: the right-hand side is `robin_operator` applied to
the exact solution, so the exact solution satisfies the condition by construction whatever
the order. `B_n` is `robin_operator(., base=1, order=n)` — the production code path, called
directly (the local argument-permuting wrapper that used to sit in `losses.outer_bc_terms`
was removed; there is one function with one argument order now).

Because `lam` enters the production loss with the same machinery (`robin_exps['lam'] = 1`),
this isolates exactly two things: the Euler product itself and the way its source is
subtracted.

## 2. The condition being tested

`theta = rho d_rho` is the radial scaling operator, and `theta rho^-k = -k rho^-k`, so
`(theta + k)` annihilates exactly `rho^-k`. Every factor is a polynomial in the *same*
operator, so the factors commute and

    B_n = prod_{i=0}^{n-1} (theta + 1 + i)      annihilates exactly   rho^-1 ... rho^-n

and nothing else in the power family: any other `rho^p` picks up `prod(p + 1 + i)`, which is
non-zero for every growing mode. Since the operator contains no angular derivatives, its
solution space is (that power window) x (arbitrary function of angle) — which is why it is
transparent to multipoles and why it is indifferent to which `l` carries the decay.

One trap, the source of the "second-order Robin condition" that does not work:

    (d_rho + a/rho)(d_rho + b/rho) = rho^-2 (theta + a - 1)(theta + b)

The two writings agree at first order but not when multiplied, because `d_rho + a/rho` and
`d_rho + b/rho` do not commute. A condition written directly in `d_rho` with the
characteristic polynomial of `theta^2 + 3 theta + 2` — i.e. `psi'' + 3 psi' + psi/rho = 0` —
is not scale covariant and annihilates no power law at all. Multiply in `theta`.

## 3. The exact solution

    lam_exact(x) = lam0 + S1 z/rho^3 + S2 (z^2 - (x^2+y^2)/2)/rho^5

* **Harmonic.** `z/rho^3` and `(z^2-(x^2+y^2)/2)/rho^5` are derivative combinations of
  `1/rho` (the `l = 1` and `l = 2` decaying harmonics). Measured: `max|Delta lam_exact| <
  1e-12` over the shell.
* **Carries the prescribed data.** On `rho = 1` the decaying harmonics reduce to the
  polynomial of the problem statement, so `lam_exact` reproduces `lam_inner_bc` exactly
  (agreement < 1e-15). The data and the reference are therefore the *same* solution, which
  is what makes this a manufactured test rather than two inconsistent datasets.
* **Sits in the `B_3` window.** It decays as `rho^-2, rho^-3` with no `rho^-1` part
  (`S1` and `S2` multiply decaying multipoles, not the monopole). So `B_3` annihilates it
  identically, while `B_1` still sees the dipole and `B_2` still sees the quadrupole.

## 4. What the source has to carry, per order

`|B_n[lam_exact - lam0]|` at `rho_out = 100` (point on the axis) and at `rho = 1.5`:

| order | source at rho=100 | source at rho=1.5 | admits the solution? |
|---|---|---|---|
| 1 | 1.010e-05 | 4.939e-02 | no — dipole and quadrupole survive |
| 2 | 1.000e-07 | 1.376e-02 | no — quadrupole survives |
| 3 | 9.780e-17 | 1.291e-15 | yes — source vanishes |
| 4 | 3.967e-16 | 2.654e-14 | yes (window is a superset) |

`B_n` is homogeneous in `rho`, so this table is radius-independent up to the `rho^-2`,
`rho^-3` prefactors — the tests assert it at `rho = 1.5, 7, 100`. With the source present
every order admits the solution, so **the source magnitude, not the solve, is what
discriminates the orders** (see §7).

## 5. The tests

| test | what it pins down |
|---|---|
| `test_manufactured_solution_is_harmonic_and_matches_the_inner_data` | `Delta lam_exact = 0`; `lam_exact` and `lam_inner_bc` agree on `rho = 1` |
| `test_robin_source_vanishes_only_from_order_three_onwards[rho]` | orders 1, 2 leave a residual > 1e-9; orders 3, 4 < 1e-13; `base = 2, order = 2` also annihilates it |
| `test_collocation_solve_ladder_reproduces_the_exact_solution[1..4]` | solve at each order with the source; each returns `lam_exact` to < 1e-9; the order-1/2 source is asserted non-zero, and the outer block of the system non-zero below order 3 and identically zero from order 3 up |
| `test_gradient_solve_reproduces_the_exact_solution` | the same loss minimised by Adam at order 3 (loss must fall by > 1000x) |
| `test_gradient_solve_with_an_order_ramp_reproduces_the_exact_solution` | curriculum 1 -> 2 -> 3 with warm-started parameters |

The solves use a **separable ansatz** rather than a tanh MLP, for a measured reason — §6.2.

## 6. Findings worth keeping

### 6.1 The raw Laplacian cannot hold the interior down; it needs the `rho^2` scaling

An unscaled `mean((Delta lam)^2)` is blind to *broad* interior error: a slowly varying hump
of amplitude `A` and radial width `L` has `Delta ~ A/L^2`, so at `L ~ rho` it contributes
`~A/rho^2` and vanishes as `rho` grows. Measured: an unscaled run whose loss had fallen to
5.6e-6 was still 0.151 away from the exact solution, with the error spread through the whole
shell (bin maxima 8.7e-2 around `rho ~ 2` and 1.5e-1 over `rho in [3, 50]`). Scaling the
residual by `rho^2` — the same thing production does with `scale_exps['lam_eq'] = 2` — took
the same run from 1.06e-01 to 1.78e-02, and made the loss actually control the field. All
tests here use the scaled form.

The boundary weights are `100`, matching `--w-inner/--w-outer 100` in production. With the
MLP that split was a capacity trade-off, not a tuning knob: raising it from 10 to 100 cut
the inner residual (rms 2.1e-3 -> 1.5e-3) while *raising* the PDE residual (1.6e-4 ->
1.5e-3) and the field error (1.07e-2 -> 3.68e-2), and 1000 was no better (3.46e-2). It does
not matter for the shipped test: the separable ansatz satisfies all three conditions at
once, and the collocation system is consistent, so the weights cannot bias the solution.

### 6.2 A tanh MLP cannot represent this solution over two decades of radius

Even *supervised* (no PDE, no boundary conditions — just fit `lam_exact` on 256 log-uniform
points and measure on fresh ones):

| model | training MSE | max error on fresh points |
|---|---|---|
| w32 d2, decay features | 6.11e-07 | 1.710e-02 |
| w48 d2 | 9.39e-07 | 1.280e-02 |
| w64 d3 + 4 Fourier modes | 1.10e-07 | 1.250e-02 |
| w32 d2, 4000 steps | 1.88e-07 | 1.217e-02 |
| w32 d2 on separable features | 3.18e-08 | 7.914e-03 |

More width, more depth and more steps leave the supervised fit at ~1e-2, and L-BFGS does not
rescue the PINN loss either. That floor is the same
size as the solution's entire structure (it deviates from `lam0` by at most 0.15), so an
MLP-based solve would either fail or force a tolerance that asserts nothing.

The test therefore learns the *coefficients* of a separable basis —
`{1, n_i, n_i n_j} x {rho^-1, rho^-2, rho^-3}`, 30 functions, each normalised to unit RMS
(the raw `rho^-3` column spans six decades, which alone makes the least-squares matrix
unusable). This is the device the production models already use for the metric
(`h = alpha delta + beta n n`): build in the tensor structure, learn the coefficients. It is
not the answer in disguise — the basis contains the `rho^-1` monopole, the `rho^-1` and
`rho^-2` dipoles and the other spurious modes, so it is the *conditions* that select the
solution. That claim is checked rather than asserted: recomputing the ladder with half the
collocation points (128/24 instead of 256/48) gives the same 2.2e-16 field.

### 6.3 At `rho_out = 100` the order cannot be discriminated by solving

The out-of-window content at the boundary is `rho^-2 ~ 1e-4` and `rho^-3 ~ 1e-6`, so an
order-1 condition that kept them would still be satisfied to ~1e-5, and the *solution* of
the BVP differs from `lam_exact` by only ~1e-6 between orders. Any solve-based
"order matters" assertion at this radius would be measuring noise. Order discrimination
lives in §4, where it is exact; the ladder in §5 checks the operator and source at each
order, not which order is needed. (At a smaller `rho_out` — say 4 — the orders separate by
~1e-3 and a solve-based discrimination would become testable.)

The same effect appears inside the ansatz space, and there it is exact rather than small.
Every basis function lies in the window `rho^-1 ... rho^-3`, so `B_3` and `B_4` annihilate
*all* of them: the outer block of the order-3 and order-4 least-squares systems is
identically zero (largest entry 6.2e-15 and 6.7e-14), against 9.9e-3 at order 1 and 2.5e-4
at order 2. The order-3 and
order-4 solves are therefore effectively PDE + inner data alone — consistent with
`lam_exact`, but they are not evidence about the condition. The ladder rows that exercise it
are orders 1 and 2, and the test asserts both cases (outer block non-zero below order 3,
zero from order 3 up), which is the annihilation property checked on all 30 basis functions
at once rather than on one field.

### 6.4 The background subtraction has to match on both sides

`B_n` sees the background: applied to a constant it gives `prod(1+i) = 6` at order 3, so a
residual built as `B[lam]` with `inf_val = 0` on one side and `B[lam_exact]` with
`inf_val = lam0` on the other carries a constant `6*lam0 = 3`. That is not a rounding-level
error: the first version of the gradient solve had exactly this, and its initial loss was
900 (`= 100 * 3^2`) instead of 0.42, with the optimiser spending its budget cancelling a
constant. Both sides must be `B[. - lam0]`; the tests assert the vanishing of the residual
rather than trusting the convention.

### 6.5 An order ramp works, but the budget split decides whether it helps

With the source every phase has the same exact solution, so the curriculum cannot
"disagree" about the target — it is a conditioning device, and its value shows up in the
budget split. Each phase restarts Adam at the full learning rate, and the first steps of a
phase make the residual of the *newly imposed* condition worse, so the highest order needs
most of the steps. All at a fixed 2400-step total:

| schedule | final max error | wall time |
|---|---|---|
| fixed order 3, 2400 | 1.11e-03 | 5.4 s |
| fixed order 1, 2400 | 1.11e-03 | 2.1 s |
| ramp 1->2->3, 800/800/800 | 3.51e-03 | 11.9 s |
| ramp 1->2->3, 400/400/1600 | **4.76e-04** | 8.8 s |
| ramp 1->2->3, 300/300/1800 | 4.40e-04 | 8.9 s |
| ramp 1->2->3, 1200/1200/1200 | 6.65e-04 | 8.9 s |
| ramp 1->2->3->3, 600 x4 | 6.77e-03 | 12.4 s |
| ramp 1->2->3->4, 600 x4 | 6.77e-03 | 21.9 s |

Stability of the shipped `400/400/1600` schedule over seeds: 4.76e-04 (seed 0), 1.03e-03,
1.68e-03, 2.29e-03 — 2.2x to 10.5x inside the 5e-3 tolerance. The test uses seed 0.

The two fixed-order rows agreeing to four digits is not a copy error. The outer block is
either vacuous (order 3, §6.3) or negligible in weight (order 1, whose residual is ~1e-5
against an inner residual of ~1e-3), so both runs are effectively solving PDE + inner data;
their coefficient vectors end up 7.8e-08 apart. Order 1 gets there in a fifth of the time
because its residual costs one radial derivative instead of three.

So **with a manufactured source, the order barely changes the solve at `rho_out = 100`**.
That is not an argument for low orders — it says this test cannot see the difference, for
the reason in §6.3. The higher orders matter in the source-free problem, where they decide
which multipoles cross the boundary, and that is what §4 measures.

## 7. What the collocation solve does, and its accuracy

`design(order)` assembles one least-squares row per collocation point and per condition: the
derivative of each residual with respect to the coefficients (reverse-mode, one sweep per
point — cheaper than evaluating the 30 basis functions one at a time). The system is
*consistent*: `lam_exact` zeroes every row, so the inner/outer weights cannot bias the
answer. Solving it with `jnp.linalg.lstsq`:

| order | max\|lam_c - lam_exact\| | cond(A) |
|---|---|---|
| 1 | 5.551e-16 | 2.03e16 |
| 2 | 6.661e-16 | 2.52e16 |
| 3 | 6.661e-16 | 3.64e16 |
| 4 | 5.551e-16 | 1.72e16 |

The condition numbers are large — the `d^3` column spans six decades even after
normalisation, so the matrix is numerically rank-deficient — but the returned function is
the exact solution to machine precision at every order and at both batch sizes tried,
because the system is consistent and that solution lies in its row space.
`TOL_SOLVE = 1e-9` leaves seven orders of magnitude of headroom.

## 8. Not covered here

* The metric and connection Robin conditions (`h` with `base = 2`, `Gamma` with `base = 3`).
  Their operator behaviour is covered by `test_robin_operator_annihilates_powers` and
  `test_robin_fourth_order_kills_only_first_four_powers` in `tests/test_pipeline.py`; the
  `h -> I` background and the reason `--no-robin-G` exists are a separate matter.
* The full Einstein coupling, the gauge condition, and anything about the inner metric data.
* Whether the production optimiser converges — §6.2 shows a plain MLP cannot even represent
  the reference here, so nothing in this file validates the production training dynamics.
* Tensor-valued fields: `robin_operator` is applied componentwise, and this file only
  exercises a scalar field.

## 9. Files

* `tests/test_laplace_robin.py` — the tests and the solves (only file added).
* `stationary/losses.py` — `robin_operator` is the operator under test; the wrapper that
  used to permute its last two arguments in `outer_bc_terms` was removed so that every call
  site now uses its single argument order `(field_fun, x, base, order, inf_val)`.
