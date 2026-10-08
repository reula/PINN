# DSGNAR in this repository

A record of the doubly-sketched Gauss–Newton optimiser: what it is, how it is wired into
`stationary`, what it cost to fit it to *this* problem (three out-of-memory failures, all of them
in the optimiser's own workspace), what is pinned by tests, and what it cannot do.

Source: *An Optimisation Framework for the Well-Conditioned Training of Physics-Informed Neural
Networks*, Webb, Jerad & Cartis, arXiv:2607.02194v1. Reference implementation
<https://github.com/wephy/physics-informed-neural-networks>. A working, smaller-scale copy lives
in this repo at `Evolution_try/` (1+1 wave equation), and `stationary/dsgnar.py` is a vendored copy
of `Evolution_try/wave_pinn/optim/dsgnar.py` with two changes: it takes `problem.Config`, and its
objective is duck-typed (`n`, `dtype`, `_residual`, `_loss`, `residual`, `loss`) instead of the
reference's `Objective` class.

---

## 1. What the method is

One iteration:

1. draw the sketch operators — a CountSketch `C` on the residual/row index and an SRCT embedding
   `Omega S` on the parameter/column index;
2. build the square doubly-sketched Jacobian `J~ = C J Omega S` in `R^{s x s}` with `s` batched
   Jacobian-vector products (`jax.linearize` once per iteration, then `vmap`), and `r~ = C r`;
3. take **one** SVD `J~ = U Sigma V^T` and use that single factorisation to solve the
   Levenberg–Marquardt subproblem at `q = 25` geometrically spaced trust-region radii;
4. lift the sketched steps back to the full parameter space, evaluate the true loss at each, and
   pick the radius whose *measured* decrease ratio is closest to (just above) a target `rho*`;
5. accept iff the loss strictly fell, move the trust region, and raise `rho*` from the
   conditioning stage (`0.15`) to the descent stage (`0.5`) once `lambda` has bottomed out.

The reason to try it here is not only speed. Our loss is a weighted sum whose weights span eight
orders of magnitude in `rho` (`scale_exps`, `scale_ref`), which is exactly the badly scaled
least-squares shape the trust region and the `lambda` solve are designed for. The reference
project's own comparison, at 2201 parameters and 2048 collocation points:

| run | optimiser | final loss | iterations | wall |
|---|---|---|---|---|
| `Evolution_try/runs/dsgnar_ref` | dsgnar | **2.98e-11** | 150 | 918 s |
| `Evolution_try/runs/ssbroyden_ref` | ssbroyden | 4.70e-10 | 9753 | 4279 s |

---

## 2. What it needs from the problem, and how it is provided

### 2.1 The loss as a residual vector — `losses.residual_vector`

DSGNAR is a least-squares method, so it needs `r(theta)` with `L = sum_i r_i^2`. Our loss is
already a sum of weighted means of squares, so this is a change of form, not of objective:

```
r_i = sqrt(weight_i / n_i) * (the i-th unreduced component),   sum_i r_i^2 == total_loss
```

`pde_terms`, `inner_bc_terms`, `outer_bc_terms` and `outer_pin_terms` were split into
`pde_raw` / `inner_bc_raw` / `outer_bc_raw` / `outer_pin_raw` (per point, unreduced) with the
existing functions now the means of their squares — identical numbers, reduction deferred.

The two pin reductions must be kept apart and are the easiest thing to get wrong: an *averaged*
pin is `mean(R)**2`, so its residual is the single number `mean(R)`, while a *value* pin is
`mean(R**2)` and keeps one row per point.

Row count at the `pq_c100` recipe (`n_coll 27768`, `n_bnd = n_bnd_outer = 4548`, metric-only model,
`eq_weights` with `compat = 0`):

```
13 * n_coll + 7 * (n_bnd + n_bnd_outer) + 1  =  424 657 rows
```

where `13 = ricci 9 + gauge 3 + lam_eq 1` per collocation point, `7 = h (packed symmetric, 6) +
lambda (1)` per boundary point, and the `+1` is the averaged lambda pin. (Checked against a
measured 33 793 rows at `n_coll 2048`, `n_bnd 512`.)

The identity `sum(r**2) == total_loss` is asserted by
`tests/test_dsgnar.py::test_the_residual_vector_squares_to_the_total_loss` in seven
configurations, which is what keeps the two paths from drifting apart.

### 2.2 The flat-parameter view — `flat_objective.FlatObjective`

`R^n -> R^M` and `R^n -> R` wrappers around the same objective, with `unflatten` from
`jax.flatten_util.ravel_pytree`. The residual is returned scaled by `sqrt(2 M)` so that

```
loss(flat) := 0.5 * mean(residual(flat)**2)  ==  total_loss(state, ...)
```

That factor is not cosmetic: DSGNAR predicts a model decrease for `0.5 * ||r||^2` and multiplies it
by `1/M` (its `scale`) to compare it with the loss it was handed, so the loss it is handed has to
be `0.5 * mean(r^2)`. With the scaling above, the number that reaches `history.json` is the same
one the other phases report.

### 2.3 The phase — `dsgnar_qn_phase` in `train.py`

`--qn-method dsgnar` enters the same quasi-Newton section as `ssbroyden`/`lbfgs` and returns the
same `(state, history, weights)` triple on the same cumulative iteration counter, so the log line,
`qn_stopped_at`, the report and the checkpoints are unchanged. Two differences, both printed
rather than silent:

* the sample is refreshed from inside the phase through the `resample` callback, on the same
  `--resample-every` counter (key `seed + 777 + cumulative iteration`, as in `ssbroyden_phase`);
* the gradient-norm **reweighting is not applied inside this phase** — the trust region and the
  `lambda` solve are what handle the scaling, and reweighting mid-phase would change the objective
  the sketched Jacobian describes.

There is no resume support for this phase: a `--resume` restarts it.

---

## 3. What it cost: three workspaces, three OOMs

The method was tuned on a 2201-parameter network with 2048 collocation points. Here the network
has **2285 parameters** (width 20, depth 6, `--decay-feature`) but **27768 collocation points and
second-order derivatives**, i.e. ~425k residual rows, so every workspace is far larger. Three
allocations had to be found by measurement, each one on the hub, each time with the failure
surfacing somewhere unhelpful:

| # | allocation | size asked for | where it surfaced | fix |
|---|---|---|---|---|
| 1 | JVP tangents: all `s` tangents through the whole residual at once | **4.78 GiB** | `runs/pq_c100_vacF_dsgnar_probe_s128` (first attempt) | `--dsgnar-chunk 16` |
| 2 | CountSketch workspace `rows x s x K` | 8.87 GiB (mis-attributed) | `jit__where` | `--dsgnar-row-chunk 4096` |
| 3 | `vmap(loss_fn)` over the `q = 25` trust-region probes | **8.87 GiB** | `dsgnar.py: rho_host = np.asarray(rhos)` — a 25-element array! | sequential `jax.lax.map` |

The third one is the instructive one. The traceback pointed at

```
rho_host = np.asarray(rhos, dtype=np.float64)
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate
8.87GiB ... [executable_name='jit__where']
```

and `rhos` holds 25 numbers. JAX executes lazily, so the allocation belongs to the computation
that *produced* `probe_losses`; `jit__where` is the last step of that chain, not the sketch. The
run with `--dsgnar-row-chunk 4096` failed identically, which is what ruled out the CountSketch.
XLA's own memory analysis settled it (CPU backend, `n_coll 2048`, 512 + 512 boundary points,
width 20 depth 6, 2285 parameters, 33 793 rows):

| piece | temp memory |
|---|---|
| loss, one evaluation | 0.021 GiB |
| `jax.vmap(loss_fn)` over the 25 probes | **0.492 GiB** |
| `jax.lax.map(loss_fn, ...)` over the same 25 | 0.057 GiB |

Scale by ~12.6 for the production row count and the vmap is the 8.87 GiB. The fix evaluates the
same 25 probes sequentially, in one workspace.

Knobs (all numerically transparent — the grouping of sums changes, not the sums):

| flag | default | what it caps |
|---|---|---|
| `--dsgnar-sketch s` | 0 (= `n/3` = 761 here) | the `rows x s` column matrix (`s = 128` → ~0.4 GiB, `s = 256` → ~0.9 GiB) |
| `--dsgnar-chunk N` | 0 (all at once) | tangents per JVP block; the launcher uses 16 |
| `--dsgnar-row-chunk N` | 4096 | CountSketch rows per block |
| `--dsgnar-steps` | 200 | iterations (each rebuilds a sketched Jacobian) |
| `--dsgnar-delta0` / `--dsgnar-omega` | 1.0 / 1e-8 | initial trust region / LM floor |

`--dsgnar-sketch 0` (the reference default, `n/3`) is **not usable here**: 761 columns make the
column matrix ~2.6 GiB before any workspace. The remaining per-run cost is the DCT matrix of the
SRCT embedding, `n^2` floats, built once per phase: 41.8 MB at 2285 parameters, but 1.5 GB at
13828 (the size quoted in the `qn_method` comment in `problem.py`), so it grows quadratically with
the network.

Commits: `44c07d6` (residual vector, adapter, vendored phase, `--qn-method dsgnar`), `5c473fb`
(JVP chunking), `fd0a91e` (CountSketch row chunking), `679d54e` (sequential probes).

---

## 4. How to run it

```bash
cd ~/serafin/Julia/PINN/Stationary
git pull
PROBE=1 SKETCH=128 bash run_dsgnar.sh      # 5 iterations from runs/pq_c100_vac3/params.pkl
SKETCH=128 STEPS=60 bash run_dsgnar.sh     # the polish
INIT=runs/pq_c100_vacF_phihyb25/params.pkl OUT=runs/pq_c100_vacG_dsgnar_F bash run_dsgnar.sh
```

`run_dsgnar.sh` starts from an existing run's `params.pkl` with `--steps 0` (the Adam warm-up *is*
the run you start from), keeps the `pq_c100` recipe otherwise, and waits for the detached run.
Environment: `INIT` (default `runs/pq_c100_vac3/params.pkl`), `OUT`, `SKETCH` (128), `CHUNK` (16),
`ROWCHUNK` (4096), `STEPS` (60), `PROBE` (0/1), `TAG`. `PY` defaults to `./.venv/bin/python` — the
system `python` has no JAX, which is how the first attempt failed before the launcher was fixed.

While it runs:

```bash
nvidia-smi                                   # the process and its memory
tail -f logs/<run>.log                       # nothing until the first [dsgnar] step line
kill -0 $(cat runs/<run>/run.pid) && echo running || echo stopped
```

Expect a long silent gap after the config block: that is XLA compiling the residual, the sketched
JVP, the `s x s` SVD and the probe evaluations, with GPU-Util at 0 %. Then read

```
[dsgnar] step  1  loss ...  rho +0.665  lam 1.2e+02  radius ...  target 0.15  (Xs)
[dsgnar] DSGNAR: loss ... -> ... in N iterations (Xs); accepted ... rejected ...
[dsgnar] timing: batched JVPs Xs, probe evaluations Ys, other Zs
```

`rho` near the target means the trust region is behaving; many rejections mean it is too large. If
the JVPs dominate the timing, lower `s`; if `other` dominates, the cost is the `s x s` SVD and the
per-iteration Python, which `s` also controls.

---

## 5. Tests — `tests/test_dsgnar.py` (13)

* `sum(residual_vector**2) == total_loss` in seven configurations: plain Robin, an averaged pin,
  the three value pins, the h pins, both combined, the radial-derivative term, and a manufactured
  Robin source;
* the residual vector vanishes identically on the exact solution (the `test_pipeline` setup);
* the weights live in the vector, not in the reduction;
* the adapter: `_loss` equals `total_loss`, and the phase lowers the loss end to end;
* chunked `jvp_columns` == unchunked, including non-divisible `s` (the JVP OOM);
* chunked `count_sketch` == unchunked, matrix and vector paths (the CountSketch workspace);
* the 25 probe losses evaluated sequentially == the vmap values, **and** the sequential
  workspace is at least 5x smaller by XLA's own analysis (the probe OOM) — so a regression that
  puts the `vmap` back fails locally, not on the hub.

---

## 6. What it cannot do

* **It minimises the same objective.** DSGNAR cannot fix a mode the loss does not constrain. The
  spurious dipole is exactly that: both outer conditions are spherical-mean objects (the order-3
  Robin on the averaged combination, the order-1 pin on the mean), there is no metric boundary
  condition in these runs (`outer_h = 0`), and the measured `S_10` in `lambda` flips sign and
  varies by 20x between runs whose losses differ by a factor of 30 (vac3 -3.2e-2, B -9.1e-2,
  C -4.9e-3, E +2.7e-2, F +3.9e-2) while the inner data impose `S_1 = 0` and the exact reference
  gives 1e-15. A better optimiser will drive the *weighted* residuals lower without necessarily
  moving that number; the fix is a per-multipole pin, i.e. putting it in the loss.
* **`--ricci-lam-source 0` leaves a scale degeneracy.** With the lambda source removed from the
  Ricci equation the system (Ricci = 0, the lambda equation, the zero-source gauge) is invariant
  under `h -> c h` for constant `c`: the Christoffel symbols are unchanged, so Ricci is; the
  lambda-equation terms scale by `1/c`; the gauge term scales by `1/c` and stays zero. Only the
  inner metric datum pins `c`, to ~3e-6 (measured: `h = (1 + c) I` with `c = -2.95e-6` in E and
  `+4.52e-6` in F, constant in `rho`; the reference asset decays instead, `g2 - 1 ~ rho^-2` and
  `h_rr - 1 ~ rho^-3`). The geometric invariants of those runs are therefore not the physical
  spatial geometry, and `--pin-h-robin` (averaged order-1, bases 3 and 2, no reference needed) is
  the knob that sees a constant offset.
* **The sketch is much smaller than the reference default.** `s = 128` against `n/3 = 761` is a
  heavily rank-deficient GN model; whether it captures the important curvature directions on this
  problem is not yet established. A row-subsampled sketch (drawing the JVP/sketch from a subset of
  the 425k rows while measuring the loss on all of them) would allow a larger `s` and is the
  obvious next lever if `s = 128` proves too small.
* **No reweighting and no resume inside the phase** (see section 2.3).

## 7. Open questions

1. Does `s = 128` on the `pq_c100` recipe lower the loss below what SSBroyden reached
   (vac3 8.7e-13, B 2.5e-13, E 8.9e-14, F 1.3e-14) and, more to the point, does it move the
   residual table or the multipoles? The first honest measurement is the probe run.
2. Is the cost per iteration acceptable at this problem size, given that each iteration rebuilds
   the sketch, and SSBroyden's iterations are ~0.2 s?
3. Should the row sketch be subsampled (`--dsgnar-sketch-rows`) to buy a larger `s`?
4. Does DSGNAR plus a per-multipole pin (`S_1 = 0` at `rho_out`, and optionally `S_2`) finally
   remove the spurious dipole, now that the pinned mode is inside the objective?
