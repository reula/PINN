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
| `--dsgnar-chunk N` | 0 (all at once) | tangents per JVP block; the launcher uses 16. Blocks are zero-padded to a multiple of N, never unrolled: the default sketch is `floor(n/3)` = 761, prime, and a divisibility rule would have silently lost the memory bound on every run that used it |
| `--dsgnar-row-chunk N` | 4096 | CountSketch rows per block |
| `--dsgnar-hashes K` | 4 (launcher: 8) | CountSketch hashes; K = 4 stalls on this problem (§7) |
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

## 7. First measurement: it runs, and a converged start has nothing to accept

`runs/pq_c100_vacF_dsgnar_probe_s128` — `--dsgnar-steps 5 --dsgnar-sketch 128 --dsgnar-chunk 16
--dsgnar-row-chunk 4096`, starting from `runs/pq_c100_vac3/params.pkl`. After the three workspace
fixes it ran to completion (exit 0) with no OOM:

```
[dsgnar] step      1  loss 1.036331e-12  rho -181400813.446  lam 4.222e-08  radius 3.333e-01  target 0.15  (29.4s)
[dsgnar] DSGNAR: loss 1.036331e-12 -> 1.036331e-12 in 5 iterations (37.9s); budget; s=128, accepted 0, rejected 5
[dsgnar] timing: batched JVPs 24.1s, probe evaluations 4.6s, other 9.2s
```

Two things to read from it.

**The cost is fine.** 37.9 s for 5 iterations, of which 29.4 s is the first-iteration compile, so
the steady state is about **2 s per iteration** (128 JVPs in blocks of 16 dominate, then the 25
probe evaluations, then the `s x s` SVD and the per-iteration Python). A 200-iteration run is
therefore minutes, not hours — against SSBroyden's ~0.2 s per iteration but thousands of them.

**The start has nothing to accept.** The starting point is `pq_c100_vac3`'s final state, i.e. the
end of 11019 Adam + SSBroyden iterations, at a plateau. For a near-stationary point the
Gauss–Newton model predicts a decrease of essentially zero, so `rho = actual / predicted` is a
0/0 artifact — the printed `-1.81e+08` — and since acceptance requires the measured loss to fall
strictly, every probe is correctly rejected. The LM regulariser was pushed to its floor
(`lam 4.2e-08` against `omega = 1e-8`) and the trust region shrank from 1.0 to 0.33 without
finding a step that helped. None of this is a defect of the optimiser or of the wiring: it
matches what the objective says, namely that this point is already a minimum of *this* loss.

So the honest test needs a starting point with room to descend. The same run's own numbers give
the bar:

| | loss |
|---|---|
| `pq_c100_vac3` at the end of Adam (step 5000) | 8.27e-3 |
| ... after its first SSBroyden block (1500 iterations) | 4.75e-11 |
| ... at the end (11019 iterations) | 8.65e-13 |

`run_dsgnar.sh` can start from any of them:

```bash
INIT=runs/pq_c100_vac3/params_adam.pkl OUT=runs/pq_c100_vacH_dsgnar_fromadam \
  SKETCH=128 STEPS=100 bash run_dsgnar.sh
```

If DSGNAR takes 8.3e-3 to ~1e-11 in ~100 iterations at ~2 s each, it is a ~10x cheaper route to
the same place; if it stalls or rejects, the sketch rank — not the point — is the suspect, and the
knobs to try are `SKETCH=256` (the column matrix grows to ~0.9 GiB) and a cold start from a random
init, which is how the reference project used the method.

### Third measurement: it works once the model is not noise-limited (and how it was found)

The stall was **not** the step, the accept rule or the trust region; it was the sketched model's
*estimated decrease*, which on this problem is both biased and noise-dominated.  Measured at
`n = 2285`, `n_coll = 1024`, from the Adam end (loss 3.76e-02), against the exact quadratic model
of the very same step (`g.p` and `½ pᵀJᵀJp/M`, two extra evaluations):

| s | sketch predicts | exact model | ratio | actual decrease | rho (exact) |
|---|---|---|---|---|---|
| 64 | 1.57e-02 | -1.15e-03 | -13.7 | -8.19e-03 (loss *rises*) | — |
| 128 | 1.51e-02 | +1.25e-03 | 12.1 | +1.05e-03 | 0.84 |
| 256 | 1.89e-02 | +2.35e-03 | 8.0 | +2.35e-03 | 1.00 |
| 512 | 1.57e-02 | +5.09e-03 | 3.1 | +3.34e-04 | 0.07 |

The sketch's prediction is *flat* (~1.6e-02) while the exact model varies by 4x: it is dominated by
the row sketch's noise, not by the curvature.  With a 8-12x over-estimate every ratio lands near
0.02 against the 0.15 target, the PCHIP always chooses the smallest radius, and 3^-28 = 1e-13 is
exactly the walk to `delta_min` -- which is why every run stopped after ~28 iterations.

Two changes, both now default and both tested (`tests/test_dsgnar.py`):

* **`--dsgnar-hashes K`** (default 4, the reference's): the row sketch's variance falls like `1/K`.
  `K = 8` is what makes the model track the function here.
* **model calibration** (`--dsgnar-no-calibrate` turns it off): divide the predicted decrease by the
  median measured/predicted ratio over the 25 probes, which already evaluate the true loss.  Only
  over-optimism is corrected (the factor is clamped to >= 1).
* **safeguarded step length** (`--dsgnar-backtrack`, `--dsgnar-backtrack-steps`): scan fractions
  1, 1/2, 1/4, ... of the *same* lifted direction and take the largest that lowers the loss; the
  trust region then follows the accepted fraction.  The algorithm's accept rule is unchanged ("the
  objective strictly fell"); this is a safety net, and with the calibration it is not needed.

With `s = 256, K = 8`, from the same Adam end at the same reduced size, 15 iterations, **15 accepted
/ 0 rejected**, all at fraction 1.0:

    3.7626e-02 -> 1.1188e-02      (70% of the loss in 15 iterations, monotone, 322 s on CPU)
    losses: 3.76e-2 3.76e-2 3.66e-2 3.60e-2 3.47e-2 3.36e-2 3.21e-2 3.00e-2 2.76e-2
            2.53e-2 2.26e-2 2.00e-2 1.73e-2 1.50e-2 1.28e-2 1.12e-2

The ablation, same start and same 15 iterations, says which knob is which:

| configuration | loss after 15 | reduction | accepted/rejected |
|---|---|---|---|
| `s = 128, K = 8`, calibrated | **4.58e-03** | **87.8%** | 15/0, every step at full length |
| `s = 256, K = 8`, calibrated | 1.12e-02 | 70% | 15/0 |
| `s = 512, K = 8`, calibrated | 9.87e-03 | 74% | 15/0 |
| `s = 256, K = 8`, NOT calibrated | 2.91e-02 | 22.6% | 12/3, mixed lengths |
| `s = 128, K = 4` (the reference default), calibrated | 3.73e-02 | 0.94% | 13/2, lengths 1/2, 1/4 ... |

So the hash count is the dominant knob -- `K = 4` is what made every run stall, and `K = 8` fixes
both the prediction and the direction, with the calibration needed to make the ratios honest (and
the safeguard a net that this configuration never needs).  The recipe to run on the GPU is
`--dsgnar-sketch 128 --dsgnar-hashes 8 --dsgnar-chunk 16` (calibration and safeguard are on by
default): 128 keeps the `M x s` column matrix at ~0.4 GiB at the production row count, and `K = 8`
only doubles the (minor) CountSketch work.  At ~13% of the loss per iteration, ~150 iterations
should take the r10 Adam end (6.6e-04) to the 1e-11 range.  The stop condition to watch is no
longer `radius below threshold`, which is what the previous three runs reported.

### Fourth measurement: the production run works, and one last piece of the radius rule

`runs/pq_r10_phihyb025_dsgnar2` -- the recipe above, from the Adam end of the r10 shell
(loss 5.399738e-04), on the GPU:

```
[dsgnar] step      1  loss 4.212520e-04  rho +0.358  lam 5.756e+02  radius 1.667e-01  target 0.15  (44.3s)
[dsgnar] DSGNAR: loss 5.399738e-04 -> 8.332133e-06 in 76 iterations (206.6s); radius below threshold
         s=128, accepted 72, rejected 4, stage2=True
[dsgnar] timing: batched JVPs 149.2s, probe evaluations 5.0s, other 52.4s
```

**65x down in 3.6 minutes, 72 of 76 steps accepted, `rho` median 0.153 against a target of 0.15** --
the trust-region control is now nominal, where before it was 4/28 with `rho ~ 0.02`.  (The report
also labels the phase DSGNAR now, which is how the earlier stall had been invisible.)

It still ended on `radius below threshold`, and the trajectory says why: the radius went
1.0 -> 0.167 -> ... -> 1.0e-14 while *72 steps were being accepted*, because the paper's rule moves
the radius to the PCHIP radius `delta_star`, and with a model whose calibration factor runs 60-80
mid-run `delta_star` is systematically smaller than the radius that just produced a good step.  So
a *successful* step shrank the trust region and the run stopped for a reason the loss knew nothing
about.  Since the safeguarded length makes a too-large radius safe (the scan trims it), the radius
now shrinks only on a **rejected** step (`--dsgnar-shrink-on-accept` restores the paper's rule).
Same local problem, 60 iterations with that change: **60 accepted / 0 rejected, 3.7626e-02 ->
1.0700e-04 (99.7%), stopped on budget**, the lengths trimmed to 1/2, 1/4, 1/8, 1/16 ... as the run
progresses.  That is the configuration to use on the GPU.

### Second measurement: from the end of Adam it stops after 28 iterations

`runs/pq_c100_vacH_dsgnar_fromadam` — the same flags, started from the Adam end (loss 8.27e-3):

```
[dsgnar] step 1  loss 8.315069e-03  rho -0.613  lam 5.634e+03  radius 3.333e-01  target 0.15
[dsgnar] DSGNAR: loss 8.315069e-03 -> 8.290791e-03 in 28 iterations (92.1s); radius below threshold
         s=128, accepted 4, rejected 24, stage2=False
[dsgnar] timing: batched JVPs 70.0s, probe evaluations 0.8s, other 21.3s
```

**0.3 % of the loss in 92 s, then `radius < delta_min` and exit.** For scale: SSBroyden's first
1500-iteration block took the same start from 8.27e-3 to 4.75e-11. Per unit wall time DSGNAR is
about nine orders of magnitude behind here, with 24 of 28 probes rejected.

The failure mode is structural, not a knob that was set wrong: the trust region lives in the
**sketch** space (`s = 128`), and the step is lifted to the 2285 parameters by the SRCT embedding,
whose columns are not unit-norm. If the lifted step overshoots, the probe is rejected, the radius
is multiplied down, and the radius is the only thing that moves -- so 24 rejections walk it from
1.0 to `delta_min = 1e-14` geometrically and the phase stops. The large `lam 5.6e3` at the first
step says the same thing from the other side: the sketched subproblem is regularisation-dominated,
i.e. the `s = 128` model of a 2285-parameter, rho^d-weighted second-order system is a poor one.

If it is worth one more try, the suspect to change is the *initial scale*, not the sketch:
`--dsgnar-delta0 1e-3` (or smaller) makes the first lifted step small enough to be acceptable, and
only then does `STEPS` mean anything. `SKETCH=256` is the other half (below). If neither moves it,
the conclusion is that this optimiser, tuned on a 2201-parameter smooth wave problem, does not
transfer to this objective, and the levers that have evidence here are the batch noise (more
points) and the missing conditions (the multipole pin) -- not the optimiser.

## 7b. The sketch size, unblocked: the fused JVP + CountSketch

Evolution_try measured the column sketch as the dominant knob there (`s = d/3 -> d/2` worth
~1000x, and the sketched model honest to 1.0-3.3x), while this problem has run at `s = 128 =
d/18` -- where the model over-predicted 12x and needed `K = 8` and the calibration as crutches.
What capped `s` was memory: the `M x s` column matrix `J Omega S`, ~3.7 GiB at `s = 1100` and
the production row count.  `dsgnar.sketched_jvp_columns` removes it: each block of `chunk`
tangents goes through the JVP and is CountSketched at once (the sketch is linear and acts column
by column), so the live workspace is `M x chunk` whatever `s` is.  Same hash draws, same columns
(`tests/test_dsgnar.py::test_the_fused_sketch_is_the_two_step_sketch`); `--dsgnar-row-chunk` is
now unused by the phase.

`run_batch_sketch.sh` measures it: one Adam warm-up, then SSBroyden and DSGNAR at
`s = 128, 512, 1100` (K = 8) from the same parameters, summary in
`logs/batch_sketch.summary.txt`.

## 8. Open questions

1. Does `--dsgnar-delta0 1e-3` (a smaller lifted first step) turn the 28-iteration stall into
   descent? This is the last cheap test before parking the optimiser: two starts have now failed
   to make progress (0/5 accepted at the plateau, 4/28 from the Adam end).
2. Is the cost per iteration acceptable at this problem size, given that each iteration rebuilds
   the sketch, and SSBroyden's iterations are ~0.2 s?
3. Should the row sketch be subsampled (`--dsgnar-sketch-rows`) to buy a larger `s`?
4. Does DSGNAR plus a per-multipole pin (`S_1 = 0` at `rho_out`, and optionally `S_2`) finally
   remove the spurious dipole, now that the pinned mode is inside the objective?

## 9. What the paper does that this phase did not (arXiv:2607.02194, re-read 2026-10-15)

Read against the PDF, not against the reference code.  All are switches, off by default;
`--dsgnar-paper` turns on the paper's set (explicit flags still override it).

| paper | here before | switch |
|---|---|---|
| per-condition reweighting, UpdateWeights, alpha = 0.05 (App. A.1) -- "essential", because CountSketch mixes every condition into the same s buckets | not applied inside the phase | `--dsgnar-reweight 0.05` |
| a step whose ratio overshoots the target by more than a margin is REJECTED and the radius expanded (3.4) | any decrease accepted | `--dsgnar-reject-overshoot` |
| the step is evaluated once | scan 1, 1/2, ... 1/16, 1e-2, 1e-3 | `--dsgnar-no-line-scan` |
| no calibration (the model is honest at s ~ d/3) | median calibration | `--dsgnar-no-calibrate` |
| radius to the PCHIP radius on accept, no floor | grow-only, floor 1e-3 of the high-water mark | `--dsgnar-shrink-on-accept --dsgnar-radius-floor 0` |
| stage-1 target 0.075 in double precision, Delta_min = 1e-8 (4.2) | 0.15, 1e-14 | `--dsgnar-stage1-ratio`, `--dsgnar-delta-min` |
| one sketch for the whole run (3.2.2; Algorithm 1 itself redraws -- the paper contradicts itself) | redrawn every iteration | `--dsgnar-fixed-sketch` |
| K = 4 hashes, s in [d/3, d/2], N >> s collocation points | K = 8 (launcher), s = 128 | `--dsgnar-hashes`, `--dsgnar-sketch` |

The reweighting is applied as a per-row scale passed to the jitted maps as an argument (no
recompilation when the weights move); each step is judged on the reweighted objective, as in the
paper, and the history's `loss` is the loss at the ORIGINAL weights, so `final_loss` stays
comparable with SSBroyden.  The conditions are the terms of `losses.residual_vector`
(`FlatObjective.blocks`, 12 on the pq recipe).

Not ported: the first ten iterations in single precision (an early-divergence guard from a poor
initialisation; the phase starts from an Adam warm-up), and the SIREN architecture.

First local check (n_coll 512, s = 300, 6 iterations from 30 Adam steps): with calibration off
the measured ratio tracked the 0.075 target (0.067-0.079), lambda fell 9.7e3 -> 18 -- the paper's
first stage -- 6/6 accepted, loss 31.4 -> 18.1.  `run_batch_sketch.sh` measures it at production
size against SSBroyden.

### 9.1 First production measurement -- on the OLD inner condition (2026-10-15, hub, `sk16k_*`)

From one Adam warm-up (5000 steps, loss 7.3e-2), n_coll 16384, n_bnd 1024, four outer shells.  These
runs imposed `h_rA = 0` on the inner sphere (fixed in 0773b6a, see PERTURBATION.md section 5), so the
problem had a floor near 2e-5 and none of the losses below is comparable with later runs; the
directories are in `runs/_old_inner_condition/` on the hub.  What carries over is the behaviour of
the trust region:

| run | final loss | wall | notes |
|---|---|---|---|
| SSBroyden | 3.1e-5 | 22 min, 4500 it. | stopped on plateau -- at the problem's floor |
| DSGNAR s = 1100, `--dsgnar-paper` | 2.7e-4 | 35 min, 208 it. | 205/208 accepted; ratio held at the 0.075 target (0.069-0.083) WITHOUT calibration; lambda 5.5e3 -> 0.2 (the paper's first stage); the reweighting then spread the 12 condition weights to 5e-7 .. 4 and the loss at the original weights stalled at ~2.8e-4 from iteration 100 |
| DSGNAR s = 1100, paper, no reweighting | 7.2e-4 at it. 70 | (killed with the container) | the loss halving every ~10 iterations; ~9 s per iteration, 90% of it the 1100 JVPs |

So at s = d/2 the sketched model is honest enough that the paper's rule works as written (no
calibration, no line scan), and the per-condition reweighting is the part that does NOT transfer to
a 12-condition problem whose conditions differ by orders of magnitude.

### 9.2 On the corrected problem (`ib16k_*`, hub, from 2026-10-15)

`run_batch_sketch.sh`: Adam -> SSBroyden; DSGNAR s = 1100 paper without and with reweighting, s = 512
paper without; each graded against the nonlinear spectral reference (PERTURBATION.md section 5).
Results (hub, 2026-10-15; Adam 5000 steps to 6.3e-2, then each arm from the same parameters):

| run | final loss | wall | iterations | stop |
|---|---|---|---|---|
| SSBroyden | **1.74e-8** | 94 min | 20000 (the cap) | still improving slowly |
| DSGNAR s = 1100, paper, no reweighting | 4.1e-5 | 17 min | 100 | radius below Delta_min |
| DSGNAR s = 1100, paper (reweighting) | 2.4e-4 | 28 min | 170 | radius below Delta_min |
| DSGNAR s = 512, paper, no reweighting | 1.3e-3 | 5 min | 49 | radius below Delta_min |

All three DSGNAR arms die the same way, at the **stage switch**.  Stage 1 works as the paper says
(the measured ratio sits on the 0.075 target, lambda falls; s = 1100 went 6.3e-2 -> 4.1e-5 by
iteration 90).  Then lambda turns up, the target jumps to 0.5, and no probe reaches it -- the
measured ratios stay at 0.07-0.3 -- so LambdaSolve returns the smallest radius every iteration, the
paper's radius rule divides the radius by 3 each time, and ~10 iterations later it is below
Delta_min = 1e-8 and the phase stops.  On this problem the sketched model is not honest enough for a
target of 0.5.  (In Evolution_try the same paper targets gave the project's best result -- README
6.13 -- so the rule is right there and wrong here.)

The nonlinear-reference comparisons of this batch failed on launch: the batch inherited the
notebook's `MPLBACKEND` (matplotlib_inline), which a plain Python process rejects.  The batch now
exports `MPLBACKEND=Agg`.

Next: keep stage 1 (target 0.075 throughout, or a reachable stage-2 target such as 0.15), and a
larger sketch (s = 1600-2000 of d = 2285).

Round 2 on Mendieta (`mendieta_submit.sh`, submitted 2026-10-16), all from the same `ib16k_adam`
parameters: DSGNAR s = 1100 with the stage-2 target at 0.075 (no switch) and at 0.15; s = 2000 at
0.075; DSGNAR's stage 1 (100 iterations) handing over to SSBroyden; SSBroyden continued from its
20000-iteration result for 40000 more.

Results (Mendieta A30, 2026-10-16; the hub's SSBroyden row above for comparison):

| run | final loss | wall | iterations | stop |
|---|---|---|---|---|
| SSBroyden (hub, from round 1) | **1.74e-8** | 94 min | 20000 | the cap |
| SSBroyden continued | 1.75e-8 | 11 min | 1000 of 40000 | plateau: no 1% gain in 3 blocks of 250 |
| DSGNAR s = 2000, target 0.075 throughout | 9.4e-8 | 68 min | 234 | radius below threshold |
| DSGNAR s = 1100, target 0.075 throughout | 1.8e-6 | 40 min | 211 | radius below threshold |
| DSGNAR s = 1100, stage-2 target 0.15 | 2.2e-5 | 23 min | 129 | radius below threshold |
| DSGNAR stage 1 only (100 it.) | 1.0e-4 | 17 min | 100 | the cap (hand-over to SSBroyden still queued) |

Read off:

1. **Removing the stage switch fixes the collapse but does not beat SSBroyden.**  With the target
   at 0.075 throughout, s = 1100 gets 1.8e-6 (round 1 with the paper's switch: 4.1e-5), and
   s = 2000 gets 9.4e-8, five times above SSBroyden's 1.74e-8, in 72% of its wall time (different
   GPUs: hub vs A30).  A reachable stage-2 target (0.15) is worse than no switch.
2. **The sketch size is again the dominant knob**: s = 1100 -> 2000 (0.48 d -> 0.88 d) is x20 in
   the final loss.  At s = 2000 each iteration costs ~17 s, 93% of it the batched JVPs.
3. **Every DSGNAR arm still ends by the radius, not by a budget**: at s = 2000 the last 20
   iterations have lambda going 3.5e2 -> 8.3e6 and the radius 1.4e-3 -> 4e-6 while the accepted
   steps keep their ratio on target (0.07-0.085, i.e. the measured decrease is ~1/13 of the
   predicted one) but gain less and less (9.84e-8 -> 9.35e-8).  Holding the ratio at 0.075 takes an
   ever smaller step: the sketched model is no longer informative at that loss level.
4. **SSBroyden is at its floor at 1.7e-8**: 1000 more iterations moved it by 1%.  The loss is
   dominated by the lambda equation (9.9e-9 of 1.75e-8).  So 1.7e-8 is where this network and
   sample stop for a quasi-Newton method, and DSGNAR has not gone below it.

So on Stationary DSGNAR is competitive in cost (s = 2000 in 68 min against 94 min) but not in the
loss it reaches (9.4e-8 against 1.7e-8).  The stage-1 -> SSBroyden hand-over (`m2_dsg_then_ssb`) was killed by
the node's memory limit after 2 h 16 min, at iteration 18250 of 20000 with the loss at 2.04e-8 and
the host RSS at 60.9 GB (SSBroyden's RSS grows ~3.3 MB per iteration): no faster to the same floor
than SSBroyden from Adam (1.74e-8 at 20000).  DSGNAR's fast first 100 iterations (6.3e-2 -> 1e-4
in 17 min) do not shorten SSBroyden's path.

Round 3 (submitted 2026-10-16): the same two optimisers with the residuals weighted by rho^4
(`--scale-exps 3,4,3,4`), the weighting under which the spectral reference's far field became
determined (PERTURBATION.md 5.2b): SSBroyden as two jobs of 10000 (the RSS limit above), DSGNAR
s = 2000 at target 0.075.

Results (Mendieta, 2026-10-16/17).  Under the rho^4 weighting the Adam warm-up's loss is 1.23e4
(the far rows, ~rho^2 = 1e4 heavier, were never trained), and:

* **SSBroyden did not move**: both jobs stopped at iteration 0 (`stop=stall`, line-search status 3)
  at 1.23e4.  The quasi-Newton start from a point that far off fails; it needs a start under the
  new weighting.
* **DSGNAR s = 2000 took it to 1.4e-6 in 300 iterations (86 min) and was still falling** (x6 per 30
  iterations at the end; stopped by the budget, not the radius; 286/300 accepted).

Its error against the rhoW nonlinear reference (`specref_summary`), against the rho^2 runs:

| run | rms dlam 1-3 | 3-10 | 10-30 | 30-100 | rms dh 1-3 | 3-10 | 10-30 | 30-100 |
|---|---|---|---|---|---|---|---|---|
| *perturbation size* | *2.0e-2* | *2.5e-3* | *7.6e-4* | *2.6e-4* | *2.2e-2* | *4.5e-4* | *3.1e-5* | *3.5e-6* |
| SSBroyden, rho^2 (loss 1.7e-8) | 4.1e-4 | 2.5e-4 | 2.1e-4 | 2.2e-4 | 5.1e-3 | 7.5e-4 | 9.2e-5 | 2.9e-5 |
| DSGNAR s = 2000, rho^2 (9.4e-8) | 8.6e-4 | 5.1e-4 | 4.0e-4 | 3.6e-4 | 1.1e-2 | 1.6e-3 | 1.5e-4 | 3.7e-5 |
| DSGNAR s = 2000, rho^4 (1.4e-6, unconverged) | 2.1e-3 | 1.0e-3 | 1.6e-4 | **1.9e-5** | 2.7e-2 | 3.9e-3 | 3.8e-4 | 3.9e-5 |

The rho^4 weighting removes the flat ~2e-4 lambda error of every rho^2 run: beyond rho = 30 the
error is 1.9e-5, 12x below SSBroyden's and below the perturbation (2.6e-4), unconverged.  Near the
inner sphere it is 2-5x worse, and h is not yet better anywhere.  (Its loss is in rho^4 units and
is not comparable with the rho^2 losses.)

Round 4 (submitted 2026-10-17), all from `m3_dsg_s2000_rho4`, rho^4 weighting: DSGNAR continued
300 iterations (`m4_dsg_rho4_cont`); SSBroyden 10000 from there (`m4_ssb_rho4`); and DSGNAR
continued with the Robin condition on rho_out only and HYBRID collocation, half log-uniform and
half uniform in volume (`m4_dsg_rho4_hyb_1sh`).  Every earlier run sampled log-uniformly only
(`radial = log`; the recipe's `radial_log_frac = 0.5` is unused unless `--radial hybrid`).
Fourth arm (`m4_dsg_far10`): the same continuation with the dimensionless exponents (1, 2, 1, 2)
and the new `--far-weight 10,4`, a LOSS weight 1 + (rho/10)^4 on the interior equations
(`Config.far_weight`; tests/test_far_weight.py).  "rho^4" (`--scale-exps 3,4,3,4`) multiplies the
loss by rho^4 from rho_in = 1 on, so the far rows outweigh the inner ones by 1e8; with the crossover
at rho_c = 10 the inner equations keep their dimensionless weight and the far rows gain 1e4.  (A
bare 1 + rho^4 would equal rho^4 beyond rho ~ 2 and change nothing.)

Results so far:

* `m4_dsg_rho4_cont`: 1.44e-6 -> 4.6e-7 in 61 iterations (18 min), then stopped by the radius
  (lambda 3e7, radius 1e-7; 45 accepted, 16 rejected).  Its error against the rhoW reference is the
  SAME as at its start, to two digits in every region (lambda 2.1e-3 / 1.0e-3 / 1.6e-4 / 1.9e-5,
  h 2.7e-2 / 3.9e-3 / 3.8e-4 / 3.9e-5): the parameters moved by at most 1e-3, and the 3x lower
  loss did not move the error.  At this point the error is not limited by the optimisation.
* `m4_ssb_rho4`: SSBroyden from the same start, 1.44e-6 -> 4.5e-7 in 1500 iterations (12 min),
  stopped by the plateau rule.  Its error against the rhoW reference equals DSGNAR's continuation
  in every region (lambda 2.1e-3 / 1.0e-3 / 1.6e-4 / 1.9e-5, h 2.6e-2 / 3.9e-3 / 3.7e-4 / 3.9e-5).
  Both optimisers reach the same loss floor (~4.5e-7, rho^4 units) and the same solution: under
  this weighting and sample the remaining error is not the optimiser's.
* `m4_dsg_rho4_hyb_1sh` (hybrid collocation, the Robin condition on rho_out only): 4.6e-6 on its
  new sample -> 4.53e-7 in 84 iterations, radius stop -- the same lam-pin floor (below).  Error
  against rhoW: lambda 2.0e-3 / 1.1e-3 / 1.6e-4 / **1.3e-5**, h 2.8e-2 / 4.0e-3 / 3.7e-4 /
  **2.7e-5**: the far field ~30% better than `m4_ssb_rho4` (1.9e-5, 3.9e-5), the rest unchanged.
  Small enough to be partly sample noise.
* `m4_dsg_far10`: cancelled before it ran (it had the lam pin); resubmitted as `m5_dsg_far10`.

**The rho^4 loss floor is the lam pin, not the optimiser** (2026-10-17).  On the same sample the
network (`m4_ssb_rho4`) and the rhoW nonlinear reference have the SAME loss, 4.5e-7, and in both
4.4e-7 of it is one term, `pin_mean_lam` = w_pin <(rho d_rho + 1)(lam - 1)>^2.  The exact spherical
solution violates that pin: lam = 1 - 2M/rho + 2M^2/rho^2 - ..., and rho d_rho + 1 leaves the
2M^2/rho^2 term, -6.59e-5 at rho_out = 100, so w_pin x that^2 = 4.34e-7.  Nothing can go below it.
On P = (1 - lam^2)/(4 lam), whose monopole has no rho^-2 term, the same pin costs 1.5e-11
(tests/test_pin_potential.py).  Under rho^2 weighting the runs instead bend lam near rho_out to
satisfy the pin, which plausibly is the flat ~2e-4 lambda error of every rho^2 run (untested).
New: `--pin-lam-potential` (the pin on P) and `--dsgnar-cycles N --dsgnar-cycle-steps K` (restart
DSGNAR N times, each on a fresh sample with a fresh trust region).

Round 5 (submitted 2026-10-17), all with the P pin: the nonlinear reference re-solved (rhoWP);
from `m4_ssb_rho4`, DSGNAR s = 2000 rho^4 in 5 cycles of <= 60 (`m5_dsg_cyc`), SSBroyden rho^4
without resampling (`m5_ssb_norsmp`), the far weight 1 + (rho/10)^4 (`m5_dsg_far10`); and a wider
network, width 40 x depth 6 (~8.6k parameters), Adam 5000 then SSBroyden 8000, rho^4 (`m5_w40`).

* `m5_dsg_cyc`: with the pin on P the loss goes 9.4e-9 -> 3.2e-9 in cycle 1 (51 iterations, 46
  accepted), and cycles 2-5 on fresh samples add little (1, 2, 6, 0 accepted; ~3.2-3.4e-9 on each
  new sample).  140x below the lam-pin floor.  Its error against the rhoW reference is AGAIN the
  same to two digits as every rho^4 run (lambda 2.1e-3 / 1.0e-3 / 1.6e-4 / 1.9e-5).

**So inside rho ~ 10 the rhoW reference is not more accurate than the networks.**  Dimensionless
residuals (rho^2 Ricci, rho^2 lambda equation; rms by region, 3000 points) and the inner data:

| | Ricci 1-3 | 3-10 | 10-30 | 30-100 | lam eq 1-3 | 3-10 | 10-30 | 30-100 | inner lam | inner h |
|---|---|---|---|---|---|---|---|---|---|---|
| `ib16k_ssbroyden` (rho^2, lam pin) | 6.8e-5 | 1.9e-5 | 1.5e-5 | 1.2e-5 | 5.6e-5 | 2.7e-5 | 4.8e-5 | 1.8e-4 | 9.8e-7 | 1.2e-6 |
| `m5_dsg_cyc` (rho^4, P pin) | 4.3e-5 | 1.3e-6 | 6.1e-8 | 4.9e-9 | 1.9e-5 | 8.0e-7 | 5.0e-8 | 4.1e-9 | 8.7e-7 | 8.7e-7 |
| rhoW reference (l <= 6, 24 radial) | 8.5e-5 | 4.4e-7 | 2.2e-8 | 1.9e-9 | 5.3e-5 | 1.1e-6 | 7.5e-8 | 1.6e-8 | 4.2e-8 | 2.4e-7 |

Near the inner sphere the network satisfies the equations better than the reference (Ricci 4.3e-5
vs 8.5e-5, lambda 1.9e-5 vs 5.3e-5); the reference fits the inner data better.  The "error"
~2e-3 near rho = 1, identical across four different rho^4 runs, is therefore mostly the
reference's own error there (its truncation l <= 6 against a 25% quadrupole), not the networks'.
The rho^2 run is clearly worse in the far field (lambda equation 1.8e-4 beyond rho = 30), which
fits its flat lambda error.  A reference good enough to grade the inner region needs more modes.

* `specref_rhoWP` (the reference with the pin on P, l <= 6, 24 radial): loss 2.6e-9 in 68 min --
  about the networks' level, as the residuals above say.
* `m5_ssb_norsmp`: SSBroyden from `m4_ssb_rho4`, P pin, no resampling: 9.4e-9 -> 2.1e-9 in 10000
  iterations (36 min), still falling slowly at the cap; the loss is now all equations (pin 3e-13).
  Its dimensionless residuals are 10-30% below `m5_dsg_cyc`'s in every region (Ricci 3.9e-5 /
  9.5e-7 / 5.4e-8 / 4.3e-9, lambda equation 1.4e-5 / 6.6e-7 / 3.4e-8 / 3.5e-9; inner lambda 8.8e-7,
  h 6.9e-7): with the floor removed SSBroyden is again ahead, by a little.
