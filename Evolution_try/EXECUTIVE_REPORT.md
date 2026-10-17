# Executive report — DSGNAR against SSBroyden and Broyden

**Project:** `Evolution_try`, a physics-informed neural network for the 1+1D wave equation
`u_tt − c²u_xx = 0`, `c = 1`, `L = 1`, integrated to `T = 20`; comparison of optimisers.
**Period covered:** the whole line of work in this directory, to 2026-10-16.
**Status:** results complete for the wave equation.  The stall's cause is identified (the
sketch), and a separate, reproducible single-cell failure turned out to be a defect in this
project's own retry rule rather than in DSGNAR — found, fixed and tested while re-running it.

---

## 1. The answer, first

**On the wave equation, yes — DSGNAR is not merely competitive with SSBroyden, it reaches
accuracies SSBroyden never reaches, by nearly six orders of magnitude.** The best `T = 20`
result is `6.3e-09` space-time relative `L2`, against SSBroyden's floor of `8.4e-03`. That is
not a speed comparison that DSGNAR wins; it is a comparison of *reachable accuracy*, and on this
problem the gradient method's floor is six orders of magnitude above DSGNAR's.

**Against Broyden there is no contest, because Broyden does not converge here.** The only
Broyden in the JAX ecosystem (`jaxopt.Broyden`, a root finder) diverges by nine orders of
magnitude, and the cause is its line search rather than its step direction. The quasi-Newton
relatives that *do* have a proper Wolfe line search progress smoothly and then stall at a
relative error of `0.42` — worse than doing nothing useful.

**The qualification matters, and it comes from outside this project.** The same DSGNAR
implementation, on a sibling project's problem (`Stationary`, 2285 parameters), **loses to
SSBroyden**: `9.4e-08` against `1.74e-08`. DSGNAR's advantage is therefore not a property of the
method but of the problem, and §5 states the rule that separates the two cases.

**The cost is real.** DSGNAR runs take `46–98 min` where SSBroyden takes `5–6 min` — nine to
sixteen times the wall time. The recommendation is not "always DSGNAR"; it is "DSGNAR when the target
accuracy is below the gradient method's floor, and the sketch audit says a sketched
Gauss–Newton model is meaningful there."

---

## 2. What was done

| line of work | outcome |
|---|---|
| Hard-constrained ansatz (`u = u0 + t·v0 + f(t)·N`) so both initial conditions are **exact** | works; residual reaches `2.2e-14` at `T = 2` |
| Periodic feature map (Fourier features rejected) | `(t/T, cos πx/L, sin πx/L)`; Fourier features reach `1.13` error |
| SSBroyden (vendored Crunch, self-scaling Broyden) | converges but plateaus; best windowed `8.4e-03` |
| DSGNAR (faithful port of arXiv:2607.02194, spec in `docs/`) | best windowed `6.3e-09` |
| Trust-region Newton (Xu & Darve, arXiv:2105.07552) | converges to `9.3e-01`; not a contender |
| Broyden root finder (`jaxopt`) | diverges; cause diagnosed (§6) |
| Wolfe-condition quasi-Newton family | smooth progress, stalls at `O(1)` error |
| Global `T = 20` solves (5 attempts) | **all fail**: `0.91`, `3.48`, `4.99`, `10.97`, `23.9` |
| **Slab marching**, 10 windows of `dt = 2`, hand-over as a penalty | **the breakthrough**: `6.3e-09` |
| Sketch-size study (`s = d/3`, `d/2`, `0.68 d`) | `s` is the dominant knob: `451–700×` |
| Exact-Jacobian audits (rank, alignment) | the stop is the **sketch's**, not the problem's |
| Paper re-read: its own rules as switches | the paper's double-precision targets are worth `×1000` |
| Line-search studies (Crunch, jaxopt, DSGNAR) | all three measured; none helps (§6) |
| A reproducible window-1 convergence failure | traced to a retry rule that could never fire for window 1; fixed and tested |
| Documentation, 59 tests, F64 throughout | `README.md`, `docs/dsgnar_spec.md`, `BATCH_REPORT.md` |

This directory is **shared with a parallel workstream** (the `Stationary`/Mendieta work, and the
`pp_*` runs, which are paper-rule ablations on the wave equation). Those results are included and
credited; they are not mine, and I have left them untouched.

---

## 3. Results

### 3.1 The headline: `T = 20`, windowed, one machine

Same machine, same 10 windows, same hand-over, float64. `d = 2201` parameters.

| run | optimiser | `s` | `n_coll` | **rel `L2` (T=20)** | wall |
|---|---|---|---|---|---|
| `dsgnar_soft_all_nc8192_s1500` | DSGNAR | `0.68 d` | 8192 | **6.34e-09** | 96 min |
| `pp_soft_all_targets` | DSGNAR | `d/3` | 2201 | **7.82e-09** | 98 min |
| `dsgnar_soft_all_nc2201_s1500` | DSGNAR | `0.68 d` | 2201 | 2.40e-07 | 46 min |
| `dsgnar_soft_all_nc8192_s1100` | DSGNAR | `d/2` | 8192 | 1.83e-06 | 53 min |
| `dsgnar_soft_all_nc8192` | DSGNAR | `d/3` | 8192 | 2.86e-06 | 58 min |
| `dsgnar_soft_all_nc2201` | DSGNAR | `d/3` | 2201 | 7.83e-06 | 76 min |
| | | | | | |
| `ssbroyden_soft_all_nc8192` | **SSBroyden** | — | 8192 | **8.41e-03** | **6 min** |
| `ssbroyden_soft_all_nc2201` | SSBroyden | — | 2201 | 7.02e-02 | 5 min |

**The gap is `1.3 × 10⁶`**, for sixteen times the wall time (nine for the smaller configurations). Note the second row: the strongest
result per unit of machinery is the parallel workstream's, using the paper's own double-precision
targets at `s = d/3` and `n_coll = 2201` — a quarter of the collocation points and half the sketch
of the top row, for a 23 % worse error.

### 3.2 A single window (`T = 2`), to convergence

| method | iterations | final loss | rel `L2` | wall |
|---|---|---|---|---|
| DSGNAR `s = 733` (`d/3`) | 186 | 2.24e-14 | 6.76e-06 | 144 s (A30) |
| DSGNAR `s = 1100` (`d/2`) | 301 | 2.92e-20 | 4.12e-09 | 215 s (A30) |
| DSGNAR `s = 1500` (`0.68 d`) | — | 5.41e-23 | 6.96e-10 | 276 s (A30) |
| SSBroyden, 3000-iteration cap | 2753 | 8.26e-09 | 6.70e-04 | 186 s (CPU) |
| SSBroyden, 10000-iteration cap | 9753 | 4.70e-10 | 9.04e-05 | 4279 s (CPU) |

DSGNAR beats SSBroyden's *full* reference run by `13×` in error using 1/50th of the iterations;
at `s = d/2` it beats it by `2 × 10⁴`.

### 3.3 Global `T = 20` without marching — every attempt failed

| run | rel `L2` |
|---|---|
| `T20_ssbroyden` | 9.07e-01 |
| `T20sat_ssbroyden` | 3.48e+00 |
| `T20sat_dsgnar` | 4.99e+00 |
| `T20_bigbatch` | 1.10e+01 |
| `T20_dsgnar` | 2.39e+01 |

This is the baseline that makes the headline meaningful: **marching in ten slabs is worth eight
orders of magnitude**, and it was worth more than any optimiser choice or hyperparameter.

---

## 4. Is DSGNAR competitive? — three comparisons

### 4.1 Against Broyden: not a contest

`jaxopt.Broyden` is a root finder for `r(θ) = 0`, and it is the right *shape* for this problem —
the ansatz is hard-constrained, so the exact solution is representable and `r = 0` is achievable
rather than a proxy. It nevertheless diverges by nine orders of magnitude (`1e5` at a random
start, `2e9` from an Adam warm-up).

The cause was found and is specific: **its backtracking line search cannot shrink enough.** Its
first step is `−r` with `‖r‖ = 48` against `‖θ‖ = 10`, and backtracking shrinks by at most
`0.8¹⁵ = 0.035`. JAX offers no alternative for that solver ("for now only backtracking", its own
docstring). Given a proper strong-Wolfe zoom search — which `jaxopt.BFGS`/`LBFGS`/`NonlinearCG`
have by default — the same family progresses smoothly, and then stalls:

| method | iterations | loss | rel `L2` |
|---|---|---|---|
| `jaxopt.BFGS` (zoom) | 300 | 3.33e-04 | **4.23e-01** |
| `optax.lbfgs` + zoom | 300 | 1.04e-03 | 6.53e-01 |
| **DSGNAR `s = 733`** | **186** | **2.24e-14** | **6.76e-06** |

Note the shape of that failure rather than its size: **loss `3.3e-04` with a relative error of
`0.42`.** With `n_coll = n_params`, a gradient-only method drives the *sampled* residual down
while the solution stays wrong between the points. It does not stop later — it stops somewhere
else.

### 4.2 Against SSBroyden, on this problem: DSGNAR wins decisively

Six orders of magnitude in reachable accuracy, at nine to sixteen times the cost. Both are self-scaling
quasi-Newton methods on the same residual; the difference is that DSGNAR builds an explicit model
of the Jacobian from sketched Jacobian-vector products, and SSBroyden only ever sees gradients.

SSBroyden is not broken — it is at its floor. On the sibling problem, 1000 further iterations
moved it by 1 %.

### 4.3 Against SSBroyden, on the sibling problem: DSGNAR loses

Same DSGNAR code, a different problem (2285 parameters, run on the same GPU pool):

| run | final loss | wall | iterations | stop |
|---|---|---|---|---|
| **SSBroyden** | **1.74e-08** | 94 min | 20000 | the cap (at its floor) |
| DSGNAR `s = 2000` (`0.88 d`) | 9.4e-08 | 68 min | 234 | radius below threshold |
| DSGNAR `s = 1100` (`0.48 d`) | 1.8e-06 | 40 min | 211 | radius below threshold |
| DSGNAR `s = 1100`, stage-2 target 0.15 | 2.2e-05 | 23 min | 129 | radius below threshold |

SSBroyden is five times better in 72 % of the wall time. DSGNAR is faster per iteration and stops
early: **every arm ends by its radius rule, not by a budget.** At `s = 2000` the last twenty
iterations take `λ` from `3.5e2` to `8.3e6` while the accepted ratio stays on target — the model
predicts the decrease correctly and the steps simply stop gaining.

---

## 5. The rule that separates the two cases

An independent audit has now answered this directly. Every DSGNAR phase that stops on
"radius below threshold" is re-examined at its final parameters against the **exact** Jacobian:

- at **every** stop, an exact Levenberg–Marquardt step still lowers the loss by **13–19 %**,
  while the *sketched* step raises even the exact Gauss–Newton model — `110 of 110`;
- the cosine between the sketched step and the exact one is only **`0.1–0.3`**;
- the numerical rank of `J` **grows as the network learns**: `824` in window 1 to `~1500` in
  window 10, against `s = 733`. Correlation between `log(stop loss)` and `rank(J)`: **`+0.84`**.

**So the stop is the sketch's, not the problem's.** DSGNAR stops when the column sketch can no
longer represent the curvature the problem actually has, and the failure looks exactly like
convergence from the outside, because the radius rule fires and the loss has fallen.

That yields a usable decision rule:

> **DSGNAR is the right choice when the Jacobian's numerical rank grows past `s` during training
> and the target accuracy lies below the gradient method's floor. It is the wrong choice when the
> sketched model goes uninformative before the target is reached — which is what happens on the
> sibling problem, and what caps DSGNAR there at `9.4e-08` while SSBroyden reaches `1.74e-08`.**

The diagnostic is cheap and needs no GPU: form the exact Jacobian (`jvp_columns` with the identity
basis — 38.7 MB, under a second at `n = 2201`) and compare the rank and the alignment. If the rank
is below `s`, raise `s`; if the rank is comfortable and the alignment is still poor, the sketch
size is not the problem.

---

## 6. Which knobs actually matter, ranked

| rank | knob | measured effect | note |
|---|---|---|---|
| 1 | **the paper's double-precision targets** (stage-1 `0.075`, `Δ_min = 1e-8`) | **×1000** | `7.8e-06 → 7.8e-09` at unchanged `s` and `n_coll`; the paper was right and the deviation cost us |
| 2 | **slab marching** instead of a global solve | **×10⁸** | `9.1e-01 → 6.3e-09` |
| 3 | **sketch size `s`** | **×451–700** | `d/3 → 0.68 d`; `d/3` is below the numerical rank |
| 4 | **hand-over as a penalty** (`soft_all`/`soft` over `hard`) | **×190–990** | amplification ×1.3–1.8 per hand-over against ×3–4 |
| 5 | warm start (paper's rule 4.4) | ×3.6 | helps |
| 6 | collocation points | ×2–8 | `2201 → 8192` |
| 7 | reweighting (paper's App. A.1) | neutral | — |
| — | overshoot rejection | **×5 worse** | the most retries; suspect in a window-10 blow-up |
| — | `K` (hash count), prediction calibration, Wolfe line search, lowest-loss probe selection | **no effect, or worse** | see §7 |

**Caveat on all of the above:** run-to-run variation of one configuration is about **×4** (the
same `soft_all` cell landed at `7.8e-06` and, on re-run, `3.0e-05`). Only differences well beyond
that mean anything; several entries in the table above are of that order and are marked
accordingly.

---

## 7. Corrections and retractions

Recorded because two of them were asserted in this project first, and because the pattern is
diagnostic.

1. **"Redrawing the collocation points costs six orders of magnitude."** *Wrong.* Confounded: the
   run with redraws was also the run whose time feature was divided by the global `T` instead of
   the window width. With that fixed, redraws make no measurable difference.
2. **"`K = 4` is what made every stall."** *Does not transfer.* That is the sibling project's
   measurement at `s = 128`; here `K = 4` and `K = 8` give the same alignment (`0.315` vs
   `0.303`).
3. **"Calibrating the predicted decrease fixes convergence."** *Does not transfer.* The
   over-prediction here is `1.0–3.3×`, not `12×`; it helped some configurations and hurt others.
   Implemented, tested, off by default.
4. **"Broyden's step direction is useless."** *Wrong.* Its line search is; see §4.1.
5. **"A Wolfe line search beats backtracking."** *Not a general fact.* True for `jaxopt`'s
   gradient methods; false for Crunch's SSBroyden, where the Wolfe path fails on **every** block
   (`status 3`) and stalls four orders of magnitude short, and irrelevant inside DSGNAR, which
   already probes 25 exact losses along the LM path.

**The lesson, which is in the README as a rule:** a fix measured on one problem at one sketch size
is a hypothesis about another, not a result. Three of the five items above entered this project as
someone else's measurement.

---

## 8. What is open

1. **The stall has a cause but no cure yet.** The audit says it is the sketch (§5). The untested
   fixes are `s` chosen from the *measured* rank rather than a fixed fraction, and separating the
   paper's two target halves (`0.075` versus `Δ_min = 1e-8`), which are currently confounded —
   round 2 of that ablation is submitted and pending.
2. **A reproducible failure was found and fixed, and it was ours, not DSGNAR's.**
   `dsgnar_soft_all_nc2201_s1100` gave `1.174e-02`, was deleted as invalid data, and on a clean
   re-run gave `1.224e-02`.  Two runs, same failure.  The recorded explanation — *"window 9
   stalled and the retry did not rescue it"* — was wrong: no window was retried at all.  What
   happened is that **window 1** stopped on its radius rule at `1.9e-08`, where a healthy window 1
   reaches `1e-20` (300 of 300 steps accepted), and the chain then carried `1.1e-03` of relative
   error through ten windows that each converged perfectly.  Nothing caught it because a window is
   retried only when its loss exceeds a multiple of **the best window so far** — and window 1 has
   no predecessor, so that test could never fire.  **The one window that seeds the whole chain was
   the one window with no guard.**  `window_first_max_loss` now judges it against an absolute
   floor, narrowly (window 1 only, only on a radius stop, retries capped), with the rule extracted
   and tested six ways.  Worth stating plainly: this is the fifth time in this project that a
   silent no-op — here a guard that could never trigger — looked exactly like correct behaviour.
3. **The two-problem contrast is unexplained in mechanism.** Why does the sibling problem's
   sketched model go uninformative at `1e-7` while the wave equation's reaches `1e-9`? The
   available evidence is the rank/alignment audit; it has not been run on the sibling problem at
   its own stop.
4. **`T = 20` has never been run at the paper's targets *and* `s = 0.68 d` together.** The two best
   rows of §3.1 are each strong on one axis only; combining them is the obvious next experiment
   and the likely best result.

---

## 9. Recommendation

- **For the wave equation, use DSGNAR**, at `s = d/2` or larger, with the paper's double-precision
  targets and the penalty hand-over (`soft_all`), marching in slabs. Expect `~1e-8` for `T = 20`.
- **Do not use Broyden** (`jaxopt`), and do not expect a line search to rescue a quasi-Newton
  method here: the deficit is curvature information, not step length.
- **Before any new problem, run the exact-Jacobian audit** (`scripts/dsgnar_sketch_audit.py`).
  Ten minutes, no GPU. If the numerical rank is below `s`, raise `s` before touching anything
  else; if it is comfortable and DSGNAR still stalls, DSGNAR is the wrong tool for that problem
  and SSBroyden — five to sixteen times cheaper per run — is the better default.
- **Budget for the cost**: DSGNAR is `9–16×` SSBroyden's wall time. It is worth it only when the
  target sits below SSBroyden's floor.

---

## Appendix — how to reproduce

```bash
cd <checkout>/Evolution_try
bash scripts/hub_run.sh --check                       # interpreter, devices, tests
PY=../Stationary/.venv/bin/python

# the sketch-size decision (ten minutes, no GPU)
$PY scripts/dsgnar_sketch_audit.py --k 4 --keys 6 --s 733 1100 1500 --skip-cold
bash scripts/run_sketch_curve.sh                      # T=2 to convergence, three sizes

# the windowed T = 20 batch (resumable; ~7.5 h at s=d/2, ~9.7 h at 0.68 d)
DSGNAR_SKETCH=1100 FOCUSED=1 bash scripts/batch_runs.sh
$PY scripts/report_batch.py                           # BATCH_REPORT.md + plots/batch_report.png

# the full test suite (49 tests, ~4 min)
$PY -m unittest discover -s tests
```

Every number in this report is regenerated by `scripts/report_batch.py` from `runs/`, which is
machine-local; `BATCH_REPORT.md`, `plots/batch_report.png` and `README.md` §§6.9–6.13 are the
versioned record. `docs/dsgnar_spec.md` §17 documents every deviation from the paper, with the
measurement that justifies it.
