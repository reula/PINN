# SBH: CCZ4 PINN for black-hole punctures

A PINN that learns the 22 vacuum CCZ4 fields of a black-hole spacetime: Brill-Lindquist /
Bowen-York punctures, moving-puncture gauge, no excision, trained window by window in time.
The code came from collaborators (the MN5 `uib108` group; the base script is the one exchanged
with Carlos). It is Python/JAX, not Julia. What it does and every flag:
[docs/summary_ccz4.txt](docs/summary_ccz4.txt). The equations: [docs/equations_ccz4.pdf](docs/equations_ccz4.pdf).

```
EEccz4_BH_puncture.py     the one script (merged 2026-10-17, see section 1)
Crunch/                   optimiser library (SSBroyden/BFGS/SCORE, Wolfe search; merged version of 2026-10-17, section 5); must sit next to the script
docs/                     summary_ccz4.txt, equations_ccz4.{txt,pdf}, README_bundle.md (the MN5 bundle's notes)
jobs/mendieta/bh.sbatch   Mendieta job (presets single | bbh | smoke | bench)
jobs/mn5/*.psubmit        the collaborators' MareNostrum 5 jobs, pointed at the merged script
dsgnar.py                 DSGNAR optimiser (vendored from Stationary/, section 4)
tools/                    load_defs.py, bench_loss.py, check_taylor.py, taylor_proto.py, smoke_local.sh, test_dsgnar_bh.py
originals/                the files as received (git-ignored), plus make_ancestor.py
runs/                     outputs (git-ignored)
```

## 1. Where the files came from, and the merge

Received in `SBH/` on 2026-10-17 (now in `originals/`):

| file | what it was |
|---|---|
| `BH_PINN_bundle_20261016.tar.gz` | MN5 production bundle: base script, `_gpus` script, `gpus.patch`, two job scripts, an **older** Crunch |
| `EEccz4_BH_puncture_gpus.py` | identical to the bundle's `_gpus` script (only CRLF line endings) |
| `EEccz4_BH_puncture.py` | a **newer** base script (2026-10-16) |
| `Crunch.zip` | the **newer** Crunch (adds SCORE, `initial_scale`, per-call stats) |

The two base scripts had forked from a common ancestor that was not in the folder:

* the bundle's version (Oct 15) added `--outdir`, `--core_points/--core_ic_points/--core_radius`,
  the blow-up guard (`--blowup_factor`, `--blowup_max`), and a fix for resuming into a different
  output directory; the `_gpus` script is that version plus `gpus.patch` (multi-GPU data
  parallelism `--n_gpus`, `--fast_hessian`, `--loss_plot_every`, `--profile_restarts`);
* the top-level version (Oct 16) instead added the optimiser updates: `--qn_update
  {ssbroyden2,score,bfgs}` with the SCORE parameters, `--h0_scaling` (default on), the excursion
  guard (`--excursion_ratio/_abs/_retries`), `--es_min_loss`, and per-window QN statistics. It
  lacked everything in the first bullet, so the production job scripts (`--blowup_factor 100`)
  and the README's best-run flags (`--core_points ...`) would not have run with it.

`originals/make_ancestor.py` rebuilt the common ancestor from the two base scripts (each diff
hunk assigned to the side that introduced it), and `git merge-file` merged `_gpus` (ours) with
the Oct-16 script (theirs) over it. The only two conflicts were the two loss guards, which sit in
the same place in the restart loop. Both are kept, in this order:

1. **excursion guard** (on by default): a restart whose loss exceeds 100 x the previous restart's
   is discarded and retried from the pre-restart weights (H reset, new batch), at most 3 times per window;
2. **blow-up guard** (off by default, `--blowup_factor 100` in the production flags): the backstop
   once the retries are used up. It rolls back to the window's *best* restart.

The new Crunch keeps `_ssbroyden_update`'s signature and module-global lookup, so `--fast_hessian`
(which swaps in a jitted copy) still works, the SCORE path included.

**Verified (CPU, 2026-10-17):** with the smoke flags (`tools/smoke_local.sh`), the merged script's
`loss_window{0,1}.csv` and `puncture_track_fine.txt` are **byte-identical** to the Oct-16
original's. A second smoke run exercising `--core_points 600 --core_ic_points 120 --blowup_factor
100 --fast_hessian --profile_restarts --loss_plot_every 2` completes. Not yet run: multi-GPU
(`--n_gpus 2`) on the merged script.

## 2. Running

**Environment.** Python with JAX (x64), flax, optax, numpy, scipy, matplotlib, tqdm. Both on the
laptop and on Mendieta the scripts use `../Stationary/.venv` (JAX 0.11.1, flax 0.12.9, optax
0.2.8); `tqdm` was added to both copies (laptop 2026-10-17, Mendieta 2026-10-18).

The script writes every output into `--outdir` (default: the current directory).

**Laptop smoke test** (about 5 min on the M2, 2 windows, tiny network):

    cd ~/Julia/PINN/SBH && bash tools/smoke_local.sh runs/smoke [extra flags]

**Mendieta** (2 x A30 per node; jobs are submitted from `SBH/`, outputs in `runs/<OUTDIR>`):

    cd ~/Julia/PINN/SBH
    sbatch -p short -t 01:00:00 --export=ALL,PRESET=bench jobs/mendieta/bh.sbatch          # timings, no training
    sbatch -p short -t 01:00:00 --export=ALL,PRESET=smoke,OUTDIR=smoke jobs/mendieta/bh.sbatch
    sbatch --export=ALL,PRESET=single,OUTDIR=single_a jobs/mendieta/bh.sbatch              # 1 puncture, t <= 10
    sbatch --gres=gpu:2 -c 16 --mem=56G --export=ALL,NGPU=2,PRESET=bbh,OUTDIR=bbh_a jobs/mendieta/bh.sbatch
    # resume after the 2-day limit:
    sbatch --export=ALL,PRESET=bbh,OUTDIR=bbh_a,EXTRA_ARGS="--resume_checkpoint runs/bbh_a/pinn_checkpoint.pkl" jobs/mendieta/bh.sbatch

The job refuses to start if JAX does not see the GPU(s), or if `OUTDIR` already holds a run
(unless resuming). It copies the script and its md5 next to the outputs. The production flags
(`PROD_ARGS`) are those of the MN5 jobs, `--fast_hessian --loss_plot_every 10 --blowup_factor
100`, plus `--deriv_mode taylor` (section 3). The README_bundle's best-run settings are not defaults; pass them through `EXTRA_ARGS`
(`--core_points 48000 --core_ic_points 4800 --core_radius 0.5 --EPOCHS_QN 24000`).
**Open:** whether the default sizes (80000 interior points, dense 9302^2 inverse Hessian) fit
an A30's 24 GB. The MN5 runs used H100s. Use `NGPU=2` if they don't.

**Forking a run** (2026-10-18): `--resume_checkpoint <run>/pinn_checkpoint.pkl --resume_windows K`
loads only the first K windows (and their puncture-track entries) and continues from window K+1,
into its own `--outdir`; `--stop_after_window W` ends after W windows, with the end-of-run plots
restricted to the trained span. The resume check compares only physics/architecture flags, so
optimizer flags may differ between branches. The RNG key is not checkpointed: a resumed run draws
different samples than the uninterrupted one would, so every arm of a comparison should be forked
the same way (including the baseline).

**Running experiment: optimizer branches from `single_a` window 15** (jobs 1180622-4, each forked
after window 15, stopped after 25, one A30 each, production flags + taylor):

| branch | extra flags | per window |
|---|---|---|
| `fork15_qn` | none: the baseline, resumed like the others | 48 QN restarts (~25 min) |
| `fork15_fixed` | `--qn_fixed_batch` | 48 QN restarts on one sample |
| `fork15_dsg` | `--EPOCHS_QN 6000 --dsgnar_order after_qn --dsgnar_steps 12 --dsgnar_sketch 1024` | 24 restarts + 12 DSGNAR its (~27 min) |

Graded per window on: the training loss, the Einstein residual on a fresh held-out sample (same
for all branches), the constraint monitors Theta and Z^i, and the puncture's position and strength.
The collaborators' 1D reference (`reference_1d/compare_pinn.py` + `ref_single_m0.5_dr0.005.npz`,
17 MB) would turn residuals into errors, but it was not in the bundle (not on the laptop or Mendieta
either, searched 2026-10-18); it has to be requested from them.

**Results (2026-10-19; held-out grading, job 1180648, `tools/eval_heldout.py`).** All three
branches took the same wall time (3 h 46, 3 h 45, 3 h 32 for 10 windows on one A30). Window 15,
shared by all, gives identical numbers in all three (the check that the grading is consistent).
Held-out total loss per window, and its ratio to the baseline:

| window (t_end) | `qn` | `fixed` | `dsg` | fixed/qn | dsg/qn |
|---|---|---|---|---|---|
| 16 (4.00) | 2.28e-5 | 4.90e-5 | 1.55e-5 | 2.2 | 0.68 |
| 17 (4.25) | 8.10e-6 | 4.82e-4 | 2.70e-5 | 60 | 3.3 |
| 18 (4.50) | 3.32e-6 | 4.98e-4 | 2.43e-5 | 150 | 7.3 |
| 19 (4.75) | 1.02e-5 | 5.47e-4 | 1.19e-5 | 53 | 1.2 |
| 20 (5.00) | 3.91e-6 | 5.50e-4 | 1.13e-5 | 141 | 2.9 |
| 21 (5.25) | 1.75e-6 | 5.70e-4 | 6.74e-6 | 326 | 3.9 |
| 22 (5.50) | 4.13e-6 | 5.58e-4 | 1.70e-5 | 135 | 4.1 |
| 23 (5.75) | 6.41e-6 | 5.84e-4 | 1.36e-5 | 91 | 2.1 |
| 24 (6.00) | 9.94e-6 | 8.36e-4 | 2.89e-5 | 84 | 2.9 |
| 25 (6.25) | 4.15e-6 | 6.00e-4 | 9.40e-6 | 145 | 2.3 |

* **`--qn_fixed_batch` overfits.** Its training loss is the lowest of the three (4e-8), but on new
  points it is 50-330x worse than the baseline, mostly in the gauge rows (1e4x worse) and the
  interior Einstein rows (20-230x); only the window-end and IC terms, whose points it fits, look
  better. One fixed sample of 80000 points in 3+1 dimensions does not represent the solution here
  (unlike Stationary's 2D problem). Do not use it for production.
* **QN -> DSGNAR (24 restarts + 12 DSGNAR iterations, s = 1024) loses to plain QN** at equal wall
  time: held-out loss 2-7x worse in 9 of 10 windows (better only in window 16). DSGNAR accepted
  every step, but 12 Gauss-Newton iterations do not make up for the 24 QN restarts they replace;
  its gauge rows end 8-15x and its IC fit 2-24x worse. DSGNAR stays available, not recommended.
* **The baseline's own generalisation gap:** held-out loss 2e-6-2e-5 against 3e-7-1e-6 logged
  in training, i.e. 5-10x. The logged loss understates the error on new points.
* The constraint monitors Theta and Z are similar in the three branches (see below for their growth).

**The single-puncture baseline drifts (single_a, t <= 10).** The tracked puncture should stay at
the origin (one non-boosted puncture). It moves, and the motion accelerates:

| t | 1 | 2 | 3 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| x | -5.6e-4 | -1.1e-3 | -1.0e-2 | -1.9e-2 | -3.5e-2 | -6.7e-2 | -0.121 |
| y | 5.6e-5 | 8.9e-4 | 2.7e-3 | 4.7e-3 | 1.0e-2 | 2.2e-2 | 0.039 |

The puncture itself is intact at t = 10 (alpha 0.033 at the centre, chi_min 0.002, g_xx up to 500).
The momentum-constraint monitor on the 1D cut grows with it: |Z^i|_L2 = 5.8e-5, 6.4e-3, 9.7e-3,
1.7e-2 at t = 0, 3.3, 6.7, 10 (|Theta| ~4e-2 after t ~ 3, roughly flat). All three window-15
branches drift by similar amounts (0.025-0.037 at t = 6.25), so it is not the optimiser. Not yet
investigated. One hypothesis: the tracking integrates dx/dt = -beta(x) from the network's shift,
and the moving chi background is centred on that track, so a small spurious shift at the puncture
moves the background, which the network then has to correct -- a possible feedback loop. Tests
that would separate the causes: `--chi_bg_mode physical`, or a fixed centre (`--bh_center_mode
fixed`), on the same single-puncture setup.

**MareNostrum 5:** `jobs/mn5/` holds the collaborators' scripts (account `uib108`), now
defaulting to the merged script. Submit from `SBH/`: `sbatch jobs/mn5/single_puncture.psubmit`.

## 3. Optimisation: `--deriv_mode taylor`

**Where the time goes.** On the laptop (production network 6 x 40, 4000 points), one
value-and-gradient of the BFGS loss costs ~170 ms, of which ~155 ms is the CCZ4 evolution residual
(the Sommerfeld boundary term ~5 ms). Inside the residual, the derivative stack (jvp in t, jacfwd
in x,y,z, jacfwd-of-jacfwd for the 11 fields that need second derivatives) is about a third of the
value-and-gradient time; the backward pass through the per-point 3 x 3 tensor algebra of the RHS
is the rest. The QN iteration itself is small next to that (the bundle's own MN5 profile: loss
evaluations are ~90 % of a restart).

**What `--deriv_mode taylor` does.** One batched second-order forward pass through the tanh MLP
gives the value, the 4 first derivatives (t,x,y,z) and the 6 *distinct* second spatial
derivatives together: for h = tanh(a), h_k = s1 a_k and h_kl = s1 a_kl + s2 a_k a_l, with
s1 = 1 - h^2, s2 = -2 h s1. That is about 1 + 4 + 6 network-sized matmul passes per point. The
autodiff stack computes 9 second-derivative directions (3 of them duplicates by symmetry) and
recomputes the primal and first derivatives inside each transform. The analytic moving chi
background is differentiated by autodiff (it is cheap) and added to the chi channel.
`--ic_mode hard` (ancestor networks inside apply_fn) falls back to autodiff with a printed note.
The default stays `autodiff`, which is bit-for-bit the previous behaviour.

**Verified (CPU, 2026-10-17):**

* `tools/check_taylor.py`: the full 22-row residual, the loss and its parameter gradient agree
  with autodiff to 1e-15 relative (max over points), in both `--chi_bg_mode moving` and `physical`;
* speed of the residual's value+gradient, idle M2, network 6 x 40:

  | points | chi mode | autodiff | taylor | speed-up |
  |---|---|---|---|---|
  | 4000 | moving | 291 ms | 157 ms | 1.9x |
  | 4000 | physical | 333 ms | 156 ms | 2.1x |
  | 16000 | moving | 1558 ms | 763 ms | 2.0x |
  | 16000 | physical | 1636 ms | 744 ms | 2.2x |

* end to end (smoke flags, 2 windows, network 4 x 20): window wall time 2:32 -> 1:25 and
  1:58 -> 1:31 (compilation included). The loss history equals autodiff's at step 0, differs by
  7e-14 after 50 QN iterations, and the optimiser then amplifies that round-off to ~1e-4 by the
  window's end. Taylor and autodiff runs are therefore **not bitwise comparable** (like runs on
  different JAX versions); compare against a baseline made in the same mode.

**GPU (Mendieta A30, job 1180555, 2026-10-18, `PRESET=bench`):** agreement 7e-16 (residual),
3e-16 (loss), 1e-15 (gradient). Times (median, network 6 x 40):

| | 8000 interior pts | 32000 interior pts |
|---|---|---|
| full BFGS loss value+grad (autodiff) | 20.9 ms | 46.9 ms |
| evolution residual value+grad (autodiff) | 16.2 ms | 46.3 ms |
| evolution residual forward only | 3.6 ms | 12.2 ms |
| Sommerfeld boundary value+grad | 1.7 ms | 3.1 ms |
| residual value+grad, autodiff -> taylor (moving chi) | | 46.5 -> 26.6 ms (**1.75x**) |
| residual value+grad, autodiff -> taylor (physical chi) | | 43.6 -> 26.4 ms (1.65x) |

So on the GPU too the residual is ~all of the loss, and taylor removes ~40 % of it. Compilation
takes 15-50 s per function (cached in `~/.cache/jax_bh`). **`--deriv_mode taylor` is in the
Mendieta job's default `PROD_ARGS`** since then; the MN5 scripts are unchanged (autodiff).
Still open: GPU memory at the default 80000 points (the bench job used at most 32000).

**Next candidates** (not done): the RHS backward pass (the other ~2/3 of the residual cost;
per-point 3 x 3 einsums under vmap) is the larger target, but it is verified physics code, so it
should only be rewritten with the GPU profile in hand and with `check_taylor.py`-style
agreement tests against the current version. `apply_model` (logging, once per restart) also
computes a gradient it discards; that is under 1 % of a restart.

## 4. DSGNAR (2026-10-18)

The doubly-sketched Gauss-Newton optimiser of Webb, Jerad & Cartis (arXiv:2607.02194), ported from
`Stationary/` with the optimiser-level schemes used there. The algorithm, its history and every
design choice are in `Stationary/DSGNAR.md`; this section covers only what is specific to the BH code.

**Code.** `dsgnar.py` is `Stationary/stationary/dsgnar.py` unchanged except that its config is
duck-typed (`_dsgnar_cfg()` in the script builds it from the flags). The script adds:

* `bfgs_residual_vector`: `compute_bfgs_loss` as a residual vector, `sum(r**2) == loss` term by
  term, 7 conditions (8 with `--lam_ic_log`): `einstein_int`, `einstein_tapa`, `gauge_int`,
  `gauge_tapa`, `bc`, `ic`, (`ic_log`), `pos`. The causal weights are already stop-gradiented in
  the QN loss, so the residual is a weighted least-squares problem with the weights held fixed,
  and the Gauss-Newton model is the right one. Rows at the default point counts: 22 x (80000 +
  8000 + 16000 + 8000) + 2 x 80000 = 2.6 M.
* `_DsgnarObjective`: the flat-parameter adapter (residual scaled by sqrt(2M), as Stationary's
  `FlatObjective`).
* `_dsgnar_phase` inside `train_window`: runs on the window's parameters, logs every iteration to
  `loss_window{w}.csv` as optimizer `dsgnar`, and refreshes the moving chi background with every
  new sample. The QN restarts continue the step count after it.

**Schemes** (flags; defaults are Stationary's):

| scheme | flags |
|---|---|
| DSGNAR then QN polish (default order) | `--dsgnar_steps N` (0 = off) `--dsgnar_order before_qn` |
| QN then DSGNAR | `--dsgnar_order after_qn` |
| DSGNAR alone | `--EPOCHS_QN 0` |
| restart cycles: fresh sample + fresh trust region | `--dsgnar_cycles C --dsgnar_cycle_steps K` |
| resampling inside a cycle (trust region kept) | `--dsgnar_resample_every R` |
| sketch size, hashes, JVP chunk | `--dsgnar_sketch` (0 = n/3), `--dsgnar_hashes 4`, `--dsgnar_chunk 4` |
| trust region | `--dsgnar_delta0 1`, `--dsgnar_delta_min 1e-14`, `--dsgnar_omega 1e-8`, `--dsgnar_radius_floor 1e-3`, `--dsgnar_shrink_on_accept` |
| step | `--dsgnar_line_scan/--no-dsgnar_line_scan`, `--dsgnar_backtrack 0.5`, `--dsgnar_backtrack_steps 4`, `--dsgnar_calibrate/--no-...` |
| targets | `--dsgnar_stage1_ratio 0.15`, `--dsgnar_stage2_ratio 0.5` |
| the paper's set (DSGNAR.md section 9) | `--dsgnar_paper` (reweight 0.05, reject overshoot, no line scan, no calibration, shrink on accept, no floor, K 4, targets 0.075/0.5, delta_min 1e-8); explicit flags override |
| per-condition reweighting, one fixed sketch, overshoot rejection | `--dsgnar_reweight 0.05`, `--dsgnar_fixed_sketch`, `--dsgnar_reject_overshoot` |
| SSBroyden on one fixed sample (Stationary's best SSBroyden scheme) | `--qn_fixed_batch` |

Not ported: Stationary's problem-specific schemes (lambda/P pins, Robin spheres, COM gauge,
z-reflection), which belong to the stationary equations; the in-phase `params_dsgnar.pkl`
checkpoint (a DSGNAR phase is part of a window, and the window checkpoint covers it).

**Verified (CPU, 2026-10-18).** `tools/test_dsgnar_bh.py`:

* `sum(r**2) == compute_bfgs_loss` to <= 8e-16 relative in 12 configurations: default, physical
  chi, Dirichlet BC, causal weights off, `--lam_ic_log 0.3`, `--deriv_mode taylor`, each at the
  initial parameters and at strongly perturbed ones (positivity rows active);
* a 6-iteration DSGNAR phase (s = 120) lowers the loss 7.24e-3 -> 4.36e-3, 6/6 accepted, and the
  reported loss equals the true loss at the returned parameters.

End to end (smoke flags, network 4 x 20, 2 windows, s = 200, `--deriv_mode taylor`):

* DSGNAR before QN, 8 iterations per window: 8/8 accepted in both windows, measured ratio
  tracking the 0.15 target; window 0 7.43e-3 -> 4.36e-3, then QN -> 3.32e-3. 83 % of the DSGNAR
  time is the s batched JVPs, as expected;
* QN then DSGNAR with `--dsgnar_paper --dsgnar_cycles 2 --dsgnar_resample_every 2`: runs, but
  every step is rejected (ratio -inf: the unscanned step raises the loss). The logged loss still
  moves, because each resample changes the loss's value. This matches Stationary's experience
  that the paper's settings do not work from a cold start;
* default flags: loss CSVs still byte-identical to the Oct-16 original; `--qn_fixed_batch` runs.

**Cost.** One iteration = s JVPs of the whole residual (forward mode through the second-order
derivatives) + 25-32 loss evaluations + an s x s SVD. At the defaults, n = 9302 and s = n/3 = 3100,
with 2.6 M rows. The production-size cost per iteration on the A30 is the open question; see
the probe jobs below. Taylor mode makes the JVPs cheaper too.

    # production-size probe: window 0 only, 3 DSGNAR iterations, no QN
    sbatch -t 06:00:00 --export=ALL,PRESET=single,OUTDIR=dsg_probe_s1024,EXTRA_ARGS="--N_window 1 --EPOCHS_QN 0 --dsgnar_steps 3 --dsgnar_sketch 1024" jobs/mendieta/bh.sbatch

## 5. Crunch update and QN scale invariance (2026-10-19)

**The new Crunch.** `Crunch/Optimizers/minimize_backtracking.py` is now the collaborators' merged
version of 2026-10-17 (received as `originals/minimize_backtracking.py`, CRLF converted to LF;
md5 after conversion `b50ed315`). It merges three diverged copies: C. Palenzuela's of 2026-10-16
(the one installed before, md5 `77cad12e`), the Boson fork (`reuse_first_trial`, extra counters,
`scale_invariant`) and a BH fork (`rescale_carried`). Every new option defaults to off, and with
them off it is **bit-identical** to the previous file. Verified: the smoke run's loss CSVs are
byte-identical to the Oct-16 original's with the new file in place (and `--qn_scale_invariant 0`).
Its header refers to `CRUNCH_MERGE_2026-10-17.md`, which we did not receive.

**The authors' recommendation, now applied here:**

* `--qn_scale_invariant 1` (**new default**): Crunch's `scale_invariant`. The curvature guards
  become relative (y.s <= 1e-14 |s||y|), and the update is computed on the rescaled pair. Same
  cost. `0` reproduces every earlier run bitwise.
* `--qn_gtol` (default 1e-9, as before): the gradient threshold stays ABSOLUTE. Lower it, or set
  0, when the loss is very small.
* `--qn_loss_norm 1` (default off): QN minimizes loss / L0, with L0 the loss at the start of the
  window's QN phase (fixed for the window, over which H is carried). The identity H of the first
  step is then well scaled at any loss level; logs and guards see the true loss; `--qn_gtol`
  becomes relative.

**Their general test, run on this code** (`tools/test_scale_invariance.py`: 3 restarts x 40 QN
iterations, production settings, H carried, gtol 0, the loss multiplied by K):

| setting | K = 1 | K = 1e8 | K = 1e-8 (loss ~3e-11) |
|---|---|---|---|
| old (`scale_invariant` off) | 2.47e-3 | 2.43e-3 (x differs 5 %) | **6.94e-3: frozen, 39/40 updates skipped** |
| `scale_invariant` on | 2.43e-3 | 2.43e-3 (x differs 13 %) | 3.07e-3: no skips, but the first step is mis-scaled |
| on + `--qn_loss_norm 1` | 2.40e-3 | identical | identical |

(loss/K after the third restart). The freezing they describe is real here: without the option,
QN stops updating its Hessian once the loss is ~1e-11. With the option on, what remains is the
identity first step, whose length scales with the loss; the loss normalization removes that, and
by construction then the optimizer sees the same O(1) objective at every scale. At the losses of
the current runs (1e-6 to 1e-7 per window) neither effect has been measured in a real run yet.
`--qn_scale_invariant 1 --qn_loss_norm 1 --fast_hessian --deriv_mode taylor` runs end to end
(smoke, CPU), with a loss history close to the old defaults' at these loss levels (~1e-3).

**Comparability.** Runs before 2026-10-19 (including `single_a` and the window-15 branches) used
`scale_invariant` off; to reproduce or extend them exactly, pass `--qn_scale_invariant 0`.

## 6. Tools

* `tools/load_defs.py`: `load(argv)` executes the script up to the training loop and returns
  its namespace (every function, no training). Used by the other tools.
* `tools/bench_loss.py [--n N] [-- script flags]`: compile time and median time of the full loss
  and of its pieces (evolution residual value+grad and forward, boundary term).
* `tools/check_taylor.py [--n N]`: taylor vs autodiff on the full residual and gradient, both chi
  modes, with timings.
* `tools/taylor_proto.py`: the derivative arrays alone (value, d_t, d_xyz, d_xyz d_xyz) against the
  autodiff stack.
* `tools/smoke_local.sh`: the laptop smoke run.
* `tools/test_dsgnar_bh.py [--quick]`: the DSGNAR tests above.
* `tools/test_scale_invariance.py`: the loss-scale test of section 5.
