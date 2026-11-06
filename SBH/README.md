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
jobs/mendieta/bh.sbatch   Mendieta job (presets single | bbh | smoke | bench | stage2 | eval)
jobs/hub/stage2.sh        stage2.py on the CCAD JupyterHub GPU (detached, resumable; 2026-10-27)
sing_pinn.py, sing_template.py   template + correction ansatz (section 7; 2026-10-28); jobs/hub/sing.sh runs it on the hub
jobs/mn5/*.psubmit        the collaborators' MareNostrum 5 jobs, pointed at the merged script
dsgnar.py                 DSGNAR optimiser (vendored from Stationary/, section 4)
tools/                    plot_losses_stages.py (both stages' losses), load_defs.py, bench_loss.py, check_taylor.py, taylor_proto.py, smoke_local.sh, test_dsgnar_bh.py, plot_1d_runs_t.py and table_1d_runs.py (sing_1d.py runs against time: plot, band-rms table)
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

**JupyterHub** (CCAD, A30; 2026-10-27, used when the Mendieta queue is slow). The hub mounts the
serafin home (`~/serafin/Julia/PINN` on the hub = `~/Julia/PINN` on serafin); `runs/` is not in git,
so the frozen run is copied to serafin first (`rsync` via the laptop). The script uses the Stationary
venv read-only and installs `tqdm` (missing there) into `SBH/.pydeps`. Paste in a hub terminal:

    cd ~/serafin/Julia/PINN && git pull && cd SBH
    OUTDIR=stage2_con10_adm_g10 STAGE2_ARGS="--sampling uniform --core_taper 1.0 --eps 1e-2 --w_adm 1 --w_gauge 10 --n_per_window 3200 --hidden 64 --layers 4 --iters 24000" bash jobs/hub/stage2.sh

It first waits until no Stationary hub batch (`Paper/hub_*.sh`, another session's) is running, so the
GPU is not shared (`WAIT_FOR=` starts at once), and lets JAX use 95 % of the GPU instead of its default
75 % (17.7 GiB): the `--w_adm 1` correction ran out of memory at 17.7 GiB on 2026-10-27. It ran out again with
95 % (2026-10-28, after iteration 500): the stage-2 H/M correction does NOT fit on the hub as it stands; it
needs fewer points per draw or a chunked loss. Relaunching the same line after a container restart
resumes from `runs/$OUTDIR/stage2_params.pkl`.

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

Submitted 2026-10-19 (jobs 1180766/7, `PRESET=single`, from t = 0 to t = 4 = 16 windows, both
with `--qn_scale_invariant 0` so the only difference from `single_a` is the chi setting):
`drift_phys` (`--chi_bg_mode physical`: no moving background, tracking still steers the sampling)
and `drift_fixed` (`--chi_bg_mode physical --bh_center_mode fixed`: no background, no tracking).
Drift is compared through the chi-minimum position (`chi_min_x` in `puncture_track.txt`), which is
independent of the tracker and agrees with it in `single_a` (-0.125 vs -0.121 at t = 10).

**Result (2026-10-20; `tools/puncture_probe.py` on the saved windows).** The chi minimum turned out
to be useless as a yardstick at these drifts: near a puncture chi ~ (2r/m)^4 is so flat (1.6e-3 at
r = 0.05, the size of the network's error there) that its minimum wanders by +-0.05 in all three
runs, even on a 0.002 grid. The discriminating quantity is the spurious shift at the origin, the
push that moves a tracked puncture (dx/dt = -beta):

| t_end | `single_a` (moving bg + tracking) beta^x(0) | `drift_phys` (tracking only) | `drift_fixed` (neither) |
|---|---|---|---|
| 0.5 | +1.2e-3 | -4.1e-4 | +1.2e-3 |
| 1.5 | +2.0e-3 | +1.0e-3 | +5.9e-3 |
| 2.75 | +2.9e-2 | +3.5e-3 | +4.0e-3 |
| 3.5 | +2.8e-2 | +2.4e-2 | +6.0e-3 |
| 4.0 | +2.8e-2 | +3.0e-2 | +2.9e-3 |

With tracking (with or without the moving background) the push grows to 2-3e-2 by t ~ 3-4; with a
fixed centre it stays at a few 1e-3, 10x smaller at t = 4. So the moving chi background is NOT the
cause; the tracking is. Since `drift_phys` has no moving background but its sampling clouds still
follow the track, the likely loop goes through the SAMPLING: a small random asymmetry in beta moves
the track, the focused clouds become off-centre, the off-centre sample teaches a larger asymmetric
beta. One run per setting (one seed each): consistent over t >= 2.75 but not repeated. Next test
(needs a small flag): tracking on, sampling clouds held at the origin. If the push stays small, the
remedy for the binary runs is in how the sampling follows the punctures (e.g. drawing the clouds
symmetrically about each tracked centre).

**Figures and the mass function (2026-10-20; `plots/2026-10-20/`, `tools/plot_report_figures.py`).**
`loss_history.png` (training loss window by window), `errors.png` (held-out loss of the window-15
branches; rms Theta and |Z^i| on the x-axis cut), `mass_function.png`. The mass function is new:
for one non-spinning puncture the spacetime is Schwarzschild, so the Misner-Sharp mass
m = (R/2)[1 - (d_x R)^2/gamma_xx + R^2 (K_yy/gamma_yy)^2], R = x sqrt(gamma_yy), on the x-axis through
the puncture must equal M = 0.5 at every radius, time and slicing (`tools/mass_function.py`;
verified: 4e-15 on the exact initial data). Unlike the loss, m/M - 1 is a coordinate-independent
error. Findings:

* max |m/M - 1| over r in [0.5, 10] grows in all three runs alike up to t = 4 (to 3-7 %), so the
  growth is not caused by the drift or the tracking; in `single_a` it keeps growing, roughly
  exponentially, to **35 % at t = 10**, with the largest error between r ~ 0.3 and 2 (the strong-field
  region around the horizon) and a clear +x/-x asymmetry.
* The training loss does not show it: the logged loss DEcreases after t ~ 2.5, from 1e-6 to 2e-7.
* |Z^i| on the cut grows to 3e-2 in `single_a`; with a fixed centre it is not smaller (1.6e-2 at
  t = 4 against 7e-3 for tracking only), so the drift and the constraint growth are separate issues.

**Clementina XXI (2026-10-20): not usable for SBH yet.** `jobs/clementina/bh.sbatch` runs the same
presets on the Intel Data Center GPU Max 1550 nodes (JAX 0.11.2 + the experimental `jax-oneapi-plugin`,
venv `~/venvs/pinn-oneapi` shared with other JAX jobs; flax 0.12.9 was ADDED to it with jax/jaxlib
pinned -- only flax and 18 new dependencies were written, nothing existing changed). The device is
found and pinned, but the float64 code does not compile reliably (`PRESET=probe`,
`tools/oneapi_probe.py`, job 1548870):

* every float64 forward product used here passes alone (network matmuls, the 9302^2 matvec and
  outer product, batched 3x3 products, the RHS einsums, 3x3 inverses and determinants);
* the GRADIENT of a trivial float64 matmul loss fails (`IndexError: _Map_base::at`); float32 passes;
* the residual's forward pass fails with the same error in autodiff mode and crashes Intel's graphics
  compiler in taylor mode (`IGC: Internal Compiler Error: Segmentation violation`), while taylor
  value+grad passes; the full BFGS loss (bench job 1548869) dies in oneDNN ("could not create a
  primitive descriptor for the matmul primitive").

Fusion-dependent compiler/plugin bugs in float64, not something to fix in this code; float32 is not
an option for this problem. SBH stays on Mendieta. Possible next attempts if Clementina is needed:
`CXXI_ZE_MODE=COMPOSITE`, a newer plugin build, or a report to the cluster's support
(tickets.clementinaxxi.org.ar) with `tools/oneapi_probe.py` as the reproducer.

**The 1D reference (received 2026-10-20, `originals/reference_1d.zip`, unpacked in `reference_1d/`).**
The collaborators' validation solution for one m = 0.5 puncture: the BH code's own `r_ccz4_evolution`
evolved in spherical symmetry (4th-order finite differences, dr = 0.005, outer boundary causally
disconnected; error <= 1e-5 for r >= 0.1), t = 0..10 at every window boundary, plus their tools
(`compare_pinn.py`, `misner_sharp.py`, `BH_invariants.py`, ...). See `reference_1d/README.md`. The two
17 MB `.npz` files are git-ignored (unzip `originals/reference_1d.zip` to get them). Their tools load
the BH script through `tests/load_module.py`, which here is a shim over `tools/load_defs.py`:

    cd reference_1d && JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python compare_pinn.py RUN_DIR ref_single_m0.5_dr0.005.npz

First comparison, `single_a` (the code's defaults = the 10-19 Oct summary's "first baseline"): alpha at
r = 0.03 is ~2x the reference from t = 0 on (0.215 vs 0.108 at t = 0.25, 0.300 vs 0.147 at t = 1, 0.081 vs
0.034 at t = 10), i.e. errors 0.108 / 0.153 -- the summary reports 0.125 / 0.156 for its first baseline, so
reference, shim and run agree. The non-spherical part (`off_form`) grows from 0.01 to 1.4 by t = 10;
mid-radius alpha L2 error from 1e-4 to 0.10.

**Why DSGNAR stops: the residual sketch (2026-10-21).** The LDC notebook gives DSGNAR's two faces on
identical code and settings: on the hub's A30 it reaches loss 2.0e-7, rel. L2 0.0093 in 210 iterations /
203 s (SSBroyden: 0.0114 after 10000), every ratio near its 0.075 target; on Clementina's Intel card the
first step's ratio is NaN, later ones -1 to -15, and it stops at 82 iterations, rel. L2 0.57 -- the oneAPI
float64 problems of the probe above, so Clementina's DSGNAR numbers are not usable. Even on the hub it
finally flattens at 2e-7 with rejections (rho ~ -0.01, radius down to 1e-13), and `tools/sketch_noise.py`
shows why. DSGNAR's gradient is J^T C^T C r; its error J^T (C^T C - I) r stays ~|J||r|/sqrt(s) while the
TRUE gradient |J^T r| shrinks as the residual becomes orthogonal to range(J) at a nonzero-residual floor:

| point (small BH problem, n = 998, s = 332) | loss | relative gradient |J^T r|/(|J||r|) | DSGNAR step: cos with g, uphill draws | exact residual: cos |
|---|---|---|---|---|
| initial network | 2.9e-2 | 4.7e-2 (~ 1/sqrt(s) = 5.5e-2) | +0.37, 0/20 uphill | +0.57 |
| after 300 SSBroyden its | 6.6e-4 | 9.3e-3 | +0.10, 4/20 | +0.57 |
| after 1500 | 6.0e-5 | 2.4e-3 | +0.04, **8/20** | +0.58 |

Once the true gradient is below the sketch noise the step is close to random and goes uphill in ~half the
draws however small the radius; the trust region collapses. **`--dsgnar_residual_sketch none`** (with
`--dsgnar_jacobian rows`) keeps the parameter sketch but uses the exact residual: G = (J Omega S)^T (J Omega S)
and b = (J Omega S)^T r accumulated over every row (`_DsgnarObjective.gram_rows`; equal to the dense
product to 2.5e-13), turned into an equivalent square system for the unchanged DSGNAR code. From the
1500-iteration stall point (paper step rule): residual-sketched 2 accepted / 13 rejected, radius 7e-8; exact
residual 30 / 0 (`tools/test_exact_residual.py`).

**Where the mass error comes from (2026-10-21; `tools/mass_function.py --both_ends`,
`tools/plot_mass_budget.py`, `plots/2026-10-21/mass_budget_*.png`).** Telescoping each window's mass at its
start and end splits the error at every radius exactly into the initial-data fit, the growth INSIDE windows
and the JUMPS at hand-offs. `single_a` at t = 10: r = 1: initial -1e-4, in-window +0.30, hand-offs -0.015,
total +0.28 (r = 0.5: +0.35 / -0.07; r = 2: +0.21 / +0.03). So it is built up inside the windows, the
hand-offs partly undo it, and the initial data are fitted to 1e-3. Its radial density d(m - m0)/d ln r is
positive only at r <~ 0.3 (around the horizon) and ~0 from 0.3 to 3: the error is PRODUCED in the core and
carried outward by the integral, with a separate oscillating part beyond r ~ 4 where the gauge pulse
passes. `w8_qn` (final settings, t = 2): +-0.5 % beyond r = 0.3, in-window and hand-off terms of similar size
and opposite sign, -2 to -4 % at r = 0.1-0.2. A constraint-violating solution still satisfies the CCZ4
EVOLUTION equations (Theta, Z^i != 0), so the residual loss sees that mode only through the initial data;
candidates: penalise Theta and Z^i (zero for the true solution) in the loss, constraint damping (weak in the
core, where alpha ~ 0.1), a mass pin.

**Constraint penalty (2026-10-21): `--w_constraint W`** adds W * mean(Theta^2 + |Z^i|^2) over the interior
points to the QN loss and the logged loss (new CSV column `loss_constraint`). Default 0 = off; with it
off every original column of the smoke run's loss history is still identical to the Oct-16 original. Not
with DSGNAR yet, and only for constraint-satisfying initial data (single puncture). Submitted with the
exact-residual DSGNAR test, all three in the `w8_qn` setting (final settings, 8 windows, one A30):
`w8_gn_exact` (DSGNAR, `--dsgnar_residual_sketch none`, paper step rule, any decrease accepted, s = n/3,
4200 s per window), `w8_qn_con1` and `w8_qn_con10` (SSBroyden + `--w_constraint 1` / `10`).

**Constraint propagation from the contracted Bianchi identity (2026-10-21): `--w_bianchi W`.** The Z4 field
equations E_mn = R_mn + nabla_m Z_n + nabla_n Z_m = 0 (vacuum, kappa = 0) satisfy, for ANY metric and
covector, nabla^m (E_mn - 1/2 g_mn E) = box Z_n + R_nm Z^m: the constraint variables obey a linear wave
equation sourced only by the DIVERGENCE of the field-equation residual -- the part of the error that the
L2 residual loss does not control and that pumps the constraint-violating (mass) modes. `--w_bianchi`
adds W * mean(B_n^2), B_n = box Z_n + R_nm Z^m, on the last `--bianchi_points` interior points (the core
cloud with the final settings, where the mass error is produced). It needs third derivatives of the
network (nested forward-mode autodiff), ~35x the per-point cost of the residual.

`z4_bianchi.py` (generic 4D: Christoffel symbols and their derivatives, Ricci, nabla Z, box) and
`EEccz4_BH_puncture.py:z4_spacetime_fields` (CCZ4 -> (g_mn, Z_m), Alic et al. conventions: Theta =
alpha Z^0, Z_i = 1/2 gt_ij (Ghat^j - Gt^j), Z_0 = beta^k Z_k - alpha Theta). Verified
(`tools/test_z4_bianchi.py`):

* the identity, on random smooth analytic metrics and covectors: to 4.7e-16 at five points (B ~ 4-10);
* exact static Schwarzschild in CCZ4 variables through the mapping: E_mn <= 7e-15 at r = 0.4 (curvature
  scale 8), B = 0;
* on the trained `w8_qn`, E_mn rms 0.2 at r in [0.3, 0.7), 7e-3 at [0.7, 1.5), 4e-4 at [3, 6) in window 1
  (CCZ4 residual 2.4e-4): **the network's spacetime is not Ricci-flat at the ~5 % level around the
  horizon**, where the mass budget puts the error source. With Z non-negligible (window 5) the Alic
  mapping gives the smallest E of the alternatives tried (Theta sign flipped, Z^i used as Z_i, Z ignored).

Submitted: `w8_qn_bia3`, `w8_qn_bia4` (the `w8_qn` setting, SSBroyden, + `--w_bianchi 1e-3` / `1e-4`,
`--bianchi_points 4096`); at W = 1e-3 the term is ~5e-8 at initialisation and comparable to the
Einstein term (~1e-6) once a window is trained.

**Why the exact-residual DSGNAR still stalls in the first windows (2026-10-21).** `w8_gn_exact` (job 1181491)
collapsed in windows 1-3 (2/16, 2/18, 0/15 accepted, rho = -inf even at radii ~1e-7) and then worked in
window 4 (rho ~ 0.07 every step, 5.7e-3 -> 1.2e-3). `tools/test_causal_consistency.py`, at the start of a
window with the production point counts (random network, core cloud), measures the loss change along the
frozen-weight gradient against the prediction, for step lengths h = 1e-1 ... 1e-6:

* causal weights on or off: no difference (they are not the cause);
* with `--min_puncture_dist 0.02` (default): only **3 of 128000** points have the network's chi below the
  1e-5 floor, but they carry ~90 % of the Einstein loss (0.19 of 0.20) and a 1000x larger gradient, and
  the measured/predicted slope ratio jumps (0.13, 1.28, 0.66, 0.96 for h = 1e-3 ... 1e-6): the hard
  floor max(chi, 1e-5) puts KINKS in the loss. With a smooth floor (new `--chi_floor_mode smooth`,
  (chi + sqrt(chi^2 + 4 floor^2))/2) the ratio rises monotonically (kink confirmed) but those points
  then dominate even more (loss 57);
* with `--min_puncture_dist 0.05` (exact chi >= 8e-4 at every point): 0 points near the floor, loss 0.034
  (Einstein 0.019), slope -1.2 instead of -1259, ratio 0.81 / 0.98 / 0.998 at h = 1e-4 / 1e-5 / 1e-6 --
  smooth.

A handful of collocation points next to the puncture make the early-window loss non-smooth and set its
curvature; a trust-region Gauss-Newton method builds its model from them and fails, while SSBroyden's line
search just takes short steps. Once the network's chi there has moved off the floor (later windows) the
loss is smooth and DSGNAR works.

Submitted (2026-10-21, jobs 1181539/40): `w8_qn_md05` (SSBroyden) and `w8_gn_exact_md05` (exact-residual
DSGNAR, paper step rule, any decrease accepted, s = n/3, 4200 s per window), both the `w8_qn` setting with
`--min_puncture_dist 0.05`, so the optimizers are compared on a smooth loss. Note it changes the sampling
of the 10-19 Oct summary (no points within 0.05 of the puncture) for both.

**Results of the penalty and DSGNAR runs (2026-10-22; 8 windows, final settings; `reference_1d/compare_pinn.py`,
`tools/mass_function.py --both_ends`).** At t = 2 against the 1D reference ("core" = L2 over 0.05 <= r < 0.5):

| run | alpha(0.03) (ref 0.281) | alpha core | Khat core | gt core | Theta max | off_form |
|---|---|---|---|---|---|---|
| `w8_qn` (no penalty) | 0.276 | 1.2e-2 | 4.5e-2 | 2.3e-2 | 0.45 | 6.6e-2 |
| `w8_qn_con1` (`--w_constraint 1`) | 0.272 | 1.5e-3 | 7.9e-3 | 2.1e-3 | 1.0e-3 | 3.8e-2 |
| `w8_qn_con10` (`--w_constraint 10`) | 0.289 | 7.1e-4 | 8.7e-3 | 1.8e-3 | 3.1e-4 | 4.2e-2 |
| `w8_qn_bia3` (`--w_bianchi 1e-3`) | 0.284 | 1.1e-3 | 3.3e-3 | 2.5e-3 | 1.4e-2 | 2.1e-2 |
| `w8_qn_md05` (`--min_puncture_dist 0.05`) | 0.141 | 1.2e-2 | 5.0e-2 | 2.3e-2 | 0.32 | 3.4e-2 |
| `w8_gn_exact` (DSGNAR, exact residual) | 0.150 | 0.29 | 0.30 | 4.5e-2 | 2.9e-2 | 0.16 |
| `w8_gn_exact_md05` | 0.737 | 0.30 | 0.28 | 1.2e-2 | 0.16 | 5.8e-2 |

Both constraint penalties cut the core field errors ~10x against the same run without them; Theta itself
drops 400-1500x with the direct penalty and 30x with the Bianchi term, which is the best on Khat and on
spherical symmetry. Cost: 12 h 16 / 12 h 36 (con1 / con10) and 15 h 08 (bia3) for 8 windows against 9 h for
`w8_qn`. `md05` has no points within 0.05 of the puncture, hence its low alpha at r = 0.03; otherwise it
equals `w8_qn`. The mass error at t = 2 is <= ~1 % in every SSBroyden variant (it was already small in
`w8_qn`; in `single_a` it grew mainly after t ~ 4), so 8 windows cannot tell whether the penalties stop the
mass drift -- that needs longer runs.

**Long runs to t = 10 (submitted 2026-10-22):** `long_con10` (`--w_constraint 10`, jobs 1181733 -> resume
1181734) and `long_bia3` (`--w_bianchi 1e-3`, jobs 1181735 -> resume 1181736), each resumed from the
8-window run's checkpoint into its own folder and continued to window 40; the second job of each pair
(`--dependency=afterany`) resumes from the newest checkpoint if the first hits the 2-day limit. To be
graded by the mass budget and the reference against the 10-19 Oct summary's final settings (+7-8 % mass
error at t = 10).

**Long-run results, t = 10 (2026-10-26; `long_con10` 45 h in one job, `long_bia3` 2 jobs).** Mass error
m/M - 1 by radius at t = 10 (ray average, `tools/mass_function.py`):

| r | `single_a` (no penalty, defaults) | `long_con10` (`--w_constraint 10`) | `long_bia3` (`--w_bianchi 1e-3`) |
|---|---|---|---|
| 0.3 | +25 % | **+0.34 %** | -5.8 % |
| 1 | +28 % | **+0.31 %** | -6.2 % |
| 2 | +24 % | **+0.15 %** | -6.0 % |
| 4 | +6.5 % | -1.3 % | -4.1 % |
| 8 | +3.5 % | +2.7 % | -16 % |
| 10 | -3.4 % | -22 % | -14 % |
| 15 | -12 % | -13 % | +7.7 % |

Against the 1D reference at t = 10: alpha(0.03) 0.081 / **0.037** / 0.041 (ref 0.0340); alpha core L2 6.6e-2 /
**6.0e-3** / 2.2e-2; alpha mid 1.0e-1 / **3.5e-3** / 2.7e-2; Khat core 6.3e-2 / **7.3e-3** / 1.8e-2; Theta
max 0.42 / **3e-5** / 2.9e-2; puncture drift 0.13 / 0.010 / 0.007; off_form 1.44 / 0.72 / 0.52.

The direct penalty keeps the HOLE's mass within 0.35 % to t = 10 (the 10-19 Oct summary's final settings
without a penalty: +7-8 %; `single_a`: +25-28 %) and the lapse at the puncture within 0.003 of the
reference (summary: 0.007). The Bianchi term holds the hole to ~-6 %, no better than the summary's run.

**What the outgoing deficit is, and a stage-2 correction network (2026-10-26).** The true Hamiltonian
constraint H = R + K^2 - K_ij K^ij (3-metric from the network, Ricci via `z4_bianchi.geometry`; note the
script's `hamiltonian_constraint` returns Theta, not H) averaged on spheres of `long_con10` is ~ -1e-3 at
r ~ 7-11 at t = 10 (positive beyond); dm/dR = R^2 H / 4 integrates it to -12 % at r = 10, -15 % at r = 11,
as the sphere-averaged Misner-Sharp mass gives (-18 %). So the deficit is a real constraint violation,
built INSIDE windows (mass budget at r = 10: in-window -21.5 %, hand-offs -0.8 %). Theta is ~1e-5 there:
the Theta/Z penalty pushes the H violation into the residual of Theta's own equation (d_t Theta = alpha H/2
+ ...), ~5e-4, which the loss barely weights: on `long_con10` 90 % of the residual is at r < 1, 8 % at
1-5, 1 % at 5-15.

`stage2.py` keeps a finished windowed run (u1, frozen) and trains ONE correction network over the whole
span: u = u1 + s N, s = eps (t/T)(1 - r^2/R^2)(1 - exp(-r^2/rc^2)) (`stage2_net.py`): u keeps u1's initial
data, outer-boundary values and, with `--core_taper rc`, the hole. Loss: the main script's residual (same
weights) + `--w_constraint` x mean(Theta^2 + |Z|^2) of the SUM, whose derivatives add: u1's per window in
closed-form Taylor mode (all windows in one vmapped call, points drawn in equal numbers per window span), N's
by the product rule. `--selftest` checks the summed residual against the main script's autodiff residual
of the summed function: 7.5e-13 (9.6e-14 with the taper) where N changes it by 3e-3. `residual_one` was
moved to module level (`_ccz4_residual_one`) for this; bit-identical in both derivative modes. Points:
`--sampling uniform` (default, evenly in the ball: the correction is for what propagates out),
`shell` (radius uniform from 0.5), `final` (the main script's clouds). The history logs the held-out loss
of u and u1 by radius band. `tools/sphere_errors.py --stage2 PKL` grades u1 + correction.

Laptop tests (first 24 windows, 300 points per window span, 3x32 network, 2000 iterations): with the
main script's clouds no gain on fresh points (the loss is the hole's); `eps` = residual rms (2e-4) too
small (no change); uniform points with `--eps 1e-2 --core_taper 1`: held-out loss 0.90x (r 5-15: 0.93x),
mass at t = 6 r = 8 -8.1 -> -7.7 %, r = 10 -6.0 -> -5.5 %, but alpha error at r = 2-8 up 1.3-1.9x (the
correction lowers the residual partly through the gauge). GPU test: `PRESET=stage2` in
`jobs/mendieta/bh.sbatch`.

**ADM constraint penalty (2026-10-26): `adm_constraints.py`, `--w_adm W`.** The true constraints
H = R + K^2 - K_ij K^ij and M_i = D_j K^j_i - d_i K (K = trace of K_ij) of the network's 3+1 data,
from the residual's derivative stack (Taylor or autodiff; also summed stacks, for `stage2.py`). The
inherited names stay as they are: `hamiltonian_constraint` returns Theta and `gauge_source_vector` Z^i
(the Z4 variables); our own penalty on them is now `zeta_penalty` (flag `--w_constraint`, unchanged).
`tools/test_adm_constraints.py`: H = M = 0 to 1e-14 for exact Schwarzschild in isotropic (K = 0) AND
Kerr-Schild (K_ij != 0) coordinates; on a trained window Taylor vs autodiff 2e-13 (H) / 3e-12 (M), and
H vs the independent route of `tools/hamiltonian_shells.py` 2e-13. `--w_adm W` adds
W * mean(chi^p (H^2 + |M_i|^2)) over the interior points (`--adm_chi_power p`, default 2: chi^2 ~ 0.008
at r = 0.3, 0.17 at r = 1, 0.67 at r = 5, fading out the puncture); off by default, and then the QN loss
and its gradient are bit-identical to before (checked); on, a value+gradient costs 1.6x (small test
network). Log column `loss_adm`. QN only.

In `stage2.py` (`--w_adm`, same weight) the laptop test (24 windows, 300 points per window span, 3x32,
2000 iterations, `--eps 1e-2 --sampling uniform --core_taper 1`, W = 1) overfits the per-point losses
(held-out 1.13x, ADM 1.33x) but halves the mass deficit: t = 4, r = 5: -3.5 -> -0.9 %; t = 6, r = 7:
-7.1 -> -3.7 %, r = 8: -8.1 -> -5.0 % (the same correction without the ADM term: -3.5, -6.9, -7.7 %).
Lapse error and Theta as with the correction alone (lapse 1.2-1.9x the run's at r = 2-8).

**Stage 2 on the GPU (2026-10-27; jobs 1184417 / 1184387, one A30, 1 h 39 / 2 h 10): correction on top of the
frozen `long_con10`, 40 windows, 128k points per draw evenly in the ball, 4x64, 24k iterations, `--eps 1e-2
--core_taper 1`; without / with `--w_adm 1`.** Names used from here on: the **stage-2 correction** (no ADM term) and the **stage-2 H/M
correction** (`--w_adm`), always graded as run + correction; the **windowed H/M run** is the 40-window run
trained with `--w_adm` (`long_con10_adm`, below). Held-out loss of the sum vs u1: 0.55x / 0.49x (r 5-15: 0.57x /
0.68x; r 1-5: 1.11x / 1.52x; ADM term 0.77x / 0.38x). Graded (`tools/sphere_errors.py --stage2`), peak of
the outgoing deficit (most negative sphere mean over r > 2), run / + correction / + correction with H, M:

| t | `long_con10` | + stage-2 correction | + stage-2 H/M correction |
|---|---|---|---|
| 4 | -3.8 % (r 5.5) | -3.0 % | **-0.1 %** |
| 6 | -8.1 % (r 7.6) | -4.8 % | **-0.8 %** |
| 8 | -14.1 % (r 10.0) | -8.8 % | **-2.9 %** |
| 10 | -18.1 % (r 11.7) | -12.3 % (r 13.8) | **-4.9 %** (r 13.8) |

The hole stays at +0.3 % in all three. The price is the gauge: the lapse error against the reference at
t = 10 is 1-2x the run's with the correction and ~2x with the H, M term (r = 3-6: 5.3-5.5e-3 -> 1.1e-2),
and Theta rises from ~3e-6 to ~2e-5 at t = 10 (still small). The mass is gauge-invariant, the lapse is not:
the correction trades agreement with the reference's gauge (its gauge rows at r 1-5 get worse) for
satisfying the constraints.

**Gauge weight 10 (2026-10-28; job 1184525, A30, 1 h 58; the stage-2 H/M correction with `--w_gauge 10`,
`runs/stage2_con10_adm_g10`).** Does weighting the gauge rows (lapse, shift equations) 10x win back the lapse?
No. Held-out loss of the sum vs u1 0.47x (ADM term 0.44x; r 1-5 1.44x, r < 1 1.76x). Graded the same way:
deficit peak -0.4 / -0.5 / -2.9 / -6.7 % at t = 4 / 6 / 8 / 10 (weight 1: -0.1 / -0.8 / -2.9 / -4.9 %);
hole +0.18 %; lapse error at t = 10, r = 3-6 1.0e-2 (weight 1: 1.1e-2; the run alone 5.2e-3), r = 10-20
3.9e-3 (4.7e-3); Theta 1.1e-5 (2.3e-5). The lapse error at r 3-6 stays at ~2x the run's: it is not a
weighting problem; the correction moves the slicing where the deficit is removed. The weight-1 correction
stays the reference version. (The same job on the JupyterHub ran out of GPU memory twice, at the 6.84 GiB
QN-norm allocation, even with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`; Mendieta's copy finished first.)

**Windowed H/M run, interim (2026-10-28, 25 of 40 windows, t <= 6.25; job 1184413 -> 14 -> 15, then its stage
2, 1184416).** `long_con10_adm` = `long_con10`'s settings and Theta/Z penalty plus `--w_adm 1` in every window.
Sphere-averaged deficit peak at t = 2 / 3 / 4 / 5 / 6 / 6.25:

| | 2 | 3 | 4 | 5 | 6 | 6.25 |
|---|---|---|---|---|---|---|
| `long_ctl` (no penalty) | -0.5 % | -1.2 % | -3.6 % | -7.5 % | -12 % | -13 % |
| `long_con10` | -0.8 % | -1.6 % | -3.8 % | -6.1 % | -8.1 % | -8.7 % |
| `long_con10` + stage-2 H/M correction | -0.2 % | -0.1 % | -0.1 % | -0.6 % | -0.8 % | -1.1 % |
| `long_con10_adm` (windowed H/M) | -0.8 % | -1.5 % | -2.3 % | -3.5 % | -4.3 % | -4.4 % |

At t = 6.25: hole +0.01 % (`long_con10` +0.30 %); alpha error r < 0.5 1.8e-3 (2.6e-3), r 3-6 6.7e-3 (3.2e-3;
stage-2 H/M 8.4e-3); Theta 1.3e-5 (2.5e-5); drift 3.4e-3 (3.8e-3). Window 24's final loss 1.9e-6 (ADM part 3.4e-7)
vs 1.3e-6. So in the windows the H/M term halves the deficit and slows its growth, with the same lapse
price as stage 2 (about 2x at r 3-6); stage 2 on the finished run still does better (-1.1 %). Its own stage 2 is queued.

**Core-cloud test (2026-10-28, job 1184804, `runs/w16_con10_nocore`).** The collaborators find, with the Theta/Z
penalty W = 10 and the code DEFAULTS (no core cloud, 12k QN), a deficit below 1 % at t = 4 (-0.8, -1.0 %) where
`long_con10` (final settings) has -3.8 %; without a penalty `single_a` vs `long_ctl` point the same way (-2.6 / -3.6 %
at t = 4, -19 / -36 % at t = 10). Their explanation: the interior residual is a plain mean, so the 48k core points
cut the uniform bulk's share from 40 % to 25 % (and the focus cloud's likewise) where the band travels. Our pairs
also differ in QN iterations, resamples and QN code version, so this run changes ONE thing: `long_con10`'s flags
without `--core_points 48000 --core_ic_points 4800` (EXTRA_ARGS="--stop_after_window 16 --EPOCHS_QN 24000
--w_constraint 10"; PROD_ARGS and PRESET=single as before), 16 windows to t = 4, from scratch. Grade: sphere
deficit at t = 2-4 vs `long_con10`, and the hole / core lapse, which the core cloud was added for.

**On hold (2026-10-28; the user: only finish and grade `long_con10_adm`, then SBH pauses): "Ricci = 0 in the same gauge".** Einstein's equations directly, with 1+log and the
Gamma-driver. Note: BSSN evolution + H = M = 0 IS Ricci = 0, so `long_con10_adm` (CCZ4 residual + Theta/Z W = 10
+ H/M W = 1) is already this system in first-order form, up to weights and K being independent. Order agreed:
(1) wait for `long_con10_adm` to finish; (2) the intermediate: the CCZ4 network without Theta and Z (Theta := 0,
Ghat^i := Gt^i[gt], i.e. BSSN) + the H/M term, so no Theta/Z weight; (3) only if worth it, the full 4-metric
version: outputs chi, gt_ij, alpha, beta^i (11), K_ij = -(d_t gamma - L_beta gamma)/(2 alpha), residual R_munu (6
K-evolution + H + M, chi-weighted) + gauge; needs d_t^2 and d_t d_i (extend the Taylor pass or autodiff). Tests:
static Kerr-Schild (residual 0) and the 1D reference's gauge; first 8 windows to t = 2 against the t = 2 runs.

**Losses of both stages (2026-10-28; `tools/plot_losses_stages.py`, `plots/2026-10-28/losses_stages.png`).**
(a) Stage 1, `long_con10`: each window's loss history from its checkpoint, drawn across its own 0.25 M. Each window
starts at 1e-3-1e-2 (the hand-off term: the new network must first match the previous window's end) and ends at
total ~1e-6, dominated by the Einstein residual (5e-7 - 1e-6) from t ~ 1 on; gauge, hand-off and the unweighted
Theta/Z term end at ~1e-7. The first 8 windows (t < 2, the earlier `w8_qn_con10` run) reach lower losses. The final
loss does not grow with t, so the growing mass deficit is not visible in the training loss. (b), (c) Stage 2:
held-out loss of the sum / u1's on the same evenly spread points. Final ratios 0.55 / 0.49 / 0.47 (correction / H/M
/ H/M with gauge x10), H/M part 0.77 / 0.38 / 0.44. Most of the gain comes in the first 3 000 iterations; the H/M
correction has one transient spike near 18 000 (recovered in ~500 iterations). Halving the residual removes
three-quarters of the deficit: the loss measures accuracy badly here too.
To redo it (from `SBH/`, runs as on Mendieta):

    python tools/plot_losses_stages.py plots/2026-10-28/losses_stages.png runs/long_con10 \
        "stage-2 correction=runs/stage2_con10/stage2_history.csv" \
        "stage-2 H/M correction=runs/stage2_con10_adm/stage2_history.csv" \
        "H/M correction, gauge x10=runs/stage2_con10_adm_g10/stage2_history.csv"

Any windowed run and any stage-2 CSVs work (e.g. `long_con10_adm` and its stage 2 when they finish).

**l-decomposition of the deficit (2026-10-26; the collaborators' `reference_1d/l_decomp.py`, received
2026-10-26: per-ray Misner-Sharp error on 300 directions about the origin, real spherical harmonics to l = 4
by least squares, energy share per l; outputs `plots/2026-10-26/l_decomp_*.txt`).** Their question: is the
band in the PENALISED run still l = 0 (a constraint violation -> H/M^i penalty), or l >= 2 (spurious waves ->
sampling, hand-offs)? l = 0 share at the band: `long_con10` 0.52-0.66 at t = 4 (r = 4-6), 0.62-0.67 at t = 7
(r = 8-10; l = 2 0.13-0.23), 0.31-0.39 at t = 10 (r = 10-12; l = 2 0.33-0.34); `single_a` 0.78-0.93,
0.79-0.83, 0.55-0.79; `long_bia3` 0.63, 0.76, 0.63. Their sphere means agree with ours (long_con10 t = 10
r = 12: -18.3 % vs -18.1 % at r = 11.7); their l = 0 shares are lower than our mean^2/mean-square (55 %
there) -- the error is spiky over the sphere (min -0.69 where the mean is -0.18), which an l <= 4 fit
spreads. Reading: l = 0 dominated through t ~ 7 in every run; in the penalised run l = 2 grows to equal
l = 0 by t = 10. With `tools/hamiltonian_shells.py` (the sphere-mean H integrates to -15 % of the -18 % at
r = 11, t = 10) most of the band is still a constraint violation, so the H/M^i penalty comes first.

**What pushes the puncture (2026-10-26; `tools/drift_dipole.py`).** dx_p/dt = -beta(x_p), and the Gamma-driver
d_t beta^i = ... + (3/4) Ghat^i - eta beta^i feeds beta(x_p) from the l = 1 part of Ghat^i at the puncture; CCZ4 splits
Ghat^i = Gt^i[gt] + 2 Z^i/chi exactly. Sphere averages (48 directions, rho = 0.3 about the tracked puncture), t = 10:

| run | |<beta>| | |<Ghat>| | |<Gt>| (metric) | |<2Z/chi>| (constraint) | drift |
|---|---|---|---|---|---|
| `single_a` | 3.9e-2 | 1.4e-2 | 0.13 | 0.14 | 0.13 |
| `long_con10` | 4.0e-3 | 2.9e-3 | 3.5e-3 | 8.3e-4 | 0.010 |
| `long_bia3` | 2.9e-3 | 2.1e-3 | 6.2e-3 | 4.9e-3 | 0.007 |

Without a penalty the conformal metric grows a large dipole that the evolved Ghat^i mostly does not follow
(2Z/chi ~ -Gt, cos ~ -1, from t ~ 3 on); the ~10 % that reaches Ghat drives beta and the drift. The penalties
make the metric's dipole 20-40x smaller and Ghat, beta 5-10x smaller: the asymmetry enters the metric through
the constraint-violating mode, and the penalties close it. (The first guess, "Z adds a dipole to Ghat", is
not what happens: Z cancels most of the metric's.) `single_a` also differs in settings; the control run
`long_ctl` separates the two.

**Sphere averages (2026-10-26; `tools/sphere_errors.py`, `tools/plot_sphere_errors.py`,
`plots/2026-10-26/sphere_errors_rt.png`).** The table above and `mass_function.py` use only the +-x rays. Averaging
over spheres (48 Fibonacci directions, 100 radii, every window end) changes the outer picture. The most negative
<m/M - 1> over 2 < r < 20:

| t | `single_a` | `long_con10` | `long_bia3` |
|---|---|---|---|
| 4 | -2.6 % (r 4.7) | -3.8 % (r 5.5) | -2.5 % (r 3.8) |
| 6 | -6.8 % (r 7.2) | -8.1 % (r 7.6) | -11 % (r 5.2) |
| 8 | -13 % (r 9.5) | -14 % (r 10.0) | -21 % (r 7.2) |
| 10 | -19 % (r 12.4) | -18 % (r 11.7) | -21 % (r 9.5) |

On the +-x rays `long_con10` reaches -36 % at r 11.6 at t = 10: the deficit is far from spherical, and two
rays can double it. On spheres the outgoing deficit is THE SAME with and without the penalties: a band that
leaves the hole at t ~ 2 and follows the gauge pulse outward, in every run (the 10-19 Oct summary's "up to
17 % at r ~ 4-12"). The penalties change only what is inside that front: the hole's mass, <m/M - 1>(r = 1) at
t = 10 = +27 % / +0.19 % / -6.2 %. The alpha error and Theta also run along the outgoing paths. Where the
pulse travels every run has only the uniform sample (the core and focus clouds sit at the hole), the first
candidate cause. The control run `long_ctl` (final settings, no penalty, resumed from `w8_qn` to t = 10; jobs
1184283 -> 1184284, submitted 2026-10-26) compares the hole and the deficit without a penalty.

**The control run (graded 2026-10-28; `plots/2026-10-28/control_sphere_errors_rt.png`).** `long_ctl` finished
in job 1184283 (35.5 h; 1184284 cancelled, not needed). Same sphere averages:

| t | `long_ctl` (final, no penalty) | `long_con10` (final + Theta/Z penalty) | `single_a` (defaults) |
|---|---|---|---|
| 4 | -3.6 % (r 5.2) | -3.8 % (r 5.5) | -2.6 % (r 4.7) |
| 6 | -12 % (r 6.9) | -8.1 % (r 7.6) | -6.8 % (r 7.2) |
| 8 | -22 % (r 8.5) | -14 % (r 10.0) | -13 % (r 9.5) |
| 10 | -36 % (r 10.0) | -18 % (r 11.7) | -19 % (r 12.4) |

At t = 10: hole <m/M - 1>(r = 1) +9.7 % (`long_con10` +0.2 %, `single_a` +27 %); rms Theta over 1 < r < 15
1.1e-2 (5.4e-6, 3.0e-2); alpha error at r 3-6 4.9e-3 (5.2e-3, 2.4e-2); drift 0.014 (0.010, 0.127). The lapse
at the tracked puncture dips below zero at t = 9.75 (-3.7e-3; ~0.035 in the penalised runs); at r = 0.03,
0.013 against the reference 0.034 (`reference_1d/compare_pinn.py`: core / mid alpha error 3.8e-2 / 3.9e-2, 6x the
penalised run's 6.0e-3 / 3.5e-3; max Theta 0.13). So the final settings alone fix the drift and the lapse
error far out (r 3-6), but not the lapse near the hole and NOT the mass: without a penalty the outgoing
deficit grows twice as fast after t ~ 5 and doubles by t = 10, and Theta grows 20x from t = 4 to 10. With
the penalty the deficit returns to the defaults' level. The settings and the penalty act on different
errors; neither removes the outgoing H violation (stage 2 with `--w_adm` is the one that does, section above).

DSGNAR, exact residual: all-or-nothing by window. Working windows accept ~85 % of the steps over the whole
70 min and lower the loss 100-1000x (5.0e-2 -> 2.7e-4; 1.4e-2 -> 8.6e-5); the others (windows 1-3 and 8 of
`w8_gn_exact`, 1-3, 6, 7 of the md05 run) reject 15 steps in a row down to the minimum radius. Keeping the
points 0.05 from the puncture did NOT remove the collapses, so the chi-floor kinks are not the whole
explanation; open.

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

### 4.1 Our DSGNAR against the paper's own code (`ldc/ldc.ipynb`, 2026-10-20)

`ldc/ldc.ipynb` (lid-driven cavity, Re = 100) drives the AUTHORS' implementation, vendored in
`Evolution_try/vendor/dsgnar_pinn` (`pinn.optimiser.Optimiser`), step by step as its `trainer64`
does; our `dsgnar.py` is Stationary's re-implementation (from `Evolution_try/wave_pinn`). The
notebooks have no saved outputs, so this compares the methods, not results.

| aspect | paper code (ldc) | our port (SBH) |
|---|---|---|
| sketched Jacobian | dense rows per macro-batch: `jacrev` vmapped over 4-point sub-batches, then DCT (parameter sketch) and CountSketch, in a `lax.scan` | s forward tangents (JVPs) through the whole residual, CountSketch fused (`sketched_jvp_columns`) |
| its cost (CPU, 6x40, 64 interior points, `tools/bench_jacobian_sketch.py`) | 200 ms at s = 1024, 534 ms at s = 3072 | 530 / 1343 ms: **2.5-2.6x slower** (same J Omega to 1.5e-15) |
| sketch | s = d/3 (a30 preset: 1665 for 5k params; paper preset 4000 for 11k) | s = 1024 = d/9 in the window-15 branch |
| role | primary optimiser from the initial network, 200 steps, then SSBroyden polish; graded by rel. L2 against a reference | 12 steps after 24 QN restarts in every window |
| step, radius | evaluated once; accept iff rho > 0 -- the code does NOT reject overshoots (it only expands the radius on a rejected overshoot), although the paper's text does; radius -> PCHIP radius | default: line scan, calibration, grow-only radius with floor; `--dsgnar_paper` follows the TEXT (rejects overshoots) |
| target ratio | 0.075 fixed in the notebook; `trainer64` switches to 0.5 after lambda bottoms out (grace 80) | 0.15 -> 0.5 (paper preset 0.075 -> 0.5) |
| weights | fixed (pde 1e-2, bc 1) in the notebook; `trainer64` reweights with alpha = 0.05 | the loss's lambda-weights; `--dsgnar_reweight` |
| precision, network | float32 pre-pass, then float64; SIREN | float64 only; tanh MLP |

So the window-15 branch was not the paper's use of DSGNAR (a third of its sketch, a tenth of its
iterations, a slower Jacobian): it showed that a short DSGNAR phase does not help QN, not that
DSGNAR cannot be a better primary optimiser here. The naive "rows" variant (`jacrev` of the whole
block at once) is 3x SLOWER than ours: the speed comes from vmapping `jacrev` over small
sub-batches. The 2.5x is a lower bound (the bench applies the parameter sketch as a dense matrix;
the paper's DCT is cheaper and nearly independent of s); not measured on the GPU.

**Rows-first DSGNAR, implemented (2026-10-20).** `--dsgnar_jacobian rows` builds DSGNAR's sketched
Jacobian the paper code's way: per-point reverse-mode Jacobians (`jacrev` vmapped over
`--dsgnar_row_batch` points), folded straight into the CountSketch buckets with each row's frozen
scale, and the parameter sketch (SRCT) applied once to the s x n result -- C J Omega S = (C J) Omega S
by linearity, which avoids even the paper's per-row DCT. `_DsgnarObjective.sketch_rows` does it;
`dsgnar.py` uses it when asked (default stays `cols`). Verified (`tools/test_dsgnar_bh.py`, test 4):
equal to the column construction to 1.6e-13 for the default, Dirichlet + log-IC and physical-chi row
layouts, and a rows-mode DSGNAR phase reproduces the cols-mode loss history. **Cost on the whole
objective** (CPU, production 6x40 network, 19680 rows, s = n/3 = 3100): rows 1.6 s, cols 18.4 s per
sketched Jacobian, **11.5x**. GPU not yet measured.

Related flags: `--dsgnar_max_seconds T` (stop a cycle after T s, for equal-time comparisons; an
addition to `dsgnar.py`), and `--dsgnar_reject_overshoot` is now on/off (`--dsgnar_paper
--no-dsgnar_reject_overshoot` = the paper's settings with "accept any step that lowers the loss",
which is what the paper's code does).

**Sampling flags for the drift (2026-10-20).** `--sampling_centers initial` keeps the focused and
core clouds at the punctures' t = 0 positions while tracking and the moving background run;
`--symmetric_focus` draws each cloud in point-reflected pairs about its centre (offsets sum to zero to
round-off). Both off by default (bitwise as before).

**Short comparisons, 8 windows (t <= 2), submitted 2026-10-20** (replacing the 40-window
final-settings run and the t <= 4 / window-15 plans; the 1D reference will cover this span):

| run | job | settings |
|---|---|---|
| `w8_qn` | 1181117 | final settings of the 10-19 Oct summary (48k core points, `--core_radius 0.5`), SSBroyden 24k its per window, 1 A30 |
| `w8_gn` | 1181118 -> cancelled, rerun with fixed weights (below) | same sampling, DSGNAR only (`--EPOCHS_QN 0`): rows Jacobian, s = n/3, `--dsgnar_paper --no-dsgnar_reject_overshoot` (accept any step that lowers the loss), 4200 s per window to match `w8_qn` |
| `w16_drift_initc` | see below | `single_a` settings + `--sampling_centers initial`, 16 windows (t <= 4) |
| `w16_drift_symm` | see below | `single_a` settings + `--symmetric_focus`, 16 windows (t <= 4) |

**`w8_gn` first attempt (job 1181118, cancelled after 41 iterations).** Rows-mode DSGNAR ran at ~15 s per
iteration on the A30 (9.7 GB, s = 3100, 128k interior points; the column method took ~125 s at s = 3100 with
88k points), all steps accepted -- but the `--dsgnar_paper` per-condition reweighting had collapsed: weights
1e-11..7, the reweighted loss 1e-13 while the actual loss stayed at 1.38e-2. Cause: the positivity
condition is identically zero while chi, alpha > 0, so (mean/(0 + eps))^0.05 raised its weight every
iteration and the normalisation drove every real condition to zero. Fixed in `dsgnar.py` (conditions with
zero loss keep their weight and are left out of the update and normalisation; verified locally: weights stay
in 0.3-1.8, loss halves in 15 steps). The rerun uses fixed weights (`--dsgnar_reweight 0`), as the `ldc`
notebook does.

**`w8_gn` (paper settings, fixed weights; job 1181134): the trust region collapses in every window.**
With the paper's step rule (step evaluated once, no line scan, radius to the PCHIP value, no floor,
delta_min 1e-8) each window stops on "radius below threshold" after 4-11 minutes of its 70: window 1
2 accepted / 14 rejected, loss 12.9 -> 11.1 (SSBroyden reaches ~1e-7 in the same window); window 2
2 / 16; window 3 27 / 12, 0.14 -> 7.6e-3. The full Gauss-Newton step overshoots about half the time on
this problem (seen already in the production probes: ratio -5e6 at the full step, +1.2 at half of it),
and without a scan every overshoot divides the radius by 3 -- the collapse Stationary saw and that
its port's safeguards were added for. Kept as the record of the paper's settings.

**`w8_gn_scan` (job below):** the same pure-GN run (rows, s = n/3, fixed weights, any step that
lowers the loss is accepted, 4200 s per window) with the port's step safeguards instead of the
paper's: the line scan (1, 1/2, ..., 1/16, 1e-3, 1e-2 of the step; first that lowers the loss is
taken), a grow-only radius with a floor at 1e-3 of its high-water mark, calibration, targets 0.15/0.5.

**First result, pure GN vs pure Broyden at equal time (2026-10-20, end-of-window training loss on each
run's own samples):**

| window | `w8_gn` (paper step rule) | `w8_gn_scan` (line scan + radius floor) | `w8_qn` (SSBroyden) |
|---|---|---|---|
| 1 | 12.9 -> 11.1, collapsed after 4 min | 12.9 -> 1.07e-2, 275 its, all accepted | -- |
| 2 | -> 0.19, collapsed | 9.0e-3 -> 5.9e-3 | 2.4e-7 |
| 3 | -> 7.6e-3, collapsed | 5.6e-3 -> 4.7e-3 | 3.8e-7 |
| 4-8 | 5.0e-3 ... 3.0e-3, collapsed every window | (running) | 5.8e-7, 8.8e-7, ... |

With the safeguards DSGNAR accepts every step and uses the whole budget (~272 iterations of ~15 s per
70-min window), and still ends each window ~1e4 above SSBroyden -- far beyond the 5-10x between
own-sample and held-out losses, so not a sampling artefact. Held-out and mass grading follow when the
runs finish.

**Against the 1D reference (2026-10-20; `w8_qn` 5 windows, `w8_gn_scan` 3, `w8_gn` 8 at the time;
`reference_1d/compare_pinn.py`).** alpha at r = 0.03 (PINN / reference) and the L2 error of alpha in the
core [0.05, 0.5) and mid [0.5, 5) bands:

| run | t | alpha(0.03) | error | alpha core | alpha mid |
|---|---|---|---|---|---|
| `w8_qn` (SSBroyden, final settings) | 0.25 | 0.130 / 0.108 | **0.022** | 1.0e-3 | 3.2e-5 |
| | 1.0 | 0.180 / 0.147 | **0.032** | 3.1e-3 | 8.5e-5 |
| `w8_gn_scan` (DSGNAR, safeguards) | 0.25 | 0.623 / 0.108 | 0.52 | 0.27 | 9.3e-2 |
| | 0.75 | 0.604 / 0.125 | 0.48 | 0.23 | 0.15 |
| `w8_gn` (DSGNAR, paper step rule) | 0.25 | 0.933 / 0.108 | 0.82 | 0.56 | 0.12 |
| | 1.0 | 0.480 / 0.147 | 0.33 | 0.14 | 0.38 |

SSBroyden with the final settings reproduces the 10-19 Oct summary's "+ both" column (0.019 / 0.024 at
t = 0.25 / 1) and improves 5x on the first baseline (`single_a`: 0.108 / 0.153). Pure DSGNAR is ~250x
worse in the core and ~3000x at mid radii and does not capture the collapsed lapse (0.6 instead of
0.11 at the puncture). **For this problem DSGNAR, in either form, is not competitive with SSBroyden.**

**Final results of the 8- and 16-window runs (2026-10-21).**

*Optimizers, all 8 windows, against the 1D reference:* SSBroyden (`w8_qn`) at t = 2: alpha(0.03) 0.276 vs
0.281 (error 0.005), core alpha L2 1.2e-2, mid 3.5e-4, max |m/M - 1| on r in [0.5, 10] = 1.1e-2.
DSGNAR with safeguards (`w8_gn_scan`) at t = 2: alpha(0.03) 0.541 vs 0.281, core 0.15, mid 0.21, mass
error 0.50. The verdict above stands.

*Drift, 16 windows (t <= 4), `single_a` settings:*

| run | puncture drift t = 4 | `off_form` t = 4 | |beta| at origin t = 2 / 3 / 4 | mass error t = 4 |
|---|---|---|---|---|
| `single_a` (clouds follow the track) | 0.020 | ~0.6 (0.36 at t = 3) | 1.1e-2 / 3.5e-2 / 2.9e-2 | 6.9e-2 |
| `w16_drift_initc` (clouds at the origin) | 0.010 | 0.32 | 4.9e-3 / 1.8e-2 / 1.7e-2 | 1.9e-2 |
| `drift_fixed` (no tracking) | -- | -- | 2.0e-3 / 8.4e-3 / 2.9e-3 | 3.4e-2 |
| **`w16_drift_symm`** (`--symmetric_focus`) | **0.0008** | **0.17** | **2.2e-3 / 8.3e-4 / 2.1e-3** | 3.4e-2 |

Point-reflected sampling clouds stop the drift (25x less than `single_a`, the push at the origin as
low as with tracking off), while holding the clouds at the origin only halves it. So the seed is the
random lopsidedness WITHIN each cloud, which the tracking then amplifies -- not mainly the clouds
following the track. Tracking can stay on with `--symmetric_focus`; it does not fix the mass error
(a separate problem, as the |Z| growth already suggested).

Both comparison runs use ONE optimizer each (Adam is off by default; `w8_gn` has `--EPOCHS_QN 0`).
The drift tests were lengthened to 16 windows (jobs 1181119/20 cancelled before starting): at t = 2
the push at the origin separates tracking from a fixed centre by 4x, at t = 4 by 10x. They keep
`single_a`'s settings so that `single_a`, `drift_phys` and `drift_fixed` are their comparisons.

**Loss plots by optimiser (2026-10-20).** `loss_window{w}.png` now shades each stretch of the
history by the optimizer that produced it (Adam, QN, DSGNAR) and names it at the top, instead of
one Adam -> BFGS line.

**Executive summary** for the PINN team, in the format of J. Masso's 10-19 Oct summary
(`originals/BH_PINN_executive_summary.pdf`): `docs/exec_2026-10-20/SBH_optimisers_drift_mass_2026-10-20.pdf`
(source `summary.html`, figures from `tools/plot_exec_summary.py`). Second summary, physics only (collocation points, constraint penalties to t = 10, mass, drift; 3 pages): `docs/exec_2026-10-26/` (figures from `tools/plot_exec_2026_10_26.py` and `tools/plot_sphere_errors.py`; the control run `long_ctl` added and "Draft" removed on 2026-10-28; section 7, stage 2, added the same day with `stage2_profiles.png` (`tools/plot_profiles_compare.py`) and `stage2_sphere_errors_rt.png`; 4 pages now; a preliminary note on the windowed H/M run at 25 of 40 windows added to section 7). Note from that summary: our runs
used the code's defaults, i.e. its "first baseline" (no core points, 12k QN steps per window); its
final settings (48k core points, 24k QN steps) gave +7-8 % mass error and 0.023 drift at t = 10
where `single_a` has 35 % and 0.127.

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
* `tools/eval_heldout.py`: held-out grading of finished runs (identical fresh samples per window).
* `tools/puncture_probe.py RUN... -- <physics flags>`: shift at the origin and fine-grid chi minimum per saved window.
* `tools/mass_function.py RUN --out X.npz -- <physics flags>`: Misner-Sharp mass, Theta, |Z^i| along the x-axis per window.
* `tools/plot_report_figures.py DATA OUT`: the figures in `plots/2026-10-20/`.
* `tools/plot_exec_summary.py DATA OUT`: the figures of the executive summary.
* `tools/sketch_noise.py`, `tools/test_exact_residual.py`: DSGNAR's step vs the true gradient; the exact-residual mode.
* `tools/sphere_errors.py RUN_DIR REF.npz --out RUN.npz -- <physics flags>`: Misner-Sharp error and alpha/chi/Khat/Theta - reference averaged over spheres (Fibonacci directions) at every window end; `tools/plot_sphere_errors.py OUT.png NAME=RUN.npz ...`: their radius-time maps. ~6 s per 40-window run on the laptop CPU.
* `stage2.py RUN_DIR --out DIR [--selftest] -- <physics flags>` (+ `stage2_net.py`): the stage-2 correction network over a finished run (above).
* `adm_constraints.py` + `tools/test_adm_constraints.py [RUN_DIR]`: the ADM constraints H, M_i from a derivative stack, and their checks (exact Schwarzschild isotropic and Kerr-Schild; Taylor vs autodiff).
* `tools/hamiltonian_shells.py RUN_DIR 0s 15 39 -- <physics flags>`: the TRUE Hamiltonian constraint H = R + K^2 - K_ij K^ij on spheres (mean = l = 0 part, rms) at window starts/ends, and the mass change dm/dR = R^2 <H>/4 it implies (long_con10, t = 10: -15 % at r = 11; t = 0: < 1e-3).
* `tools/l0_share.py RUN.npz ...`: l = 0 share (mean^2 / mean-square) of the sphere-sampled Misner-Sharp error at the peak of the outgoing deficit, from `sphere_errors.py` output (long_con10: 79/83/77/55 % at t = 4/6/8/10; single_a 88-91 %).
* `tools/drift_dipole.py RUN_DIR -- <physics flags>`: sphere-averaged beta, Ghat and its metric / constraint parts (Gt, 2Z/chi) at the tracked puncture, every 4 windows: what drives the drift.
* `tools/plot_mass_budget.py RUN.npz OUT.png`: the mass-error budget (initial / in-window / hand-off) by radius.
* `tools/plot_clouds.py OUT.png`: the interior points around a single puncture, default vs `--symmetric_focus` (`plots/2026-10-21/sampling_clouds.png`; centroid of the points within r < 1: 2.8e-3 vs 1e-5).
* `tools/bench_jacobian_sketch.py [--s 1024]`: rows (paper) vs columns (ours) for DSGNAR's sketched Jacobian, one block of points.

## 7. Template + correction ansatz: `sing_pinn.py` (2026-10-28)

A new script, separate from `EEccz4_BH_puncture.py`: the puncture's singular structure is no longer
learned by the network but built in, and the network learns only the departure from it. Per window and
puncture A, with xi_A = x - a_A(t), r_A = |xi_A| (full description in the script's docstring):

    W_A = exp(-(r_A/r_c)^4),   rho = prod_A r_A^2/(r_A^2 + r_c^2)               (r_c = 0.2 by default)
    chi, alpha:  u = exp(sum_A W_A ln T_A) * exp(rho N)        (multiplicative: vanishing order and sign kept)
    beta:        u = sum_A W_A (T_A^beta - da_A/dt) + rho N     (beta(a_A) = -da_A/dt: no tracker)
    the rest:    u = flat + sum_A W_A (T_A - flat) + rho N

* **Template T** (`sing_template.py`): the 1D spherical solution of `reference_1d/sph_reference.py` as a
  tensor-product cubic spline in (r, t), mirrored through r = 0 with each radial function's parity, lifted to
  the 22 Cartesian fields. `--template bl`: the time-dependent evolution from Brill-Lindquist data
  (`reference_1d/sing_bl_m0.5_dense.npz`: dr = 0.005, snapshots every 0.02, r <= 1, t <= 10); at t = 0 it is
  the BL puncture (chi to 2e-9, alpha to 2e-6). `--template trumpet`: the snapshot at t_s = 10 (20 M) of
  `reference_1d/ref_single_m0.5_dr0.005_t20.npz` (the reference run extended to t = 20, r <= 30), frozen; the
  run then starts from that whole slice (initial data = the 1D solution at t_s, graded against it at t_s + t).
* **Why the cutoff:** inside r_c ~ 0.2 (inside the horizon) the template carries the r^4 / r^2 zero of chi,
  the kink of alpha and the direction-dependent limits of the tensors at the puncture; outside it the network
  learns everything, so the 1D reference stays an honest yardstick there (it is used for the template only
  for r <~ 0.4, where W > e^-16).
* **Centre a(t)** = a0 + v0 (t - t0) + s_a (p1 tau^2 + p2 tau^3), a0, v0 from the previous window (continuous
  position and velocity), p1, p2 trained with N (`--fixed_center` to freeze).
* **Sampling: no exclusion zone** (there is no `--min_puncture_dist`). The interior clouds are drawn about a(t)
  in antithetic pairs (xi, -xi): an inner Gaussian (`--inner_sigma 0.05`, `--inner_frac 0.125`), the core
  (sigma 0.5, 0.375) and focused (sigma 2, 0.25) Gaussians, and a uniform bulk. Per 32k-point batch that is ~3-9
  points within 0.01 of the puncture, ~70 within 0.02 and ~750 within 0.05 (the sigma 0.5 core alone: 5 within
  0.05). The script prints the counts at each window start, and `sing_history.csv` splits the held-out
  Einstein + gauge loss by distance to the puncture (d < 0.01, 0.01-0.05, 0.05-0.5, > 0.5).
* **Near r = 0:** the lifted template stays finite down to r = 1e-5. Its own Einstein residual there is
  2e-4-7e-4 at t = 0.1, 2e-3-2e-2 at t = 1, and 0.06-0.4 for r <= 0.01 at t = 5 (trumpet slice: 0.02-0.3). That
  is the 1D reference's error at its innermost points (1st-2nd order for r < 0.1, Kreiss-Oliger dissipation, the
  chi floor). Since rho ~ r^2/r_c^2 there, N can hardly change it: those points mainly fix a(t), and they set a
  floor to the loss in the d < 0.01 band. A finer 1D run near the centre would lower it (not done). The table's
  chi dips to -6e-7 at its innermost point (r = 0.0025), so the log blend uses the smooth floor
  (v + sqrt(v^2 + 4 eps^2))/2, eps = 1e-10.
* **Loss:** the main script's terms and weights (Einstein + gauge rows, lam_tapa x the same at t1, Sommerfeld
  BC, lamg0 x soft IC), + `--w_constraint` (default 10, the setting that held the mass in `long_con10`).
  No causal weights. SSBroyden, fresh points every `--n_change`.

**Checks (`--selftest`, CPU, 2026-10-28):**

| check | result |
|---|---|
| derivative stacks (Taylor N + autodiff template, product/chain rule) vs autodiff of the whole u, random N, moving off-centre puncture, points down to 1e-4 from it | max residual difference 6e-12 (bl) / 5e-11 (trumpet), residual up to ~150 |
| shift at the puncture vs -da/dt | equal to 1e-4 (the probe sits 1e-4 from the centre) |
| CCZ4 residual of the lifted template itself (no cutoff), dense table, rms Einstein rows | t = 0.1: 5e-5 / 2e-5 / 6e-6 for r in [0.05,0.1) / [0.1,0.3) / [0.3,0.6); t = 1: 6e-4 / 7e-5 / 1e-5; t = 5: 2e-4 / 1e-5 / 7e-6 |

With the 0.25-spaced table (the original reference) the template residual at t = 0.1 was 4e-2: the early
lapse collapse needs the dense time table. A 2-window CPU smoke run (tiny network, 100 iterations) completes
and writes the outputs.

**The two templates near the centre** (`tools/plot_sing_templates.py` -> `plots/2026-10-28/`: `bl_rt.png`,
`trumpet_rt.png`, `trumpet_drift.png` r-t maps, `profiles.png` radial profiles; r from the first grid point
0.0025 to 1, log axis, dashed lines at r_c and 2 r_c):

* chi goes from ~ r^4 (BL, 1e-8 at r = 0.0025) to ~ r^2 (trumpet, 2e-5 there) during t ~ 2-4; alpha stays ~ r
  but drops ~5x at fixed small r over the same span, after the transient rise to ~0.25 near r ~ 0.01 at
  t ~ 2 (the physical rise noted in reference_1d/README.md). Khat at the centre: 0 -> 1.1 (t = 5) -> 0.54
  (t = 10) -> 0.60 (t = 20); At's n_in_j coefficient d -> -2.7 at the centre.
* The transition is fastest inside r ~ 0.05, i.e. well inside r_c: the bl template carries it; the trumpet
  template starts after it.
* The trumpet is not exactly stationary at t = 10: over t = 10-20 Khat drifts by up to +0.09 (largest at r ~
  0.005-0.02), d by -0.05, alpha by +-0.007, chi, b, beta, g by <~0.01, mostly at r >~ 0.1
  (`trumpet_drift.png`). That is what N must supply inside the cutoff in the trumpet run, where rho is small.
* The 1D data are unreliable at the innermost 3-4 grid points (r < 0.02): Theta oscillates there (+-1e-3, up
  to 0.027 at t = 2), and chi at r = 0.0025 is <= 0 for t <~ 2.3 (the white strip in `bl_rt.png`'s chi panel;
  the chi floor and the dissipation). That is the template residual floor at d < 0.01 noted above.

Outputs in `runs/OUTDIR`: `window_KKK.pkl`, `sing_history.csv` (per draw: held-out loss and parts, a(t1)),
`sing_report.csv` (per window end: rms error of each field group vs the 1D reference in r in [0.05,0.5),
[0.5,5), [5,20) on 8 rays, and alpha at r = 0.03), `config.json`.

**Run on the hub** (`jobs/hub/sing.sh`; it generates the two template tables on the CPU first if they are
missing, ~20 + 40 min, since `*.npz` is not in git):

    cd ~/serafin/Julia/PINN && git pull && cd SBH
    OUTDIR=sing_bl SING_ARGS="--template bl" bash jobs/hub/sing.sh
    OUTDIR=sing_tr SING_ARGS="--template trumpet" bash jobs/hub/sing.sh

**First hub run, `sing_bl` (2026-10-28, 8 windows, 3 h 25 min).** ~25 min per window (12000
SSBroyden iterations, vs ~90 min for `w8_qn_con10`). Every window ends at held-out loss 3-6e-4, the d < 0.01
band at ~1e-8 (the template carries it), but the error against the 1D reference grows from window to window:

| window | t | alpha core | Khat core | gt core | At core | Ghat core | alpha inner (r < 0.05) | a_z |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.25 | 1.7e-3 | 3.8e-3 | 1.9e-3 | 6.4e-3 | 1.4e-2 | 6.3e-4 | -4e-5 |
| 1 | 0.5 | 3.3e-3 | 6.9e-3 | 3.1e-3 | 7.1e-3 | 1.8e-2 | 8.1e-4 | -9e-5 |
| 2 | 0.75 | 6.5e-3 | 9.8e-3 | 4.0e-3 | 8.6e-3 | 1.3e-2 | 1.8e-3 | -3e-5 |
| 3 | 1.0 | 1.0e-2 | 1.4e-2 | 4.7e-3 | 1.2e-2 | 1.1e-2 | 4.0e-3 | +2e-4 |
| 4 | 1.25 | 1.4e-2 | 1.8e-2 | 5.1e-3 | 1.6e-2 | 1.3e-2 | 8.2e-3 | +6e-4 |
| 5 | 1.5 | 1.7e-2 | 2.3e-2 | 5.5e-3 | 2.2e-2 | 1.8e-2 | 1.6e-2 | +1.1e-3 |
| 6 | 1.75 | 2.0e-2 | 2.7e-2 | 6.0e-3 | 3.1e-2 | 2.3e-2 | 2.9e-2 | +1.8e-3 |
| 7 | 2.0 | 2.0e-2 | 3.3e-2 | 6.4e-3 | 4.4e-2 | 3.0e-2 | 4.4e-2 | +2.7e-3 |

alpha(0.03) stays within 0.6% of the reference (0.2383 vs 0.2369 at t = 1.5), but at t = 1.5 alpha core is
already ~20x and gt ~3x the `w8_qn_con10` values at t = 2. The trained centre drifts along z from window 3 on
(a single puncture at rest should stay at the origin), and the inner error grows fastest; `--fixed_center`
is the control for that.

**`--pre_iters N` (`--pre_windows K`, default all windows).** Each window starts with N SSBroyden
iterations on the IC + Sommerfeld terms alone (no interior, end or constraint points are drawn), then the
`--iters` on the full loss. Motivation: in window 0 the initial loss is almost all IC (0.126 of 0.151), so
the interior residual was being minimised around initial data not yet fitted. CPU smoke run (2 windows, tiny
network) completes; the history has a `phase` column (pre / full) and the log marks pre iterations with `p`.

**Results of the three hub runs (2026-10-29; 8 windows each, ~3.5 h each).** At t = 2 against the 1D
reference ("core" = rms over 0.05 <= r < 0.5 on 8 rays; trumpet: the reference at t + 10):

| run | alpha(0.03) (ref) | alpha core | Khat core | gt core | At core | Ghat core | Theta core | alpha inner | alpha far | a_z(t=2) |
|---|---|---|---|---|---|---|---|---|---|---|
| `w8_qn_con10` (baseline, 12 h) | 0.289 (0.281) | 7.1e-4 | 8.7e-3 | 1.8e-3 | | | (max 3.1e-4) | | | |
| `sing_bl` | 0.2821 (0.2814) | 2.0e-2 | 3.3e-2 | 6.4e-3 | 4.4e-2 | 3.0e-2 | 1.2e-3 | 4.4e-2 | 1.9e-3 | +2.7e-3 |
| `sing_bl_pre` (`--pre_iters 2000`) | 0.2816 (0.2814) | 1.3e-2 | 1.9e-2 | 5.6e-3 | 2.1e-2 | 2.9e-2 | 7.2e-4 | 2.1e-2 | 3.9e-3 | -1.8e-3 |
| `sing_tr` (trumpet, t = 10-12) | 0.0348 (0.0347) | 2.7e-3 | 4.2e-2 | 2.2e-3 | 1.9e-2 | 4.3e-3 | 6.6e-5 | 1.5e-3 | 1.2e-2 | -3.1e-4 |

* The template gets the lapse at the puncture right: alpha(0.03) is within 0.25% of the reference at t = 2
  in both BL runs (`w8_qn_con10`: 2.8%). Everywhere else the error is 3-30x the baseline's, and it grows
  steadily from window to window although every window ends at held-out loss 3e-4 to 1e-3 and IC 4e-5:
  the errors accumulate through the window chain.
* `--pre_iters 2000` helps modestly (+5% time): at t = 2 alpha core 1.5x, Khat 1.7x, At 2x, alpha inner 2x
  lower; window 0 alpha core 7.7e-4 vs 1.7e-3. It does not stop the accumulation.
* The centre drifts along z in both BL runs, with opposite signs (+2.7e-3, -1.8e-3): a weakly determined
  mode, not an effect of the IC fit. The inner error grows with it. Next: `--fixed_center` as the control.
* Trumpet: the losses fall through the windows (held 5.7e-4 -> 1.5e-4) and most fields stay at ~2e-3, but
  Khat core grows linearly (8.7e-3 -> 4.2e-2) and alpha(0.03) stays frozen at 0.0348 while the reference
  moves 0.0353 -> 0.0358 -> 0.0347: the template's drift (`trumpet_drift.png`) is not supplied by N inside
  r_c, where rho ~ r^2 is small, as anticipated. alpha far is 1e-2 from window 0 on (the t = 10 slice
  probably has an outgoing gauge pulse in r > 5 that the IC fit does not resolve; not checked).

**Mass and loss histories (2026-10-29; `sing_pinn.py --eval_spheres NPZ` = `tools/sphere_errors.py` for
these runs, same npz layout; baseline = `tools/sphere_errors.py` on `long_con10`'s first 8 windows, which
are `w8_qn_con10`; figures `tools/plot_sing_mass.py` and `tools/plot_sphere_errors.py` ->
`plots/2026-10-29/`: `sing_mass_profiles.png`, `sing_mass_time.png`, `sing_losses.png`,
`sing_sphere_errors_rt.png`).**

* The sing runs' mass error is set by the INITIAL-DATA fit, not by the template or the evolution: already at
  t = 0, <m/M - 1> is ~ +1 % over 0.2 < r < 3, +3 to +6 % at r ~ 3-8 and -4 to -10 % beyond r ~ 10, in
  all three runs (the trumpet's from its own t = 10 slice), and that pattern persists to t = 2. The
  baseline's t = 0 error is < 1e-4 over 0.2 < r < 10 and ~ -0.1 to -0.4 % at t = 2. The IC term ends every
  window at ~4e-5 (sing_bl) but the far field is clearly not fitted well enough for the mass, which
  depends on first derivatives of the metric. Not yet diagnosed: candidates are the sampling (the IC
  points use the same Gaussian clouds as the interior, so r > 5 gets only the uniform bulk) and the
  baseline's extra IC terms (`loss_g0`, `loss_gt0`, `loss_ic_log`) that sing_pinn.py does not have.
* The centre drifts in all three components, not only z: |a(2)| = 6.5e-3 (`sing_bl`), 2.7e-3
  (`sing_bl_pre`), 1.9e-3 (`sing_tr`), growing steadily from window 1.
* `--pre_iters` on warm windows is destructive: at the start of every window after the first, the
  IC + boundary-only phase drives the Einstein residual from ~1e-4 back up to ~0.1 (`sing_losses.png`,
  middle top) and the full phase spends ~2000 iterations recovering. Its gain at t = 2 comes from window 0;
  `--pre_windows 1` should keep that and drop the cost.
* The Einstein + gauge loss at 0.01 <= d < 0.05 from the puncture rises ~50x over windows 5-7 in both BL
  runs (t = 1.5-2, the start of the chi ~ r^4 -> r^2 transition, and while the centre drifts); the trumpet
  run's stays flat.

**Next pair (2026-10-29): fit the initial data on the whole point budget, and a lighter lid.**
`--pre_n_ic N` sets the IC points per draw in the `--pre_iters` phase (same clouds as the interior, all at
t = t0; the phase takes no derivatives except on the boundary, so 150000 points are affordable, against
8000 normally). Both runs: `--pre_iters 4000 --pre_windows 1 --pre_n_ic 150000` (window 0 only, since the
phase hurt warm windows). `sing_bl_ic` keeps `lam_tapa` = 2; `sing_bl_ic_t05` passes `-- --lam_tapa 0.5`, so
the end-slice term no longer dominates the loss. Centre still trained (`--fixed_center` not used), so the
two changes are judged alone against `sing_bl` / `sing_bl_pre`.

**Point counts (2026-10-29): the runs above used FEWER points than the baseline.** `sing_pinn.py`'s
defaults were 32000 interior + 4000 end + 8000 IC + 4000 boundary = 48000 per batch; `w8_qn_con10` uses
Nint 80000 + core_points 48000 = 128000 interior, NI 8000 end, NI 8000 + core_ic_points 4800 = 12800 IC,
Nb 16000 boundary (~165000; the `Nint: 80000` lines in the sing logs are the main script's unused
defaults printed by load_defs). So `sing_bl`, `sing_bl_pre`, `sing_tr` are not a like-for-like comparison
(3.5x fewer points, 4x fewer on the boundary). From now on the defaults ARE the baseline counts:
`--n_int 128000 --n_end 8000 --n_ic 12800 --n_bc 16000` (the interior shares are unchanged, so ~4x the
point counts quoted above near the puncture). All future sing runs use them.
The same holds for the iterations: the baseline runs `--EPOCHS_QN 24000` SSBroyden iterations per window,
the runs above 12000; `--iters` now defaults to 24000. (Network 40 x 6, loss weights, `--w_constraint 10`
and the 250-iteration restarts already matched.)

**`--hard_ic` (2026-10-29): exact initial data built into the ansatz.**
    u = h(x) + B(t, x) + s(t) rho N        (additive fields; for chi, alpha the same in ln u),
    h = U0 - B(0, .)  (ln U0 - ln B(0, .) for chi, alpha),     s = t / t1 in window 0, s = 1 later,
with U0 the exact data (BL, or the trumpet's 1D slice) and B the template part at t = 0 about the initial
centres. h is fixed (no trained parameters, no t), so it and its x-derivatives are computed once per draw at
the batch's points (`hard_stacks`), not per iteration. In window 0, u(0) = U0 identically (to 1e-10, the chi
floor; the IC loss is 0 and the `--pre_iters` phase is skipped there); later windows keep the soft hand-off.
h stays in EVERY window because it carries the exact data's far field (psi^-4 - 1 etc.): a first version
with h in window 0 only left that whole field to the network of window 1 (hand-off IC error 0.27 in a small
CPU smoke run, 0.044 with h kept). Unlike the main script's `--ic_mode hard` there is no chain through
earlier windows' networks. `--selftest` checks the hard stacks against autodiff in window 0 and in a later
window (8e-15, 2e-10).

**Laptop CPU NaN (2026-10-29).** With the full 40 x 6 network and the default 16000 boundary points, the
jit-compiled loss is NaN at the very first evaluation on the M2 CPU, while the same function run
de-optimized (`JAX_DEBUG_NANS=True`) is finite; 2000 boundary points are fine, and the hub GPU computes the
same configuration without trouble (also with the code before `--hard_ic`). An XLA CPU compilation issue,
not the ansatz: run full-size configurations on the GPU only, and use small `--n_bc` for CPU smoke tests.

**State after the IC phase.** Every run now saves `pre_000.pkl` at the end of window 0's `--pre_iters`
phase; `--stop_after_pre` exits there; `--eval_spheres X.npz --eval_pre` grades that state. `--eval_spheres`
also stores `ic_rms`: each field group's rms error against the exact initial data on the t = 0 spheres.
`tools/plot_ic_error.py OUT.png NAME=X.npz ...` plots the t = 0 errors by radius (mass, alpha, chi, Khat,
Theta against the 1D reference for every run; gt, At, Ghat, beta against the exact data for sing runs).

**IC phase alone, `sing_bl_ic_pre` (2026-10-29; hub, `--pre_iters 4000 --pre_windows 1 --pre_n_ic 150000
--stop_after_pre`, 326 s, peak 4.3 GB).** IC loss 8.7e-8 at the end. `plots/2026-10-29/sing_ic_error.png`
(baseline, `sing_bl` window 0 after full training, and this state; all at t = 0):
- gt, At, Khat, Theta, Ghat, beta are exactly 0: the BL template already is the exact data in those fields,
  the output layer starts at zero, and the IC + Sommerfeld loss has zero gradient in their columns, so
  they stay zero. Only chi and alpha are fitted.
- chi: rms 3-6e-4 for r > 0.2 (baseline 5e-6 to 1e-5, i.e. ~50x better); `sing_bl` after full training 2e-3
  to 6e-3. alpha: 1e-4 to 1.5e-3, close to the baseline beyond r ~ 3, worse in 0.1 < r < 1.
- mass: |<m/M - 1>| up to 1e-2 (sign changes through r), against the baseline's 1e-5 to 1e-4 for r > 0.2.
  The IC loss averages over 22 fields of which 20 are exact, so 8.7e-8 means rms ~3e-4 in chi and alpha;
  the mass uses chi's radial derivative and amplifies that. The soft fit is not converged at this level,
  which is the case for `--hard_ic`.

**`sing_bl_hard` first launch (2026-10-29): died before its first iteration.** Run alone after
`sing_bl_ic_pre`, it failed ~5 min after starting with RESOURCE_EXHAUSTED "Failed to load in-memory CUBIN ...
out of memory" in `jit_parts` (MEM_FRACTION 0.95, no other run of ours on the GPU). Cause not yet known:
the hard stacks themselves are ~0.2 GB at baseline counts. `nvidia-smi` afterwards: GPU empty, so no other
process. The traceback is at the first held-out `parts_jit` call, i.e. after the first 250 iterations had run:
the BFC pool (no preallocation, may grow to MEM_FRACTION 0.95 and never returns memory) had left too little
outside it to load a newly compiled kernel. `sing_pinn.py --mem_report` (compiles window 0's value+gradient
step at full size, soft and hard, prints XLA's memory analysis; on the laptop CPU: temp 18.52 GB soft, 18.58 GB
hard, arguments 0.01 vs 0.22 GB) shows the hard ansatz does not enlarge the step. Fixes: `parts_jit` is
compiled on the held-out set before training starts, and `jobs/hub/sing.sh` defaults to MEM_FRACTION 0.85
(~3.7 GB left for kernels compiled later: end-of-window report, next window's IC).

**`sing_bl_hard`, windows 0-5 (2026-10-30; hub, baseline settings, `--template bl --hard_ic`, ~2.7 h per
window, peak 10.47 GB).** The hub stopped during window 6; relaunched with the same command (no `rm`), it
resumed after window 5. The IC loss is 4.6e-34 throughout window 0 (exact); later windows fit the hand-off
softly (1.8e-5 to 2.6e-5). Core rms vs the 1D reference at the window ends:

| t | alpha | chi | Khat | gt |
|---|---|---|---|---|
| 0.25 | 4.8e-4 | 1.4e-4 | 2.7e-4 | 8.6e-4 |
| 0.5 | 1.8e-3 | 5.8e-4 | 2.4e-3 | 1.9e-3 |
| 1.0 | 2.1e-3 | 4.8e-4 | 7.8e-3 | 3.5e-3 |
| 1.5 | 4.6e-3 | 5.4e-4 | 8.7e-3 | 4.0e-3 |

Window 0 is 3-14x better than `sing_bl` / `sing_bl_pre` (those used ~3.5x fewer points and half the
iterations), but alpha and Khat grow from window 1 on; at t = 1.5 Khat equals the baseline's t = 2 value
and alpha is ~6x the baseline's. |a| reaches 7e-4 at t = 1.5. Mass (`plots/2026-10-30/sing_mass_*.png`,
`tools/plot_sing_mass.py` now includes this run and drops a resumed window's first attempt from the loss
history): exact at t = 0 (1e-15), -1.8e-3 at r = 1 by t = 0.25 (baseline -4.5e-5) and ~-1e-2 from t = 0.5,
the same level as the soft runs and ~10x the baseline; at r = 10 it drifts positive to +2.5e-2 by t = 1.5
(baseline -3.3e-3). Exact initial data remove the IC error but not the error the evolution adds: the mass is
lost in the first two windows, mostly at r ~ 1-10.

**Point density on spheres (2026-10-30, `tools/plot_point_density.py` -> `plots/2026-10-30/point_density.png`).**
Same totals (128000 interior, 12800 IC), different mixtures. Baseline: 60% of Nint in a s = 2 Gaussian + 40%
uniform, plus core_points 48000 at s = 0.5, nothing below r = 0.02 (min_puncture_dist). sing_pinn.py:
12.5% s = 0.05 (inner), 37.5% s = 0.5, 25% s = 2, 25% uniform. So sing_pinn.py moves 16000 points from the
s = 2 cloud to the puncture: ~12000 interior points at r < 0.1 (baseline ~100) and 45900 in 1 <= r <= 10
against 61400 (-25%), i.e. fewer exactly where sing_bl_hard loses its mass; identical for r < 1 above
0.2 and for r > 8. (The IC set has the same shapes at 1/10 the count.)
The plot also shows a candidate log-uniform sampler (uniform in ln r on 0.02 <= r <= 24, n ~ r^-3, ~18000
points per e-fold). Interior counts by band (r < 0.1 / 0.1-1 / 1-10 / 10-24): baseline 101 / 36833 / 61379 /
29685; sing_pinn.py 11918 / 40522 / 45873 / 29685; log-uniform 29053 / 41570 / 41570 / 15803. It puts 23% of
the points inside r = 0.1, which the template already covers, and has fewer in 1-10 than either run. Since
the loss is a mean over the points, the sampling density is also the weight of the residual: log-uniform
weights the residual squared by r^-3. Log-uniform on 0.2-24 instead would give ~43000 / 61600 / 23400 in
0.2-1 / 1-10 / 10-24, close to the baseline.

**`sing_bl_hard` finished (2026-10-30 18:46 UTC; two hub stops, resumed with the same command each time,
windows 6 and 7 restarted from their start). Final grades at t = 2, core rms vs the 1D reference:**

| run | alpha | chi | Khat | gt | alpha(0.03) (ref 0.2814) |
|---|---|---|---|---|---|
| baseline w8_qn_con10 | 7.1e-4 | - | 8.7e-3 | 1.8e-3 | - |
| sing_bl_hard | 7.6e-3 | 7.8e-4 | 1.1e-2 | 4.7e-3 | 0.2820 |
| sing_bl_pre | 1.3e-2 | 8.3e-4 | 1.9e-2 | 5.6e-3 | - |
| sing_bl | 2.0e-2 | 1.9e-3 | 3.3e-2 | 6.4e-3 | 0.2821 |

Mass <m/M - 1> at t = 2: baseline -3.6e-3 / -2.1e-3 / -3.9e-3 / -4.9e-3 at r = 0.3 / 1 / 3 / 10;
sing_bl_hard -1.1e-2 / -1.1e-2 / -3.6e-2 / +2.8e-2; sing_bl_pre -1.4e-2 / -1.6e-2 / +2.2e-3 / +7.9e-2.
Centre drift |a(2)| = 1.8e-3 (sing_bl_pre 2.7e-3, sing_bl 6.5e-3). Loss: window 6's end term jumped to
1.3e-3 from 2.4e-4 at the end of window 5 and stayed there in window 7 (held-out total 1.5e-3 -> 1.0e-3);
the restarts do not explain it (window 6 reached the same values in all three attempts).
Verdict: exact initial data are the best of the sing variants (alpha 1.7x, Khat 1.7x better than
sing_bl_pre at t = 2, mass 2-3x smaller at r <= 3), but the run is still 10x the baseline in alpha, 2.5x in
gt, 3-10x in mass; Khat is comparable. The IC is not the limiting error; the evolution in 1 <= r <= 10 is.
Plots: `plots/2026-10-30/sing_mass_*.png`, `sing_losses.png`, `sing_ic_error.png` (now with sing_bl_hard:
exactly 0 at t = 0 in every panel).

**Where the error is: the template seam (2026-10-30; `tools/plot_error_by_radius.py` ->
`plots/2026-10-30/error_by_radius.png`, alpha/Khat/chi rms vs r at t = 0.25, 1, 2, sing_bl_hard vs baseline).**
Inside r = 0.2 the hard run beats the baseline at every time (Khat at t = 2: 6.4e-3 vs 1.5e-2 in 0.05-0.2;
alpha at t = 1: 1.3e-3 vs 8.6e-3): the template does its job. The alpha and Khat errors peak in
r_c <= r <= 3 r_c = 0.2-0.6, where W falls from e^-1 to e^-81 and rho rises from 1/2 to 0.9: already at
t = 0.25 (alpha 4.3e-4 vs 1.1e-4 in 0.2-1, 4x the baseline), 1.1e-2 vs 5.4e-4 at t = 2 (20x). chi's error
peaks just outside, 0.5-3, and the 1-10 band (where the mass is lost) is 10-15x the baseline in all three
fields at t >= 1: the seam error propagates outward. So the point-count deficit in 1-10 and the lam_tapa
weight are not the mechanism; the sharp hand-over between the template and the network is. Candidates:
a larger r_c (0.5 or 1: the seam sits where the fields are smoother and slower), a gentler cutoff (r^2 in
the exponent, or a wider ratio R_TPL/r_c), or dropping W altogether beyond r_c so the template's far field
is kept and only the correction is learned.
`plots/2026-10-30/sing_sphere_errors_rt.png` (`tools/plot_sphere_errors.py`, now with Khat and chi rows)
shows the same in (t, r): the hard run's alpha and Khat error appears as a band at r = 0.2-0.6 in window 1
and darkens in place; chi's band sits at r = 1-3 and spreads outward; the baseline's error instead grows
from the puncture outward along a front r ~ t.

**Inner network at the seam, `--r_in` (2026-10-30).** The seam is a resolution problem: N takes x/R with
R = 24, so the shell 0.2-0.6 is 0.01-0.025 in input units. `--r_in R_in` adds a second, smaller network at
the seam's scale, N = N_out(t, x/24) + exp(-(r/R_in)^2) N_in(t, (x - a0)/R_in), with a0 the window-start
centre (constant in the window: no extra time derivative; r = |x - a0|) and N_in of `--inner_hidden` x
`--inner_layers` (32 x 4, ~4000 parameters). Template, rho, the hard IC and the IC/hand-off are untouched:
N_in's stacks go through _taylor_mlp with the input scaled by 24/R_in (chain rule), times the window
(product rule), and add to N_out's before rho. The two networks live at very different scales, so they
are trained in blocks rather than jointly (`--schedule`, default `inner:8000,outer:8000,inner:4000,outer:4000`
= 24000 iterations per window, replacing `--iters`): in a block only that network's parameters move (outer
= N_out and the centre; `joint` = everything) and the optimiser restarts as it does every `--n_change`
anyway; inner blocks draw their bulk interior/end/IC points inside 3 R_in (w = e^-9 there), the held-out
set is the full draw. Freezing is exact (the optimiser sees w[idx] only), done outside the jitted loss, so
no extra compilation. The history's `phase` column holds the block name. Precedent: Stationary's
`--stage_base` (a correction network trained on a frozen run). `--selftest --r_in 1.5 --hard_ic`: stacks vs
autodiff 1.8e-10, hard IC 8e-15 (random N_in output layer, so the product/chain rule is exercised).
Alternative not taken: moving the seam with a larger r_c.
CPU smoke run (2 windows, 1000 interior points, `inner:250,outer:250,joint:250`): 13362 parameters, IC
6.7e-34, hand-off and `--eval_spheres` fine. `--mem_report --r_in 1.5`: temp 22.8 GB vs 18.5 GB without the
inner network (+23%; the single-network run peaked at 10.45 GB on the A30, so expect ~13 GB). First hub run:
`sing_bl_inner` = `--template bl --hard_ic --r_in 1.5`, baseline settings, default schedule.

**`sing_bl_inner`, window 0 (2026-10-31; inner-first schedule): the outer blocks stalled.** Held-out loss
1.947e-4 after the first inner block, 1.942e-4 after 8000 outer iterations; 1.368e-4 -> 1.367e-4 in the
second pair (flat to three digits at every 250; only the boundary term moved, at 1e-8). N_out starts with
a zero output layer, so its hidden layers get no gradient until the output layer moves; with the residual
already small and small-scale after the inner block, the near-constant features (inputs x/24, |x|/24 < 0.2
inside r = 5) cannot reduce it and the bootstrap never happens. The single network escapes this only
because it starts against the large, smooth residual of the raw template. The default `--schedule` is now
`outer:8000,inner:8000,outer:4000,inner:4000` (outer first; in window 0 the first 8000 iterations then
replicate sing_bl_hard's, same seeds); the inner-first run is kept as `sing_bl_inner_ifirst`.
Its window 0 is nevertheless an "inner network only" result (13362 parameters, 4054 trained): at t = 0.25,
core rms alpha 5.9e-4, chi 7.7e-5, Khat 2.8e-4, gt 1.1e-3 (sing_bl_hard 4.8e-4 / 1.4e-4 / 2.7e-4 / 8.6e-4),
held-out 1.37e-4 (7.6e-5). By band (alpha / Khat / chi rms; hard -> inner): seam 0.2-1: 4.3e-4 -> 5.2e-4 /
3.4e-4 -> 3.3e-4 / 2.7e-4 -> 1.5e-4; 1-3: 9.9e-5 -> 1.0e-4 / 1.2e-4 -> 2.0e-4 / 1.5e-4 -> 8.6e-5; 3-10:
9.4e-5 -> 7.3e-5 / 6.4e-5 -> 1.1e-4 / 1.2e-4 -> 1.4e-5; mass at r = 1: -1.7e-3 -> -3.9e-4 (baseline -4.5e-5).
So a 1.5-scale network matches the 24-scale one at the seam and beats it in chi, the mass and the far
field, but does not resolve the alpha/Khat seam error: at x/1.5 the seam 0.2-0.6 is still only 0.13-0.4
input units. Next run: outer first and `--r_in 0.75` (seam at 0.27-0.8 input units; w = 0.53 at r = 0.6,
0.17 at r = 1, 0.018 at r = 1.5).

**`sing_bl_inner` relaunched outer-first with `--r_in 0.75` (2026-10-31 09:05 UTC).** Window 0: the outer
blocks now train (2.15e-4 at 8000, not bit-identical to sing_bl_hard's 1.76e-4: the extra zero terms change
the compiled arithmetic and 8000 quasi-Newton iterations amplify that; within the run-to-run spread), the
inner blocks continue the descent (1.44e-4 at 16000, 9.0e-5 at 24000; sing_bl_hard 7.6e-5). At t = 0.25
(alpha / Khat / chi rms by band, hard -> inner 0.75): seam 0.2-1: 4.3e-4 -> 4.8e-4 / 3.4e-4 -> 4.1e-4 /
2.7e-4 -> 1.9e-4; 1-3: 9.9e-5 -> 9.7e-5 / 1.2e-4 -> 1.6e-4 / 1.5e-4 -> 1.6e-4; mass at r = 1: -1.7e-3 ->
-8.0e-4. Neither the 1.5- nor the 0.75-scale inner network reduces the alpha/Khat seam error: it is not a
resolution problem.

**The seam error is a coherent bump the loss prefers (`tools/plot_seam_profile.py` ->
`plots/2026-10-31/seam_profile.png`: signed sphere-mean errors vs r with W and rho overlaid).** At t = 0.25
the alpha error of sing_bl_hard and sing_bl_inner coincide: one smooth positive bump, +1e-4 at r = 0.1
rising to +8e-4 at r = 0.5, back to 0 at r = 0.8. Two different networks giving the same error means the
loss, minimized to the same level, prefers this wrong solution; capacity is not the issue. The bump starts
inside the template region (r = 0.1-0.2, where W = 1 and the template alone is exact): there u = T(t) + rho N,
so rho N is wrong by +1e-4 to +5e-4 where the right value is 0. Khat's error at t = 0.25 is a +-1.5e-4
dipole across r = 0.15-0.4; by t = 0.5 both are 3e-3 bumps over 0.1-1 (alpha +, Khat + then -). The baseline's
errors have the opposite sign inside r = 0.2 (its puncture failure). Hypotheses that remain: the W T(t) +
(1 - W) U0 background's steep seam sources a residual that the network cancels only approximately, leaving
a PDE-consistent (small-residual) but wrong mode; or rho lets the network act too far in (rho(0.1) = 0.2).
Both are testable with window 0 alone (`--stop_after_window 0`, 2.7 h on the hub): a sweep in r_c (0.1,
0.5, 1.0; moves W and rho together) shows whether the bump moves with the seam. The BL template table
(`reference_1d/sing_bl_m0.5_dense.npz`) only reaches r = 1, so r_c <= 0.33 (R_TPL = 3 r_c) without
regenerating it: the sweep is r_c = 0.1 and 0.3 around the current 0.2, plus `--rho_power 4` at r_c = 0.2
(rho = r^p/(r^p + r_c^p); p = 4 keeps the correction out of the template region, rho(r_c/2) = 0.06 instead
of 0.2; selftest with p = 4, r_c = 0.3: stacks 1.5e-10, hard IC 7e-15). Three window-0 runs, ~2.8 h each:
`sing_w0_rc01`, `sing_w0_rc03`, `sing_w0_rho4` (`--template bl --hard_ic --stop_after_window 0` + the flag).

**Mid-window checkpoints (2026-10-31).** The hub container restarted three times in three days (window 6 of
sing_bl_hard twice, then the sweep's first run at iteration 10250 of 24000: 2.5 h lost each time, since
windows only saved at their end). `--ckpt_every N` (default 2000, ~15 min of work) writes `partial.pkl`
(parameters, iteration, RNG key, the held-out set's key) inside a window; `--resume` continues from it
(same held-out set, same key sequence; the `seconds` column restarts), removes it when the window is saved,
and drops a stale one. `--stop_after_window K` with `--resume` exits at once when window K is already saved,
so a sweep loop can be relaunched as is. Laptop test: killed after the first checkpoint, resumed at 250,
finished, third launch "already done".

**Window-0 seam sweep, first two runs (2026-11-01; `sing_w0_rc01` done 01:13 UTC, `sing_w0_rc03` 03:59; the
hub stopped again ~04:51 during `sing_w0_rho4`, checkpoint at 6000).** At t = 0.25, core rms alpha / chi /
Khat / gt and held-out loss: r_c = 0.1: 5.4e-4 / 1.5e-4 / 4.5e-4 / 1.7e-3, 4.1e-4; r_c = 0.2 (sing_bl_hard):
4.8e-4 / 1.4e-4 / 2.7e-4 / 8.6e-4, 7.6e-5; r_c = 0.3: 2.6e-4 / 1.3e-4 / 2.6e-4 / 4.7e-4, (not logged).
By band (alpha / Khat / chi rms, r_c 0.1 -> 0.2 -> 0.3; baseline): 0.05-0.2: 7.0e-4 -> 4.1e-4 -> 1.8e-4 (2.8e-3)
/ 5.8e-4 -> 1.7e-4 -> 1.8e-4 (2.9e-3) / 3.2e-5 -> 2.6e-5 -> 2.8e-5 (1.6e-5); 0.2-0.6: 3.5e-4 -> 5.1e-4 -> 3.1e-4
(1.3e-4) / 3.5e-4 -> 3.8e-4 -> 3.2e-4 (2.2e-4) / 2.7e-4 -> 2.4e-4 -> 2.1e-4 (4.3e-5); 0.6-1: 2.1e-4 -> 9.4e-5 ->
7.3e-5 (2.6e-5) / 2.4e-4 -> 1.9e-4 -> 1.9e-4 (5.5e-5) / 3.4e-4 -> 3.4e-4 -> 2.8e-4 (2.2e-5); 1-3: 1.8e-4 -> 9.9e-5
-> 5.0e-5 (3.6e-5) / 1.9e-4 -> 1.2e-4 -> 1.2e-4 (4.2e-5) / 1.8e-4 -> 1.5e-4 -> 1.5e-4 (7.4e-6); mass at r = 1:
-1.5e-3 -> -1.7e-3 -> -9.2e-4 (-4.5e-5). `plots/2026-11-01/seam_profile.png` (signed, W for the three r_c):
- alpha: the positive bump's inner edge moves with the seam (plateau from r = 0.1 at r_c = 0.1; rise at 0.2
  / 0.3 for r_c = 0.2 / 0.3) but its outer edge stays at r ~ 0.6-0.8 for all three; its height is 8e-4 at
  r_c = 0.1 and 0.2 and 4e-4 at 0.3. So the bump fills the region between the seam and r ~ 0.7 (where the
  lapse evolves fastest in window 0), and shrinks as the template covers more of it.
- Khat: the same +/- dipole for all three, shifted with r_c (sign change at r = 0.17 / 0.22 / 0.27), same
  amplitude 3e-4: it is a seam feature.
- chi: identical for all three (2-3e-4 bump over 0.3-2, baseline 5e-5): not a seam feature.
- r_c = 0.1 is worse everywhere (loss 5x) : the network has to do more of the fast region.
Conclusion so far: a larger template region helps monotonically (0.3 halves alpha's error and the mass
loss vs 0.2), which argues for regenerating the BL template table beyond r = 1 (it is a 1D evolution,
cheap) and trying r_c = 0.5 and 1; for two punctures the limit is their separation. chi's error (and the
mass lost at r ~ 1-3) needs a different explanation; `--rho_power 4` (pending) tests rho.

**1D twin, `sing_1d.py` (2026-11-01; the problem is radial, so explore it in 1D first).** The same ansatz
with a RADIAL network N(t, r) (inputs 2(t - t0)/(t1 - t0) - 1 and r/`--x_scale`, 24 as in 3D) on the 10
radial functions of sing_template.py (template T^W-blended, rho N correction, multiplicative chi/alpha,
`--hard_ic` with h = U0 - B(0) and s = t/t1 in window 0); the Cartesian fields on the x-axis are lift(f),
their derivative stacks come from autodiff (jvp in t, jacfwd in x, y, z: the cartoon reduction of
reference_1d/sph_reference.py), the residual is the main script's `_ccz4_residual_one`, and the loss has
sing_pinn.py's terms and weights (Sommerfeld on the radial functions at r = 24, Theta/Z constraint). Points
are the RADII of sing_pinn.py's clouds, so each shell is weighted as in 3D. Flags for the sweep: `--r_c`,
`--w_power` (W = exp(-(r/r_c)^p)), `--rho_power`, `--x_scale`, `--hard_ic`, `--template_file` (the r <= 3
table `reference_1d/sing_bl_m0.5_dense_r3.npz`, generated by jobs/hub/sing.sh when missing, allows r_c up to
1). Report per window end (`report.npz`): signed f - f_ref for the 10 radial functions on r in [0.03, 20]
against the fine reference `ref_single_m0.5_dr0.005_t20.npz` (which IS this problem's exact solution) and
the Misner-Sharp m/M - 1; `tools/plot_1d_profiles.py OUT.png NAME=RUN_DIR ...` draws them with W. Hard IC
check: max |u(0) - U0| = 1e-16. Laptop: 88 s per 250 iterations with 500 interior points; the smoke run
(500 iterations) already shows alpha's error peaking at the seam (1.4e-3 at r = 0.22). Hub:
`SCRIPT=sing_1d.py OUTDIR=1d_rc02 SING_ARGS="--hard_ic" bash jobs/hub/sing.sh`.

**1D twin of sing_bl_hard, window 0 (2026-11-01, laptop, 2000 interior points, 4000 iterations, ~1 h once the
CPU was free; `plots/2026-11-01/sing_1d_rc02.png`).** Held-out loss 2.2e-7 (3D: 7.6e-5, 350x); at t = 0.25,
rms vs the reference by band (alpha / Khat / chi): 0.05-0.2: 3.5e-5 / 2.7e-5 / 7.7e-6; 0.2-0.6: 1.0e-5 /
6.0e-5 / 2.3e-5; 0.6-1: 2.3e-6 / 2.2e-5 / 1.4e-5; 1-3: 6.4e-6 / 7.1e-6 / 5.3e-6 (3D sing_bl_hard: alpha 4.1e-4
/ 5.1e-4 / 9.4e-5 / 9.9e-5; the baseline 2.8e-3 / 1.3e-4 / 2.6e-5 / 3.6e-5). Largest signed errors: alpha
+8.6e-5 at r = 0.04, Khat +1.0e-4 at r = 0.41, chi +2.9e-5 at 0.30; mass -2.0e-4 / -6.8e-5 / -5.2e-5 / +1.6e-4
at r = 0.3 / 1 / 3 / 10 (3D: -1.7e-3 at r = 1). No seam bump: the same ansatz, cutoffs, weights and shell
weighting solve window 0 to 1e-5 in spherical symmetry, 50x better than the 3D run and 10x better than the
baseline in the seam band. So the 3D seam error is not a property of the template ansatz; it is the 3D run
failing to reach the loss level the ansatz allows (7.6e-5 vs 2.2e-7), and the seam is simply where the
sharpest radial structure the network must supply sits. What differs in 3D: the network is a 3D MLP that
has to represent a radial function (a feature at r = 0.3 is a sphere of radius 0.0125 in x/24 input units,
in every direction), the residual has angular parts that must also vanish, 128000 points instead of 2000.
The inner-network runs (same loss plateau ~9e-5 with 1.5x the parameters) say capacity alone is not it.
Next, to separate representation from optimisation in 3D: `--radial_net`, the 3D run with a network of
(t, |x - a0|) (one puncture), i.e. the 1D ansatz inside the 3D loss, points and optimiser.
Implemented (2026-11-01): `--radial_net`, a 2-input network of (t, r/24) with the same width/depth, its
stacks from autodiff in (t, r) per point and the chain rule (d_i r = n_i, d_i d_j r = (delta - n n)/r; rho ~
r^2 keeps rho N smooth at the puncture). Selftest: stacks vs autodiff 6e-12, hard IC 8e-15. Hub run
`sing_w0_radial` = `--template bl --hard_ic --radial_net --stop_after_window 0`: if its loss reaches the 1D
twin's level (1e-7) the 3D MLP's representation of radial functions is the obstacle; if it stays at 1e-4,
the 3D loss/sampling/optimisation is.

**1D sweep on the hub (2026-11-01, 9 min per run on the A30, two windows to t = 0.5, 2000 interior points,
4000 iterations per window; `plots/2026-11-01/sing_1d_sweep.png`).** rms vs the reference at t = 0.5 by
band 0.05-0.2 / 0.2-0.6 / 0.6-1 / 1-3 / 3-10 (alpha | Khat), mass at r = 1, final held-out loss:

| run | alpha | Khat | mass | loss |
|---|---|---|---|---|
| r_c 0.1 | 3e-4 1e-4 5e-5 5e-5 1e-5 | 1e-3 2e-4 8e-5 2e-5 3e-5 | -5.7e-4 | 7.5e-7 |
| r_c 0.2 (twin) | 9e-5 1e-4 1e-5 8e-6 2e-5 | 3e-4 4e-4 5e-5 4e-5 1e-5 | -3.1e-4 | 2.9e-7 |
| r_c 0.3 | 4e-5 4e-5 1e-5 3e-6 2e-6 | 8e-5 1e-4 1e-5 4e-6 7e-6 | -8.9e-5 | 1.8e-7 |
| r_c 0.5 | 4e-5 2e-5 3e-6 7e-6 6e-7 | 2e-5 4e-5 2e-5 2e-6 1e-6 | -6.0e-5 | 1.5e-7 |
| r_c 1.0 | 3e-5 4e-6 5e-6 5e-6 3e-6 | 4e-6 9e-6 1e-5 6e-6 6e-6 | -3.5e-5 | 1.5e-7 |
| rho_power 4 | 7e-5 2e-5 5e-6 2e-6 5e-6 | 6e-5 2e-4 8e-6 1e-5 7e-6 | +1.9e-5 | 1.9e-7 |
| w_power 2 | 1e-4 4e-5 1e-5 4e-6 1e-5 | 1e-4 7e-5 2e-5 7e-6 8e-6 | -4.4e-5 | 1.8e-7 |
| x_scale 1 | 4e-5 3e-5 7e-6 6e-6 5e-6 | 2e-4 1e-4 8e-6 2e-5 4e-6 | -1.3e-4 | 1.8e-7 |
| soft IC | 1e-4 1e-4 6e-6 2e-5 4e-5 | 2e-4 3e-4 4e-5 4e-5 2e-5 | -2.6e-6 | 2.7e-7 |

In 1D everything is at 1e-5 to 1e-4 and the second window roughly triples the twin's window-0 error
(alpha 2.5e-5 -> 6.4e-5, mass -6.8e-5 -> -3.1e-4): the hand-off error grows here too, just 30x lower.
Larger r_c is monotonically better (1.0: 10-40x better than 0.2 in the seam band and the mass, with the
r <= 3 table), a gentler W (p = 2) helps Khat and the mass 3-7x, rho^4 helps the mass but not Khat in the
seam, the input scale hardly matters, and the hard IC buys 2x in alpha/Khat at t = 0.5 but nothing in the
mass. Laptop window-0 runs agree (r_c 0.5: alpha 1.9e-5, Khat 4.1e-6, mass +2.0e-5; r_c 1.0: Khat 2.2e-6;
soft: alpha 4.0e-5, mass +9.0e-5).

**`sing_w0_radial` (3D code, radial network; hub, 13:34 UTC) stalls at 3.6e-3** from iteration 1000 on
(3.5e-2 at 250, 8.6e-3 at 500, 3.9e-3 at 1000, flat to 22500; 815 s for the first 250 iterations, 3x the
MLP's: the per-point jacfwd stacks), the loss spread over all radii (d0.05-0.5 1.0e-3, d>0.5 8e-4), while
the same network, shell weighting, weights and optimiser give 1.5e-6 at 1000 iterations in sing_1d.py, and
the laptop smoke run (2000 points) was already 300x behind the 1D twin at 500 iterations. So it is not the
point count; something in the 3D path blocks the radial network. Tests: the trained centre frozen
(`--fixed_center`; the one freedom the 1D twin lacks) and no inner cloud (`--inner_frac 0`; the 1/r term of
the chain rule).
Both still stall (1.2e-2 / 6.7e-3 at 750 iterations). The cause was the ansatz: that `--radial_net` gave
each of the 22 CARTESIAN components as a function of r, which is a spherically symmetric field only for the
scalars; gt, At, Ghat, beta must be a delta + b n n, g n, B n (the 1D twin outputs the 10 radial functions
and lifts them), so the solution was outside the ansatz and the loss had a floor. Fixed: the radial network
outputs the 10 radial functions, `radial_point` lifts them (sing_template.lift), the stacks are autodiff of
the lifted function (as sing_1d.py); selftest 6e-12. Laptop, 2000 points: held-out 7.7e-3 / 4.9e-5 /
1.3e-5 at 250 / 500 / 750 iterations, the 1D twin's track (9.3e-3 / 5.9e-5), against 1.9e-2 / 1.5e-2 /
1.2e-2 for the Cartesian version. So the 3D loss, points and optimiser are not the obstacle; the 3D MLP's
representation of a radial tensor field is. The hub run with the wrong ansatz is kept as
`sing_w0_radial_cart`; `sing_w0_radial` is relaunched with the lifted network.

**`sing_w0_radial`, lifted radial network in the 3D code (hub, 2026-11-01 18:02 UTC; 0.5 s per iteration,
peak 15 GB).** Held-out 8.8e-7 / 5.6e-8 / 4.7e-8 / 4.7e-8 at 1000 / 4000 / 8000 / 10250 iterations
(sing_bl_hard: 9.9e-3 / 3.8e-4 / 1.8e-4 / 1.6e-4; the 1D twin 1.5e-6 at 1000), evenly spread over all
distance bands (1e-9 to 1e-8), the floor from ~4000 on presumably the template table's own accuracy.
Graded from the checkpoint at 10250 (`--eval_spheres` on partial.pkl), t = 0.25, rms by band 0.05-0.2 /
0.2-0.6 / 0.6-1 / 1-3 / 3-10: alpha 3e-5 / 7e-6 / 1e-6 / 2e-6 / 1e-6 (sing_bl_hard 4e-4 / 5e-4 / 9e-5 / 1e-4 /
9e-5; baseline 3e-3 / 1e-4 / 3e-5 / 4e-5 / 1e-4), Khat 1e-5 / 4e-5 / 9e-6 / 3e-6 / 2e-6, chi 6e-6 / 2e-6 /
3e-6 / 5e-7 / 3e-6; mass -1.1e-5 / -4.5e-5 / -1.1e-5 / +9.8e-5 at r = 0.3 / 1 / 3 / 10 (baseline -3.6e-4 /
-4.5e-5 / -2.6e-5 / -9.2e-5; sing_bl_hard -1.2e-3 / -1.7e-3 / +4.6e-4 / +4.3e-5). Conclusion: with the
symmetry built into the network the 3D code reaches the 1D level; the 3D MLP's representation of a
spherically symmetric tensor field was the whole seam problem. For two punctures: one lifted radial
network per puncture about a_A(t) plus a small 3D MLP for the non-spherical remainder.
Final window 0 (24000 iterations, 3.2 h; held-out 4.2e-8, flat since 4000): core rms alpha 2.2e-5, chi
4.9e-6, Khat 1.4e-5, gt 8.3e-6 (sing_bl_hard 4.8e-4 / 1.4e-4 / 2.7e-4 / 8.6e-4); by band alpha 3e-5 / 6e-6 /
8e-7 / 1e-6 / 9e-7, Khat 7e-6 / 2e-5 / 4e-6 / 4e-6 / 3e-6, chi 6e-6 / 2e-6 / 2e-6 / 7e-7 / 1e-6; mass -1.6e-5 /
-2.9e-5 / -2.3e-5 / +9.9e-5 at r = 0.3 / 1 / 3 / 10. Added to `plots/2026-11-01/seam_profile.png`.

**Early stopping (2026-11-01).** The loss reaches its floor long before 24000 iterations (the radial run
by 4000). `--stop_floor F --stop_tol 1.05 --stop_patience N`: a window ends once the held-out loss has
stayed within 1.05 F for N iterations; `--stop_rel X --stop_patience N`: once the best held-out loss of the
last N iterations is less than X below the best of the N before (no floor needed; later windows have
higher floors; the min-vs-min form is robust to the 2x check-to-check noise near the floor seen in the
laptop test). Both check every --n_change, the pre/IC phase excluded. Laptop test (2 windows, plateau
rule): both windows ended early, the run continued and saved. Tonight's run: `sing_radial`, 8 windows to t = 2 with the
lifted radial network, `--stop_rel 0.02 --stop_patience 1000`, continuing from sing_w0_radial's window 0.

**No template, `--no_template` (2026-11-01; both scripts).** W = 0 and rho = 1: u = flat + N, with the hard-IC
offset h so chi = psi^-4 exp(N_chi) and alpha = psi^-1 exp(N_alpha) carry the exact data's puncture structure
and the network supplies the departure from the initial data everywhere, puncture included (`--hard_ic`
required). The question (the author's): is the template redundant once the network is radial? Tests: the
1D twin on the laptop (`notpl`, window 0) and `sing_w0_radial_notpl` on the hub after sing_radial
(`--template bl --hard_ic --radial_net --no_template --stop_after_window 0 --stop_rel 0.02
--stop_patience 1000`), against sing_w0_radial (held-out 4.2e-8, alpha 6e-6 in the seam band). Selftest with
`--no_template --radial_net`: stacks 1.8e-12, hard IC 6e-14. Caveat built into the ansatz: with h fixed, chi's
zero at the puncture stays r^4 (the BL form) times a smooth factor; the 1+log evolution turns it into r^2
(trumpet), which the template carried and this ansatz cannot. The chi floor (1e-5, r < 0.015 at t = 0)
masks the innermost region in the loss.
1D result (laptop, window 0, `plots/2026-11-02/sing_1d_notemplate.png`): the radial network alone reaches
held-out 5.8e-8 at 4000 iterations (the template run 2.2e-7) and the same errors: by band 0.03-0.1 / 0.1-0.2
/ 0.2-0.6 / 0.6-1 / 1-3 / 3-10, alpha 4e-5 / 2e-5 / 2e-5 / 3e-6 / 5e-6 / 1e-5 (template run 5e-5 / 3e-5 / 1e-5 /
2e-6 / 6e-6 / 9e-6), Khat 2e-5 / 2e-5 / 2e-5 / 9e-6 / 4e-6 / 1e-5 (6e-6 / 4e-5 / 6e-5 / 2e-5 / 7e-6 / 8e-6),
chi 8e-6 / 2e-5 / 1e-5 / 1e-5 / 3e-5 / 2e-5 (5e-6 / 1e-5 / 2e-5 / 1e-5 / 5e-6 / 7e-6); mass +2e-4 / -2e-4 / +4e-4
at r = 0.3 / 1 / 3 (template run -2e-4 / -7e-5 / -5e-5). So in window 0 the template is not needed once the
network is radial and the exact data enter as the offset: same accuracy in the fields, the mass 3-8x worse
in 1-3. The puncture zero (r^4 vs r^2) did not bite in window 0 (0.03-0.1: alpha 4e-5 vs 5e-5); whether it
does by t = 2 is for the multi-window comparison.

**`sing_radial`, the lifted radial network over 8 windows to t = 2 (hub, 2026-11-01 21:42 - 2026-11-02 00:50
UTC; windows 1-7 with the plateau early stop, 2000-6000 iterations each, 15-50 min).** Core rms vs the 1D
reference at the window ends (alpha / chi / Khat / gt): t = 0.25: 2.2e-5 / 4.9e-6 / 1.4e-5 / 8.3e-6; 0.5:
2.6e-5 / 8.3e-6 / 4.8e-5 / 2.3e-5; 0.75: 5.3e-5 / 1.2e-5 / 1.7e-4 / 3.6e-5; 1: 1.0e-4 / 1.0e-5 / 3.2e-4 / 4.1e-5;
1.25: 2.0e-4 / 6.1e-6 / 4.7e-4 / 4.4e-5; 1.5: 2.5e-4 / 4.2e-5 / 6.8e-4 / 9.9e-5; 1.75: 2.2e-4 / 4.4e-5 / 1.1e-3 /
1.6e-4; 2: 6.3e-4 / 2.8e-4 / 1.5e-3 / 4.4e-4 (baseline at t = 2: 7.1e-4 / - / 8.7e-3 / 1.8e-3; sing_bl_hard
7.6e-3 / 7.8e-4 / 1.1e-2 / 4.7e-3). Held-out loss at the window ends: 4.2e-8, 3.1e-7, 2.2e-7, 6.5e-7, 1.1e-6,
4.1e-6, 4.6e-5, 4.0e-5: a 10x jump into window 6 (t = 1.5-1.75; the Einstein and end terms both), where
sing_bl_hard's end term also jumped; windows 6 and 7 are still 2x below the 3D MLP's window 0. Mass
<m/M - 1> at r = 0.3 / 1 / 3 / 10: 1e-4 or less through t = 1.5 (e.g. t = 1: -1.4e-4 / +9e-6 / -3.5e-4 /
+1.6e-4; baseline -5.3e-4 / -3.3e-4 / -6.4e-4 / -1.3e-3), +6.9e-4 / +4.6e-4 / -2.0e-3 / +9.4e-4 at 1.75, then
+3.6e-3 / +1.4e-3 / +3.2e-3 / +2.1e-3 at t = 2 (baseline -3.6e-3 / -2.1e-3 / -3.9e-3 / -4.9e-3; sing_bl_hard
-1.1e-2 / -1.1e-2 / -3.6e-2 / +2.8e-2). By band at t = 2 (0.05-0.2 / 0.2-1 / 1-3 / 3-10): alpha 5e-4 / 6e-4 /
3e-4 / 9e-5 (baseline 1e-3 / 5e-4 / 3e-4 / 8e-4), Khat 2e-3 / 4e-4 / 5e-4 / 2e-4 (1e-2 / 2e-3 / 4e-4 / 2e-4), chi
1e-4 / 3e-4 / 1e-4 / 6e-5 (1e-4 / 4e-4 / 3e-4 / 3e-4). Plots: `plots/2026-11-02/sing_mass_time.png`,
`sing_mass_profiles.png`, `sing_losses.png`, `error_by_radius.png`, `sing_sphere_errors_rt.png`.
Reading: through t = 1.5 the radial run is 3-10x better than the baseline everywhere and 10-100x better
than the 3D MLP; at t = 2 it is comparable to the baseline in the seam band and the mass (opposite sign),
and still 5x better in Khat inside r = 0.2 and 4-9x in alpha/chi beyond r = 3. The error grows ~2x per
window from a 1e-5 start (the hand-off), with a step at window 6 that needs understanding (the 2%
plateau rule ended window 6 after the minimum 2000 iterations at 4.6e-5: maybe stopped too soon, maybe
a real change at t ~ 1.5 - the lapse front, the trumpet transition at the puncture that the BL offset
cannot follow). Total GPU time for windows 1-7: 3 h.

**`sing_radial_w67` (windows 6-7 rerun from window_005.pkl with 8000 fixed iterations; 2026-11-02).**
Window 6 ends at held-out 4.7e-5 (best 4.1e-5) after 8000 iterations, the early-stopped one at 4.6e-5
after 2000: the 10x step into window 6 is a real floor, not the plateau rule. The field errors at the same
loss differ, though: the 8000-iteration window 6 has core rms alpha 1.8e-3, chi 3.4e-4, Khat 9.8e-4, gt
5.7e-4 against 2.2e-4 / 4.4e-5 / 1.1e-3 / 1.6e-4 for the 2000-iteration one, i.e. at this loss level the
solution is no longer pinned down by the loss (a 4.6e-5 floor admits alpha errors from 2e-4 to 2e-3).
What raises the floor at t >= 1.5 is open; the no-template run does NOT show the step (its floors grow
smoothly, see below), so it is tied to the template (the table's accuracy or the W T(t) + (1 - W) U0
seam at late times) rather than to the physics; the 3D MLP's end term jumped there too.

**`sing_radial_notpl`, the radial network alone over 8 windows (hub, 2026-11-02 07:46-21:05 UTC with two
hub stops; `plots/2026-11-03/sing_mass_time.png`, `sing_sphere_errors_rt.png`, `error_by_radius.png`).**
Held-out loss at the window ends (iterations : loss), windows 0-7: 11250 : 1.2e-8, 10750 : 2.6e-8,
5500 : 9.7e-8, 7500 : 5.7e-8, 3250 : 2.2e-7, 4750 : 2.3e-7, 2750 : 1.0e-6, 2250 : 8.0e-6 (sing_radial:
24000 : 4.2e-8, 6000 : 3.1e-7, 2500 : 2.2e-7, 3000 : 6.5e-7, 3500 : 1.1e-6, 2250 : 4.1e-6, 2000 : 4.6e-5,
3250 : 4.0e-5): without the template the loss is 2-5x LOWER in every window and grows smoothly (~3x per
window) with no step at window 6, while the field errors near the puncture are 2-3x larger: the loss
floor and the error are not tied one to one at this level. (The windows 6-7 rerun finished 2026-11-02
23:19 UTC: at t = 2 with 8000 fixed iterations per window, core rms alpha 5.0e-4, chi 1.7e-4, Khat 1.1e-3, gt
3.1e-4 against the early-stopped 6.3e-4 / 2.8e-4 / 1.5e-3 / 4.4e-4: 4x the iterations buy 1.3-1.6x.)
**To T = 8:** `sing_radial_notpl` continues from window 7 with `--tfinal 8 --N_window 32` (same 0.25
windows, resumed; launched 2026-11-02 23:19 UTC, ~1 h per window), to see whether the radial ansatz keeps
its advantage once the puncture is a trumpet and whether the far-field mass drift saturates; the
baseline's sphere file for 32 windows is `eval_data/long_con10_w32.npz`. At t = 4 (window 15, reached
2026-11-03 03:51 UTC, ~35 min per window): core rms alpha 1.7e-3, chi 5.1e-4, Khat 6.2e-4, gt 1.2e-4 (Khat
peaked at 4.3e-3 at t = 2.25 and fell as the solution settles; alpha grows slowly, 3e-4 -> 1.7e-3 over
t = 2-4); by band 0.05-0.2 / 0.2-1 / 1-3 / 3-10 (alone | baseline): alpha 7e-4 / 3e-3 / 2e-3 / 6e-4 | 2e-3 /
4e-4 / 8e-4 / 2e-3; Khat 7e-4 / 6e-4 / 2e-4 / 3e-4 | 9e-3 / 2e-3 / 1e-3 / 4e-4; chi 3e-4 / 8e-4 / 4e-4 / 4e-4 |
2e-4 / 4e-4 / 4e-4 / 8e-4. Mass at r = 0.3 / 1 / 3 / 10: t = 2.5: +5.8e-3 / +4.6e-3 / -3.6e-4 / +4.3e-3 |
-2.5e-3 / -2.1e-3 / -6.1e-3 / -8.5e-3; t = 3: +4.4e-3 / +4.8e-3 / +2.1e-3 / +4.5e-3 | +7e-5 / +8e-5 / -5.6e-3 /
-1.0e-2; t = 4: +4.8e-3 / +4.6e-3 / +6.1e-3 / -5.6e-3 | +3.6e-3 / +3.4e-3 / -1.1e-2 / -1.9e-2; min over r at
t = 4: -1.5e-2 | -3.9e-2. So by t = 3-4 the radial ansatz without a template is at parity with the
baseline overall: still 3-10x better in Khat everywhere and 2-3x in the mass beyond r = 3, but 2-5x
worse in alpha in 0.2-3; the 10x advantage of t <= 1.5 is gone by t ~ 3 (a step of +5e-3 in the mass
inside r = 1 between t = 2 and 2.5).
**T = 8 reached (2026-11-03 12:54 UTC; 32 windows; `plots/2026-11-03/sing_T8_summary.png`,
`sing_sphere_errors_rt_T8.png`).** Core rms vs the 1D reference saturates from t ~ 4: alpha 6e-4 to 2.0e-3
(slowly rising again from t = 6), chi 6.4-7.1e-4, Khat 1.1-1.9e-3, gt 0.8-1.7e-4; alpha(0.03) tracks the
reference to 1e-4 throughout (0.0236 at t = 8). By band 0.05-0.2 / 0.2-1 / 1-3 / 3-10 (alone | baseline) at
t = 6: alpha 3e-4 / 1e-3 / 2e-3 / 1e-3 | 3e-3 / 2e-3 / 2e-3 / 3e-3, Khat 2e-3 / 1e-3 / 6e-4 / 2e-4 | 9e-3 / 3e-3 /
6e-4 / 5e-4, chi 3e-4 / 1e-3 / 9e-4 / 8e-4 | 8e-4 / 1e-3 / 1e-3 / 1e-3; at t = 8: alpha 1e-3 / 3e-3 / 3e-3 / 1e-3 |
4e-3 / 2e-3 / 3e-3 / 4e-3, Khat 1e-3 / 2e-3 / 6e-4 / 2e-4 | 8e-3 / 4e-3 / 1e-3 / 8e-4, chi 3e-4 / 1e-3 / 9e-4 /
1e-3 | 1e-3 / 2e-3 / 2e-3 / 2e-3. Mass at r = 0.3 / 1 / 3 / 10 (min over r): t = 5: +5.6e-3 / +5.8e-3 / +6.0e-3 /
-2.0e-2 (-2.3e-2) | +2.6e-3 / +2.6e-3 / -2.3e-3 / -3.3e-2 (-6.1e-2); t = 6: +5.8e-3 / +6.4e-3 / +7.1e-3 /
-3.1e-2 (-3.4e-2) | +1.9e-3 / +2.8e-3 / -1.7e-4 / -5.9e-2 (-8.1e-2); t = 8: +6.0e-3 / +6.9e-3 / +7.1e-3 /
-6.9e-2 (-8.1e-2) | +1.4e-3 / +2.7e-3 / +2.5e-3 / -1.4e-1 (-1.4e-1). Reading at late times: the fields are
at the baseline's level or 2-5x better (Khat 4-8x better inside r = 1, where the baseline falls apart as
the trumpet forms; alpha 1.5x worse in 0.2-1 at t = 8); the mass inside r = 3 sits at a constant +6e-3
offset from t = 2.5 on (the baseline +1.5-3e-3), and the far-field mass drifts negative in BOTH runs,
the radial one half as fast (-7e-2 vs -1.4e-1 at r = 10 by t = 8; the far field has only the network and
the Sommerfeld condition, and the baseline's drift shows the problem is the far field itself, not the
ansatz). Total GPU time for windows 8-31: 13.5 h (35 min per window with the plateau rule).
**The 1D twin to T = 8, `1d_T8` (hub, 2026-11-03 16:11-18:02 UTC, 3.5 min per window; sing_1d.py `--hard_ic --no_template --N_window 32
--tfinal 8`, 4000 iterations per window; `plots/2026-11-03/T8_compare.png`, `tools/plot_T8_compare.py`).**
Same ansatz, scheme, loss and shell weighting as sing_radial_notpl, in spherical symmetry. Band rms
0.05-0.2 / 0.2-1 / 1-3 / 3-10 (1D | 3D radial | baseline) at t = 4: alpha 3e-4 / 3e-4 / 6e-5 / 3e-4 | 7e-4 /
3e-3 / 2e-3 / 6e-4 | 2e-3 / 4e-4 / 8e-4 / 2e-3, chi 7e-5 / 2e-4 / 9e-5 / 1e-4 | 3e-4 / 8e-4 / 4e-4 / 4e-4 | 2e-4 /
4e-4 / 4e-4 / 8e-4; at t = 8: alpha 2e-4 / 5e-4 / 3e-4 / 4e-4 | 1e-3 / 3e-3 / 3e-3 / 1e-3 | 4e-3 / 2e-3 / 3e-3 /
4e-3, Khat 7e-4 / 3e-4 / 2e-4 / 2e-4 | 1e-3 / 2e-3 / 6e-4 / 2e-4 | 8e-3 / 4e-3 / 1e-3 / 8e-4, chi 1e-4 / 5e-4 /
2e-4 / 3e-4 | 3e-4 / 1e-3 / 9e-4 / 1e-3 | 1e-3 / 2e-3 / 2e-3 / 2e-3. Mass at r = 0.3 / 1 / 3 / 10 at t = 8:
+2.5e-3 / +2.7e-3 / +1.9e-3 / +2.5e-2 | +6.0e-3 / +6.9e-3 / +7.1e-3 / -6.9e-2 | +1.4e-3 / +2.7e-3 / +2.5e-3 / -1.4e-1.
Readings: (1) the windowed scheme itself has a ceiling: the 1D twin, with losses of 1e-8 to 1e-6, also
stops improving and saturates at 1e-4 to 6e-4 in the fields from t ~ 2; the 3D radial run sits 3-10x
above it in alpha and chi in 0.2-3 at the same loss (realisation-to-realisation spread at fixed loss, seen
already in the windows 6-7 rerun, and/or 3D effects), the baseline 5-10x above. (2) The far-field mass
drifts in all three runs, the 1D one included (+2.5e-2 at r = 10 by t = 8, after a sign change at t ~ 3.3;
the 3D radial run changes sign at t ~ 3.6): the drift belongs to the scheme (windows, boundary, far-field
sampling), not to the ansatz or to 3D. The mass at radius r magnifies the field-level error by r / 2M
(10x at r = 10): 2M |m/M - 1| / r is 1e-3 to 1e-2 there in all runs. (3) The 1D held-out loss spikes at a
few window ends (5 at t = 2.5, 0.1 at t = 4.25, 1e-4 at t = 4.75 and 7.25) without any trace in the band
errors (r >= 0.05): outliers among the held-out points of the sigma = 0.05 inner cloud, r << 0.05; the
plateau rule keeps the LAST state of a window, so a window can end on such an excursion (to fix: keep the
best held-out state).
**Next: a parallel 1D batch to T = 8 (2026-11-03), one change each against 1d_T8**, five runs at once on the
A30 (XLA_PYTHON_CLIENT_MEM_FRACTION 0.17 each; a 1D run has ~2700 points per draw): `1d_T8_seed8` (`--seed 8`:
the spread between realisations at fixed loss), `1d_T8_ic100` (`-- --lamg0 100`: the hand-off weight, since
each window starts from the previous one's end as soft data), `1d_T8_R48` (`-- --R_domain 48`: the outer
boundary twice as far; the reference is free-space and exact to r = 30 for t <= 20), `1d_T8_far`
(`--focus_sigma 5`: the 25% focus cloud at sigma 5 instead of 2, more points in r = 3-10), `1d_T8_w05`
(`--N_window 16 --iters 8000`: windows of 0.5, half the hand-offs, the same iteration budget). All five
flags checked with zero-iteration runs on the laptop.
**Derivative (Sobolev) hand-off, `--ic_d1 W1 --ic_d2 W2` (sing_1d.py, 2026-11-03; the author's idea).** The
IC term of a window matches not only the previous window's end values but also their radial derivatives:
lamg0 [mean|f - U|^2 + W1 mean|r f' - r U'|^2 + W2 mean|r^2 f'' - r^2 U''|^2] on the IC points, the targets
(U, r U', r^2 U'') computed by nested jvp of the frozen previous window (of U0 in window 0, where the hard IC
makes them exact anyway). r d/dr is scale-free: near the puncture it tames chi's and alpha's steep slopes,
far out it decays like the field, so W = 1 weighs the derivative mismatch like the value mismatch. Why it
should help: the soft hand-off fixes the values at t0 but leaves the slopes free, and the next window's
residual is driven by the slopes (the spatial derivatives in the RHS); a slope error at t0 is a source of
error for the whole window. Laptop smoke test (2 windows, 500 points, both orders on): the window-1 IC term
(values + both orders) goes 1.4e-5 -> 4.3e-6 in 500 iterations, same cost per iteration. Added to the
parallel batch: `1d_T8_icd1` (`--ic_d1 1`) and `1d_T8_icd12` (`--ic_d1 1 --ic_d2 1`); if it helps in 1D it
goes into sing_pinn.py (the 3D IC points then need the d_xyz stacks of the previous window, as u_derivs gives).
**The colleagues' 3 Nov update (`docs/BH_PINN_progress_2026-11-03.pdf`, J. Masso).** New reference run to
t = 10: 6x80 network, 8k core points, Theta/Z penalty, soft excision (At, Khat, Theta, Ghat residuals x chi),
r^2-weighted Hamiltonian/momentum penalty (lambda = 30), 12k QN steps per window. Hole mass +0.03 %, mass at r =
12 / 15 / 23.5 -0.35 / -0.72 / -1.38 %, chi and alpha at r = 1 4e-4, but the lapse peak at t ~ 2 is 20 % low
(0.067; they suspect the reduced core cloud). Against ours (t = 8): their far-field mass is 10-20x better
than the 3D radial run's (-6.9 % at r = 10) and their r = 1 errors match the 1D twin's; our runs track the
lapse at r = 0.03 to <= 1e-3 throughout (peak 4e-4), with ~12000 points inside r = 0.1. Their error bound
is two thirds hand-over jumps. Their remaining far-field deficit is a one-signed <H> ~ -1e-4 at r = 8-14;
the finished 1d_T8 run shows the same (sphere-mean H -4e-5 to -6e-5 at r = 8-14 for t >= 4).
**The ADM constraint penalty, division-free (2026-11-03).** Our runs had only the Theta/Z penalty
(--w_constraint, the CCZ4 constraint VARIABLES); the true H and M_i penalty existed only in the main script
(--w_adm, chi^p-weighted, f8016b1; off in the baseline) and was never in sing_pinn.py / sing_1d.py. H
carries 1/chi factors (R = chi R~ + 2 D~^2 chi - (5/2) |d chi|^2 / chi), so near the puncture it is
O(1/chi) noise: in 1d_T8, H ~ -0.1 to -0.2 and M_r ~ 0.5 at r < 0.05, and an r^2-weighted H^2 penalty would
spend most of its weight there. Following the author's suggestion, `adm_constraints.conformal_constraints_one`
computes chi H and chi M_i with every 1/chi cancelled by hand:
chi H = chi^2 R~ + 2 chi D~^2 chi - (5/2) gt^ij d_i chi d_j chi + chi [(tr A)^2 - A_ij A^ij],
chi M_i = chi (gt^jk D~_j A_ki - d_i tr A) + (1/2) (d_i chi tr A - 3 A_il gt^lm d_m chi),
A_ij = At_ij + gt_ij K / 3, K = Khat + 2 Theta, indices with gt^-1, D~ and R~ of gt: polynomial in chi, no
division. `tools/test_adm_constraints.py` (check 4): equal to chi times the direct H, M to 3e-16 on generic
fields (At not trace-free, det gt != 1, Theta != 0); zero to round-off on exact Schwarzschild (isotropic, Kerr-
Schild) down to r = 1e-3. sing_1d.py: `--w_adm W --adm_r_power p` adds W mean(r^p [(chi H)^2 + |chi M|^2]) on
the interior points; `--eval_constraints` writes H, M_r, chi H, chi M_r and the density per window end to
constraints.npz. On the trained 1d_T8 windows under the run's own point distribution, p = 0 puts 65-80 % of
the penalty inside r = 1, p = 2 puts 85-95 % at r = 1-24 (mostly 3-8, and 10-35 % at 8-24 at late times),
where the deficit is: p = 2 is the choice (the colleagues' r^2, without the 1/chi noise). Laptop smoke
test (W = 30): same cost per iteration; the adm part falls 4.7e-5 -> 3.0e-5 in window 0. Hub runs:
`1d_T8_adm30`, `1d_T8_adm100` (their lambda and their next value), one change each against 1d_T8.
**The first parallel 1D batch ran out of memory (2026-11-03 18:44 UTC).** All seven runs, capped at
XLA_PYTHON_CLIENT_MEM_FRACTION 0.12 (2.9 GB) each, died in their first iterations allocating 581 MiB: that is
8730^2 x 8 bytes, the SSBroyden dense inverse Hessian, kept in several copies. A 1D run is small in points
but needs a few GB because of the optimiser. `jobs/hub/sing1d_batch.sh` now runs NPAR at a time (default 3,
MEM_FRACTION 0.9/NPAR each), starting the next as one ends, resumable item by item; sing_1d.py prints the
peak device memory on each progress line (as sing_pinn.py) so NPAR can be raised if there is room.
Peak per 1D run: 3.13 GB. A second batch launched while the first ran (my suggestion) failed at start-up:
each run's memory pool grows to its cap and keeps it (reserved > peak in use), so the first three at 0.30
held most of the card. Rule: never two batches at once; NPAR within one batch sets the caps. Throughput (2026-11-04):
four 1D runs at once take 880 s per window each, one alone 210 s: one run saturates the A30 (the dense
SSBroyden matrix), parallel runs gain nothing; run them in priority order, NPAR 1-2.
**First results, 1D to t ~ 7 (2026-11-04; the hub stopped at 04:42 UTC with adm30 at 28 / 32 windows,
adm100 and icd1 at 27, w05 at 13 / 16; `plots/2026-11-04/sing_1d_adm.png`, `tools/plot_1d_runs_t.py`).**
At t = 6.75, band rms 0.05-0.2 / 0.2-1 / 1-3 / 3-10 for alpha | chi | Khat, and m/M - 1 at r = 1 / 3 / 10:
1d_T8 (control) 2e-4 6e-4 2e-4 2e-4 | 1e-4 4e-4 2e-4 3e-4 | 8e-4 3e-4 2e-4 2e-4 | +1.9e-3 +2.2e-3 +2.1e-2;
adm30 3e-4 7e-4 7e-4 7e-4 | 4e-5 1e-4 3e-4 3e-4 | 1e-3 5e-4 2e-4 3e-4 | +3.3e-4 +2.7e-4 -2.0e-4;
adm100 1e-4 3e-4 4e-4 2e-3 | 1e-5 1e-4 1e-4 2e-4 | 2e-3 6e-4 3e-4 4e-4 | -4.3e-4 -4.2e-4 -9.0e-4;
icd1 7e-5 5e-4 8e-4 3e-3 | 4e-5 2e-4 4e-4 1e-3 | 9e-4 3e-4 4e-4 8e-4 | +4.2e-4 +2.9e-3 +2.1e-2.
The constraint penalty (division-free, r^2) removes the mass drift: at lambda = 30 the mass stays within
+-5e-4 at r = 1, 3, 10 for all t (the control drifts to +2.5e-3 at r = 1 and +2e-2 at r = 10: 6-100x), the
fields at parity except alpha in r = 3-10, 2-3x worse at t > 5. lambda = 100 overshoots: a steady -5e-4 at
r = 1, 3 and -1e-3 at r = 10, alpha in 3-10 5x worse. The slope hand-off (ic_d1 1) helps in the first
windows (Khat 2x at t <= 2) but not later: alpha and chi in 3-10 grow to 3e-3 and 1e-3, the far mass
drifts as the control's. Windows of 0.5: worse (mass +3.5e-2 at r = 10 by t = 6).
At t = 8 (adm30 and adm100 completed 2026-11-04 11:22 / 11:28 UTC; band rms 0.05-0.2 / 0.2-1 / 1-3 / 3-10 for
alpha | chi | Khat; m/M - 1 at r = 0.3 / 1 / 3 / 10; max |m/M - 1| over r >= 0.2 and all t):
1d_T8 2e-4 5e-4 3e-4 4e-4 | 1e-4 5e-4 2e-4 3e-4 | 7e-4 3e-4 2e-4 2e-4 | +2.5e-3 +2.7e-3 +1.9e-3 +2.5e-2 | 3.7e-2;
adm30 4e-4 7e-4 8e-4 1e-3 | 2e-5 5e-5 3e-4 5e-4 | 7e-4 6e-4 2e-4 5e-4 | +2.9e-4 +3.3e-4 +2.9e-4 -1.1e-4 | 1.2e-3;
adm100 1e-4 4e-4 1e-3 2e-3 | 2e-5 3e-4 4e-4 2e-4 | 1e-3 6e-4 7e-4 3e-4 | -3.9e-4 -3.6e-4 -3.5e-4 -7.2e-4 | 1.6e-3.
lambda = 30: the mass is flat at +3e-4 inside r = 3 and -1e-4 at r = 10 (control 8x / 230x larger), the worst
mass error anywhere 30x smaller; chi 5-10x better inside r = 1; alpha 2.5x worse in r = 1-10 (8e-4 - 1e-3 vs
3-4e-4), Khat 2.5x worse in 3-10. lambda = 100 trades a uniform -4e-4 mass bias for alpha 5x worse in 3-10.
Whether the alpha differences exceed the run-to-run spread is for the seed runs (adm30_seed8, seed8).

**Seed pair to t = 8 (2026-11-04; batch finished 20:52 UTC; `plots/2026-11-04/sing_1d_seeds.png`, `tools/plot_1d_runs_t.py`).**
`1d_T8_seed8` and `1d_T8_adm30_seed8`: the same flags with `--seed 8`. Columns as above (band rms 0.05-0.2 /
0.2-1 / 1-3 / 3-10 for alpha | chi | Khat at t = 8 against the 1D reference; m/M - 1 at r = 0.3 / 1 / 3 / 10;
max |m/M - 1| over r >= 0.2 and all t):
1d_T8_seed8 5e-4 1e-3 1e-3 8e-4 | 7e-5 3e-4 7e-4 4e-4 | 2e-3 1e-3 4e-4 3e-4 | +1.6e-3 +1.5e-3 +3.4e-3 +3.3e-2 | 4.4e-2;
adm30_seed8 1e-4 2e-4 1e-3 3e-3 | 6e-6 1e-4 7e-4 1e-3 | 6e-4 6e-4 5e-4 6e-4 | -1.2e-5 +8.7e-6 +2.9e-5 -2.5e-4 | 1.4e-3.
1. The mass gain does not depend on the seed: with lambda = 30 the worst mass error is 1.2e-3 / 1.4e-3 against
   3.7e-2 / 4.4e-2 for the two controls (30x), and at r = 10 it ends at -1e-4 / -2.5e-4 against +2.5e-2 / +3.3e-2.
2. Seed-to-seed spread in the fields is large: the two controls differ by 2-3x in alpha everywhere (seed 8 is
   1e-3 in r = 0.2-3 by t = 6).
3. Inside r = 1 lambda = 30 is at least as good as the control (adm30_seed8: alpha 2e-4, chi 1e-4 in 0.2-1 at
   t = 8, both controls 5e-4 - 1e-3 / 3e-4 - 5e-4). In r = 1-3 the two lambda = 30 runs lie inside the
   controls' spread.
4. In r = 3-10 the cost is real: both lambda = 30 runs are above both controls from t ~ 5 on, alpha 1e-3 / 3e-3
   against 4e-4 / 8e-4 at t = 8 (~3x), chi 5e-4 / 1e-3 against 3e-4 / 4e-4, Khat 5e-4 / 6e-4 against 2e-4 /
   3e-4. adm30_seed8's outer alpha grows steadily from t = 4 (3e-4 -> 3e-3). That is where the r^2 weight puts
   the penalty (85-95 % of it in r = 1-24, mostly 3-8). Next: `1d_T8_adm10` (does a gentler penalty keep the
   mass and give back the outer fields?).

**Constraint penalty on the hand-off slice, `--w_adm_ic W` (sing_1d.py, 2026-11-04; the user's choice over a
slope hand-off).** W mean(r^p [(chi H)^2 + |chi M|^2]) at the IC points on t = t0, the same division-free density
as `--w_adm`, independent of it. Why: H and M_i on a slice involve only the fields and their spatial (tangential)
derivatives there, so a soft value-only hand-off that leaves the slopes free injects constraint violation at
the start of every window. `--ic_d1/--ic_d2` attacked this by matching the previous window's slopes (and so
its slope errors); this asks instead that each window start on a constraint-satisfying slice. In window 0 the
hard IC makes the term vanish (5.5e-32). Mini CPU smoke test (2 windows, 500 interior / 200 IC points, 500
iterations, `--w_adm 10 --w_adm_ic 10`): window 1's adm_ic part falls 2.4e-6 -> 1.1e-6, same time per
iteration (42 s per 250). Fallback if it does not help: the first-derivative hand-off alone (`--ic_d1`, no
`--ic_d2`) together with the interior penalty.

**`1d_T8_adm10`, the gentler penalty (lambda = 10; hub 2026-11-04 21:58 - 23:50 UTC, stopped by a hub restart at
31 / 32 windows, t = 7.75; `plots/2026-11-05/sing_1d_adm10.png`; table by the new `tools/table_1d_runs.py`).**
At t = 7.75 (band rms 0.05-0.2 / 0.2-1 / 1-3 / 3-10 for alpha | chi | Khat; m/M - 1 at r = 0.3 / 1 / 3 / 10;
max |m/M - 1| over r >= 0.2 and t' <= t):
control 2e-4 5e-4 2e-4 3e-4 | 1e-4 4e-4 2e-4 3e-4 | 6e-4 2e-4 2e-4 2e-4 | +2.4e-3 +2.5e-3 +2.0e-3 +2.5e-2 | 3.6e-2;
control seed 8 5e-4 1e-3 2e-3 9e-4 | 7e-5 3e-4 7e-4 4e-4 | 1e-3 1e-3 4e-4 3e-4 | +1.5e-3 +1.5e-3 +3.0e-3 +3.3e-2 | 4.3e-2;
adm10 9e-5 2e-4 4e-4 1e-3 | 4e-5 1e-4 5e-5 3e-4 | 2e-3 7e-4 2e-4 3e-4 | -9.5e-4 -9.4e-4 -9.7e-4 -2.2e-3 | 3.7e-3;
adm30 4e-4 7e-4 7e-4 1e-3 | 2e-5 5e-5 3e-4 4e-4 | 9e-4 4e-4 2e-4 5e-4 | +2.8e-4 +3.1e-4 +2.7e-4 -1.3e-4 | 1.2e-3;
adm30 seed 8 8e-5 1e-4 8e-4 2e-3 | 7e-6 7e-5 6e-4 1e-3 | 6e-4 4e-4 4e-4 6e-4 | -2.3e-5 -9.4e-6 +1.3e-5 -3.5e-4 | 1.4e-3.
1. Mass: lambda = 10 is 10x better than the controls on the worst value (3.7e-3 vs 3.6e-2 / 4.3e-2) but 3x worse
   than lambda = 30, and it drifts: a uniform negative offset growing from -3e-4 (t = 4) to -9.5e-4 inside r = 3
   and -2.2e-3 at r = 10 by t = 7.75 (lambda = 30 stays flat at +3e-4).
2. Fields: inside r = 3 at or below the controls (alpha 2e-4 / 4e-4 in 0.2-1 / 1-3, chi 1e-4 / 5e-5). In r = 3-10
   it does NOT give the outer fields back: alpha 1e-3 as lambda = 30 (controls 3e-4 / 9e-4), chi 3e-4, Khat 3e-4.
3. So the outer-alpha cost does not scale with lambda between 10 and 30, while the mass does: lambda = 30 stays
   the choice. The outer error has to be attacked another way (the hand-off-slice penalty `--w_adm_ic`, or the
   first-derivative hand-off, both queued as single changes against `1d_T8_adm30`).
   (The last window, run 2026-11-05: at t = 8 adm10 is alpha 7e-5 2e-4 4e-4 1e-3, mass -9.9e-4 / -9.5e-4 / -9.9e-4 /
   -2.2e-3, worst 3.7e-3: the reading stands.)

**`1d_T8_adm30_ic`: lambda = 30 inside plus lambda = 30 on the hand-off slice (`--w_adm 30 --w_adm_ic 30`; hub
2026-11-05 07:31 - 09:30 UTC, 32 windows, same time per window as adm30; `plots/2026-11-05/sing_1d_adm30_ic.png`,
adm30 drawn black there).** One change against `1d_T8_adm30`. At t = 8 (columns as above):
adm30 4e-4 7e-4 8e-4 1e-3 | 2e-5 5e-5 3e-4 5e-4 | 7e-4 6e-4 2e-4 5e-4 | +2.9e-4 +3.3e-4 +2.9e-4 -1.1e-4 | 1.2e-3;
adm30 seed 8 1e-4 2e-4 1e-3 3e-3 | 6e-6 1e-4 7e-4 1e-3 | 6e-4 6e-4 5e-4 6e-4 | -1.2e-5 +8.7e-6 +2.9e-5 -2.5e-4 | 1.4e-3;
adm30_ic 3e-4 7e-4 5e-4 4e-4 | 1e-5 1e-4 2e-4 3e-4 | 2e-3 7e-4 2e-4 3e-4 | -4.8e-4 -4.0e-4 -4.4e-4 -1.1e-3 | 2.5e-3.
1. The outer fields come back: alpha in r = 3-10 stays at 4-5e-4 from t = 4 to 8 (flat, even falling at the end)
   while both adm30 seeds grow to 1e-3 / 3e-3; that is the controls' level (4e-4 / 8e-4). chi 3e-4 and Khat 3e-4
   in 3-10 are also back at the controls' level (adm30: 5e-4 - 1e-3, 5e-4 - 6e-4). Below both adm30 seeds, so
   more than the seed spread, though one seed only.
2. The mass gets worse than adm30: flat near zero to t = 5 (-1e-5 inside r = 3 at t = 4), then a negative
   drift, -4.4e-4 inside r = 3 and -1.1e-3 at r = 10 by t = 8; worst 2.5e-3 (adm30 1.2e-3 / 1.4e-3, controls
   3.7e-2 / 4.4e-2: still 15x better than the controls).
3. Inner fields (r < 1) at the adm30 level; Khat in 0.05-0.2 2e-3 (adm30 7e-4 / 6e-4, controls 7e-4 / 2e-3).
4. Reading: the slice penalty removes the outer-field cost of the interior penalty, which supports the picture
   that unconstrained hand-off slices feed the outer error; the mass then drifts slowly negative, as with
   lambda = 10. Open: whether the slice weight (30, untested otherwise) sets that drift; the first-derivative
   hand-off (`1d_T8_adm30_icd1`) is the competing fix, running.

**`1d_T8_adm30_icd1`: lambda = 30 plus the first-derivative hand-off (`--w_adm 30 --ic_d1 1`; hub 2026-11-05
09:30 - 11:22 UTC, 32 windows; `plots/2026-11-05/sing_1d_adm30_icd1.png`, adm30 black).** One change against
`1d_T8_adm30`. At t = 8 (columns as above):
adm30_icd1 6e-4 2e-3 2e-3 7e-4 | 8e-6 1e-5 2e-4 4e-4 | 2e-3 3e-4 3e-4 3e-4 | +2.6e-4 +3.6e-4 +3.4e-4 -1.4e-4 | 9.9e-4.
1. Mass: the best of all runs: flat at +2 - 6e-4 inside r = 3 at every t, -1.4e-4 at r = 10, worst 9.9e-4 (adm30
   1.2e-3 / 1.4e-3, adm30_ic 2.5e-3). No late drift.
2. chi: the best inside r = 1 (8e-6 / 1e-5 in 0.05-0.2 / 0.2-1 at t = 8; adm30 2e-5 / 5e-5).
3. alpha: in r = 3-10 7e-4 at t = 8, between adm30 (1e-3 / 3e-3) and adm30_ic (4e-4), after a bump to 1e-3 at
   t = 5-6; but in r = 0.2-3 it grows after t = 5 to 2e-3 / 2e-3 at t = 8 (adm30 7e-4 / 8e-4, adm30_ic 7e-4 / 5e-4).
4. Reading: the two hand-off fixes trade differently. The slope hand-off keeps the mass best and chi best but
   moves the alpha error inwards (r = 0.2-3); the slice constraint penalty gives the best outer fields with a
   late negative mass drift. One seed each.

**Outer-boosted weights, `--adm_r0 R0 --adm_q Q` and `--res_r0 R0 --res_q Q` (sing_1d.py, 2026-11-05; the user's
weight from another study, where penalising the outer region more heavily helped).** The constraint density
(interior and hand-off slice) becomes r^p (1 + (r/R0)^Q) [(chi H)^2 + |chi M|^2]; the Einstein and gauge
residual means on the interior and t1 points get the factor (1 + (r/R0)^Q). r^p keeps the term scale-free, the
second factor tilts the penalty outwards. Chosen R0 = 5, Q = 2: 1x at r = 0, 2x at r = 5, 5x at r = 10, 24x at
R = 24. Off by default (R0 = 0). Mini CPU smoke test (2 windows, 500 points, 250 iterations, both on): runs,
same time per iteration (42 s per 250). Hub: `1d_T8_adm30_ic_aw` (`--adm_r0 5 --adm_q 2`) and `1d_T8_adm30_ic_rw`
(`--res_r0 5 --res_q 2`), each one change against `1d_T8_adm30_ic`.

**`1d_T8_adm30_ic_aw` / `_rw`, results (hub 2026-11-05 11:23 - 15:20 UTC, 32 windows each;
`plots/2026-11-05/sing_1d_outer_boost.png`, adm30_ic black).** At t = 8 (columns as above):
adm30_ic 3e-4 7e-4 5e-4 4e-4 | 1e-5 1e-4 2e-4 3e-4 | 2e-3 7e-4 2e-4 3e-4 | -4.8e-4 -4.0e-4 -4.4e-4 -1.1e-3 | 2.5e-3;
aw 1e-3 2e-3 3e-3 2e-3 | 4e-5 1e-4 3e-4 6e-4 | 2e-3 2e-3 5e-4 3e-4 | -1.1e-3 -1.1e-3 -1.1e-3 -1.6e-3 | 2.1e-3;
rw 2e-4 8e-4 1e-3 1e-3 | 2e-5 2e-4 3e-4 3e-4 | 6e-4 1e-3 6e-4 2e-4 | +4.8e-4 +5.3e-4 +5.3e-4 -2.5e-4 | 1.5e-3.
1. Boosting the constraint penalty outwards (aw) makes things worse: alpha grows steadily from t = 4 in every
   band, 2e-3 in r = 3-10 at t = 8 (adm30_ic 4e-4), chi 6e-4 there (3e-4), and the negative mass drift starts
   earlier (t = 3.5) and is 2x larger (-1.1e-3 inside r = 3).
2. Boosting the residual outwards (rw) removes the late negative mass drift: the mass stays at +5e-4 from t = 4 to
   8 inside r = 3, like adm30 (+3e-4), worst 1.5e-3. But the outer alpha goes back to the adm30 level (1e-3 in
   r = 1-3 and 3-10 at t = 8): it undoes what the slice penalty bought. Khat 3-10 at 2e-4, the best of the three.
3. Neither is a step forward for the outer fields, the error the boost was aimed at. In this problem the outer
   region is not under-weighted in the residual: the trained runs' residual is already ~1e-8 at r = 1-24 (see the
   (t, r) maps below); the outer error comes from elsewhere. One seed each; not pursued further.

**Points down to the puncture: `--log_frac F --log_rmin RMIN` (sing_1d.py, 2026-11-05; the user's proposal after the
(t, r) maps).** A share F of every point set (interior, t1, IC, held) is drawn log-uniform in r on [RMIN, R], the
same number of points per decade; the cloud shares are scaled by (1 - F). Why: the 3D clouds' radial density
goes like r^2 near zero, so with the defaults only 2.7e-7 of the points lie below r = 1e-3, 2.6e-4 below 0.01 and
2.5e-2 below 0.05, and the residual there is unconstrained (1e1 - 1e3 at r = 1e-3). With F = 0.25, RMIN = 1e-3:
5.7e-2 of the points below 0.01, 1.2e-1 below 0.05. (The inner sigma 0.05 is the narrowest the code has used;
before 2026-10-28 there was an exclusion zone and no inner cloud.) Note: the held loss is then measured on a
different point set and is not comparable with the earlier runs' held loss; the errors against the 1D reference
are. Mini CPU smoke test (1 window, 1000 points, 500 iterations, `--w_adm 30`): with F = 0.25 held 4.5e-3,
without 1.5e-3, errors at t = 0.25 alike (alpha 1.9e-3 / 1.8e-3, chi 9e-3 / 3e-3): runs; too short to judge.
Hub: `1d_T8_adm30_ic_log25` (`--log_frac 0.25`), one change against `1d_T8_adm30_ic`.

**`1d_T8_adm30_ic_log25`, result (hub 2026-11-05 19:24 - 21:33 UTC, 32 windows; `plots/2026-11-06/sing_1d_log25.png`):
it fails in window 0.** Window 0's loss stops at train 1.3e-3 after 4000 iterations (adm30_ic: ~1e-7), and the
errors at t = 0.25 are already 1e-3 - 1e-2 in every band (alpha 2e-3 2e-3 9e-4 8e-4, chi 3e-3 1e-2 8e-3 2e-3) with
m/M - 1 = +7e-2 at every r (adm30_ic: 1e-5 - 5e-5, mass 1e-5). Later windows train to ~1e-4 but carry the bad
start: at t = 8 alpha 3e-2 4e-2 3e-2 2e-2, chi 5e-3 3e-2 3e-2 1e-2, mass +1.4e-1 (adm30_ic 3e-4 - 7e-4, -4e-4).
Reading: the residual inside r = 0.01 is out of reach for this network (the value and Sobolev fits to the
answer have 4e2 - 2e8 there too, see the (t, r) maps); 5.7 % of the points there make that part dominate the
loss, and the optimiser trades the whole solution for it. So the unconstrained region at r < 0.01 is not what
holds the error at 1e-4: sampling it costs 100x. The current clouds stay. (A milder version, a small share
with --log_rmin 0.01, would only test the band 0.01 - 0.05; not run.)

**Fit to the reference, then the PINN loss (`--fit_ref NPZ --fit_window K` in sing_1d.py and sing_pinn.py,
`jobs/mini/fit_ref.sh`, Mac mini CPU, 2026-11-05; the user's experiment).** The network (same ansatz, `--hard_ic
--no_template`) is trained on one window with the pointwise loss mean|f - f_ref|^2 against the 1D reference (1D:
the 10 radial functions on the reference's own (t, r) nodes; 3D: the 22 fields of the reference lifted to 3D,
on the run's clouds with radii snapped to the reference grid). Every 250 iterations the PINN loss parts of the
fitted network are evaluated on held-out points (the IC term against the reference at t0); at the end, the
run's usual error report at t1. Questions: (1) how far the representation goes when the optimiser is given the
answer (representation vs optimisation), (2) what PINN loss a network that close to the exact solution has,
the yardstick for the PINN runs' loss floors. It needs the reference inside the windows: the same 1D evolution
saved every 0.025, `reference_1d/ref_single_m0.5_dr0.005_t8_snap0025.npz` (`sph_reference.py --dr 0.005
--tfinal 8 --snap_dt 0.025 --r_save 24.5`, 4 min on the mini; identical to ref_single_m0.5_dr0.005_t20.npz at
the shared snapshots). Windows: 1D 0, 16, 31 (hub-run settings, 4000 iterations, constraint parts at weight 30
reported, not fitted); 3D 16, 31 (plain 3D MLP, 32000 fit points per draw, 8000 iterations).

**Value-only fits, results (2026-11-05 10:47 - 12:14 UTC; 1D ~10 min per window, 3D ~28 min).** Fit error = held
mean|f - f_ref|^2; errors at t1 as the run reports (band rms 0.05-0.2 / 0.2-1 / 1-3 / 3-10 for alpha | chi | Khat);
PINN loss = held total of the run's loss parts (constraint parts at weight 30), against `1d_T8_adm30_ic`'s held
loss at the end of the same window.
- 1D window 0 (t = 0-0.25): fit 4.5e-10; alpha 3e-5 4e-6 2e-6 4e-6 | chi 2e-5 9e-6 3e-6 4e-6 | Khat 1e-5 6e-6 4e-6 2e-6
  (adm30_ic: alpha 5e-5 2e-5 3e-6 5e-6, chi 4e-6 5e-6 7e-6 7e-6, Khat 6e-5 2e-5 8e-6 7e-6); PINN loss 3.4e-4
  (Einstein 2.8e-4) against the trained run's 8.2e-8: 4000x higher with errors 1-5x smaller.
- 1D window 16 (t = 4-4.25): fit 4.3e-7; alpha 3e-4 2e-4 2e-4 2e-4 | chi 5e-5 2e-4 1e-4 2e-4 | Khat 6e-4 4e-4 4e-4 3e-4,
  m/M - 1 +7.6e-4 / +6.7e-4 / +7.0e-3 / +1.1e-3 at r = 0.3 / 1 / 3 / 10 (adm30_ic at t = 4.25: alpha 3e-4 3e-4 4e-4
  5e-4, chi 4e-6 2e-4 4e-4 2e-4, Khat 7e-4 4e-4 3e-4 1e-4, mass within 4e-4); PINN loss 3.9 (Einstein 2.3, end 1.6).
  (The trained run's held loss at that window is 0.25, almost all of it the end term: one of the held-out spikes
  from points at r << 0.05 noted in HANDOFF; its training loss is ~1e-7.)
- 1D window 31 (t = 7.75-8): fit 8.2e-8; alpha 5e-4 3e-4 8e-5 2e-4 | chi 8e-4 4e-4 2e-4 2e-4 | Khat 4e-4 3e-4 8e-5 1e-4,
  mass +5.0e-3 / +2.3e-3 / -4.0e-3 / +5.7e-3 (adm30_ic at t = 8: alpha 3e-4 7e-4 5e-4 4e-4, chi 1e-5 1e-4 2e-4 3e-4,
  mass within 1.1e-3); PINN loss 2.5 (Einstein 2.3) against the trained run's 3.0e-8.
- 3D window 16 / 31 (plain MLP, from scratch): fit 1.4e-4 / 1.6e-4 after 8000 iterations, still falling slowly;
  rms errors 1e-3 - 3e-2 (alpha 3e-2 1e-2 2e-2 2e-3 in r = 0-0.05 / 0.05-0.5 / 0.5-5 / 5-20 at t = 4.25); PINN loss
  3.3e3 / 1.0e4.
Reading. (1) In 1D a network fitted to the answer over one window, from scratch, has about the same field error as
the PINN after 16 - 31 windows of time marching (t = 4, 8), and a worse mass (it is not asked for it); at t = 0 it
is 1-5x better. So 1e-4 is where this network and optimiser get on these fields in one window either way;
the PINN's error level is not far above what the representation gives. (2) The PINN loss is no proxy for the
error: networks with the PINN's error level or better have PINN losses 1e3 - 1e8 times the trained runs'. A
pointwise fit leaves the derivatives free, and the residual is made of derivatives (and 1/chi near the
puncture). The Sobolev fits (below) test that directly. (3) In 3D the plain MLP fitted from scratch on one window
with 32000 points and 8000 iterations stays at 1e-2: this says nothing yet about its representation limit (the
3D PINN runs warm-start window after window and reach 1e-4 - 1e-3), only that a cold single-window fit is slow.

**Where the residual sits: (t, r) maps (`sing_1d.py --eval_tr PKL --eval_tr_out NPZ`, `tools/plot_tr_loss.py`;
`plots/2026-11-05/tr_loss_1d.png`; data in `eval_data/tr_loss/`).** Per point, lam_einstein x mean of the 18
Einstein rows squared + the gauge part, on 41 t x 240 log-spaced r in [1e-3, 24] over the window, for the trained
`1d_T8_adm30_ic` window networks and the value-only fits. Band averages (over t and the log-r grid):

| window | network | r = 0.001-0.01 | 0.01-0.05 | 0.05-1 | 1-24 |
|---|---|---|---|---|---|
| 0 | PINN | 1.9e-2 | 4.3e-6 | 2.2e-7 | 4.9e-9 |
| 0 | value fit | 6.4e2 | 3.0e-2 | 1.1e-5 | 4.5e-9 |
| 16 | PINN | 3.5e2 | 7.5e-4 | 3.6e-8 | 1.7e-8 |
| 16 | value fit | 4.7e7 | 2.9e2 | 1.2e-2 | 2.2e-6 |
| 31 | PINN | 6.6e1 | 1.7e-6 | 1.5e-8 | 8.6e-9 |
| 31 | value fit | 7.8e6 | 4.2e2 | 7.6e-2 | 1.5e-6 |

1. In every network the residual falls steeply with r: it is concentrated inside r = 0.01, where the 3D clouds put
   almost no points (the inner Gaussian has sigma 0.05 in 3D, so its radial density goes like r^2). The trained
   runs' loss of ~1e-7 is the r >~ 0.01 part; at r = 1e-3 their residual is 1e1 - 1e3, unconstrained.
2. With about the same field errors (1e-4), the value fit's residual is 100 - 10^6 times the PINN's at every r
   below 1, and 100x at r = 1-24 in the late windows: the PINN has found a network that solves the equations
   far better than the pointwise-closest one, at the same distance from the solution. The error level is set
   elsewhere (time marching, hand-offs, the representation), not by the residual the PINN reaches.

**Sobolev fits, results (`--fit_d1 1 --fit_d2 1`, 2026-11-05 12:15 - 15:38 UTC on the mini; 1D ~10 min per window,
3D ~85 min).** Loss = mean|f - f_ref|^2 + mean|r (f' - f'_ref)|^2 + mean|r^2 (f'' - f''_ref)|^2 (3D: r grad u and
r^2 grad grad u of all fields): a discrete, scale-free H^2 norm of the error in r at fixed t; no time derivatives.
Columns as for the value-only fits (fit = held Sobolev loss, "values" its value part):
- 1D window 0: fit 1.4e-7 (values 4.8e-10); alpha 2e-4 2e-5 5e-6 4e-6 | chi 2e-5 3e-6 5e-7 7e-7 | Khat 6e-5 2e-5 1e-5 7e-6;
  PINN loss 5.2e-4 (Einstein 2.8e-4, end 2.4e-4); value-only fit 3.4e-4.
- 1D window 16: fit 5.4e-4 (values 5.5e-6), not converged; alpha 3e-3 6e-3 4e-3 8e-4 | chi 3e-3 1e-3 6e-4 3e-4 |
  Khat 1e-2 6e-3 1e-3 5e-4, mass +2.3e-2 at r = 0.3; PINN loss 6.4e2 (Einstein 1.2e2, end 5.2e2); value-only 3.9.
- 1D window 31: fit 8.3e-6 (values 1.1e-7); alpha 2e-3 2e-3 1e-3 3e-4 | chi 7e-4 4e-5 2e-4 2e-4 | Khat 2e-3 8e-4 7e-5 1e-4;
  PINN loss 3.0 (Einstein 2.5); value-only 2.5.
- 3D window 16 / 31: fit 5.6e-4 / 6.6e-4 (values 3.1e-5 / 4.0e-5, 4x below the value-only 3D fits' 1.4e-4 /
  1.6e-4); rms errors 1e-3 - 1e-2 (Khat 1e-1 in r < 0.05); PINN loss 5.2 / 1.3e2 against the value-only 3.3e3 / 1.0e4.
(t, r) residual, band averages as in the table above (r = 0.001-0.01 / 0.01-0.05 / 0.05-1 / 1-24), Sobolev fit:
window 0 3.8e2 / 2.5e-2 / 2.4e-5 / 3.2e-8; window 16 2.1e8 / 2.4e4 / 2.1e-1 / 2.0e-4; window 31 6.4e7 / 5.8e2 /
9.9e-3 / 1.2e-5 (third column of `plots/2026-11-05/tr_loss_1d.png`).
Reading. (1) In 1D matching r f' and r^2 f'' does not bring the residual down: it is at the value-only fit's level
in windows 0 and 31 and 10-100x above it in window 16 (where the fit did not converge and the field errors are
10x worse), so 10^3 - 10^6 x the PINN's at every r. Where the 1D values fit to 1e-7 - 1e-10, spatial derivatives
that agree with the reference's grid derivatives still leave the Einstein residual large; the missing piece is
the time derivative (the residual is d_t u - RHS(u, u', u''), and the fit never constrains d_t u) plus the
reference's own finite-difference error in f'' near the puncture. A d_t term from the reference's RHS would test
that. (2) In 3D the derivative terms help the cold single-window fit (values 4x better, PINN loss 1e2 - 1e3 x
lower) but it stays at 1e-2 errors and PINN loss >= 5: the plain 3D MLP fitted cold is far from the PINN runs.
(3) Taken together: a network close to the solution, in values or in values and spatial derivatives, does not
have a small PINN loss; the trained PINN reaches residuals 3-6 orders lower at the same 1e-4 field error, so its
error level is not set by how small it drives the residual.

Core rms vs the 1D reference at t = 0.25 ... 2 (alpha / chi / Khat / gt): 2.1e-5 / 9.6e-6 / 1.7e-5 /
1.3e-5; 8.1e-5 / 5.1e-6 / 8.4e-5 / 1.6e-5; 2.1e-4 / 6.9e-6 / 4.3e-4 / -; 2.3e-4 / 1.6e-5 / 9.5e-4 / -; 1.6e-4 /
9.0e-5 / 1.4e-3 / 7.4e-5; 2.3e-4 / 4.0e-5 / 1.9e-3 / 8.7e-5; 3.6e-4 / 7.6e-5 / 2.2e-3 / 1.9e-4; 3.0e-4 / 1.1e-4 /
3.8e-3 / 3.0e-4 (sing_radial at t = 2: 6.3e-4 / 2.8e-4 / 1.5e-3 / 4.4e-4; baseline 7.1e-4 / - / 8.7e-3 /
1.8e-3). By band at t = 2 (0.05-0.2 / 0.2-1 / 1-3 / 3-10; alone | with template | baseline): alpha 2e-4 /
4e-4 / 5e-4 / 7e-5 | 5e-4 / 6e-4 / 3e-4 / 9e-5 | 1e-3 / 5e-4 / 3e-4 / 8e-4; Khat 4e-3 / 2e-3 / 5e-4 / 1e-4 |
2e-3 / 4e-4 / 5e-4 / 2e-4 | 1e-2 / 2e-3 / 4e-4 / 2e-4; chi 5e-5 / 2e-4 / 1e-4 / 5e-5 | 1e-4 / 3e-4 / 1e-4 / 6e-5 |
1e-4 / 4e-4 / 3e-4 / 3e-4. Mass at r = 0.3 / 1 / 3 / 10 (alone | with template | baseline): t = 1: -1.9e-4 /
+8.9e-6 / +8.1e-5 / +1.4e-3 | -1.4e-4 / +9.3e-6 / -3.5e-4 / +1.6e-4 | -5.3e-4 / -3.3e-4 / -6.4e-4 / -1.3e-3;
t = 1.5: -6.4e-4 / -9.9e-5 / +1.2e-3 / +4.0e-3 | +4.6e-4 / +1.4e-4 / -2.6e-4 / +1.9e-4 | -1.3e-3 / -6.4e-4 /
-2.5e-3 / -3.3e-3; t = 2: +1.2e-4 / +1.0e-3 / +2.5e-4 / +5.5e-3 | +3.6e-3 / +1.4e-3 / +3.2e-3 / +2.1e-3 |
-3.6e-3 / -2.1e-3 / -3.9e-3 / -4.9e-3.
Reading: the two ansatze are in the same league through t = 2, both at or below the baseline in the
fields; the template buys 2-5x in Khat inside r = 1 (the puncture region, where chi's r^4 -> r^2 change
is what the frozen BL offset lacks) and keeps the far-field mass (r = 3-10) 5-20x smaller until t = 1.75
(the no-template run's mass drifts to +5.5e-3 at r = 10 from t = 1 on, steadily); without the template
the mass inside r = 1 is better at t = 2. Verdict for the design: the template is not needed for the
accuracy level reached; a lifted radial network per puncture with the exact data as the fixed offset
carries the puncture, and a third network can take the rest. Keeping the template remains a 2-5x option
near the puncture. Evaluation data now live in `SBH/eval_data/` (git-ignored; the macOS temp cleaner
removed the scratch copies after three days).

**`sing_w0_radial_notpl`, 3D, the radial network alone (hub, 2026-11-02 00:50-02:36 UTC; early stop at 4000
iterations, 40 min).** Held-out 3.2e-5 / 4.7e-8 at 1000 / 4000 (template run 8.8e-7 / 5.6e-8: slower start,
same floor); at t = 0.25 core rms alpha 2.1e-5, chi 9.6e-6, Khat 1.7e-5, gt 1.3e-5 (template run 2.2e-5 /
4.9e-6 / 1.4e-5 / 8.3e-6); by band 0.05-0.2 / 0.2-0.6 / 0.6-1 / 1-3 / 3-10: alpha 3e-5 / 2e-6 / 1e-6 / 2e-6 /
1e-6 (3e-5 / 6e-6 / 8e-7 / 1e-6 / 9e-7), Khat 2e-5 / 9e-6 / 5e-6 / 3e-6 / 4e-6 (7e-6 / 2e-5 / 4e-6 / 4e-6 / 3e-6),
chi 9e-6 / 1e-5 / 3e-6 / 5e-6 / 6e-6 (6e-6 / 2e-6 / 2e-6 / 7e-7 / 1e-6); mass +3.4e-5 / +1.9e-6 / +2.3e-5 /
+1.0e-4 at r = 0.3 / 1 / 3 / 10 (-1.6e-5 / -2.9e-5 / -2.3e-5 / +9.9e-5). Confirmed in 3D: in window 0 the
template is redundant once the network is radial and the exact data enter as the fixed offset. Next:
the same over 8 windows (`sing_radial_notpl`, continuing from this window 0), against sing_radial, where
the r^4 puncture zero frozen into the offset may start to matter; and sing_radial's windows 6-7 rerun from
window_005.pkl with 8000 fixed iterations (`sing_radial_w67`) to tell an early-stop artefact from a real
change at t ~ 1.5.

**Two at once does not fit (2026-10-29).** `sing_bl_ic` and `sing_bl_ic_t05`, launched together on the hub
A30 with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.47` each, both died with RESOURCE_EXHAUSTED (8.06 GiB in one
allocation) at the first full-phase iteration: the allocator kept the ~4.3 GB of the IC phase and the full
phase needs ~10 GB. One run per GPU (peak 10.1 GB alone). Measured alone at baseline settings: 87 s per 250
iterations, ~2.3 h per window, ~19 h per 8-window run.
