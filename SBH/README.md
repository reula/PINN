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
tools/                    plot_losses_stages.py (both stages' losses), load_defs.py, bench_loss.py, check_taylor.py, taylor_proto.py, smoke_local.sh, test_dsgnar_bh.py
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
