# SBH: CCZ4 PINN for black-hole punctures

A PINN that learns the 22 vacuum CCZ4 fields of a black-hole spacetime: Brill-Lindquist /
Bowen-York punctures, moving-puncture gauge, no excision, trained window by window in time.
The code came from collaborators (the MN5 `uib108` group; the base script is the one exchanged
with Carlos). It is Python/JAX, not Julia. What it does and every flag:
[docs/summary_ccz4.txt](docs/summary_ccz4.txt). The equations: [docs/equations_ccz4.pdf](docs/equations_ccz4.pdf).

```
EEccz4_BH_puncture.py     the one script (merged 2026-10-17, see section 1)
Crunch/                   optimiser library (SSBroyden/BFGS/SCORE, Wolfe search); must sit next to the script
docs/                     summary_ccz4.txt, equations_ccz4.{txt,pdf}, README_bundle.md (the MN5 bundle's notes)
jobs/mendieta/bh.sbatch   Mendieta job (presets single | bbh | smoke | bench)
jobs/mn5/*.psubmit        the collaborators' MareNostrum 5 jobs, pointed at the merged script
tools/                    load_defs.py, bench_loss.py, check_taylor.py, taylor_proto.py, smoke_local.sh
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
0.2.8); `tqdm` was added to the laptop's copy on 2026-10-17 and is **still missing on Mendieta**:

    ssh mendieta.ccad.unc.edu.ar '~/.local/bin/uv pip install --python ~/Julia/PINN/Stationary/.venv/bin/python tqdm'

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
(`PROD_ARGS`) are those of the MN5 jobs: `--fast_hessian --loss_plot_every 10 --blowup_factor
100`. The README_bundle's best-run settings are not defaults; pass them through `EXTRA_ARGS`
(`--core_points 48000 --core_ic_points 4800 --core_radius 0.5 --EPOCHS_QN 24000`).
**Open:** whether the default sizes (80000 interior points, dense 9302^2 inverse Hessian) fit
an A30's 24 GB. The MN5 runs used H100s. Use `NGPU=2` if they don't.

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

**Not yet measured: the GPU.** `PRESET=bench` on Mendieta runs `tools/bench_loss.py` and
`tools/check_taylor.py` at 8000/32000 points. Until it has run, `--deriv_mode taylor` is not in
the production flags.

**Next candidates** (not done): the RHS backward pass (the other ~2/3 of the residual cost;
per-point 3 x 3 einsums under vmap) is the larger target, but it is verified physics code, so it
should only be rewritten with the GPU profile in hand and with `check_taylor.py`-style
agreement tests against the current version. `apply_model` (logging, once per restart) also
computes a gradient it discards; that is under 1 % of a restart.

## 4. Tools

* `tools/load_defs.py`: `load(argv)` executes the script up to the training loop and returns
  its namespace (every function, no training). Used by the other tools.
* `tools/bench_loss.py [--n N] [-- script flags]`: compile time and median time of the full loss
  and of its pieces (evolution residual value+grad and forward, boundary term).
* `tools/check_taylor.py [--n N]`: taylor vs autodiff on the full residual and gradient, both chi
  modes, with timings.
* `tools/taylor_proto.py`: the derivative arrays alone (value, d_t, d_xyz, d_xyz d_xyz) against the
  autodiff stack.
* `tools/smoke_local.sh`: the laptop smoke run.
