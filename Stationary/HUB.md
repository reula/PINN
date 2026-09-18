# Running on a JupyterHub server

Target setup: a **plain single-user JupyterHub server** (no Slurm/PBS), with
**NVIDIA GPUs** on the nodes, and a home directory that persists.

The one thing to internalise: **a JupyterHub server is not a batch system.** Your
JupyterLab terminal, your notebook kernels and any process you start from them all
live inside one single-user container. A notebook kernel dies with the server; a
terminal process dies with that terminal's session. So:

| where you start it | survives a browser disconnect | survives a server restart |
|---|---|---|
| notebook cell | no | no |
| terminal, plain `&` | only if `nohup`ed | no |
| `./run_hub.sh ...` (this repo) | yes | yes, from the last checkpoint |

A 20 000-step Adam phase takes **~1–2 h** here (`runs/m2R4_realrobin` = 7 435 s,
`runs/m2R3_symhybrid` = 7 866 s), so this matters: never launch one from a notebook.

---

## 1. One-time setup

Python, JAX and friends. **Do not copy the local `.venv`** — it is a macOS ARM
build. Recreate it on the hub:

```bash
cd ~/PINN/Stationary                      # clone of github.com/reula/PINN
python -m venv ~/venvs/pinn && source ~/venvs/pinn/bin/activate

# Pick ONE jax install, then the pinned requirements (see requirements.txt for why
# jaxlib is not pinned there):
pip install "jax[cuda12]==0.11.1"                       # GPU, NVIDIA PyPI wheels
# pip install "jax[cuda12-local]==0.11.1"               # GPU, use the container's CUDA
# pip install "jax[cpu]==0.11.1"                        # CPU only
pip install -r requirements.txt
```

Use `cuda13` instead of `cuda12` if `nvidia-smi` reports a CUDA 13 driver; the
`-local` variants skip ~3 GB of pip wheels but need CUDA already in the image.

**Confirm the GPU is actually visible before trusting it:**

```bash
nvidia-smi
python -c "import jax; print(jax.__version__, jax.devices())"
```

JAX falls back to CPU *silently* when no accelerator is visible, so a
misconfigured container turns a 20-minute run into a 2-hour one with no warning.
The usual cause is not something you can fix from inside: the JupyterHub spawner
was configured without a GPU resource request. Ask your admin.

## 2. Always run the check first

```bash
./run_hub.sh --check
```

This verifies the interpreter, prints the JAX devices (and warns if it sees CPU
only), imports the package, runs the test suite (~4–5 min, 23 tests) and does a
200-step smoke run. It is much cheaper than discovering a broken environment two
hours into a real run.

## 3. Launch a run

```bash
./run_hub.sh --steps 20000 --n-coll 4096 --arch sym_hybrid --rho-out 100
```

Everything after the script name is forwarded verbatim to
`python -m stationary.train`. The script:

* detaches the job into its own session (`setsid` + `nohup`) so it does not die
  with your terminal,
* writes results to `<repo>/runs/<timestamp>` and the log to `<repo>/logs/` (both
  inside the checkout: everything stays with the project; `logs/` is gitignored while
  `runs/` is tracked on purpose),
* checkpoints every **500** Adam steps (`ckpt.pkl`, ~165 KB, overwritten), and
* writes **`<outdir>/resume.sh`** — the exact command to continue that run.

Useful overrides (environment variables):

| variable | default | meaning |
|---|---|---|
| `OUTDIR` | `<repo>/runs/<timestamp>` | output directory (inside the checkout) |
| `CKPT_EVERY` | `500` | checkpoint period in Adam steps (`0` disables) |
| `TRAIN_THREADS` | `4` | caps CPU threads — be a good neighbour on a shared node |
| `LOGDIR` | `<repo>/logs` | where the log goes |
| `PY` | `python` | interpreter if your venv is not activated |

The script warns if `OUTDIR` is under `/tmp` (usually wiped with the container) or
if `JAX_ENABLE_X64` is set (see gotchas).

## 4. Monitor, resume, analyse

```bash
tail -f logs/<name>.log                                    # progress (Ctrl-C is safe)
kill -0 $(cat runs/<name>/run.pid) && echo running || echo stopped
nohup runs/<name>/resume.sh > logs/<name>.resume.log 2>&1 &   # continue, DETACHED
python -m stationary.evaluate --outdir runs/<name>         # figures/diagnostics afterwards
python -m stationary.profile  --outdir runs/<name>         # lambda vs rho (+ exact overlay)
python -m stationary.report   --outdir runs/<name>         # one-screen text report (paste-able)
```

`resume.sh` ends in `exec python ...`, so if you run it bare in a terminal it dies with
that terminal (and Ctrl-C stops it): wrap it in `nohup ... &` as above. It checkpoints
every `CKPT_EVERY` steps, so stopping and resuming costs at most that many steps.

Resuming is **exact**, not approximate: I verified that a run SIGKILLed mid-flight
and resumed from its checkpoint reproduces an uninterrupted run *bit-for-bit*
(identical final loss, identical parameters, identical history), including the
resampling cadence and the path-dependent gradient-norm reweighting. Pass the
**same `--steps`** as the original: the cosine LR schedule and the resample cadence
are functions of the step index, and a mismatch is reported at startup rather than
silently changing the trajectory.

Two limits worth knowing:

* If the run dies before the first checkpoint (i.e. within the first
  `CKPT_EVERY` steps ≈ 3 min), there is nothing to resume from — it restarts.
* `ckpt.pkl` lives in the run's `outdir`, so resuming requires that directory to
  have survived. Keep `OUTDIR` on `$HOME`.

## 5. Gotchas that will bite you

* **Precision: training is float32.** `train.py` never enables x64, and every
  result under `runs/` was produced that way. `evaluate.py`, `invariants.py`,
  `multipoles.py` and the tests *do* enable x64, so the pattern is *train in
  float32, post-process in float64*. If you export `JAX_ENABLE_X64=1` you change
  the optimisation trajectory (and lose ~2x throughput on GPU), and **none of the
  existing numbers will reproduce**. `run_hub.sh` warns if it is set.
* **`--init-from` is not a resume.** It restores only the network weights
  (`saved["net"]`); for a Robin run with a learnable `lam_inf` it re-initialises
  `lam_inf` to `--lam-inf-init`, and it never restores the optimizer state or the
  step count. Use `--resume` (which restores all of it) to continue a run.
* **Headless plotting.** Figures are on by default (`make_figures`), so set
  `MPLBACKEND=Agg` and a writable `MPLCONFIGDIR` — otherwise matplotlib tries to
  build its font cache in an unwritable directory on every import. `run_hub.sh`
  sets both.
* **"Out of memory while trying to allocate 9.32GiB" on a 12 GB card is JAX's
  preallocation, not the model.** JAX reserves 75% of the visible device at startup; the
  smoke run's actual tensors are tens of MB. `run_hub.sh` exports
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` -- confirm it with
  `grep -n PREALLOC run_hub.sh` on the machine you are actually running, and if in doubt
  export it in the launching shell as well (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.4` caps the
  share instead, `XLA_PYTHON_CLIENT_ALLOCATOR=platform` is the fully on-demand allocator).
  The `--check` smoke run now uses the light production architecture (`sym_hybrid`,
  `n_coll=256`) rather than the 25-field 3-D model.
* **Check how much GPU you were given, not just that you got one.** `nvidia-smi` on
  this hub reports `0MiB / 750MiB` for an A30 -- i.e. a small vGPU slice, not the card.
  The `--check` smoke run alone needs ~750 MiB, so it dies with
  `cuBlas allocation failure` / `HAMI OOM` while the (much smaller) test suite passes.
  `run_hub.sh --check` now prints the GPU name and memory and warns when it is under
  4 GiB. There is no configuration of this project that fits in 750 MiB usefully: ask
  the admin for a larger GPU profile (GBs), or run on CPU with
  `JAX_PLATFORMS=cpu ./run_hub.sh ...`.
* **Shared-GPU allocation failures.** `INTERNAL: ... gpublasCreate(&handle) failed:
  cuBlas allocation failure` from something as trivial as `jit_add` means JAX could not
  get GPU memory at all -- almost always because it asked for 75% of the device up
  front while another user held it. `run_hub.sh` now exports
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` for you. If it still fails, check `nvidia-smi`
  (free memory, other processes), pick a free card with `CUDA_VISIBLE_DEVICES=1`, cap
  the share with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.2`, or fall back to
  `JAX_PLATFORMS=cpu ./run_hub.sh ...`: this workload is 13.8k parameters and runs
  perfectly well on CPU (that is how every number under `runs/` was produced).
* **GPU sizing.** The network is small — 13 828 parameters (width 64 × depth 4),
  `n_coll` 4096, a 165 KB checkpoint — so a GPU gives a modest speedup, not a
  dramatic one: the L-BFGS line search is inherently sequential. Raising
  `--n-coll` is the lever that actually feeds a GPU. One GPU is plenty — do not
  request several.
* **Push before you clone.** A clean clone contains only what is committed. Run
  `git status` and `git push` before starting work on the hub, or the hub gets stale
  code (a missing import for the figures is the failure mode). If you must move an
  uncommitted tree instead: `rsync -av --exclude .venv --exclude .pip-cache \
  --exclude .mplcache --exclude __pycache__ Stationary/ hub:~/PINN/Stationary/`.
* **`runs/` is tracked on purpose** — the run outputs are committed as reference
  results, so `.gitignore` deliberately does not exclude them. Large `ckpt.pkl`
  files inside them *are* ignored.

## 6. Reference numbers

Measured on the 8-core macOS CPU machine, `n_coll=4096`, width 64, depth 4:

| run | steps (Adam + LBFGS) | wall time |
|---|---|---|
| `n2_dipole` | 10 000 + 1 000 | 4 905 s |
| `m2R4_realrobin` | 20 000 + 2 000 | 7 435 s |
| `m2R3_symhybrid` | 20 000 + 2 000 | 7 866 s |

with **13 828 parameters** (55 KB of weights, ~165 KB per checkpoint). This is a
small job: request modest CPU/RAM rather than many cores or several GPUs.

## 7. The two production runs (separate, run in sequence)

**They are two independent runs, not one run combining both cases.** A single run has one
set of inner boundary data, so the spherically symmetric case (`S1 = 0`) and the dipole
(`S1 = 0.1`) are necessarily separate jobs, each with its own output directory, log and
checkpoints. Launch them one after the other — one GPU, two jobs at a time will just fight
for memory.

**Why run the `S1 = 0` case at all:** the dipole has no known solution, so it cannot
validate itself. The `S1 = 0` run does — the exact family member (`R0 = 1/sqrt(3)`,
`k = 1`) solves it, so `lambda(rho = 100)` must come out **0.9885** — and it uses the *same*
model, the same inner-data machinery and the same fourth-order Robin conditions, differing
only in `S1`. Passing it therefore validates exactly the code path the dipole uses; if it
fails, the dipole number means nothing.

Shared physics of both: second-order (metric-only) scheme with `Gamma` derived from `h`;
inner sphere at `rho = 1` carrying the round metric of areal radius 1 with
`lambda = 1/3 + S1 z/rho_in`; fourth-order Robin conditions on `h` and `lambda`
(`{r^-1..r^-4}` for lambda, `{r^-2..r^-5}` for h — what lets the quadrupole pass the outer
boundary unpenalised); `lambda -> 1` at `rho = 100`. `--no-robin-G` drops the redundant
Gamma condition, which in this scheme only buys fifth derivatives of the network.

Set the size once. The first line is the CPU reference; override it on the GPU:

```bash
SIZE=(--n-coll 4096 --n-bnd 256 --width 64 --depth 4 --fourier 8)
# GPU:      SIZE=(--n-coll 16384 --n-bnd 1024 --width 256 --depth 6 --fourier 16)
COMMON=(--steps 20000 --lbfgs-steps 1000 --ref-solution --ref-asymptotic 1.0 \
        --R0 0.5773502691896258 \
        --rho-in 1.0 --inner-radius 1.0 --rho-out 100 --lam0 0.3333333333333333 \
        --outer-bc robin --robin-orders h=4,lam=4 --no-robin-G --lam-inf 1.0 \
        --no-inner-h-rr --decay-feature --radial log --pde-ramp-steps 500 \
        --w-inner 100 --w-outer 100 --reweight-every 1500)
```

**(1) Control first — `S1 = S2 = 0`.** Output goes to `runs/<timestamp>` inside the
checkout (override with `OUTDIR=`; `run_hub.sh` never writes outside the project):

```bash
./run_hub.sh "${COMMON[@]}" "${SIZE[@]}" \
    --arch axisym_hybrid --outdir runs/control
```

Check it before going further — any of:

```bash
python -m stationary.evaluate --outdir runs/control     # prints max_dh vs the exact solution
python -m stationary.profile  --outdir runs/control     # lambda vs rho, exact overlay + table
```

`lambda` must reach **0.9885** at `rho = 100` and `max_dh` should be ~1e-4 or smaller.

**(2) Then the dipole — `S1 = 0.1`:**

```bash
./run_hub.sh "${COMMON[@]}" "${SIZE[@]}" \
    --arch axisym_hybrid --lam-bc-S1 0.1 --lam-bc-S2 0.0 --outdir runs/dipole
```

`--ref-solution --ref-asymptotic 1.0 --R0 1/sqrt(3)` is in the shared block so that the
dipole run also carries the *spherical* reference for comparison: its diagnostics then show
how far the dipole solution departs from the spherical one, and `stationary.profile` can
overlay it without warning. It does not enter the loss unless `--robin-source` is also
given.

**Cost:** each is one full run, so budget 2x a single run (on the CPU here ~1.5-2.5 h
each; measure ms/step on the GPU with the `--check` smoke run before launching 20000
steps). If GPU time is tight, run the control at the smaller size — it is a correctness
check, not a resolution study — and spend the big configuration on the dipole.

The analysis is produced by the run itself: `lambda_inner.png` (imposed `lambda` on the
inner sphere vs the network), `lambda_multipoles_outer.png` (the `l = 0, 1, 2` angular
dependences at `rho_final = 100`, with amplitudes), `lambda_multipole_decay.png`
(amplitudes vs `rho` against the expected `-(l+1)`), and `report.json` with the residuals,
boundary values and multipole content.

## 8. Notebooks

A Jupyter kernel inherits **none** of what `run_hub.sh` sets up, so a notebook needs its
own preamble. Two prerequisites and one cell:

**8.1 The kernel must be the environment that has JAX.** Register it once:

```bash
source ~/venvs/pinn/bin/activate          # or Stationary/.venv
pip install ipykernel
python -m ipykernel install --user --name pinn --display-name "PINN (jax)"
```

then pick "PINN (jax)" in the notebook's kernel menu. A kernel from the base conda env
will not have `jax`, or will have the CPU-only build.

**8.2 The preamble cell must come first.** Three settings only take effect if they are
made *before* the first `import jax` / `import matplotlib` in that kernel:

* `XLA_PYTHON_CLIENT_PREALLOCATE=false` -- otherwise the kernel grabs 75% of the GPU and
  either fails with `cuBlas allocation failure` or blocks your own training run;
* `MPLCONFIGDIR` -- a writable font-cache directory, else matplotlib rebuilds its cache on
  every import (and fails if the home directory is read-only);
* `jax_enable_x64` -- post-processing and the tests are float64 by convention, while
  training stays float32.

**If you have already imported jax in this kernel, restart it** -- environment variables
read at import time cannot be changed afterwards.

**8.3 The cell.** From inside `Stationary/`:

```python
%run notebook_setup.py
```

It sets the three things above, puts the project on `sys.path`, prints the device summary,
and defines three helpers:

```python
pf, cfg = load("runs/m2R4_realrobin")               # or params_file="ckpt.pkl" for a live run
rhos, curves = lambda_vs_rho(pf, cfg, thetas=(0.0, 0.7))
plot_lambda_vs_rho(pf, cfg, save="lambda_vs_rho.png")
```

`plot_lambda_vs_rho` is written so the figure explains itself: the two horizontal guides
are labelled as the asymptotic and the imposed inner value (not as bare symbols), the
caption names the run and its architecture, and if the solution does not depend on the
polar angle (every spherical run) the per-angle curves are collapsed to one line that
says so, instead of drawing three coincident curves and a five-entry legend. When the
angle *does* matter it draws one curve per angle plus the min/max envelope. It also prints
a small table (pass `table=False` to silence it), which is a quick check that the inner
boundary data were imposed: on `runs/n2_dipole` it reproduces `1/3 + 0.1 cos(theta)`.

For anything else, the fields are `pf(x)` -> `(h, Gamma, lambda)` at a point `x` (a length-3
array in the harmonic coordinates), and `cfg` carries `rho_in`, `rho_out`, `lam0`,
`lam_inf`, `inner_radius` and the Robin orders, so the same pattern extends to `Gamma`,
`h_rr` or the multipoles (`from stationary.multipoles import lambda_multipoles`).

**8.4 A text report to paste into a discussion.** `python -m stationary.report --outdir
runs/<name>` prints one screen with: the code provenance (git HEAD, and whether the
boundary-weight fix is present in `train.py`), the full configuration (including `S1`,
`S2`, `lambda_0`, Robin orders), the loss trajectory with its groups, the inner boundary
imposed versus achieved at three angles, `lambda` at the outer sphere with its distance
from `lambda_inf`, the multipole amplitudes and their fitted decay powers, the PDE
residuals, the comparison with the exact reference when there is one, and the
`[reweight]` history from the log -- the last of which is where a silently de-weighted
boundary condition shows up. It plots nothing, so it is safe headless.

**8.5 Which call draws what.** A notebook displays *every* figure a cell creates, and the
figure-producing calls differ a lot in how many they make:

| call | figures | axes in total |
|---|---|---|
| `%run notebook_setup.py` | 0 | - (defines helpers, prints one summary line) |
| `show_run("runs/<name>")` | **1** | 3 |
| `plot_lambda_vs_rho(pf, cfg)` | 1 | 1 |
| `python -m stationary.profile --outdir RUN` | 1 | 2 |
| `python -m stationary.evaluate --outdir RUN` | 4 | 15 |
| a training run (`run_hub.sh ...`) | 0 | writes the same 4 files, displays nothing |

So calling `evaluate()` in a notebook explains a screenful of plots: `diagnostics.png`
(6 axes) plus `lambda_inner.png` (4), `lambda_multipoles_outer.png` (3) and
`lambda_multipole_decay.png` (2) are each drawn and displayed as they are created. Use
`show_run` for a single three-axis summary, or pass `--no-plots` to `evaluate`.

Every figure carries its provenance in the caption: the run directory, the architecture,
and the inner boundary data `lambda_0`, `S1`, `S2` -- so a PNG or a notebook cell can be
told apart from the next one. The plotting helpers here call `plt.close("all")` first, so
re-running a cell replaces its plot instead of stacking another one.

