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

> **On this hub the CUDA environment is the checkout's own `.venv`**, not a separate
> one: `<checkout>/.venv/bin/python -c "import jax; print(jax.devices())"` reports
> `[cuda:0]`, while `~/venvs/pinn` was created without the CUDA jax and is CPU-only.
> Either use the checkout's venv -- `PY=$PWD/.venv/bin/python ./run_hub.sh ...`, see
> section 1b -- or install `jax[cuda12]` into `~/venvs/pinn` as below. Do not create yet
> another environment expecting it to have a GPU: a venv is CPU-only unless the CUDA
> jax is installed into it *first*, with `requirements.txt` afterwards (it pins `jax`
> but deliberately not `jaxlib`, so installing it first would install the CPU wheel).

Python, JAX and friends. **Do not copy the local `.venv`** from the Mac — it is a macOS
ARM build. Recreate it on the hub:

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

## 1b. Which environment am I in? (and where is the checkout?)

It is easy to end up with two virtualenvs (say `<checkout>/.venv` and `~/venvs/pinn`)
and two paths to the same directory (`~/serafin/...` versus `/serafin/<user>/...`). XLA
picks up whichever interpreter you launch, silently falling back to CPU if that one has
the CPU build, so check rather than assume.

**On this hub: the checkout's `.venv` is the CUDA one; `~/venvs/pinn` is CPU-only.**
Quickest confirmation and the recommended way to launch either of them:

```bash
pwd -P                              # the physical path of the checkout you are in
readlink -f ~/serafin/Julia/PINN/Stationary

for py in "$PWD/.venv/bin/python" ~/venvs/pinn/bin/python; do
  [ -x "$py" ] || continue
  printf '%-40s ' "$py"
  "$py" -c "import jax; print(jax.__version__, jax.devices())" 2>&1 | tail -1
done
```

Whichever line prints `[cuda:0]` is the environment to use. Switching between them:

```bash
deactivate 2>/dev/null; source ~/venvs/pinn/bin/activate     # or any other env
which python && python -c "import jax; print(jax.devices())"
```

To give a CPU-only env the GPU build, install **jax first**, because `requirements.txt`
pins `jax` but deliberately not `jaxlib`:

```bash
source ~/venvs/pinn/bin/activate
pip install "jax[cuda12]==0.11.1"     # cuda13 if nvidia-smi reports CUDA 13
pip install -r requirements.txt
```

The most robust habit, and the one that avoids all activating confusion, is to name the
interpreter explicitly -- `run_hub.sh` honours `PY`, and the detached job inherits it:

```bash
PY=$PWD/.venv/bin/python ./run_hub.sh --check
PY=$PWD/.venv/bin/python ./run_hub.sh --steps 20000 ... 
```

For notebooks the kernel matters instead: register one **from the environment that has the
GPU** (the command runs the kernel spec against *that* interpreter, not against whichever
name you give it):

```bash
$PWD/.venv/bin/python -m ipykernel install --user --name stationary --display-name "Stationary (cuda)"
```

and pick it in the kernel menu. If a previous registration points at the wrong env, simply
re-register with the same name.

## 2. Always run the check first

```bash
./run_hub.sh --check
```

This verifies the interpreter, prints the JAX devices (and warns if it sees CPU
only), imports the package, runs the test suite (~7 min, 35 tests) and does a
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
* checkpoints every **500** Adam steps (`ckpt.pkl`, ~165 KB, overwritten),
* writes **`<outdir>/resume.sh`** — the exact command to continue that run, and
* when the training process ends — **even if it crashed** — runs `postprocess.sh` on the
  run directory, so a finished run already contains all five `.png` figures and
  `<outdir>/report.txt` (the paste-able text report). The whole chain is recorded in
  **`<outdir>/job.sh`**, which you may re-run by hand.

> Do not run `python -m stationary.train` directly if you want figures and a report:
> that only *trains*. `run_hub.sh` (or `postprocess.sh`, see §4) is what produces the
> figures, the λ vs ρ profile and `report.txt`.

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

./run_hub.sh --post --outdir runs/<name>    # re-make ALL figures + report.txt (no training)
cat runs/<name>/report.txt                  # the text report, ready to paste
```

**After a run finishes, everything is already there** — you do not have to run anything
else. The end of `run_hub.sh`'s job automatically executes `postprocess.sh`, so the run
directory holds:

| file | what it is |
|---|---|
| `diagnostics.png` | loss history, λ(ρ), h_rr(ρ), tangential metric, errors, PDE residuals |
| `lambda_vs_rho.png` | **λ against ρ** along several polar angles, exact solution overlaid |
| `lambda_inner.png`, `lambda_multipoles_outer.png`, `lambda_multipole_decay.png` | inner-sphere λ map, outer-sphere multipoles, per-multipole decay |
| `report.txt` | **the one-screen text report** (same text as `python -m stationary.report`) |
| `config.json`, `history.json`, `report.json`, `report_eval.json` | the numbers behind the figures |
| `params.pkl`, `params_adam.pkl`, `ckpt.pkl` | final parameters and the resumable checkpoint |
| `job.sh`, `resume.sh`, `run.pid` | what was run, how to continue it, is it alive |

`./run_hub.sh --post` (no `--outdir` = the most recent run) redoes the figures and the
report at any time without touching the parameters; it works on a **crashed** run too, in
which case it falls back to the last checkpoint `ckpt.pkl`. Each step prints how long it
took and the output is unbuffered, so it is clear which part you are waiting for:

```bash
./run_hub.sh --post                                  # newest run, all three steps
./run_hub.sh --post --outdir runs/<name> --only report   # just the text report (fastest)
POST_THREADS=16 ./run_hub.sh --post ...              # widen the thread cap
JAX_CACHE=0 ./run_hub.sh --post ...                  # disable the compilation cache
```

The first `--post` for a given architecture is compilation-bound (~1 min); later ones
reuse `.jaxcache` and take ~30 s, or ~10 s with `--only report`. See the gotcha in §5.

The three steps individually, if you want one of them alone:

```bash
python -m stationary.evaluate --outdir runs/<name>     # figures/diagnostics
python -m stationary.profile  --outdir runs/<name>     # lambda vs rho (+ exact overlay + table)
python -m stationary.report   --outdir runs/<name>     # text report;  --out FILE also saves it
```

A run that consumed itself in the middle of the night therefore still leaves a complete
directory: the figures and `report.txt` are written from the last checkpoint, and the
report says how far it got.

**Comparing several runs** — after a ladder, this is the analysis step:

```bash
python -m stationary.compare runs/control_ord1 runs/control_ord1b runs/control_ord2 \
                              runs/control_ord4_x64 runs/control_ord4_big runs/dipole_x64
```

It recomputes every quantity, with the current code and in float64, for all of the runs —
so the columns stay comparable even though the runs were made with different Robin orders,
different precision, or different versions of the code — and prints one table:
precision, size, steps, wall time, `lambda` on both spheres against the exact values, the
inner/outer boundary-condition residuals, the four PDE residuals, the chart-independent
`(R0, k)` read off the solution, and the `l = 1,2,3` amplitudes.  The row
`config fields it predates` counts how many `Config` fields the run's own `config.json`
does not contain: a nonzero value means that run was made before those fields existed, so
its *own* logged residuals may use older definitions and its recomputed numbers here are
the ones to trust.  `--json FILE` dumps the raw numbers, `--lam-tol`/`--bc-tol` set the
PASS/FAIL thresholds.

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
* **`$PY ./run_hub.sh ...` runs Python on a bash script.** `run_hub.sh` is a *shell* script;
  `PY` is an environment variable it reads, so the assignment goes on the same command:
  `PY=/path/to/python ./run_hub.sh --steps ...`. Writing it as two lines, or using
  `$PY ./run_hub.sh`, sends the script to the interpreter, which reports a Python
  `SyntaxError` about a line with `--outdir) ... ;;` (it tried to parse bash `case`). If you
  see that, nothing ran and nothing was written.
* **Your flags may not have arrived.** Every physics setting in this project is a flag, and
  a shell array that was defined in another terminal expands to nothing, so
  `./run_hub.sh "${COMMON[@]}" ...` quietly trains the *default* configuration instead.
  Two defences, both automatic now: the log begins with `== command ==` and the expanded
  command line, and `stationary.train` prints an `effective configuration` banner (with the
  reference's `lambda(rho_in)`, `lambda(rho_out)` and asymptotic `k`) before the first
  Adam step. Compare those against §7 and stop the run if they differ. Use the literal
  commands in §7 rather than arrays.
* **Post-processing is compilation-bound, so it is slow the first time and fast
  afterwards.** `evaluate` + `profile` + `report` build ~15 separate XLA graphs between
  them and each module is its own process, so the first `--post` spends most of its time
  in the compiler, not in the arithmetic. Two things are on by default now:
  a **persistent compilation cache** in `<repo>/.jaxcache` (same arch + same code = reuse),
  and a **thread cap** (`POST_THREADS`, default 4). Measured on an 8-core machine, on the
  same run: 53 s cold, **32 s warm** (report alone 27 s cold → **8 s warm**). On a
  128-core node the uncapped default made this worse, not better: XLA's CPU pool takes
  *every* core for graphs this small, so it oversubscribes and fights with anything else
  on the box. Use `--only report` when you just want the text, `POST_THREADS=8` to widen
  it, `JAX_CACHE=0` to disable the cache (do that if `$HOME` is NFS and the cache makes
  things *slower*). Output is line-buffered (`python -u`) and every step prints
  `[post] NAME finished in Ns`, so a long step is visible as progress rather than a hang.
* **Check the two lines of the report that say whether the run was well posed.** In
  `report.txt`, `reference vs the imposed inner data: ... <- CONSISTENT` and the
  `over the shell max |dh|` line (as opposed to `at rho_out`) are the ones that catch a
  run whose boundary conditions contradict each other. A run can match the reference to
  1e-6 at `rho_out` and still be 20% off at the inner sphere; the shell-wide number is
  the honest one. `train.build` prints a WARNING before the run starts if the exact
  reference violates the imposed inner data (the usual cause: `--inner-radius` not
  matching the inner sphere of the exact solution — see README §5).
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

### Precision, capacity, and how high a Robin order you can ask for

The order-`n` Robin condition is `prod_{i<n}(rho d_rho + base + i)(field - field_inf) = 0`,
i.e. `n` radial derivatives.  Each `rho d_rho` multiplies the content at t-frequency
`omega` by `omega / log(rho_out/rho_in)`, and the Fourier features of the network reach
`omega_max = pi * fourier`.  So the condition both **amplifies round-off** and gets
**stiffer** as `n` or `fourier` grows — and round-off is set by the *arithmetic*, not by
the network size.  Measured on the exact solution of the control geometry at `rho = 100`
(`/tmp` one-off: `robin_operator` applied to `exact.reference_fields_asymptotic`), float32
vs float64 evaluation of the *same* exact fields:

| Robin order | true residual (float64) | float32 evaluation | verdict in float32 |
|---|---|---|---|
| 1, `lambda` | 6.6e-05 | 6.6e-05 | representable: the limit is the condition itself, not round-off |
| 2, `lambda` | 7.4e-07 | 6.9e-07 | representable — order 2 should work in float32 |
| 3, `lambda` | 2.5e-08 | 1.1e-06 | floor 45x above the target |
| 4, `lambda` | 8.3e-10 | 1.2e-05 | floor 15000x above the target |
| 2, `h` | 5.6e-11 | 6.6e-07 | floor 12000x above |
| 4, `h` | 5.9e-12 | 4.4e-05 | floor 7.5e6x above |

**These measurements were then tested (20 Sept, `runs/control_ord*`, 20000 Adam + 3000
L-BFGS each, identical otherwise) and the outcome was not what the round-off argument
alone predicted:**

| run | order | precision | `lambda(100)` | error | outer BC rms (h, lam) | lam_eq rms | spurious l=2 at rho_out |
|---|---|---|---|---|---|---|---|
| `control_ord1b` | 1 | float32 | 0.9882606 | **3.0e-04** | 5.0e-05, 4.5e-05 | 6.4e-07 | 2.3e-07 |
| `control_ord2` | 2 | float32 | 0.8263355 | 1.6e-01 | 1.5e-04, 1.8e-04 | 2.9e-05 | 3.4e-03 |
| `control_ord4_x64` | 4 | float64 | 0.9303993 | 5.9e-02 | 3.5e-04, 7.7e-04 | 5.1e-06 | 2.2e-04 |

* x64 did rescue order 4 (the earlier float32 attempt parked at 0.406 against 0.930 now),
  so round-off is real — order 4 is floored at 1.2e-05 in float32 against 8.3e-10 in
  float64.  But it was not what broke order 2, whose float32 floor (6.9e-07) is four
  orders of magnitude below its error.
* **The actual cause was the outer weight.**  Look at the order-2 loss trajectory: its
  final loss is 7.05e-04 of which 6.84e-04 is the `lam_eq` group, while every boundary term
  is at 1e-08…1e-09.  It did not fail on the boundary condition — it stopped solving the
  equation, because the stiff outer term dominated the globally clipped gradient.  The
  order-`n` Robin residual at initialisation is `(gain)^n` larger (8.8e-04 for `n = 1`,
  2.1e-01 for `n = 2`), so `w_outer = 100` is a different weight at every order.
  A fixed-budget sweep (6000 Adam + 500 L-BFGS, `n_coll = 512`, same seed):

  | run | `w_outer` | final loss | `lam_eq` rms | `ricci` rms |
  |---|---|---|---|---|
  | order 1 | 100 | 3.43e-04 | 1.79e-02 | 4.25e-03 |
  | order 2 | 100 | 1.17e-03 | 3.05e-02 | 5.62e-03 |
  | order 2 | **10** | **2.29e-04** | **1.20e-02** | **2.33e-03** |
  | order 2 | 1 | 4.50e-04 | 7.76e-03 | 7.89e-04 |

  With a lower weight the order-2 run's PDE residuals improve 2.5x and its loss 5x (part of
  which is just the outer term being counted less -- trust the weight-independent PDE and
  boundary residuals, not the loss).  At order 4 (x64, 3000 steps, `n_coll = 256`) the same
  sweep shows a **trade-off rather than a correct setting**: as `w_outer` falls 100 -> 0.1
  the `lam_eq` residual improves 2.83e-05 -> 1.49e-05 while the outer `lambda` residual
  degrades 6.2e-02 -> 2.2e-01 and `lambda(100)` falls 0.661 -> 0.490.  Hence the verdict on
  order 2 comes from `runs/control_ord2_w10`, the full-budget run with `--w-outer 10`, judged
  on its **outer BC residual**.  There is **no simple scaling law** for that weight: the initial outer residual is 8.8e-04 (order 1), 2.1e-01
  (order 2) and **4.4e+03** (order 4, where it is 99.99% of the loss at step 1) — a factor
  5e6 across three orders — yet the best weight at order 2 is 10, not the 0.4 that matching
  the initial magnitudes would suggest.  **Measure it**: a three-point sweep
  (`--w-outer 100, 10, 1`) at 3000-6000 steps takes minutes.  The signature of too large a
  weight is in `report.txt`: the PDE residuals stall while all boundary terms sit at
  1e-08…1e-09.  Step 4 of the ladder re-runs order 2 at full budget with `--w-outer 10`.
* **Order 1 is still what the dipole runs use for now** — not because higher order is
  worse, but because the dipole's `l = 1` tail at `rho_out = 100` is only
  `S1 (rho_in/rho_out)^2 ~ 1e-05`, so an order-1 condition biases it by ~1e-05, thirty
  times below the accuracy the control reaches (3e-04).  Higher order becomes necessary
  below `~1e-04`, since the exact solution itself violates the order-1 condition by
  6.6e-05 in `lambda` (7.4e-07 at order 2, 8.3e-10 at order 4).

Consequences, in the order they were tried:

1. **Order 1 in float32** is the right first run, and its accuracy ceiling is the
   condition's own residual (6.6e-05 at `rho = 100` for `lambda`), not the optimiser.
2. **Order 2 in float32** is representable (floor 6.9e-07) and should be clearly better;
   the earlier `n2_dipole` failure was an inconsistent `lam_inf` vs `k`, which `build()` now
   refuses to start silently.
3. **Order 3 and above, and any combination with a bigger `fourier`, need x64**
   (`JAX_ENABLE_X64=1`): `fourier 16` doubles every amplification factor above, so in
   float32 order 4 would floor around 2e-04 for `lambda` before the optimiser even starts.
   Note that a bigger network does *not* lower that floor — only the arithmetic does.
   x64 costs ~2x throughput and `run_hub.sh` warns that its results do not reproduce the
   float32 ones.
4. With x64 the parameters are really float64 (`stationary/model.py` routes every layer
   through a dtype helper: Flax otherwise keeps `float32` whatever `jax_enable_x64` says,
   which would silently defeat the point).  Both the run banner and `report.txt` print the
   precision actually used, read off the stored parameters.

### What the numbers must come out to

Every run sits on the **`lambda -> 1` branch** (`k = 1`).  `lambda -> c lambda` is an exact
symmetry of the system (Ricci is unchanged and so is the right-hand side), so `k = lambda`
at infinity is a free normalisation; `k = 1` is the physically interesting one, and
`lambda_0` is now *derived* from the geometry rather than being a flag:
`lambda_0 = k (rho_g - R0)/(rho_g + R0)`, `rho_g = sqrt(inner_radius^2 + R0^2)`.

| configuration | `lambda` at the inner sphere | at the outer sphere | inner areal radius |
|---|---|---|---|
| M1-style check: `R0 = 1`, `rho_in = sqrt5 = 2.2360680`, areal radius 2 (default), `dirichlet_exact` | **0.3819660** (derived) | `lambda(20) = 0.9047619`, `lambda -> 1` | **2.000000** |
| M2 control (§7): `R0 = 1/sqrt3`, `rho_in = 1`, areal radius 1, `lam0 = 1/3` (derived), Robin | **0.3333333** | `lambda(20) = 0.9438861`, `lambda(100) = 0.9885193` | 1.000000 |

The second row is the one with the acceptance criterion: **`lambda(100) = 0.9885`**.  The
banner and `report.txt` print `lambda_0` and the `k` it implies; `k != 1` means an explicit
`--lam0` was given and the run is on another branch.  For reference, the stored runs from
before this rule: `m2R2_asym1`, `m2R3_symhybrid`, `m2R4_realrobin` were already `k = 1`;
`m1_sym`, `m1_3d`, `m2R1_trivial` used `lambda_0 = 1` (`k = phi^2 = 2.618`) and `n2_dipole`
`k = 1.943`; their `lambda` numbers must be divided by that `k` to be compared.

## 6b. The Laplacian recipe (small net + dense quasi-Newton)

`tests/test_laplace_robin.py` + `Laplace_Robin.md` solve a *flat Laplacian* with the same
Robin operator in a shell of the same ratio, and they pin down a solver recipe that is worth
carrying over.  **The two structural choices are now the defaults** — the scaled shell and
the 20x6 network — so a plain `./run_hub.sh ...` run uses them, and the ladder's steps 1-7
(scaled shell too) are directly comparable with the recipe steps.  It is wired into the
ladder as `./run_ladder.sh recipe`:

```bash
PY=$PWD/.venv/bin/python ./run_ladder.sh recipe   # 8 control, 9 dipole, 10 ramp, 11 big net,
                                                 # 12 order 3 alone (equal-cost ramp check)
PY=$PWD/.venv/bin/python ./run_ladder.sh 8        # just the control
```

| ingredient | value | why |
|---|---|---|
| network | `--width 20 --depth 6 --fourier 0` (**the default now**) | a dense quasi-Newton carries an `n_params^2` inverse Hessian: **0.04 GB** at 2285 parameters against **1.57 GB** at the production net's 14 533. No Fourier features: that file measures them as not worth the extra stiffness the higher-order condition sees |
| shell | `--R0 0.005773502691896258 --rho-in 0.01 --inner-radius 0.01 --rho-out 1` (**the default in the ladder**) | the same problem with `rho` rescaled by 1/100. A relabelling — verified numerically: with the default residual normalisation (local `rho`) the two shells agree to `7e-8` at step 1 and to 4% after 400 steps, i.e. float32 round-off. **Never mix normalisations**: `--scale-ref-rho-in` on one side only breaks it by `100^p` |
| points | `--n-coll 16384 --n-bnd 1024` — far more than the 2000 asked for | 3x the round numbers is where that file measures the field error to saturate |
| optimiser | `--qn-method ssbroyden` (default), `--lbfgs-steps 2000` after `--steps 2000` Adam | Crunch's self-scaling Broyden with a line search; `initial_scale=True` is required after an Adam warm-up (with `H = I` the first step is `-grad` and the Wolfe search cannot bracket it: 0 iterations, status 3) |
| precision | `JAX_ENABLE_X64=1` | the recipe is float64, and the dense Hessian is cheap at 2285 parameters |
| order | 1 for the control and the dipole | that file's section 6.3 shows the orders cannot be discriminated at shell ratio 100 (the out-of-window content at `rho_out` is `1e-5`), which is what our own estimate said |
| order ramp | `recipe_ramp_ord2 -> recipe_ramp_ord3`, warm-started with `--init-from` and **zero Adam steps** (straight into SSBroyden) | its section 6.6: the ramp ends an order of magnitude below a fixed-order solve.  Restarting Adam at lr = 1e-3 on an already converged solution is how such a phase diverges, so warm starts skip it; the quasi-Newton phase cannot diverge, as Crunch's line search returns the state unchanged when it fails.  `recipe_ord3_alone` remains the equal-cost control for "does the ramp reach the same accuracy in fewer total iterations?" |

**Crunch ships with this repo.** `Jax/` (holding `Crunch/Optimizers`) is tracked in the same
git repository, so `git pull` on the hub brings it and the sibling-directory import resolves.
`CRUNCH_ROOT` overrides the location; when it is missing the quasi-Newton phase falls back to
`optax.lbfgs` with a printed reason (`--qn-method lbfgs` forces that).

## 6c. The quadrupole production run and the VTK export for VisIt

```bash
PY=$PWD/.venv/bin/python ./run_ladder.sh 15       # production_quad, the ramp route
```

`production_run_quad` uses the inner data

    lambda = lambda_0 - lambda_0 (1 - eps) (z^2 - (x^2+y^2)/2) / r^2 ,   eps = 0.1

which in this code's parameterisation is `S1 = 0`, `S2 = -lambda_0 (1-eps) = -0.3` at
`lambda_0 = 1/3`: `--lam-bc-S2 -0.3`.  Everything else is the control's recipe — scaled shell,
20x6 net, float64, 16384/1024 points — and it is **one run, cold-started, at Robin order 3**,
the condition whose inconsistency with the exact solution is 8.3e-10 against order 1's
6.6e-05 (measured; the order-1 control is floored at 6.4e-05 in `lambda(rho_out)` for exactly
that reason).  There is no exact solution for this data (a spherical reference cannot carry
`S2`), so the reference rows in the report are departure indicators, not errors; the guard
prints a note saying so.

**No Adam phase.**  Production runs go straight into the quasi-Newton phase (`--steps 0`):
measured, a cold start reaches 3.2e-02 in 100 iterations from a random initialisation, and the
Adam warm-up was only needed because it left a gradient too large for the Wolfe search to
bracket (`initial_scale` exists to rescue that).  If a phase reports `0 iterations, status 3
(zoom failed)`, put a short warm-up back with `--steps 500`.  Watch out that the adaptive PDE
reweighting, the `pde_ramp_steps` ramp and the resampling live inside the Adam loop, so with
`--steps 0` they do not run and the PDE groups keep their configured weights.

**Budget.** `QN_CAP` sets the quasi-Newton iteration cap (default 20000; the plateau rule
stops the run earlier whenever it can):

```bash
QN_CAP=8000 PY=$PWD/.venv/bin/python ./run_ladder.sh 15
```

Order 3 costs about 0.85 s per iteration at 16384 points on this hub, so 20000 iterations is
up to ~4.7 h in the worst case, ~85 minutes if it stops near 6000 as the dipole runs did.  A
warm-started alternative exists if that is too slow: run order 1 into `runs/production_quad`,
then order 2 with `--init-from runs/production_quad/params.pkl`, then order 3 likewise — that
is what made the control reach 4e-16 in 906 iterations, at the price of two extra runs.

**VTK for VisIt.** `--vtk` on the last phase makes `postprocess.sh` write
`runs/production_quad/vtk/solution.vtk` when that run ends.  Since the shell-conforming grid
became the default (`stationary/vtk.py`, `--grid spherical`, used by `postprocess.sh` without
an override) this is the SAME export the Weyl runs use:

* **shell-conforming spherical grid in PHYSICAL coordinates** (`1 … 100`, i.e. the run's scaled
  shell `[0.01, 1]` multiplied by `1/rho_in`): radial levels geometric so both spheres are hit
  exactly, uniform in theta and phi, every cell inside the shell by construction, and the
  inner sphere — where the field varies fastest — resolved the same amount in every direction.
  The polar rings are wedges (VTK type 13), not degenerate hexahedra.  On the default
  36 x 24 x 48: 41 k points, 41 k cells, ~8 MB;
* `--grid cartesian` still gives the older graded box (points per half axis via `--vtk-n-half`,
  default 20 → 41 per axis; only cells whose centre is inside the shell), kept for comparison:
  it grades the three AXES only, so most of the inner sphere is comparatively bare;
* point data: `lambda`, `lambda_minus_1`, `r_areal`, `ricci_scalar` (R of h), `ricci_sq`
  (`R_ab R^ab` — in three dimensions the Weyl tensor vanishes, so this is the rest of the
  curvature information beyond the scalar), `res_ricci`, `res_lam_eq` as quality maps, and —
  when the run carried a reference — `lambda_err`, `h_err`, `lambda_exact`.

Options: `--n-rho/--n-theta/--n-phi` (defaults 36/24/48), `--grid cartesian` with
`--n-half N`, `--physical-inner` (default 1.0), `--no-error` (skip the reference fields) and
`--lambda-only` (just the two `lambda` arrays, ~4x cheaper: the others need second
derivatives).  Regenerate by hand with

```bash
python -m stationary.vtk --outdir runs/production_quad                    # spherical, all fields
python -m stationary.vtk --outdir runs/production_quad --grid cartesian   # the old box
python -m stationary.vtk --outdir runs/production_quad --lambda-only
```

**VTK files are not in git**: `runs/*/vtk/` is in `.gitignore` (they are large and
regenerable from `params.pkl`).  To copy one to your laptop:
`scp <hub>:~/serafin/Julia/PINN/Stationary/runs/production_quad/vtk/solution.vtk .`,
then open it in VisIt (`File → Open`), and use `Contour`/`Slice` on `lambda_minus_1`, or
`Volume` on `res_ricci` to see where the solution is least accurate.

## 6d. Far-field value pins: what the Robin conditions cannot see

A Robin condition is a *differential* combination, and its kernel contains the leading
decaying modes of the exact solution: `prod_i (rho d_rho + base + i)` annihilates
`rho^-base … rho^-(base+order-1)`, i.e. `rho^-1, rho^-2, rho^-3` for `lambda` (base 1) and
`rho^-2, rho^-3, rho^-4` for `h` (base 2).  So it says nothing about the *amplitude* of those
modes — and the amplitude is the branch.  Measured at `rho_out = 1` on the production
geometry, with the reference's own numbers:

```
the order-3 lambda combination is a cancellation of terms of order 0.1
  rho^3 lam''' = +6.7705e-02
  9 rho^2 lam'' = -2.0429e-01
  18 rho lam'   = +2.0547e-01
  6 (lam - 1)   = -6.8884e-02
  sum           = -2.5e-08          (analytic floor 12 R0^4 = 1.3e-08)
```

so a wrong level can be paid for out of the derivatives; and the exact `h - delta` IS the
`rho^-2` kernel mode (`h_rr = 1` identically for any amplitude), so the `h` condition is
satisfied *identically* whatever that amplitude is.  The three quadrupole runs show it:

| run | `lambda(rho_out)` (target 0.9885193) | outer Robin residual, rms | final loss |
|---|---|---|---|
| `production_quad` (S2 = −0.3)     | 0.0238 | 2.6e-04 | 5.6e-04 (cap) |
| `production_quad_half` (−1/6)     | 0.0467 | 4.6e-06 | 6.3e-06 (cap) |
| `production_quad_quarter` (−1/12) | 0.31, flat | 7.2e-06 | 1.3e-05 (plateau) |

The quarter run is the sharpest case: it *converged* (plateau at 2460 of 8000 iterations)
onto the `lambda = const` branch — the lambda-equation is satisfied trivially, the
anisotropic inner data are absorbed in a thin inner layer — and the loss it minimised is 91 %
`pde_ricci`.  Its metric at `rho_out` was nearly right (max `|dh|` = 3.6e-03) while its lambda
was 0.68 off: smaller anisotropy makes that branch *more* attractive, so an amplitude sweep
maps the same failure rather than finding a threshold.

**The pins.**  `--pin-lam`, `--pin-h-tan`, `--pin-h-rr` (or `--pin-far` for all three) add
mean-square *value* differences against the exact reference at `rho_out`, weighted by
`--w-pin` (default 100, independent of `--w-outer`).  A value cannot be traded against a
derivative, so the branch is fixed.  `--pin-lam` acts on the spherical **MEAN** of lambda —
the monopole, which is where the branch lives — deliberately leaving the `l >= 1` content
free for the Robin conditions; `--pin-h-tan` pins the tangential metric angle by angle (the
areal-radius content) and `--pin-h-rr` the radial gauge component, separately, because the
inner data leaves `h_rr` free.

The pinned numbers are the reference's own values at `rho_out` (no constant is hard-coded;
`[pin]` in the log prints them):

```
lambda = 0.9885193     h_tan = 0.9999670     h_rr = 0.9999993
```

and the true quadrupole solution differs from them by `|S2| (rho_in/rho_out)^3 = 8.3e-08`,
i.e. ~800x below the order-1 Robin floor (6.6e-05) and ~30x below order 2's (7.4e-07).  So
pinning costs no accuracy; it is the same kind of known asymptotic data as `lambda_0` being
derived from `k = 1`.  At the quarter run's flat state the pin residual was 0.68, i.e. a
weighted cost of ~46 against ~1e-13 for the true solution, and a boundary layer cannot hide it
(a jump of 0.68 confined to a width `delta` costs ~`1/delta^3` in the lambda-equation loss
near `rho_out`: ~1e6 at `delta = 0.01`).

**Two stages.**  The pins exist to land on the right branch, not to be the final answer: once
the solution is converging, drop them (a warm-started `--steps 0` run from the pinned
`params.pkl`) so the order-3 Robin conditions determine the multipole content — the monopole's
`rho^-1, rho^-2, rho^-3` coefficients included — from the equations.  Stage 2 starts on the
right branch, which is a minimum of the unpinned loss, so the level has no reason to drift.

```bash
# stage 1: pinned, cold
JAX_ENABLE_X64=1 PY=$PWD/.venv/bin/python ./run_hub.sh \
  --arch axisym_hybrid --outdir runs/production_quad_quarter_pin \
  --steps 500 --lbfgs-steps 8000 \
  --R0 0.005773502691896258 --rho-in 0.01 --inner-radius 0.01 --rho-out 1.0 \
  --ref-solution --ref-asymptotic 1.0 \
  --outer-bc robin --robin-exps 2,3,1 --robin-orders h=3,lam=3 --no-robin-G --lam-inf 1.0 \
  --decay-feature --radial log --pde-ramp-steps 500 \
  --w-inner 100 --w-outer 100 --reweight-every 1500 \
  --n-coll 16384 --n-bnd 1024 --pin-far --w-pin 100 --lam-bc-S2 -0.08333333333333333

# stage 2: Robin only, warm (drop --pin-far; --steps 0 = no Adam, straight to SSBroyden)
JAX_ENABLE_X64=1 PY=$PWD/.venv/bin/python ./run_hub.sh \
  --arch axisym_hybrid --outdir runs/production_quad_quarter_robin \
  --steps 0 --lbfgs-steps 4000 --init-from runs/production_quad_quarter_pin/params.pkl \
  --R0 0.005773502691896258 --rho-in 0.01 --inner-radius 0.01 --rho-out 1.0 \
  --ref-solution --ref-asymptotic 1.0 \
  --outer-bc robin --robin-exps 2,3,1 --robin-orders h=3,lam=3 --no-robin-G --lam-inf 1.0 \
  --decay-feature --radial log --w-inner 100 --w-outer 100 \
  --n-coll 16384 --n-bnd 1024 --lam-bc-S2 -0.08333333333333333
```

What to look at: the per-block `[qn]` number is the *outer Robin + pins* combined (a run whose
boundary is moving is not converged), the report prints the achieved value differences next to
the reference's values, and `report.json` carries `pin_lam`, `pin_h_tan`, `pin_h_rr`.  The
`[pin]` line in the log states what is pinned and to what; the Config refuses a pin without a
reference (`--ref-solution`), and `tests/test_pins.py` checks the reference satisfies all three
to < 1e-20, that a flat `lambda = 0.31` costs 0.46, that a wrong metric tail is caught, and
that the lambda pin sees the monopole but not the quadrupole.

## 6e. The radial derivative of the lambda equation (`--w-lam-eq-radial`)

The other half of the same story as §6d.  The lambda-equation is *second order*, so a loss
built from it is blind to `lambda'''` — and `lambda'''` at `rho_out` is exactly where the
order-3 Robin condition lets a wrong level hide (its kernel is `rho^-1, rho^-2, rho^-3`, and
its residual is a cancellation of terms of order 0.1).  This term closes that hole from inside
the loss instead of by imposing a value:

```
w_lam_eq_radial * rho^3 * d/d rho [ box(lambda) - (1/lambda) |grad lambda|^2 ]
```

`rho^3` is the dimensionally consistent factor, taken from `scale_exps['lam_eq'] = 2` rather
than hard-coded (the residual is 1/length², its radial derivative 1/length³), which makes the
term invariant under the shell rescaling `rho -> rho/s` with the fields relabelled — checked
in `tests/test_pde_radial.py` to 1e-9.  It is ramped in with the other equation terms
(`--pde-ramp-steps`) and carries its own weight `--w-lam-eq-radial` (0, the default, drops the
term *and* its cost: nothing is differentiated).

Measured on the production geometry (4096 points, reference built with `--ref-asymptotic 1`):

| field | plain `lam_eq` | radial term | ratio |
|---|---|---|---|
| exact reference | 8.8e-32 | 1.2e-30 | — (a zero of both) |
| `lambda + a/rho`, a = 0.05 (Robin kernel mode) | 3.3e-01 | 4.93 | **15x** |
| `lambda + 0.1` (level shift) | 6.8e-04 | 9.1e-03 | **13x** |
| radial wiggle `1e-3 sin(32(rho-1))` | 2.8e-02 | 19.8 | 703x |
| radial wiggle `1e-5 sin(200(rho-1))` | 4.4e-03 | 110.0 | 2.5e4x |

So it is 13-15x more sensitive than the equation itself to the two deviations the Robin
condition cannot see, and its sensitivity grows with the radial wavenumber — hiding the
mismatch in higher radial derivatives gets more expensive, not cheaper.  Cost, production
network at 16384 points, value+gradient, CPU: the four equation groups 2.08 s, plus this term
2.44 s, i.e. **1.17x** (an earlier formulation that built the full Jacobian and contracted
with `n` cost 1.89x; `jax.jvp` along `n` is what makes the difference).

```bash
# the same quarter run, no pins, only the extra equation term
JAX_ENABLE_X64=1 PY=$PWD/.venv/bin/python ./run_hub.sh \
  --arch axisym_hybrid --outdir runs/production_quad_quarter_lamrad \
  --steps 500 --lbfgs-steps 8000 \
  --R0 0.005773502691896258 --rho-in 0.01 --inner-radius 0.01 --rho-out 1.0 \
  --ref-solution --ref-asymptotic 1.0 \
  --outer-bc robin --robin-exps 2,3,1 --robin-orders h=3,lam=3 --no-robin-G --lam-inf 1.0 \
  --decay-feature --radial log --pde-ramp-steps 500 \
  --w-inner 100 --w-outer 100 --reweight-every 1500 \
  --n-coll 16384 --n-bnd 1024 --w-lam-eq-radial 1 --lam-bc-S2 -0.08333333333333333
```

The `[adam]` line then carries `lam_rad=…` next to `lam=…`, and the final value lands in the
run's `report.json` as `pde_lam_eq_radial`.  `w_lam_eq_radial` is also folded into the
`lam_eq` group for the gradient-norm reweighting, so a reweighting run sees the whole gradient
of that equation.  A weight of 1 is the natural starting point: at the correct solution the
term is ~1e-30, so it costs the solution nothing, while at a hidden-level state it is an order
of magnitude larger than the term it corrects.

**Both diagnostics in one chain.**  `./run_quad_chain.sh` runs the pinned quarter problem and
then the radial-term quarter problem back to back (each launched through `run_hub.sh`, each
identical to §6d/§6e apart from its own mechanism, both cold), prints the decisive lines of
each `report.txt` as it finishes, and ends with one `stationary.compare` table over the three
quarter runs — the original failed one included.  Everything on its command line is appended
to both runs (`--vtk`, `--lbfgs-steps 16000`, …), `TAG=_trial` suffixes the run directories,
`FORCE=1` redoes a finished one, and `TAG=_trial ./run_quad_chain.sh --steps 3 --lbfgs-steps 1
--n-coll 128 --n-bnd 32 --width 8 --depth 2 --no-figures` is the ~6-minute plumbing trial.
Since it only waits, run it under `nohup`/`tmux` if the terminal may close; the runs
themselves are detached and survive regardless.

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

**Copy-paste the whole command, and do not hide the physics flags in shell arrays.** An
array defined in an earlier terminal, or in a different tab, is *empty* in this one;
`./run_hub.sh "${COMMON[@]}" ...` then expands to nothing, every physics flag is silently
missing, and the run becomes the **default milestone-1 configuration** (`R0 = 1`,
`lambda_0 = 1`, `rho_out = 20`, `dirichlet_exact`) — a valid run of something you did not
ask for. That is exactly what happened to `runs/control_ord1` on 18 Sept. The log now
starts with `== command ==` and the fully expanded command line, followed by the effective
configuration, so this takes two seconds to spot.

**(1) Control first — `S1 = S2 = 0`.** Output goes to `runs/<timestamp>` inside the
checkout (override with `OUTDIR=`; `run_hub.sh` never writes outside the project). This is
the **order-1** Robin variant we are testing at the moment. Note that `--lam0` is *not*
given: it is derived so that `lambda -> 1` (here `lambda_0 = 1/3`), and that `h_rr` is not
constrained by default either:

```bash
cd <checkout>
rm -rf runs/control_ord1                 # drop an earlier attempt, if any
PY=$PWD/.venv/bin/python ./run_hub.sh --arch axisym_hybrid --outdir runs/control_ord1 \
    --steps 20000 --lbfgs-steps 1000 --ref-solution --ref-asymptotic 1.0 \
    --R0 0.5773502691896258 \
    --rho-in 1.0 --inner-radius 1.0 --rho-out 100 \
    --outer-bc robin --robin-orders h=1,lam=1 --no-robin-G --lam-inf 1.0 \
    --decay-feature --radial log --pde-ramp-steps 500 \
    --w-inner 100 --w-outer 100 --reweight-every 1500 \
    --n-coll 4096 --n-bnd 256 --width 64 --depth 4 --fourier 8
```

For the GPU, change only the last line to
`--n-coll 16384 --n-bnd 1024 --width 256 --depth 6 --fourier 16`; for the fourth-order
Robin conditions change `--robin-orders h=1,lam=1` to `h=4,lam=4`.

**The ladder for the Robin order and the capacity.** It is one script, run in sequence on
the single GPU, and it ends by comparing everything it produced:

```bash
cd <checkout>
PY=$PWD/.venv/bin/python ./run_ladder.sh --dry-run all   # show the commands, launch nothing
PY=$PWD/.venv/bin/python ./run_ladder.sh --clean          # list what a clean would delete
PY=$PWD/.venv/bin/python ./run_ladder.sh --clean --force  # delete this ladder's runs only
PY=$PWD/.venv/bin/python ./run_ladder.sh sweep            # measure w_outer per order (~10 min)
W_OUTER_ORD2=10 W_OUTER_ORD4=1 ./run_ladder.sh 1 2 3   # the controls (NUMBERS, not prose)
PY=$PWD/.venv/bin/python ./run_ladder.sh 4 5 6 7    # capacity, then the dipoles
PY=$PWD/.venv/bin/python ./run_ladder.sh --compare  # just re-print the table
```

**Run the sweep first.**  One fixed `w_outer` is not a fair comparison across Robin
orders: the order-`n` residual at initialisation is `(gain)^n` larger (`8.8e-04` at order 1,
`2.1e-01` at order 2, `4.4e+03` at order 4), so `w_outer = 100` starves the equation at
order 2 and 4 and makes order 1 look better than it is.  `sweep` runs six short jobs
(5000 steps, `n_coll 1024`) — order 2 and order 4 (x64) at `w_outer = 100, 10, 1` — and
prints the weight-independent metrics.  Judge on **`lam_eq`/`ricci` and the outer BC
residual**, not on the loss: lowering `w_outer` shrinks the loss by itself.  Then pass the
chosen weights as `W_OUTER_ORD2` / `W_OUTER_ORD4` to the full-budget controls.

A step already done (its `runs/<name>/params.pkl` exists) is skipped unless you pass
`--force`; a step that fails to launch stops that step but not the ladder. The steps (see
§6 for the measurements behind them):

`PY` is optional for the ladder: without it the checkout's own `.venv` is used (which on
this hub is the CUDA environment) and checked by importing the package.  `W_OUTER_ORD2` /
`W_OUTER_ORD4` are optional **numbers** with defaults 10 and 1; anything that is not a
number is refused up front — writing `<from the sweep>` there makes bash try to read a file
by that name.

| step | run directory | what | expect | cost |
|---|---|---|---|---|
| 1 | `runs/control_ord1b` | order 1, float32, 64x4 f8, 3000 L-BFGS | λ(100) ≈ 0.9882, outer BC rms ~3e-05 | ~7 min |
| 2 | `runs/control_ord2` | order 2, float32, same size | floor 6.9e-07 is representable, so it should beat order 1 | ~7 min |
| 3 | `runs/control_ord4_x64` | order 4, **x64**, same size | the decisive test of the round-off argument (floor 1.2e-05 in float32 vs 8.3e-10 in float64) | ~15 min |
| 4 | `runs/control_ord1_big` | order 1, float32, `--n-coll 16384 --n-bnd 1024 --width 256 --depth 6 --fourier 16` | capacity: does the bigger net beat 3.0e-04? | 20-40 min |
| 5 | `runs/dipole_small` | the dipole (`--lam-bc-S1 0.1`), order 1, 64x4 f8 | first look at the physics | ~7 min |
| 6 | `runs/dipole_big` | the dipole, order 1, big net | the physics at capacity | 20-40 min |
| 7 | `runs/dipole_ord2_small` | the same dipole at order 2 with the swept weight | tests the claim that order 1 suffices: the `l = 1` amplitude and `lambda(rho_out)` should agree with step 5 to ~1e-05 | ~7 min |

Steps 2 and 3 are cheap and settle the order question before the big runs.  Note that a
bigger network does **not** lower the round-off floor and that in float32 the floor grows
as `2^4 = 16x` per doubling of `--fourier`, so "higher order *and* bigger network" only
makes sense in x64.  If step 4 dies with a memory error on a small GPU slice, retry it
with `LADDER_EXTRA="--n-coll 8192 --width 192 --depth 5"`.

The first screen of `logs/control_ord1.log` must show the physics you asked for — with
anything else, stop the run:

```
== command ==
env ... -m stationary.train --outdir runs/control_ord1 ... --rho-out 100 --lam-inf 1.0 ...
=============
========================================================================
effective configuration (every value below is a flag; check them)
  model       axisym_hybrid   64 x 4, fourier 8   (Gamma derived from h)
  exact data  R0 = 0.57735   lambda_0 = 0.3333333   S1 = 0   S2 = 0   ->  lambda -> k = 1
  domain      rho in [1, 100]   inner sphere areal radius 1   h_rr free
  outer BC    robin   lambda_inf = 1.0 (learnable)   orders {'h': 1, 'lam': 1} ...
[ref] exact reference: lambda(1) = 0.333333 (imposed 0.333333)   lambda(100) = 0.988519   lambda -> k = 1.000000
```

Check the run when it ends — **it does this for you** (§4): read
`runs/control_ord1/report.txt` and look at `runs/control_ord1/lambda_vs_rho.png`. To redo
them by hand, or to process a run that stopped early:

```bash
./run_hub.sh --post --outdir runs/control_ord1     # figures + lambda_vs_rho.png + report.txt
python -m stationary.evaluate --outdir runs/control_ord1   # prints max_dh vs the exact solution
python -m stationary.profile  --outdir runs/control_ord1   # lambda vs rho, exact overlay + table
```

`lambda` must reach **0.9885** at `rho = 100` and `max_dh` over the shell should be ~1e-4 or
smaller.

**(2) Then the dipole — `S1 = 0.1`:** the same command with two changes,
`--lam-bc-S1 0.1` and `--outdir runs/dipole`.

`--ref-solution --ref-asymptotic 1.0 --R0 1/sqrt(3)` is in the shared block so that the
dipole run also carries the *spherical* reference for comparison: its diagnostics then show
how far the dipole solution departs from the spherical one, and `stationary.profile` can
overlay it without warning. It does not enter the loss unless `--robin-source` is also
given.

**Cost:** each is one full run, so budget 2x a single run (on the CPU here ~1.5-2.5 h
each; measure ms/step on the GPU with the `--check` smoke run before launching 20000
steps). If GPU time is tight, run the control at the smaller size — it is a correctness
check, not a resolution study — and spend the big configuration on the dipole.

The analysis is produced by the run itself (figures + `report.txt`, see §4):
`lambda_vs_rho.png` (λ against ρ at several angles, exact solution overlaid),
`lambda_inner.png` (imposed `lambda` on the inner sphere vs the network),
`lambda_multipoles_outer.png` (the `l = 0, 1, 2` angular dependences at `rho_final = 100`,
with amplitudes), `lambda_multipole_decay.png` (amplitudes vs `rho` against the expected
`-(l+1)`), and `report.txt`/`report.json` with the residuals, boundary values and
multipole content.

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

**8.4 A text report to paste into a discussion.** Every run writes one to
`runs/<name>/report.txt` when it ends (add `--out FILE` to the command below to save it
yourself), and `python -m stationary.report --outdir runs/<name>` prints one screen with:
the code provenance (git HEAD, and whether the boundary-weight fix is present in
`train.py`), the full configuration (including `S1`,
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
| `run_hub.sh ...` / `postprocess.sh RUN` | 0 | writes the 5 files (profile included), displays nothing |

So calling `evaluate()` in a notebook explains a screenful of plots: `diagnostics.png`
(6 axes) plus `lambda_inner.png` (4), `lambda_multipoles_outer.png` (3) and
`lambda_multipole_decay.png` (2) are each drawn and displayed as they are created. Use
`show_run` for a single three-axis summary, or pass `--no-plots` to `evaluate`.

Every figure carries its provenance in the caption: the run directory, the architecture,
and the inner boundary data `lambda_0`, `S1`, `S2` -- so a PNG or a notebook cell can be
told apart from the next one. The plotting helpers here call `plt.close("all")` first, so
re-running a cell replaces its plot instead of stacking another one.

