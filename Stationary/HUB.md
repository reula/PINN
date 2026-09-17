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

A 20 000-step Adam phase takes **~1.5–2 h** here (`runs/m2R4_realrobin` = 7 435 s,
`runs/n1_control` = 5 419 s), so this matters: never launch one from a notebook.

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
* writes results to `$HOME/runs/<timestamp>` (persistent storage) and the log to
  `$HOME/logs/`,
* checkpoints every **500** Adam steps (`ckpt.pkl`, ~165 KB, overwritten), and
* writes **`<outdir>/resume.sh`** — the exact command to continue that run.

Useful overrides (environment variables):

| variable | default | meaning |
|---|---|---|
| `OUTDIR` | `$HOME/runs/<timestamp>` | output directory; keep it on `$HOME` |
| `CKPT_EVERY` | `500` | checkpoint period in Adam steps (`0` disables) |
| `TRAIN_THREADS` | `4` | caps CPU threads — be a good neighbour on a shared node |
| `LOGDIR` | `$HOME/logs` | where the log goes |
| `PY` | `python` | interpreter if your venv is not activated |

The script warns if `OUTDIR` is under `/tmp` (usually wiped with the container) or
if `JAX_ENABLE_X64` is set (see gotchas).

## 4. Monitor, resume, analyse

```bash
tail -f ~/logs/<name>.log                                  # progress
kill -0 $(cat ~/runs/<name>/run.pid) && echo running       # is it alive?
~/runs/<name>/resume.sh                                    # continue from the last checkpoint
python -m stationary.evaluate --outdir ~/runs/<name>       # figures/diagnostics afterwards
python -m stationary.train --outdir ~/runs/<name> --resume auto --steps 20000 ...   # what resume.sh runs
```

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
* **GPU sizing.** The network is small — 13 828 parameters (width 64 × depth 4),
  `n_coll` 4096, a 165 KB checkpoint — so a GPU gives a modest speedup, not a
  dramatic one: the L-BFGS line search is inherently sequential. Raising
  `--n-coll` is the lever that actually feeds a GPU. One GPU is plenty — do not
  request several.
* **Commit and push before you clone.** The current working tree has uncommitted
  work (including `stationary/multipoles.py`, which `train.py` imports for
  figures, as a brand-new untracked file). A clean clone at the current `HEAD`
  gets **stale code** and will fail on figures. Push first, or `rsync` the tree.
* **`runs/` is tracked on purpose** — the run outputs are committed as reference
  results, so `.gitignore` deliberately does not exclude them. Large `ckpt.pkl`
  files inside them *are* ignored.

## 6. Reference numbers

Measured on the 8-core macOS CPU machine, `n_coll=4096`, width 64, depth 4:

| run | steps (Adam + LBFGS) | wall time |
|---|---|---|
| `n2_dipole` | 10 000 + 1 000 | 4 905 s |
| `n1_control` | 12 000 + 1 000 | 5 419 s |
| `m2R4_realrobin` | 20 000 + 2 000 | 7 435 s |

with **13 828 parameters** (55 KB of weights, ~165 KB per checkpoint). This is a
small job: request modest CPU/RAM rather than many cores or several GPUs.
