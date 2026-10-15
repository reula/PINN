# Mendieta (CCAD-UNC): the SLURM cluster with A30s

Mendieta is a real batch system: a job submitted with `sbatch` runs to completion whether or not
anyone is logged in, unlike the JupyterHub server (Stationary/HUB.md), where the container is torn
down roughly an hour after the browser closes and takes every process with it.  Use the hub for
interactive work; use Mendieta for anything long, and for running several jobs at once.

Everything below was checked on 2026-10-15 unless marked otherwise.

## 1. Access

    ssh mendieta.ccad.unc.edu.ar          # user reula, key-based; account (SLURM) ifeg

`mendieta.unc.edu.ar` (the old name, and the `mendieta` alias in `~/.ssh/config`) does **not**
resolve; use the `ccad` name.  Help: <https://wiki.ccad.unc.edu.ar/empezar/conexion.html>.

## 2. Hardware and partitions

| | |
|---|---|
| GPU nodes | `ivb[02-20]`: 19 nodes, **2 GPUs each** (`gres gpu:2`), 20 CPUs, 64 GB RAM |
| NVIDIA driver | 535.230.02 (CUDA 12.2) -- enough for `jax[cuda12]` |
| OS | Rocky Linux 8.10, glibc 2.28 (manylinux_2_27 wheels install) |
| partitions | `multi`: 2 days limit;  `short` (default): **1 hour** limit |
| account | `ifeg` (pass `-A ifeg`) |

The queue is shared and can be full: at 14:10 UTC on 2026-10-15 `multi` had 152 jobs (29 running)
and a 5-minute `short` test job waited with `PENDING (Resources)` for over 10 minutes.  Submit and
walk away; do not wait at the terminal.

## 3. Filesystem

Mendieta's `/home` is its **own** NAS (`nas-ivb-ib:/export/ivb_home`, 333 TB), **not** the hub's.
The tree is

    /home/reula/Julia/PINN            # git clone of github.com/reula/PINN
    /home/reula/Julia/PINN/Stationary/.venv     # the Python environment (section 4)

so results have to be copied back (`scp`/`rsync` from the Mac or from serafin), and code changes
reach Mendieta by `git pull` there, as on the hub.  The login node has internet access (GitHub and
PyPI both reachable), so cloning and `pip` work directly.

## 4. Python environment

The system Python is 3.6.8, far too old.  The environment is a `uv`-managed Python 3.13 venv with
**the same package versions as the hub** (read from the hub's `.venv`):

    curl -LsSf https://astral.sh/uv/install.sh | sh          # installs ~/.local/bin/uv (once)
    cd ~/Julia/PINN/Stationary
    ~/.local/bin/uv venv --python 3.13 .venv
    ~/.local/bin/uv pip install --python .venv/bin/python \
        "jax[cuda12]==0.11.1" flax==0.12.9 optax==0.2.8 numpy==2.5.3 scipy==1.18.1 \
        matplotlib==3.11.2 pytest==9.1.1

Done on 2026-10-15 (uv 0.13.1, Python 3.13.16).  On the **login node** `import jax` reports
`CUDA_ERROR_NO_DEVICE` and falls back to the CPU -- expected, the login node has no GPU.

Evolution_try needs no environment of its own: `Evolution_try/scripts/env.sh` falls back to
`../Stationary/.venv/bin/python`.  Stationary's SSBroyden finds Crunch inside the checkout.

**GPU check: pending.**  Job 1180327 (`short`, 1 GPU, 5 min) runs `nvidia-smi` and a float64 JAX
matmul on the GPU; its output goes to `Stationary/logs/slurm_jaxtest_<jobid>.out`.  Until it has
run, "JAX sees the A30" is expected from the driver version, not verified.

## 5. Submitting a job

A minimal single-GPU job (one run of `stationary.train`):

    #!/bin/bash
    #SBATCH -p multi
    #SBATCH -A ifeg
    #SBATCH --gres=gpu:1
    #SBATCH -c 8
    #SBATCH --mem=28G
    #SBATCH -t 12:00:00
    #SBATCH -J <name>
    #SBATCH -o logs/slurm_%x_%j.out
    cd ~/Julia/PINN/Stationary
    export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
    export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax
    .venv/bin/python -m stationary.train --outdir runs/<name> <flags...>

Half a node per job (`--gres=gpu:1 -c 8 --mem=28G`) lets two jobs share a node.  `-t` must fit the
partition (`short` 1 h, `multi` 2 days); a run that may exceed it should checkpoint
(`--ckpt-every`) so it can be resumed with `--resume auto`.

Useful commands:

    sbatch job.sh                 # submit
    squeue -u $USER               # your jobs (PD = pending, R = running)
    squeue -j <id> -o "%T %R"     # why it waits (Resources, Priority, ...)
    scancel <id>                  # cancel
    sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode
    sinfo -p multi                # node states

Dependencies chain jobs without waiting at the terminal, e.g. the optimiser arms after the Adam
warm-up: `sbatch --dependency=afterok:<adam_jobid> arm.sh`.

## 6. Differences from the hub

| | JupyterHub (HUB.md) | Mendieta |
|---|---|---|
| survives the browser closing | no (~1 h idle cull kills the container) | **yes** |
| GPUs | one A30, 13.5 GiB share | 19 nodes x 2 A30 |
| parallel runs | no (two jobs on the share OOM) | yes, one job per GPU |
| home | `~/serafin/Julia/PINN` (= serafin `~/Julia/PINN`) | `/home/reula/Julia/PINN`, separate |
| start | immediate | queued, can wait |
| who launches | the user, from a notebook cell | `sbatch`; Claude can submit and monitor over ssh |
