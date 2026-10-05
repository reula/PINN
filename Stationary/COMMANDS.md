# Commands that work, and the traps that made them fail

Every entry here cost a run or an hour. Nothing in this file is a preference.

## 1. `PY` defaults to the WRONG python

```bash
PY="${PY:-python}"        # run_hub.sh:37
```

`python` on this host is the base conda interpreter, which has **no jax**. Any `run_hub.sh`
invocation that does not set `PY` fails with `ModuleNotFoundError: No module named 'jax'` — in
training *and* in `--post`. Always:

```bash
PY=$PWD/.venv/bin/python ./run_hub.sh ...
```

## 2. `git pull` first

New flags (`--pin-h-robin`, `--h-robin-bases`, `--log-resources`) do not exist until the
checkout has them, and the failure is a bare
`train.py: error: unrecognized arguments: ...` followed by `status 2`. If `git pull` reports
`Unable to create '.git/index.lock'`, the lock is stale when `ls -l .git/index.lock` shows an
old timestamp and `ps -ef | grep '[g]it'` shows nothing — `rm -f` it and retry.

## 3. Training environment

GPU (the fast path, ~0.09 s per iteration on this problem):

```bash
cd ~/serafin/Julia/PINN/Stationary
export JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache-$USER
nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader   # want 12288 MiB, not 752

JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PY=$PWD/.venv/bin/python ./run_hub.sh --outdir runs/<name> --qn-block 250 ...
```

CPU (a fallback, ~22x slower, ~2 s per iteration at 16 threads):

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=16 \
  PY=$PWD/.venv/bin/python ./run_hub.sh --outdir runs/<name> --qn-block 1500 ...
```

**Never set `XLA_PYTHON_CLIENT_MEM_FRACTION`.** Raising it was recommended and retracted twice;
each failure it was blamed for belonged to something else.

## 4. `--resume` TAKES A VALUE

```bash
--resume auto          # <outdir>/ckpt.pkl
--resume <path>        # an explicit checkpoint
--init-from <path>     # a bare parameter tree (params.pkl), no optimiser state
```

A bare `--resume` is an argparse error (`expected one argument`, `status 2`). Since the
quasi-Newton phase checkpoints itself, `--resume auto` re-enters that phase with its iteration
count and Hessian; `--init-from` restarts the optimiser from a field.

## 5. `--qn-block` is bounded by the backend, differently

Every quasi-Newton block recompiles `minimize_bfgs`'s line search and the objective inside it
(Crunch is not jitted; it is not our file).

| backend | cost per block | bound | use |
|---|---|---|---|
| CPU | **+2534 mapped regions**, RSS flat | 65530 total, ~25 blocks | `--qn-block 1500` |
| GPU | **+1 mapping**, **+281 MB RSS** | ~1300 blocks of RSS | `--qn-block 250` |

The phase now measures this on its first block and warns with the block it expects to die at,
and the `--qn-block` that would fit. `--log-resources` adds `traces= maps= rss=` per block.

## 6. Post-processing: CPU, and `PY`

```bash
JAX_PLATFORMS=cpu POST_THREADS=16 PY=$PWD/.venv/bin/python \
  ./run_hub.sh --post --outdir runs/<name>
```

Runs `evaluate`, `profile`, `report`, `vtk`, `plane`. `--only report` subsets it. **CPU, not
GPU**: the same five steps took 236 s on CPU against ~17 min on the GPU, because they are
reference-quadrature work rather than compiled graphs. If it dies on
`<repo>/.jaxcache` with `Input/output error`, add `JAX_CACHE=0`.

The training run writes `lambda_inner.png`, `lambda_multipole_decay.png`,
`lambda_multipoles_outer.png` and `report.json` itself, **before** any post-processing — so the
multipole figures do not need `--post`, and `report.json` existing is the proof that the run
reached the end of `train()` rather than being killed.

## 7. Operational traps

- **One run at a time.** Two trainers on one slice produced every "OOM" in this project.
- **A fresh `--outdir` name**, not `rm -rf`: that filesystem fails with
  `Directory not empty` and leaves stale `params.pkl` behind, which post-processing then reads.
- **`kill -9`, and stop the LOOP's shell, not the trainer.** Interactive bash ignores
  SIGTERM, so `kill <loop shell>` silently does nothing while the trainer it spawned dies and
  is relaunched — which looks exactly like "it keeps restarting". The loop's shell is the
  trainer's *ancestor*, so kill the chain outside-in. Killing the trainer alone always
  relaunches it.
- **`maps`/`rss`/`traces` come from `/proc`**, so they print on Linux and are silently absent
  elsewhere.

## Units and the chart convention

The problem is scale-invariant at a fixed shell RATIO.  The network's inputs
(`t = log(rho/rho_in)/log(rho_out/rho_in)` and `n = x/rho`), the loss, the log-uniform sampling and
the Robin conditions all depend on the ratio alone, so a run in a rescaled chart and a run in
physical units are the *same computation*.  There is no numerical reason to prefer either, and no
need to carry two conventions.

**Convention: the longer scale is the physical one.**  For ratio 200 that is rho in [1, 200], and a
run should use

    --rho-in 1.0 --rho-out 200.0 --inner-radius 1.0 --R0 0.5773502691896258
    --vtk-physical-inner 1.0        (or omit it: chart and physical coincide)

R0 is a length, so it scales with the chart: R0 = 0.5773502691896258 is 1/sqrt(3) at rho_in = 1.

Runs under `runs/` made before this convention used the SHORT scale as physical: `pq_r400_hrobin`
at [0.01, 4], and `pq_u100_s2_12` / `pq_u200b_s2_12` whose charts [1, 100] / [1, 200] stood for
physical [0.01, 1] / [0.01, 2].  To read one of those in the new convention multiply every length
by 100 -- the fields, the multipoles and the residuals are unchanged, only the axes move.

The one thing that must agree is `--vtk-physical-inner` with `--rho-in`: equal when the chart IS
physical, the physical value otherwise.  Getting it wrong mislabels the VTK and plane axes and
nothing else, which is what makes it easy to miss.
