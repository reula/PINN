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
