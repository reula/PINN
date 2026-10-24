#!/bin/bash
# ---------------------------------------------------------------------------
# THE PAPER CONFIGURATION through the PAPER'S OWN TRAINER, with the reference-grid
# evaluation split across BOTH A30s of a Mendieta node (the reason it OOMed on one).
#
# The vendored trainer's DSGNAR step already fits (it builds the Jacobian per macro-batch:
# 1.04 GiB at the paper batch).  What did not fit is `predict_to_current_slab` ->
# `_eval_model_jit`: on the GPU backend XLA's lowering of the vmap'd SIREN materialises
# per-lane weight products and the 16641-point reference grid needs ~27 GiB.  The vendored
# trainer is patched (LOCAL PATCH in trainer64.py/trainer32.py) to evaluate the grid in
# PINN_EVAL_CHUNK-point batches and dispatch successive chunks to successive local devices,
# so the same numbers are produced with a per-device peak of order
# 27 GiB * chunk / 16641 and the two chunks in flight overlap.
#
# Paper configuration kept exactly: 60-60-60-60, 2^15/2^14 points, s=4000, probe 256,
# batch 2^12, float32 pre-pass 10, 200 float64 DSGNAR steps.  Memory fraction 0.9 (the card
# is 24 GiB; JAX's default 0.75 is not the hardware).
#
#   git -C ~/Julia/PINN worktree add ~/ldc_paper_run origin/main    # once
#   cd ~/ldc_paper_run && sbatch clementina/22_mendieta_paper_trainer_parallel.sbatch
# ---------------------------------------------------------------------------
#SBATCH -J ldc_paper_par
#SBATCH -p multi
#SBATCH -A ifeg
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o logs/mendieta_paper_par_%j.out
#SBATCH -e logs/mendieta_paper_par_%j.err

set -uo pipefail
REPO="${LDC_REPO:-$HOME/ldc_paper_run}"
cd "$REPO" || { echo "FATAL: no $REPO" >&2; exit 1; }
mkdir -p logs runs
# A worktree has no untracked .venv: point PY at the CUDA interpreter.
export PY="${LDC_PY:-$HOME/Julia/PINN/Stationary/.venv/bin/python}"
source Evolution_try/scripts/env.sh
PY="$(find_python)" || { echo "FATAL: no interpreter with jax" >&2; exit 1; }
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export PINN_EVAL_CHUNK="${PINN_EVAL_CHUNK:-1024}"
export PINN_EVAL_DEVICE_PARALLEL="${PINN_EVAL_DEVICE_PARALLEL:-1}"
export MPLCONFIGDIR="$HOME/.cache/mpl"
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax_ldc"
mkdir -p "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR"
OUT="$REPO/runs/paper_trainer_par_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"

echo "host $(hostname) | job ${SLURM_JOB_ID:-?} | $(date -u '+%F %T UTC') | HEAD $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "outdir $OUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── preflight: interpreter, two GPUs, eval-chunk memory budget, tiny jit ─────────────
"$PY" - <<'PYEOF' || { echo "FATAL: preflight failed" >&2; exit 2; }
import os, subprocess
import jax, jax.numpy as jnp
import equinox, jaxtyping, optax, scipy        # noqa: F401  (the trainer's stack)
devs = jax.devices()
print("[pre] jax", jax.__version__, "| backend", jax.default_backend(),
      "| devices", [str(d) for d in devs], flush=True)
assert jax.default_backend() == "gpu", "no GPU visible"
assert len(devs) >= 2, f"this run wants 2 devices, sees {devs}"
st = devs[0].memory_stats() or {}
lim = int(st.get("bytes_limit") or st.get("bytes_reservable_limit") or 0)
frac = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
chunk = int(os.environ.get("PINN_EVAL_CHUNK", "1024"))
n_eval = 16641
# the unpatched single-shot evaluation needs ~27 GiB at 16641 points (measured on this
# backend); a chunk scales that roughly linearly
eval_peak = 27.0 * chunk / n_eval
jac = int(os.environ.get("LDC_BATCH", "4096")) * 3 * 11347 * 8 / 2**30
gpu_total = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
    capture_output=True, text=True).stdout.split()[0]
print(f"[pre] nvidia-smi total {gpu_total} MiB | fraction {frac} -> "
      f"bytes_limit {lim/2**30:.1f} GiB", flush=True)
print(f"[pre] dense Jacobian (batch 4096) {jac:.2f} GiB | eval chunk {chunk} of {n_eval} points "
      f"~{eval_peak:.2f} GiB/device | 4x headroom {(4*jac + 2*eval_peak):.1f} GiB", flush=True)
assert lim == 0 or (4 * jac + 2 * eval_peak) < lim, "does not fit: lower PINN_EVAL_CHUNK/--batch"
x = jnp.ones((8, 2)); y = jax.jit(lambda z: z @ z.T)(x); y.block_until_ready()
print("[pre] tiny jit OK", flush=True)
PYEOF

# The paper's own batch is 2^12; LDC_BATCH lets the same script drop to 2048/1024 if the
# DSGNAR step (not the evaluation) turns out not to fit a 21.2 GiB budget.
WARM_ARGS=""
[ -n "${LDC_WARM_START:-}" ] && WARM_ARGS="--warm-start ${LDC_WARM_START}"
echo "warm start: ${LDC_WARM_START:-<none, random init>}"
"$PY" -u clementina/ldc_paper_run.py --mode paper --batch "${LDC_BATCH:-4096}" \
      ${WARM_ARGS} --float32-steps "${LDC_F32:-10}" --skip-fields \
      --outdir "$OUT" --label paper_trainer_parallel_a30
E=$?
echo "trainer exit $E | $(date -u '+%F %T UTC')"
(cd "$OUT" && "$PY" - <<'PYEOF'
import json, pathlib
try:
    s = json.loads(pathlib.Path("summary.json").read_text())
    m = s.get("final_rel_l2_slab", {})
    print(f"PAPER TRAINER (parallel eval, paper config): relL2 mean "
          f"{s.get('final_rel_l2_slab_mean', float('nan')):.6f} "
          f"(u {m.get('u', float('nan')):.5f} v {m.get('v', float('nan')):.5f} "
          f"p {m.get('p', float('nan')):.5f}), {s.get('wall_seconds', 0)/60:.1f} min")
except Exception as exc:
    print(f"PAPER TRAINER: no summary ({type(exc).__name__}: {exc})")
PYEOF
)
exit $E
