#!/bin/bash
# ---------------------------------------------------------------------------
# THE PAPER CONFIGURATION, through the PAPER'S OWN TRAINER (vendored pinn.train64):
# 60-60-60-60, 2^15/2^14 points, s=4000, batch 2^12, float32 pass (10) then float64,
# 200 DSGNAR steps.  This pipeline reached rel error 6.37e-2 on the Intel tile; here it
# runs on a WHOLE A30 (24 GB, not the hub's shared 19.3 GiB slice), which is the
# measurement that settles whether the paper configuration fits a single card.
#
#   git -C ~/Julia/PINN worktree add ~/ldc_paper_run origin/main   # once
#   cd ~/ldc_paper_run && sbatch clementina/20a_mendieta_paper_trainer.sbatch
# ---------------------------------------------------------------------------
#SBATCH -J ldc_paper_tr
#SBATCH -p multi
#SBATCH -A ifeg
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/mendieta_paper_tr_%j.out
#SBATCH -e logs/mendieta_paper_tr_%j.err

set -uo pipefail
REPO="${LDC_REPO:-$HOME/ldc_paper_run}"
NTFY="${NTFY_URL:-https://ntfy.sh/dipsiq_delamac}"
cd "$REPO" || { echo "FATAL: no $REPO" >&2; exit 1; }
mkdir -p logs runs
# A git worktree has no untracked .venv, so find_python()'s "Stationary venv next
# door" candidate is missing there: point PY at the known CUDA interpreter first
# (find_python returns $PY when it imports jax/numpy/scipy/optax).
export PY="${LDC_PY:-$HOME/Julia/PINN/Stationary/.venv/bin/python}"
source Evolution_try/scripts/env.sh
PY="$(find_python)" || { echo "FATAL: no interpreter with jax" >&2; exit 1; }
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# nvidia-smi reports the whole card (24576 MiB); JAX's default MEM_FRACTION=0.75
# caps bytes_limit at ~18 GiB, which is NOT the hardware.  Use ~90% of the card.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export MPLCONFIGDIR="$HOME/.cache/mpl"
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax_ldc"
mkdir -p "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR"
OUT="$REPO/runs/paper_trainer_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
echo "host $(hostname) | job ${SLURM_JOB_ID:-?} | $(date -u '+%F %T UTC') | HEAD $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "outdir $OUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
preflight() {  # LDC_N_PARAMS / LDC_BATCH in the environment
  "$PY" - <<'PYEOF'
import os, jax, jax.numpy as jnp
import subprocess
try:
    gpu_total_mib = int(subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15).stdout.split()[0])
except Exception:
    gpu_total_mib = -1
n = int(os.environ.get("LDC_N_PARAMS", "0") or 0)
b = int(os.environ.get("LDC_BATCH", "0") or 0)
print("[pre] jax", jax.__version__, "| backend", jax.default_backend(),
      "| devices", [str(d) for d in jax.devices()], flush=True)
assert jax.default_backend() == "gpu", "jax does not see a GPU"
dev = jax.devices()[0]
st = dev.memory_stats() or {}
lim = int(st.get("bytes_limit") or st.get("bytes_reservable_limit") or 0)
frac = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
est = b * 3 * n * 8                      # dense Jacobian per macro-batch
print(f"[pre] nvidia-smi total {gpu_total_mib} MiB | JAX mem fraction {frac} -> "
      f"bytes_limit {lim/2**30:.1f} GiB | dense Jacobian at batch {b}: "
      f"{est/2**30:.2f} GiB | x4 headroom = {4*est/2**30:.2f} GiB", flush=True)
x = jnp.ones((8, 2)); y = jax.jit(lambda z: z @ z.T)(x); y.block_until_ready()
print("[pre] tiny jit OK", flush=True)
assert lim == 0 or 4 * est < lim, (
    f"config does not fit: dense Jacobian {est/2**30:.2f} GiB, device {lim/2**30:.1f} GiB; "
    "reduce --batch")
PYEOF
}

LDC_N_PARAMS=11347 LDC_BATCH=4096 preflight || { echo "FATAL: preflight failed" >&2; exit 2; }
# The paper's own batch (2^12) is kept: the earlier failure was JAX's 0.75 memory
# fraction, not the card (24 GiB), and the fraction is now 0.9.  If the preflight or
# the run still refuses to plan, drop to --batch 2048/1024 (the batch only chunks the
# per-macro-batch Jacobian; in the exact-residual driver it is mathematically free).
"$PY" -u clementina/ldc_paper_run.py --mode paper --batch 4096 --skip-fields \
      --outdir "$OUT" --label paper_trainer_a30
E=$?
echo "trainer exit $E | $(date -u '+%F %T UTC')"
MSG=$(cd "$OUT" && "$PY" - <<'PYEOF'
import json, pathlib
try:
    s = json.loads(pathlib.Path("summary.json").read_text())
    m = s.get("final_rel_l2_slab", {})
    print(f"PAPER TRAINER, paper config, whole A30: relL2 mean {s.get('final_rel_l2_slab_mean', float('nan')):.6f} "
          f"(u {m.get('u', float('nan')):.5f} v {m.get('v', float('nan')):.5f} p {m.get('p', float('nan')):.5f}), "
          f"{s.get('wall_seconds', 0)/60:.1f} min")
except Exception as exc:
    print(f"PAPER TRAINER: no summary ({type(exc).__name__}: {exc})")
PYEOF
)
echo "$MSG"
curl -s -m 20 -H 'Title: Mendieta: paper config (paper trainer)' -d "$MSG" "$NTFY" >/dev/null || true
exit $E
