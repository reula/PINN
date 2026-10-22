#!/bin/bash
# ---------------------------------------------------------------------------
# THE PAPER CONFIGURATION on Mendieta, one run per A30 (two whole cards on the node,
# 64 GB host RAM, whole 24 GB per card -- the memory problem was the hub's SHARED
# 19.3 GiB HAMi slice, not an A30).
#
#   GPU 0: the vendored paper's own trainer at the paper configuration
#          (pinn.train64: 60-60-60-60, 2^15/2^14, s=4000, batch 2^12, float32 pass
#          then float64) -- this is the pipeline whose 6.37e-2 on the Intel tile is
#          the closest anyone has come to the paper's result.
#   GPU 1: our hand-driven driver at the same paper configuration but with the
#          exact-residual mode and float64 only -- the control that tells us whether
#          the rel L2 ~0.54 plateau is the configuration or the Intel backend.
#
# Run from a checkout of origin/main (so the other agent's checkout/job is untouched):
#   git -C ~/Julia/PINN worktree add ~/ldc_paper_run origin/main
#   cd ~/ldc_paper_run && sbatch clementina/20_mendieta_paper_2gpu.sbatch
# ---------------------------------------------------------------------------
#SBATCH -J ldc_paper2
#SBATCH -p multi
#SBATCH -A ifeg
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH -t 02:30:00
#SBATCH -o logs/mendieta_paper2_%j.out
#SBATCH -e logs/mendieta_paper2_%j.err

set -uo pipefail
REPO="${LDC_REPO:-$HOME/ldc_paper_run}"
NTFY="${NTFY_URL:-https://ntfy.sh/dipsiq_delamac}"
cd "$REPO" || { echo "FATAL: no $REPO" >&2; exit 1; }
mkdir -p logs runs
source Evolution_try/scripts/env.sh
PY="$(find_python)" || { echo "FATAL: no interpreter with jax" >&2; exit 1; }
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR="$HOME/.cache/mpl"
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax_ldc"
mkdir -p "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR"

RUNS="$REPO/runs/paper_2gpu_${SLURM_JOB_ID:-local}"
mkdir -p "$RUNS"
echo "host $(hostname) | job ${SLURM_JOB_ID:-?} | $(date -u '+%F %T UTC')"
echo "repo $REPO | HEAD $(git rev-parse --short HEAD) | outdirs $RUNS/{paper_trainer,paper_driver}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$PY" - <<'PYEOF' || { echo "FATAL: jax does not see two GPUs" >&2; exit 2; }
import jax
print("[mend] jax", jax.__version__, [str(d) for d in jax.devices()], jax.default_backend())
assert jax.default_backend() == "gpu" and len(jax.devices()) >= 2, jax.devices()
PYEOF

echo "=== GPU 0: the paper's own trainer, paper configuration ==="
CUDA_VISIBLE_DEVICES=0 "$PY" -u clementina/ldc_paper_run.py --mode paper --skip-fields \
    --outdir "$RUNS/paper_trainer" --label paper_trainer_a30 \
    > "$RUNS/paper_trainer.log" 2>&1 &
P0=$!
echo "=== GPU 1: hand-driven driver, paper configuration, exact residual, f64 only ==="
CUDA_VISIBLE_DEVICES=1 "$PY" -u clementina/ldc_run.py --mode paper --exact-residual \
    --float32-steps 0 --batch 1024 --skip-fields \
    --outdir "$RUNS/paper_driver" --label paper_driver_a30 \
    > "$RUNS/paper_driver.log" 2>&1 &
P1=$!
wait $P0; E0=$?
wait $P1; E1=$?
echo "trainer exit $E0 | driver exit $E1 | $(date -u '+%F %T UTC')"

MSG=$(cd "$RUNS" && "$PY" - <<'PYEOF'
import json, pathlib
lines = []
try:
    s = json.loads(pathlib.Path("paper_trainer/summary.json").read_text())
    m = s.get("final_rel_l2_slab", {})
    lines.append("PAPER TRAINER (pinn.train64, paper config): relL2 mean "
                 f"{s.get('final_rel_l2_slab_mean', float('nan')):.6f} "
                 f"(u {m.get('u', float('nan')):.5f} v {m.get('v', float('nan')):.5f} "
                 f"p {m.get('p', float('nan')):.5f}), {s.get('wall_seconds', 0)/60:.1f} min")
except Exception as exc:
    lines.append(f"PAPER TRAINER: no summary ({type(exc).__name__})")
try:
    s = json.loads(pathlib.Path("paper_driver/summary.json").read_text())
    d = s.get("dsgnar", {}); g = s.get("grades_rel_l2", {})
    rl = d.get("rel_l2", {})
    lines.append("DRIVER exact-residual paper config: DSGNAR "
                 f"{d.get('steps')} steps, loss {d.get('loss', float('nan')):.3e}, "
                 f"relL2 {rl.get('mean', float('nan')):.6f}; best "
                 f"{s.get('best_by_accuracy')} relL2 {g.get(s.get('best_by_accuracy', ''), float('nan')):.6f}")
except Exception as exc:
    lines.append(f"DRIVER: no summary ({type(exc).__name__})")
print("\n".join(lines)[:1000])
PYEOF
)
echo "$MSG"
curl -s -m 20 -H 'Title: Mendieta: paper configuration, 2 x A30' -d "$MSG" "$NTFY" >/dev/null || true
echo "=== done ==="
