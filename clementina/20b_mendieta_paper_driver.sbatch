#!/bin/bash
# ---------------------------------------------------------------------------
# THE PAPER CONFIGURATION, through OUR hand-driven driver with the exact-residual mode
# and float64 only: 60-60-60-60, 2^15/2^14 points, s=4000, 200 DSGNAR steps, then the
# two SSBroyden runs.  The control for the paper trainer run: if this plateaus near
# rel L2 0.54 (as it did on the Intel tile) while the paper trainer converges, the
# problem is the driver/recipe, not the backend; if both converge, the Intel stall was
# the backend.
#
#   cd ~/ldc_paper_run && sbatch clementina/20b_mendieta_paper_driver.sbatch
# ---------------------------------------------------------------------------
#SBATCH -J ldc_paper_drv
#SBATCH -p multi
#SBATCH -A ifeg
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=28G
#SBATCH -t 02:00:00
#SBATCH -o logs/mendieta_paper_drv_%j.out
#SBATCH -e logs/mendieta_paper_drv_%j.err

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
export MPLCONFIGDIR="$HOME/.cache/mpl"
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax_ldc"
mkdir -p "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR"
OUT="$REPO/runs/paper_driver_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
echo "host $(hostname) | job ${SLURM_JOB_ID:-?} | $(date -u '+%F %T UTC') | HEAD $(git rev-parse --short HEAD)"
echo "outdir $OUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$PY" - <<'PYEOF' || { echo "FATAL: jax does not see a GPU" >&2; exit 2; }
import jax
print("[mend] jax", jax.__version__, [str(d) for d in jax.devices()], jax.default_backend())
assert jax.default_backend() == "gpu", jax.devices()
PYEOF
"$PY" -u clementina/ldc_run.py --mode paper --exact-residual --float32-steps 0 --batch 1024 \
      --skip-fields --outdir "$OUT" --label paper_driver_a30
E=$?
echo "driver exit $E | $(date -u '+%F %T UTC')"
MSG=$(cd "$OUT" && "$PY" - <<'PYEOF'
import json, pathlib
try:
    s = json.loads(pathlib.Path("summary.json").read_text())
    d = s.get("dsgnar", {}); g = s.get("grades_rel_l2", {}); rl = d.get("rel_l2", {})
    best = s.get("best_by_accuracy")
    print(f"DRIVER, paper config, exact residual: DSGNAR {d.get('steps')} steps, "
          f"loss {d.get('loss', float('nan')):.3e}, relL2 {rl.get('mean', float('nan')):.6f} "
          f"({d.get('seconds', 0)/60:.1f} min); best {best} relL2 "
          f"{g.get(best, float('nan')):.6f}")
except Exception as exc:
    print(f"DRIVER: no summary ({type(exc).__name__}: {exc})")
PYEOF
)
echo "$MSG"
curl -s -m 20 -H 'Title: Mendieta: paper config (exact-residual driver)' -d "$MSG" "$NTFY" >/dev/null || true
exit $E
