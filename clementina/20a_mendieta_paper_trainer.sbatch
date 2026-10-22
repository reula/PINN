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
source Evolution_try/scripts/env.sh
PY="$(find_python)" || { echo "FATAL: no interpreter with jax" >&2; exit 1; }
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR="$HOME/.cache/mpl"
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax_ldc"
mkdir -p "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR"
OUT="$REPO/runs/paper_trainer_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
echo "host $(hostname) | job ${SLURM_JOB_ID:-?} | $(date -u '+%F %T UTC') | HEAD $(git rev-parse --short HEAD)"
echo "outdir $OUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$PY" - <<'PYEOF' || { echo "FATAL: jax does not see a GPU" >&2; exit 2; }
import jax
print("[mend] jax", jax.__version__, [str(d) for d in jax.devices()], jax.default_backend())
assert jax.default_backend() == "gpu", jax.devices()
PYEOF
"$PY" -u clementina/ldc_paper_run.py --mode paper --skip-fields --outdir "$OUT" --label paper_trainer_a30
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
