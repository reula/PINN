#!/bin/bash
# ---------------------------------------------------------------------------
# One windowed T = 20 Evolution_try run on Mendieta (SLURM, one A30).  See ../../Mendieta.md.
#
#   sbatch scripts/mendieta_run.sbatch <label> <window_ic> <dsgnar_sketch> [--set k=v ...]
#
# Submitted by scripts/mendieta_submit.sh, which is the normal way in.  The run is the same cell
# as the windowed batch (T = 20, 10 windows, t2 ansatz, n_coll 2201, DSGNAR 300 iterations x 5
# rounds per window, stop audits on); the extra --set flags select what is being tested.
# ---------------------------------------------------------------------------
#SBATCH -p multi
#SBATCH -A ifeg
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=28G
#SBATCH -t 08:00:00
#SBATCH -o logs/slurm_%x_%j.out
set -uo pipefail

LABEL="$1"; IC="$2"; SKETCH="$3"; shift 3
HERE="${SLURM_SUBMIT_DIR:-$HOME/Julia/PINN/Evolution_try}"
cd "$HERE"
source scripts/env.sh
PY="$(find_python)" || { echo "FATAL: no interpreter with jax" >&2; exit 1; }

export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR="$HOME/.cache/mpl"
mkdir -p logs runs "$MPLCONFIGDIR"

echo "=== $LABEL  job ${SLURM_JOB_ID:-?} on $(hostname)  $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# A JAX that cannot see the GPU falls back to the CPU silently and the run then takes days:
# refuse to start instead.
"$PY" - <<'EOF' || { echo "FATAL: JAX does not see a GPU on $(hostname); not running" >&2; exit 2; }
import jax
assert jax.default_backend() == "gpu", jax.devices()
print("jax", jax.__version__, jax.devices())
EOF

if [ -f "runs/$LABEL/report.md" ]; then
    echo "runs/$LABEL/report.md exists: done already"; exit 0
fi
rm -f "runs/$LABEL/dsgnar_stop_audits.jsonl"     # a previous, interrupted attempt's records

"$PY" -m wave_pinn.cli --label "$LABEL" --outdir "$HERE/runs/$LABEL" \
    --set T=20 --set windows=10 --set ansatz=t2 \
    --set window_ic="$IC" --set w_ic=100 \
    --set n_coll=2201 --set sampler=random --set resample_every=0 \
    --set "snapshot_times=[0,2,4,6,8,10,12,14,16,18,20]" \
    --set optimizer=dsgnar --set dsgnar_steps=300 --set dsgnar_sketch="$SKETCH" \
    --set dsgnar_delta0=1.0 --set dsgnar_delta_min=1e-14 --set resample_rounds=5 \
    --set dsgnar_audit_stops=true "$@"
rc=$?
echo "=== $LABEL exit $rc  $(date -u '+%F %T UTC')"
exit $rc
