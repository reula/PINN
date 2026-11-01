#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# A bigger spectral basis for S2 = -0.30 (ms_s2_030_w2), whose l <= 14 / 48 radial reference
# (tag damp) stalled at |r| = 1.15e-3, loss 1.3e-6, against 4.6e-5 at S2 = -0.25: both the
# reference and the networks are then at the resolution limit.  Which truncation matters?  One
# direction at a time, the rows kept at 60418 (n_coll 4000, n_bnd 600) so that the host memory of
# the least-squares step (14.5 GB at 1824 unknowns, 28 GB requested) grows only with the unknowns:
#   big_l20   l <= 20 (11 even modes), 48 radial: 2544 unknowns  (~20 GB host)
#   big_r72   l <= 14, 72 radial:                  2736 unknowns  (~22 GB host)
# Backtracking Gauss-Newton as in hub_hardref.sh (default rcond: 1e-9 was worse).  Each variant
# prints its peak GPU memory per step.  Submit with --mem=36G.
#   bash Paper/hub_bigref.sh --run [amp]    amp 030 (default) or 025; log logs/hub_bigref.log
# About 1.5 h per variant on one A30.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
mkdir -p logs
AMP=${2:-030}
r=runs/ms_s2_${AMP}_w2
PY="$HERE/.venv/bin/python"
export JAX_ENABLE_X64=1 MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=(--nonlinear --n-coll 4000 --n-bnd 600 --iters 14)
echo "=== hub_bigref $AMP $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD)"
var() {   # var <tag> <ls> <n-radial>
    [ -f "$r/specref_nonlinear_$1.npz" ] && { echo "--- $1: done, skipped"; return; }
    echo "--- specref $r $1 (l <= ${2##*,}, $3 radial)  $(date -u '+%T UTC')"
    "$PY" -u -m stationary.specref "$r" "${BASE[@]}" --ls "$2" --n-radial "$3" --tag "$1"
    "$PY" -u scripts/gh_moments_runs.py "$r" --ref "$r/specref_nonlinear_$1.npz" --tag "$1" || true
}
var big_l20 0,2,4,6,8,10,12,14,16,18,20 48
var big_r72 0,2,4,6,8,10,12,14 72
echo "=== hub_bigref finished $(date -u '+%F %T UTC')"
