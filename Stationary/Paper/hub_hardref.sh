#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The spectral references of the hard end of the S2 scan (ms_s2_025_w2, ms_s2_030_w2), redone: the
# first ones (tag fine, l <= 14, 48 radial, 1824 unknowns) DIVERGED -- |r| rose 0.98 -> 5.2 on the
# first step and the solver kept that iterate; its cond(J) was 1.5e11 (4.6e9 at S2 = -1/6).
# stationary/specref.py now backtracks the step and has --rcond.  Variants, each its own tag:
#   damp    backtracking only, rcond 1e-13 (the old cutoff)
#   rc9     backtracking and rcond 1e-9 (drops the near-null directions of the bigger basis)
# on ms_s2_025_w2 first; the winner is then used for ms_s2_030_w2 (tag as given in $2).
#   bash Paper/hub_hardref.sh --run [30 <tag>]
# About 25 min per solve on one A30.  Log: logs/hub_hardref.log.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
mkdir -p logs
PY="$HERE/.venv/bin/python"
export JAX_ENABLE_X64=1 MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false
HARD=(--nonlinear --ls 0,2,4,6,8,10,12,14 --n-radial 48 --n-coll 4000 --n-bnd 600 --iters 12)
ref() {   # ref <run> <tag> <rcond>
    local r=runs/$1
    [ -f "$r/specref_nonlinear_$2.npz" ] && { echo "--- $1 $2: done, skipped"; return; }
    echo "--- specref $1 $2 (rcond $3)  $(date -u '+%T UTC')"
    "$PY" -u -m stationary.specref "$r" "${HARD[@]}" --rcond "$3" --tag "$2"
    "$PY" -u scripts/gh_moments_runs.py "$r" --ref "$r/specref_nonlinear_$2.npz" || true
}
echo "=== hub_hardref $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD)"
if [ "${2:-}" = 30 ]; then
    ref ms_s2_030_w2 "$3" "$([ "$3" = rc9 ] && echo 1e-9 || echo 1e-13)"
else
    ref ms_s2_025_w2 damp 1e-13
    ref ms_s2_025_w2 rc9 1e-9
fi
echo "=== hub_hardref finished $(date -u '+%F %T UTC')"
