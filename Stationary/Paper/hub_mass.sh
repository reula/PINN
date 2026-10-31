#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The mass as a function of the quadrupole amplitude S2 (paper, Sec. IV.A): the back-reaction
# M0 - 1/sqrt(3) should be O(S2^2).  Two more amplitudes besides the paper's S2 = -1/12, each as
#   ms_<tag>_*   the paper's run (ab_none: final formulation + centre of mass, Adam 5000 then
#                2 x 10000 SSBroyden/Wolfe), with only --lam-bc-S2 changed
#   specref_nonlinear_fine.npz in ms_<tag>_w2: the fine spectral reference of that problem
#                (l = 0..10, 36 radial modes, 2500 points), solved on the run's own formulation
# and the fine reference of ab_none_w2 itself (S2 = -1/12 in the final formulation; the paper's
# reference was solved with the extra outer conditions of ib16k_ssbroyden).  Then the GH moments of
# every network and reference (scripts/gh_moments_runs.py).
#   s2h  S2 = -1/24 (half)       s2d  S2 = -1/6 (double; lambda on the sphere in [1/6, 5/12])
# and two amplitudes with the spectral reference only (no training: a 10-step Adam run only provides
# the run directory the spectral solve reads its problem from), for the fit of the reference masses:
#   s2t  S2 = -1/8               s2p  S2 = +1/12 (the sign separates an S2^3 term from S2^2)
# Added 2026-10-31, the hard end of the scan (networks and references, same settings), where the
# datum nearly reaches a horizon: lambda = 1/3 + S2 P2 has its minimum 1/3 + S2 at the poles
#   s2_025  S2 = -0.25 (lambda_min = 1/12)      s2_030  S2 = -0.30 (lambda_min = 1/30)
# Their references use a bigger basis (l <= 14, 48 radial, 4000 points): at -1/6 the l <= 10 one
# already stalled at |r| = 3.7e-5 (ms_s2d_w2/specref_nonlinear_fine.json).
#
#   bash Paper/hub_mass.sh         launches itself detached and returns
# Resumable.  Log: logs/hub_mass.log.  About 4.5 h (2 x 25 min of training, 5 x ~40 min of spectral); the hard end about 3 h more.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
mkdir -p logs
if [ "${1:-}" != "--run" ]; then
    setsid nohup bash "$HERE/Paper/hub_mass.sh" --run >> logs/hub_mass.log 2>&1 < /dev/null &
    echo "launched (pid $!); follow with:  tail -f $HERE/logs/hub_mass.log"
    exit 0
fi

PY="$HERE/.venv/bin/python"
export JAX_ENABLE_X64=1 MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false
source recipe_pq16k.sh
FINAL=(--scale-exps 3,4,3,4 --outer-shell-fracs 1.0 --no-pin-h-robin --no-pin-lam)
COM=(--com-pin 1,1.5,3,10,30,100)
SSB=(--steps 0 --qn-method ssbroyden --qn-linesearch wolfe --lbfgs-steps 10000 --resample-every 500
     --plateau-min-iters -1)
SPEC=(--nonlinear --ls 0,2,4,6,8,10 --n-radial 36 --n-coll 2500 --n-bnd 400 --iters 10 --tag fine)
SPEC_HARD=(--nonlinear --ls 0,2,4,6,8,10,12,14 --n-radial 48 --n-coll 4000 --n-bnd 600 --iters 12 --tag fine)
echo "=== hub_mass $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD)"

stage() {   # stage <name> <init run> <flags...>
    local name=$1 init=$2; shift 2
    if [ -f "runs/$name/report.json" ]; then echo "--- $name: done, skipped"; return; fi
    local res=()
    [ -f "runs/$name/ckpt.pkl" ] && res=(--resume auto) && echo "--- $name: resuming from its checkpoint"
    echo "--- $name from $init  $(date -u '+%T UTC')"
    "$PY" -u -m stationary.train --outdir "runs/$name" "${S_RECIPE[@]}" "${SSB[@]}" \
        --init-from "runs/$init/params.pkl" "${res[@]}" "$@"
}
adam() {    # adam <name> <flags...>
    local name=$1; shift
    if [ -f "runs/$name/report.json" ]; then echo "--- $name: done, skipped"; return; fi
    echo "--- $name (Adam)  $(date -u '+%T UTC')"
    "$PY" -u -m stationary.train --outdir "runs/$name" "${S_RECIPE[@]}" --steps 5000 --plateau-min-iters -1 \
        --lbfgs-steps 0 "$@"
}
spec() {    # spec <run> [hard]: the fine spectral reference of the run's problem, then grade and moments
    local r=runs/$1
    if [ ! -f "$r/specref_nonlinear_fine.npz" ]; then
        echo "--- specref $1 ${2:-}  $(date -u '+%T UTC')"
        if [ "${2:-}" = hard ]; then "$PY" -u -m stationary.specref "$r" "${SPEC_HARD[@]}"
        else "$PY" -u -m stationary.specref "$r" "${SPEC[@]}"; fi
    fi
    [ -f "$r/fresh_samples.json" ] || "$PY" -u scripts/fresh_sample_loss.py "$r" --n 5
    "$PY" -u scripts/grade_rotated.py "$r" --ref "$r/specref_nonlinear_fine.npz"
    "$PY" -u scripts/gh_moments_runs.py "$r" --ref "$r/specref_nonlinear_fine.npz"
}

for v in "s2h:-0.041666666666666664" "s2d:-0.16666666666666666"; do
    tag=ms_${v%%:*}; S2=${v#*:}
    adam  ${tag}_adam            "${FINAL[@]}" "${COM[@]}" --lam-bc-S2 "$S2"
    stage ${tag}_w1 ${tag}_adam  "${FINAL[@]}" "${COM[@]}" --lam-bc-S2 "$S2"
    stage ${tag}_w2 ${tag}_w1    "${FINAL[@]}" "${COM[@]}" --lam-bc-S2 "$S2"
done
for v in "s2t:-0.125" "s2p:0.08333333333333333"; do
    tag=ms_${v%%:*}; S2=${v#*:}
    if [ ! -f "runs/${tag}_cfg/report.json" ]; then
        "$PY" -u -m stationary.train --outdir "runs/${tag}_cfg" "${S_RECIPE[@]}" --steps 10 \
            --plateau-min-iters -1 --lbfgs-steps 0 "${FINAL[@]}" "${COM[@]}" --lam-bc-S2 "$S2"
    fi
done
spec ms_s2h_w2
spec ms_s2d_w2
spec ab_none_w2
for t in s2t s2p; do      # reference only: its moments (the network part of the output is noise)
    r=runs/ms_${t}_cfg
    [ -f "$r/specref_nonlinear_fine.npz" ] || "$PY" -u -m stationary.specref "$r" "${SPEC[@]}"
    "$PY" -u scripts/gh_moments_runs.py "$r" --ref "$r/specref_nonlinear_fine.npz"
done
for v in "s2_025:-0.25" "s2_030:-0.30"; do           # the hard end (2026-10-31)
    tag=ms_${v%%:*}; S2=${v#*:}
    adam  ${tag}_adam            "${FINAL[@]}" "${COM[@]}" --lam-bc-S2 "$S2"
    stage ${tag}_w1 ${tag}_adam  "${FINAL[@]}" "${COM[@]}" --lam-bc-S2 "$S2"
    stage ${tag}_w2 ${tag}_w1    "${FINAL[@]}" "${COM[@]}" --lam-bc-S2 "$S2"
    spec ${tag}_w2 hard
done
echo "=== hub_mass finished $(date -u '+%F %T UTC')"
curl -s -H "Title: hub_mass finished" -d "mass vs S2: ms_s2h, ms_s2d and the five fine references done on the hub" \
     https://ntfy.sh/pinn-reula-f0798d1ec301 > /dev/null || true
