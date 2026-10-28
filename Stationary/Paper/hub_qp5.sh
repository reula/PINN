#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Runs in the paper's final formulation (pointwise Robin at rho_out only, no extra outer spheres,
# no averaged pins; rho^4 weights), Adam 5000 then SSBroyden/Wolfe stages of 10000:
#   fn_nocom_*   without the centre-of-mass condition: the partner of ab_none_w2 for Table I
#                (the qp_com / qp_nocom pair carried the extra outer conditions)
#   fn_tan_*     ab_none + --inner-tangential: the inner condition in H^1 (rho grad_T of the
#                lambda and h_AB residuals; h_rr, h_rA stay free), axisymmetric network
#   rot_tan_*    the same on the off-axis datum, 3-D network width 40, 32768 points
#                (qp_rot4_n32768 + --inner-tangential), four stages
#
#   bash Paper/hub_qp5.sh         launches itself detached and returns
# Resumable like hub_qp.sh.  Log: logs/hub_qp5.log.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
mkdir -p logs
if [ "${1:-}" != "--run" ]; then
    setsid nohup bash "$HERE/Paper/hub_qp5.sh" --run >> logs/hub_qp5.log 2>&1 < /dev/null &
    echo "launched (pid $!); follow with:  tail -f $HERE/logs/hub_qp5.log"
    exit 0
fi

PY="$HERE/.venv/bin/python"
export JAX_ENABLE_X64=1 MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false
source recipe_pq16k.sh
FINAL=(--scale-exps 3,4,3,4 --outer-shell-fracs 1.0 --no-pin-h-robin --no-pin-lam)
COM=(--com-pin 1,1.5,3,10,30,100)
ROT=(--n-coll 32768 --n-bnd 2048 --n-bnd-outer 2048 --arch hybrid --width 40 --depth 6
     --lam-bc-axis 1,0,1)
SSB=(--steps 0 --qn-method ssbroyden --qn-linesearch wolfe --lbfgs-steps 10000 --resample-every 500
     --plateau-min-iters -1)
REF=runs/ib16k_ssbroyden/specref_nonlinear_fine.npz
echo "=== hub_qp5 $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD)"

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
check() {   # check <run>
    [ -f "runs/$1/fresh_samples.json" ] || "$PY" -u scripts/fresh_sample_loss.py "runs/$1" --n 5
    "$PY" -u scripts/grade_rotated.py "runs/$1" --ref "$REF"
}
chain() {   # chain <tag> <number of SSBroyden stages> <flags...>
    local tag=$1 n=$2; shift 2
    adam ${tag}_adam "$@"
    local prev=${tag}_adam
    for i in $(seq 1 "$n"); do stage ${tag}_w$i $prev "$@"; prev=${tag}_w$i; done
    check $prev
}

chain fn_nocom 2 "${FINAL[@]}"
chain fn_tan   2 "${FINAL[@]}" "${COM[@]}" --inner-tangential
chain rot_tan  4 "${ROT[@]}" "${FINAL[@]}" "${COM[@]}" --inner-tangential
echo "=== hub_qp5 finished $(date -u '+%F %T UTC')"
curl -s -H "Title: hub_qp5 finished" -d "fn_nocom, fn_tan, rot_tan done on the hub" \
     https://ntfy.sh/pinn-reula-f0798d1ec301 > /dev/null || true
