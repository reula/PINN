#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The off-axis quadrupole run in the final formulation: no Robin ladder and no averaged pins
# (the ablations ab_noladder / ab_nopins / ab_none of hub_qp.sh were as good or better than
# qp_com without them), with the centre-of-mass condition (48 x 32 quadrature) and the rho^4
# weights.  qp_rot2 (hub_qp2.sh, with ladder and pins) stalled at 1.35e-6, rms dh 1.8e-3.
#   qp_rot3_adam (Adam 5000), qp_rot3_w1 .. w4 (SSBroyden, strong Wolfe, 10000 each), then the check.
#
#   bash Paper/hub_qp3.sh         launches itself detached and returns
# Resumable like hub_qp.sh.  Log: logs/hub_qp3.log.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
mkdir -p logs
if [ "${1:-}" != "--run" ]; then
    setsid nohup bash "$HERE/Paper/hub_qp3.sh" --run >> logs/hub_qp3.log 2>&1 < /dev/null &
    echo "launched (pid $!); follow with:  tail -f $HERE/logs/hub_qp3.log"
    exit 0
fi
echo "=== hub_qp3 $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD)"

PY="$HERE/.venv/bin/python"
export JAX_ENABLE_X64=1 MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false
source recipe_pq16k.sh
F=(--arch hybrid --width 20 --depth 6 --lam-bc-axis 1,0,1 --scale-exps 3,4,3,4 --pin-lam-potential
   --com-pin 1,1.5,3,10,30,100 --outer-shell-fracs 1.0 --no-pin-h-robin --no-pin-lam)
SSB=(--steps 0 --qn-method ssbroyden --qn-linesearch wolfe --lbfgs-steps 10000 --resample-every 500
     --plateau-min-iters -1)
REF=runs/ib16k_ssbroyden/specref_nonlinear_fine.npz

if [ ! -f runs/qp_rot3_adam/report.json ]; then
    echo "--- qp_rot3_adam (Adam)  $(date -u '+%T UTC')"
    "$PY" -u -m stationary.train --outdir runs/qp_rot3_adam "${S_RECIPE[@]}" --steps 5000 \
        --plateau-min-iters -1 --lbfgs-steps 0 "${F[@]}"
fi
prev=qp_rot3_adam
for i in 1 2 3 4; do
    name=qp_rot3_w$i
    if [ ! -f "runs/$name/report.json" ]; then
        res=(); [ -f "runs/$name/ckpt.pkl" ] && res=(--resume auto)
        echo "--- $name from $prev  $(date -u '+%T UTC')"
        "$PY" -u -m stationary.train --outdir "runs/$name" "${S_RECIPE[@]}" "${SSB[@]}" \
            --init-from "runs/$prev/params.pkl" "${res[@]}" "${F[@]}"
    fi
    prev=$name
done
[ -f runs/qp_rot3_w4/fresh_samples.json ] || "$PY" -u scripts/fresh_sample_loss.py runs/qp_rot3_w4 --n 5
"$PY" -u scripts/grade_rotated.py runs/qp_rot3_w4 --ref "$REF"
echo "=== hub_qp3 finished $(date -u '+%F %T UTC')"
curl -s -H "Title: hub_qp3 finished" -d "qp_rot3 (no ladder, no pins, COM) done on the hub" \
     https://ntfy.sh/pinn-reula-f0798d1ec301 > /dev/null || true
