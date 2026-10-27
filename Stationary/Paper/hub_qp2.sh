#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The off-axis quadrupole run again, from scratch, with the corrected centre-of-mass quadrature
# (48 x 32 for 3-D networks, stationary/losses.com_quadrature; qp_rot used 24 x 8, which aliases
# on the inner sphere for data about a tilted axis).  Same settings as qp_rot otherwise:
#   qp_rot2_adam (Adam 5000), qp_rot2_w1 .. w3 (SSBroyden, strong Wolfe, 10000 each), then the check.
#
#   bash Paper/hub_qp2.sh         launches itself detached; it WAITS until Paper/hub_qp.sh has
#                                 finished (its "=== hub_qp finished" line), so it never shares the GPU
# Resumable like hub_qp.sh.  Log: logs/hub_qp2.log.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
mkdir -p logs
if [ "${1:-}" != "--run" ]; then
    setsid nohup bash "$HERE/Paper/hub_qp2.sh" --run >> logs/hub_qp2.log 2>&1 < /dev/null &
    echo "launched (pid $!); it waits for hub_qp.sh to finish.  Follow with:  tail -f $HERE/logs/hub_qp2.log"
    exit 0
fi
echo "=== hub_qp2 $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD): waiting for hub_qp.sh"
while ! tail -n 5 logs/hub_qp.log 2>/dev/null | grep -q "=== hub_qp finished"; do sleep 120; done
echo "=== hub_qp.sh has finished; starting $(date -u '+%T UTC')"

PY="$HERE/.venv/bin/python"
export JAX_ENABLE_X64=1 MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false
source recipe_pq16k.sh
F=(--arch hybrid --width 20 --depth 6 --lam-bc-axis 1,0,1 --scale-exps 3,4,3,4 --pin-lam-potential
   --com-pin 1,1.5,3,10,30,100)
SSB=(--steps 0 --qn-method ssbroyden --qn-linesearch wolfe --lbfgs-steps 10000 --resample-every 500
     --plateau-min-iters -1)
REF=runs/ib16k_ssbroyden/specref_nonlinear_fine.npz
"$PY" -c "from stationary.losses import com_quadrature; from stationary.train import parse_args; \
import sys; print('[hub_qp2] COM quadrature', com_quadrature(parse_args(sys.argv[1:])))" --outdir runs/x "${F[@]}"

if [ ! -f runs/qp_rot2_adam/report.json ]; then
    echo "--- qp_rot2_adam (Adam)  $(date -u '+%T UTC')"
    "$PY" -u -m stationary.train --outdir runs/qp_rot2_adam "${S_RECIPE[@]}" --steps 5000 \
        --plateau-min-iters -1 --lbfgs-steps 0 "${F[@]}"
fi
prev=qp_rot2_adam
for i in 1 2 3; do
    name=qp_rot2_w$i
    if [ ! -f "runs/$name/report.json" ]; then
        res=(); [ -f "runs/$name/ckpt.pkl" ] && res=(--resume auto)
        echo "--- $name from $prev  $(date -u '+%T UTC')"
        "$PY" -u -m stationary.train --outdir "runs/$name" "${S_RECIPE[@]}" "${SSB[@]}" \
            --init-from "runs/$prev/params.pkl" "${res[@]}" "${F[@]}"
    fi
    prev=$name
done
[ -f runs/qp_rot2_w3/fresh_samples.json ] || "$PY" -u scripts/fresh_sample_loss.py runs/qp_rot2_w3 --n 5
"$PY" -u scripts/grade_rotated.py runs/qp_rot2_w3 --ref "$REF"
echo "=== hub_qp2 finished $(date -u '+%F %T UTC')"
curl -s -H "Title: hub_qp2 finished" -d "qp_rot rerun with the 48x32 COM quadrature done on the hub" \
     https://ntfy.sh/pinn-reula-f0798d1ec301 > /dev/null || true
