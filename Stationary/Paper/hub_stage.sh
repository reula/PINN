#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# A second-stage network on the rotated Weyl test (paper Sec. IV.D; base runs/weyl_rot45_hyb:
# metric-only 3-D network width 20 depth 6, SSBroyden from cold, stopped on its plateau after 7000
# iterations at loss 5.1e-9, max errors 1.6e-4 in h and 1.7e-5 in lambda).  The first stage is
# frozen and a correction network of scale eps is trained on the same loss (model.StagedNet,
# --stage-base).  Every arm has the base's settings (Paper/commands_weyl.sh) except what it tests:
#   st_ctl   the control: the base network itself continued for the same budget
#   st_a     correction width 20 depth 6, eps 1e-4 (about the base's error)
#   st_b     the same with 4 Fourier features in t (the base's error is concentrated near the
#            inner sphere, i.e. of higher radial frequency than the solution)
#   st_c     correction width 40 depth 6 with the 4 Fourier features (9007 parameters): capacity,
#            which limited the 3-D network on the off-axis quadrupole (width 20 -> 40: loss / 270)
#   st_w40   the control for st_c: ONE width-40 network (8687 parameters) from scratch with the
#            base's settings -- is a second network better than a bigger first one?
#
#   bash Paper/hub_stage.sh        launches itself detached and returns
# Resumable.  Log: logs/hub_stage.log.  About 2.5 h.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
mkdir -p logs
if [ "${1:-}" != "--run" ]; then
    setsid nohup bash "$HERE/Paper/hub_stage.sh" --run >> logs/hub_stage.log 2>&1 < /dev/null &
    echo "launched (pid $!); follow with:  tail -f $HERE/logs/hub_stage.log"
    exit 0
fi

PY="$HERE/.venv/bin/python"
export JAX_ENABLE_X64=1 MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=runs/weyl_rot45_hyb
[ -f "$BASE/params.pkl" ] || { echo "FATAL: no $BASE/params.pkl"; exit 2; }
ROT=(--weyl --weyl-half-length 0.028571428571 --weyl-half-length-b 0.002857142857
     --weyl-half-gap 0.014285714286 --weyl-rotate-deg 45 --rho-in 0.1 --rho-out 1.0 --arch hybrid
     --n-coll 8192 --n-bnd 512 --w-inner 10 --w-outer 10 --reweight-every 2000 --steps 0
     --qn-method ssbroyden --lbfgs-steps 20000 --ckpt-every 500 --vtk-physical-inner 0.1 --no-pin-h-robin)
echo "=== hub_stage $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD)"

arm() {     # arm <name> <flags...>
    local name=$1; shift
    if [ -f "runs/$name/report.json" ]; then echo "--- $name: done, skipped"; return; fi
    local res=()
    [ -f "runs/$name/ckpt.pkl" ] && res=(--resume auto) && echo "--- $name: resuming from its checkpoint"
    echo "--- $name  $(date -u '+%T UTC')"
    "$PY" -u -m stationary.train --outdir "runs/$name" "${ROT[@]}" "${res[@]}" "$@"
}
summary() {
    for r in "$@"; do
        "$PY" - "$r" <<'PYEOF'
import json, sys
r = sys.argv[1]; d = json.load(open(f"runs/{r}/report.json"))
print(f"[stage] {r:10s} loss {d['final_loss']:.3e}  max|dh| {d['max_dh']:.3e}  max|dlam| {d['max_dlam']:.3e}"
      f"  rms|dh| {d['rms_dh']:.3e}  rms|dlam| {d['rms_dlam']:.3e}  iters {d.get('qn_stopped_at')}"
      f"  {d['wall_seconds']:.0f} s")
PYEOF
    done
}

arm st_ctl --init-from "$BASE/params.pkl"
arm st_a   --stage-base "$BASE" --stage-eps 1e-4 --width 20 --depth 6 --fourier 0
arm st_b   --stage-base "$BASE" --stage-eps 1e-4 --width 20 --depth 6 --fourier 4
arm st_c   --stage-base "$BASE" --stage-eps 1e-4 --width 40 --depth 6 --fourier 4
arm st_w40 --width 40 --depth 6
summary weyl_rot45_hyb st_ctl st_a st_b st_c st_w40
echo "=== hub_stage finished $(date -u '+%F %T UTC')"
curl -s -H "Title: hub_stage finished" -d "second-stage network on rotated Weyl: st_ctl, st_a, st_b, st_c, st_w40 done" \
     https://ntfy.sh/pinn-reula-f0798d1ec301 > /dev/null || true
