#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# DSGNAR polish of an existing solution: start from a finished run's parameters and let the
# doubly-sketched Gauss-Newton phase drive the residuals further down.
#
# Why this shape.  DSGNAR needs a residual-vector objective and its Jacobian-vector products;
# it is worth trying here because in Evolution_try the same method reached a 16x lower loss than
# SSBroyden in 65x fewer iterations (2.98e-11 in 150 iterations / 918 s against 4.70e-10 in
# 9753 iterations / 4279 s), and because our loss weights span eight orders of magnitude in rho
# -- exactly the badly scaled least-squares problem its trust region and its lambda solve are
# built for.  Starting from an existing run removes the Adam warm-up from the experiment: the
# only question is whether the same loss can be pushed lower from the same point.
#
# Cost.  One iteration costs `--dsgnar-sketch` batched JVPs over the whole residual vector plus
# one SVD of that size, and the tangent batch is s x n_rows; at s = 128 the tangent batch is a
# few hundred MB for this problem.  MEASURE FIRST with PROBE=1 (5 iterations, printed timings
# and memory), then launch the real one.
#
# Usage, on the hub node (see HUB.md):
#
#     cd ~/serafin/Julia/PINN/Stationary
#     git pull
#     PROBE=1 SKETCH=256 bash run_dsgnar.sh                  # 5 iterations, to size it
#     SKETCH=256 STEPS=60 bash run_dsgnar.sh                 # the real polish
#     INIT=runs/pq_c100_vacB_logphi/params.pkl OUT=runs/pq_c100_vacG_dsgnar_B bash run_dsgnar.sh
#     DRY=1 bash run_dsgnar.sh                               # print, launch nothing
#
# Env: INIT (default runs/pq_c100_vac3/params.pkl), OUT, SKETCH (128), STEPS (60),
#      PROBE (0), TAG (suffix for OUT), plus any extra arguments forwarded to run_hub.sh.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-$HERE/.venv/bin/python}"
if [ ! -x "$PY" ]; then
    echo "run_hub.sh needs the project's python (jax lives there, not in PATH):" >&2
    echo "  PY=\$PWD/.venv/bin/python $0" >&2
    exit 1
fi
DRY="${DRY:-0}"
INIT="${INIT:-runs/pq_c100_vac3/params.pkl}"
SKETCH="${SKETCH:-128}"
PROBE="${PROBE:-0}"
TAG="${TAG:-}"
SKETCH_STR="${SKETCH}"

if [ "$PROBE" = "1" ]; then
    STEPS="${STEPS:-5}"
    OUT="${OUT:-runs/pq_c100_vacF_dsgnar_probe_s${SKETCH_STR}${TAG}}"
else
    STEPS="${STEPS:-60}"
    OUT="${OUT:-runs/pq_c100_vacG_dsgnar_s${SKETCH_STR}${TAG}}"
fi

if [ ! -f "$INIT" ]; then
    echo "no parameters to start from: $INIT" >&2
    echo "pass INIT=<run>/params.pkl (params_adam.pkl for the end of the Adam phase)" >&2
    exit 1
fi

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/jaxcache-$USER}"

ARGS=(
    --outdir "$OUT"
    --arch axisym_hybrid
    --steps 0                       # no Adam: the warm-up is the run we start from
    --lbfgs-steps 30000             # gates entry into the quasi-Newton section; DSGNAR uses
                                    # --dsgnar-steps, so this is only a ceiling
    --qn-method dsgnar
    --dsgnar-steps "$STEPS"
    --dsgnar-sketch "$SKETCH_STR"
    --init-from "$INIT"
    --n-coll 27768 --n-bnd 4548 --n-bnd-outer 4548 --ckpt-every 500
    --R0 0.5773502691896258 --rho-in 1.0 --rho-out 100.0 --inner-radius 1.0
    --vtk-physical-inner 1.0
    --lam0 0.33333333333333337 --lam-inf 1.0 --lam-bc-S2 -0.08333333333333333
    --outer-bc robin --robin-orders h=3,lam=3 --no-robin-G
    --eq-weights compat=0,ricci=1,gauge=1,lam_eq=1,inner_h=0,outer_h=0
    --ricci-lam-source 0 --pin-lam-robin --ref-solution --ref-asymptotic 1.0
    --reweight-every 0 --resample-every 500 --decay-feature --log-resources --vtk
)

echo "DSGNAR polish:  init $INIT"
echo "                out  $OUT"
echo "                $STEPS iteration(s), sketch s = $SKETCH_STR"
echo "                (each iteration = $SKETCH_STR batched JVPs + one ${SKETCH_STR}x${SKETCH_STR} SVD)"

if [ "$DRY" = "1" ]; then
    printf 'DRY  %s run_hub.sh' "$PY"
    printf ' %q' "${ARGS[@]}"
    printf '\n'
    exit 0
fi

PY="$PY" bash "$HERE/run_hub.sh" "${ARGS[@]}" "$@"

pidfile="$OUT/run.pid"
for _ in $(seq 1 5760); do
    if [ -f "$pidfile" ]; then
        pid="$(cat "$pidfile" 2>/dev/null || true)"
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
    fi
    sleep 30
done
echo "finished: $OUT   (log: logs/$(basename "$OUT").log)"
echo
echo "read, in the log:  [dsgnar] step ... lines (loss, rho, lambda, radius, target)"
echo "                   [dsgnar] DSGNAR: loss ... ; and the timing line"
echo "and in the report: the S_lm tables (lambda and the Geroch-Hansen potential), the"
echo "                   SPURIOUS DIPOLE lines, and the raw residual table -- compare with"
echo "                   runs/pq_c100_vac3, and against the parameters you started from."
