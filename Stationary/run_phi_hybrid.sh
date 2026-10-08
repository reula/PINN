#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Two runs with the LOG form of the lambda equation and a HYBRID radial sampling.
#
# Everything else is the pq_c100_vac3 setting, unchanged, as asked:
#   * metric boundary conditions untouched (eq_weights inner_h = 0, outer_h = 0);
#   * the Ricci equation solved WITHOUT its lambda source (--ricci-lam-source 0);
#   * the averaged Robin pins kept (--pin-lam-robin), since they behaved;
#   * rho in [1, 100], inner areal radius 1, lambda_0 = 1/3, S2 = -1/12, Robin h=3/lam=3,
#     n_coll 27768, 5000 Adam + 30000 SSBroyden, x64.
#
# What changes:
#
#   E  pq_c100_vacE_phihyb50   --lam-eq-form log --radial hybrid --radial-log-frac 0.5
#   F  pq_c100_vacF_phihyb25   --lam-eq-form log --radial hybrid --radial-log-frac 0.25
#
#   --lam-eq-form log  imposes Delta_h(log lambda) = 0 instead of Delta_h lambda =
#       |d lambda|^2/lambda.  Same solution set, different loss: the lambda residual is
#       homogeneous of degree one in lambda, so lambda -> 0 makes it vanish; dividing by lambda
#       charges exactly that collapse.  pq_c100_vacB_logphi showed it recovers the same
#       physical solution as pq_c100_vac3 (lambda(rho_out) = 0.98906 against 0.98852 exact)
#       while running with the UNWEIGHTED local-rho loss.
#
#   --radial hybrid  draws --radial-log-frac of the points log-uniformly in rho (so the inner
#       boundary layer keeps its resolution) and the rest uniformly in volume (so the far field
#       is represented per unit volume).  Pure volume (pq_c100_vacC_volpts) left 0.19 of 27768
#       points inside rho = 2 and converged to a wrong, too-flat profile; pure log leaves the
#       far field thin.  At frac = 0.5 the expected count inside rho = 2 is ~2090 and outside
#       rho = 10 is ~20800.
#
# Usage, on the hub node (see HUB.md):
#
#     cd ~/serafin/Julia/PINN/Stationary
#     git pull
#     nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader
#     PY=$PWD/.venv/bin/python ./run_phi_hybrid.sh              # both, sequentially
#     PY=$PWD/.venv/bin/python ./run_phi_hybrid.sh --only F     # one
#     DRY=1 bash ./run_phi_hybrid.sh                            # print, launch nothing
#
# Each run is launched by run_hub.sh (detached, logged, resumable) and this script WAITS for it
# to finish before starting the next, so the two never share the GPU.  Extra arguments (e.g.
# --steps 500) are forwarded to both.
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
TAG="${TAG:-}"
DRY="${DRY:-0}"
ONLY=""
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        --only=*) ONLY="${1#*=}"; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/jaxcache-$USER}"

BASE=(
    --arch axisym_hybrid --steps 5000 --lbfgs-steps 30000 --qn-block 1500
    --n-coll 27768 --n-bnd 4548 --n-bnd-outer 4548 --ckpt-every 500
    --R0 0.5773502691896258 --rho-in 1.0 --rho-out 100.0 --inner-radius 1.0
    --vtk-physical-inner 1.0
    --lam0 0.33333333333333337 --lam-inf 1.0 --lam-bc-S2 -0.08333333333333333
    --outer-bc robin --robin-orders h=3,lam=3 --no-robin-G
    --eq-weights compat=0,ricci=1,gauge=1,lam_eq=1,inner_h=0,outer_h=0
    --ricci-lam-source 0 --pin-lam-robin --ref-solution --ref-asymptotic 1.0
    --reweight-every 0 --decay-feature --log-resources --vtk
)

run_one() {
    local name="$1"; shift
    local out="runs/${name}${TAG}"
    if [ "$DRY" = "1" ]; then
        printf 'DRY  %s\n     %s run_hub.sh' "$out" "$PY"
        printf ' %q' "${BASE[@]}" "$@" "--outdir" "$out" ${ARGS[@]+"${ARGS[@]}"}
        printf '\n'
        return 0
    fi
    echo "=================================================================="
    echo "== $name  ->  $out"
    echo "=================================================================="
    PY="$PY" bash "$HERE/run_hub.sh" "${BASE[@]}" "$@" --outdir "$out" \
        ${ARGS[@]+"${ARGS[@]}"}
    local pidfile="$out/run.pid" pid
    for _ in $(seq 1 5760); do          # up to 48 h at 30 s
        if [ -f "$pidfile" ]; then
            pid="$(cat "$pidfile" 2>/dev/null || true)"
            if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
                break
            fi
        fi
        sleep 30
    done
    echo "== $name finished: $out  (log: logs/$(basename "$out").log)"
}

if [ -z "$ONLY" ] || [ "$ONLY" = "E" ]; then
    run_one pq_c100_vacE_phihyb50 --lam-eq-form log --radial hybrid --radial-log-frac 0.5
fi
if [ -z "$ONLY" ] || [ "$ONLY" = "F" ]; then
    run_one pq_c100_vacF_phihyb25 --lam-eq-form log --radial hybrid --radial-log-frac 0.25
fi

echo
echo "compare against:"
echo "  runs/pq_c100_vac3         local rho + log sampling   lambda(100) = 0.98906 (exact 0.98852)"
echo "  runs/pq_c100_vacB_logphi  log form  + log sampling   lambda(100) = 0.98906"
echo "  runs/pq_c100_vacC_volpts  local rho + pure volume    lambda(100) = 0.99812, wrong profile"
for n in E_phihyb50 F_phihyb25; do
    d="runs/pq_c100_vac${n}${TAG}"
    [ -d "$d" ] && echo "  $d"
done
echo
echo "in every report, read the multipole sections in the S_lm form (coefficient of"
echo "Y_lm/rho^(l+1), no rho factor) and the SPURIOUS DIPOLE line: the inner data impose S_1 = 0,"
echo "so |S_10| at rho_out and its drift are the numbers to watch."
