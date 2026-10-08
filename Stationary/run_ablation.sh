#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Four one-change ablations of the pq_c100_vac3 setting, run back to back.
#
# The starting point is runs/pq_c100_vac3's configuration (rho in [1, 100], inner sphere areal
# radius 1, lambda_0 = 1/3, S2 = -1/12, Robin orders h=3/lam=3, --ricci-lam-source 0,
# --pin-lam-robin, radial sampling LOG, x64, 5000 Adam + 30000 SSBroyden, n_coll 27768).
# Each run changes exactly ONE thing:
#
#   A  pq_c100_vacA_L40      --scale-ref 40
#        A fixed reference length: the rho^d weights become constants (40, 1600, 1600), so the
#        loss has no spatial bias.  L = 40 is the log-average of the old local-rho weighting
#        over [1, 100] (sqrt(<rho^2>) = 33 for the first-order groups, <rho^4>^(1/4) = 48 for
#        the second-order ones).  L = rho_in = 1 (runs/pq_c100_vac6_fixref) removed the
#        far-field enforcement and failed: lambda collapsed to 1e-7 inside rho = 2 and the
#        saved solution had lambda(rho_out) = 0.43 against 0.9885.
#
#   B  pq_c100_vacB_logphi   --lam-eq-form log
#        The lambda equation written for phi = log lambda, i.e. Delta_h phi = 0 -- the SAME
#        equation (Delta_h phi = lam_eq/lam) with a different loss: the lambda residual is
#        homogeneous in lambda, so lambda -> 0 makes it vanish, while the log form divides by
#        lambda and charges for exactly that collapse.
#
#   C  pq_c100_vacC_volpts   --radial volume
#        The local-rho weights kept, but the collocation uniform in VOLUME: rho^3 uniform on
#        [rho_in^3, rho_out^3], so the number of points per sphere grows like rho^2 and the far
#        field is sampled as densely per unit volume as the near field.  (vac3 used log-uniform
#        rho: equal points per decade, density ~ rho^-3 per unit volume.)
#
#   D  pq_c100_vacD_relterm  --relative-terms
#        Log sampling kept, but every residual is divided by the sum of the absolute values of
#        the terms it is made of: dimensionless, in [-1, 1], no rho^d weight applied at all
#        (the exponents are ignored in this mode).  A residual that is small only because its
#        terms are small -- the lambda -> 0 collapse -- is charged at its true relative size.
#
# Usage, on the hub node (the GPU lives there; see HUB.md):
#
#     cd ~/serafin/Julia/PINN/Stationary
#     git pull
#     nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader
#     PY=$PWD/.venv/bin/python ./run_ablation.sh                 # all four, sequentially
#     PY=$PWD/.venv/bin/python ./run_ablation.sh --only B        # just one of them
#     DRY=1 bash ./run_ablation.sh                               # print, launch nothing
#
# Each run is launched by run_hub.sh (detached, logged, resumable); this script WAITS for the
# previous one to finish before starting the next, so the four never share the GPU.
# Extra arguments (e.g. --steps 500 --n-coll 4096) are forwarded to every run.
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

# x64 and no GPU preallocation, exactly as the pq_c100_vac3 recipe (HUB.md section 5 warns that
# x64 changes the trajectory: pq_c100_vac3 was x64, and that is what we compare against).
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
    # run_hub.sh detaches and returns: wait for ITS job shell (which also runs the
    # post-processing) before starting the next run, so the GPU is never shared.
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

if [ -z "$ONLY" ] || [ "$ONLY" = "A" ]; then
    run_one pq_c100_vacA_L40 --scale-ref 40
fi
if [ -z "$ONLY" ] || [ "$ONLY" = "B" ]; then
    run_one pq_c100_vacB_logphi --lam-eq-form log
fi
if [ -z "$ONLY" ] || [ "$ONLY" = "C" ]; then
    run_one pq_c100_vacC_volpts --radial volume
fi
if [ -z "$ONLY" ] || [ "$ONLY" = "D" ]; then
    run_one pq_c100_vacD_relterm --relative-terms
fi

echo
echo "all requested runs launched/finished.  Compare against:"
echo "  runs/pq_c100_vac3          the local-rho baseline       lambda(rho_out) = 0.9891"
echo "  runs/pq_c100_vac6_fixref   L = rho_in (the failure)     lambda(rho_out) = 0.4327"
for n in A_L40 B_logphi C_volpts D_relterm; do
    d="runs/pq_c100_vac${n}${TAG}"
    [ -d "$d" ] && echo "  $d"
done
