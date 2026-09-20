#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The control ladder, run in sequence (ONE GPU: never in parallel), followed by a
# side-by-side comparison of everything it produced.
#
#   PY=$PWD/.venv/bin/python ./run_ladder.sh            # steps 1-3  (~30 min)
#   PY=$PWD/.venv/bin/python ./run_ladder.sh 3 4        # chosen steps
#   PY=$PWD/.venv/bin/python ./run_ladder.sh all        # 1-5 (~2-4 h, unattended)
#   PY=$PWD/.venv/bin/python ./run_ladder.sh --dry-run  # print the commands, launch nothing
#   PY=$PWD/.venv/bin/python ./run_ladder.sh --compare  # only re-print the comparison table
#
#   step 1  order 1, float32, 64x4 f8      runs/control_ord1b      ~7 min
#   step 2  order 2, float32, same size    runs/control_ord2       ~7 min
#   step 3  order 4, x64,     same size    runs/control_ord4_x64   ~15 min
#   step 4  order 1, float32, BIG net      runs/control_ord1_big   ~20-40 min
#   step 5  the dipole, order 1, BIG net   runs/dipole_big         ~20-40 min
#   step 6  the dipole, order 1, small net runs/dipole_small       ~7 min
#
# Steps 1-3 were run on 20 September and settled the order question: order 1 wins by two
# orders of magnitude in lambda(rho_out) (3.0e-4 against 1.6e-1 for order 2 in float32 and
# 5.9e-2 for order 4 in float64).  The higher-order conditions are stiffer -- they amplify
# the high radial frequencies -- and they are also more permissive, which costs accuracy on
# a spherically symmetric solution that has no high multipoles to let through.  They are
# not needed for the dipole either: its l = 1 tail at rho_out is ~S1 (rho_in/rho_out)^2 =
# 1e-5, so the order-1 condition biases it by ~1e-5, thirty times below the accuracy we
# already have.  Hence steps 4 and 5 use order 1 and spend the time on capacity instead.
# See HUB.md section 6.
#
# Each step is launched with run_hub.sh, which detaches it, logs it, checkpoints it and
# runs the figures + report when it ends; this script only waits for the run to finish
# before starting the next one.  A step whose run directory already has params.pkl is
# skipped (use --force to redo it).
#
# Environment: PY (interpreter, as in run_hub.sh), plus the usual run_hub.sh overrides.
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="${PY:-python}"
FORCE=0
DRY=0
COMPARE_ONLY=0
WANT=()

# The shared physics: the M2 control geometry, k = 1 (derived), lambda -> 1, order-1..4
# Robin conditions chosen per step, weights and ramp as in HUB.md section 7.
BASE=(--arch axisym_hybrid --steps 20000 --ref-solution --ref-asymptotic 1.0
      --R0 0.5773502691896258 --rho-in 1.0 --inner-radius 1.0 --rho-out 100
      --outer-bc robin --no-robin-G --lam-inf 1.0 --decay-feature --radial log
      --pde-ramp-steps 500 --w-inner 100 --w-outer 100 --reweight-every 1500)
SMALL=(--n-coll 4096  --n-bnd 256  --width 64  --depth 4 --fourier 8)
BIG=(--n-coll 16384 --n-bnd 1024 --width 256 --depth 6 --fourier 16)
NICE=(--lbfgs-steps 3000)

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --force)   FORCE=1; shift ;;
        --compare) COMPARE_ONLY=1; shift ;;
        all)       WANT=(1 2 3 4 5 6); shift ;;
        [1-6])     WANT+=("$1"); shift ;;
        *) echo "unknown argument '$1' (want 1..6, all, --dry-run, --force, --compare)" >&2
           exit 1 ;;
    esac
done
[ ${#WANT[@]} -eq 0 ] && WANT=(1 2 3)

# ------------------------------------------------------------------ one step
run_one() {                     # run_one NAME X64(0|1) -- <extra flags...>
    local name="$1" x64="$2"; shift 3
    local out="runs/$name"
    if [ "$FORCE" -eq 0 ] && [ -f "$out/params.pkl" ]; then
        echo "== $name: already done ($out/params.pkl) -- --force to redo"
        return 0
    fi
    # NOTE the invocation: run_hub.sh is a SHELL script, so it is run by bash with PY in
    # its environment (env VAR=value bash script ...).  Writing "$PY" run_hub.sh would
    # hand the script to Python and produce a SyntaxError about a `case` line.
    local -a cmd=(env "PY=$PY")
    [ "$x64" = "1" ] && cmd+=(JAX_ENABLE_X64=1)
    # LADDER_EXTRA overrides anything, for a quick trial of the machinery:
    #   LADDER_EXTRA="--steps 20 --lbfgs-steps 0 --n-coll 64 --n-bnd 32" ./run_ladder.sh 1
    local -a tail_args=("$@")
    # (never expand an empty array: bash 3.2 with set -u calls that unbound)
    if [ -n "${LADDER_EXTRA:-}" ]; then
        local -a extra=()
        read -r -a extra <<< "$LADDER_EXTRA"
        tail_args+=("${extra[@]}")
    fi
    cmd+=(bash "$HERE/run_hub.sh" "${BASE[@]}" "${tail_args[@]}" --outdir "$out")
    echo
    echo "===================================================================="
    echo "== step $name"
    printf '  %q' "${cmd[@]}"; echo
    echo "===================================================================="
    if [ "$DRY" = "1" ]; then return 0; fi
    "${cmd[@]}" || { echo "!! $name: run_hub.sh failed to launch" >&2; return 1; }
    local pid
    pid="$(cat "$out/run.pid" 2>/dev/null || true)"
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        echo "!! $name: not running -- see logs/$(basename "$out").log" >&2
        tail -20 "logs/$(basename "$out").log" 2>/dev/null
        return 1
    fi
    echo "-- $name running as pid $pid; waiting (log: logs/$name.log)"
    while kill -0 "$pid" 2>/dev/null; do sleep 20; done
    echo "-- $name finished"
    # the run wrote its report itself; show the lines that decide the next step
    if [ -f "$out/report.txt" ]; then
        grep -E "final loss|lambda\(rho_out\)|outer BC residuals|inner BC residuals|inner sphere areal|family read off|max \|lambda - lambda_fam" \
             "$out/report.txt" | sed 's/^/   /' || true
    else
        echo "   (no report.txt -- check logs/$name.log)" >&2
    fi
}

for s in "${WANT[@]}"; do
    case "$s" in
        1) run_one control_ord1b   0 -- "${SMALL[@]}" "${NICE[@]}" --robin-orders h=1,lam=1 ;;
        2) run_one control_ord2    0 -- "${SMALL[@]}" "${NICE[@]}" --robin-orders h=2,lam=2 ;;
        3) run_one control_ord4_x64 1 -- "${SMALL[@]}" "${NICE[@]}" --robin-orders h=4,lam=4 ;;
        4) run_one control_ord1_big 0 -- "${BIG[@]}"   "${NICE[@]}" --robin-orders h=1,lam=1 ;;
        5) run_one dipole_big       0 -- "${BIG[@]}"   "${NICE[@]}" --robin-orders h=1,lam=1 \
                    --lam-bc-S1 0.1 ;;
        6) run_one dipole_small     0 -- "${SMALL[@]}" "${NICE[@]}" --robin-orders h=1,lam=1 \
                    --lam-bc-S1 0.1 ;;
    esac || true
done

[ "$DRY" = "1" ] && { echo; echo "(--dry-run: nothing launched)"; exit 0; }

# ------------------------------------------------------------------ comparison
RUNS=()
for n in control_ord1b control_ord2 control_ord4_x64 control_ord1_big dipole_big dipole_small; do
    [ -f "runs/$n/config.json" ] && RUNS+=("runs/$n")
done
[ ${#RUNS[@]} -eq 0 ] && { echo "no runs to compare yet" >&2; exit 0; }

echo
echo "=== comparing ${RUNS[*]} ==="
mkdir -p logs
MPLBACKEND=Agg MPLCONFIGDIR="${MPLCONFIGDIR:-$HERE/.mplcache}" \
    "$PY" -m stationary.compare "${RUNS[@]}" --json logs/compare.json \
    2>&1 | tee logs/ladder_compare.txt

echo
echo "the table above is also in logs/ladder_compare.txt (raw numbers: logs/compare.json)"
