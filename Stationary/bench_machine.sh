#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Machine benchmark: a short run of the paper's recipe (ab_none: axisymmetric quadrupole,
# S2 = -1/12, width 20, depth 6, 16384 + 1024 + 1024 points, redrawn every 500), shortened to
#   Adam 1000, then SSBroyden (strong Wolfe) 1000,
# with a fixed seed, on the CPU. JAX compiles from an empty cache, so the times include compilation.
#
#   bash bench_machine.sh            (from Stationary/, with .venv set up; ~10-30 min on a laptop)
#   SMOKE=1 bash bench_machine.sh    tiny points/iterations, checks the flags only (not logged)
#
# Writes runs/_bench/<host>/{adam,ssb}/ and appends one line per stage plus the total to
# logs/bench_machines.txt (host, chip, cores, seconds). Nothing else may be running on the machine.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="$HERE/.venv/bin/python"
HOST=$(hostname -s)
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)
CORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo ?)P+$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo ?)E
CACHE=$(mktemp -d)
export JAX_ENABLE_X64=1 MPLBACKEND=Agg JAX_PLATFORMS=cpu JAX_COMPILATION_CACHE_DIR="$CACHE"
OUT=runs/_bench/$HOST
rm -rf "$OUT"; mkdir -p "$OUT" logs

NUM=(--depth 6 --ckpt-every 500 --resample-every 500 --decay-feature --no-figures
     --outer-bc robin --robin-orders h=3,lam=3 --no-robin-G
     --eq-weights compat=0,ricci=1,gauge=1,lam_eq=1 --scale-exps 3,4,3,4
     --outer-shell-fracs 1.0 --no-pin-h-robin --no-pin-lam
     --arch axisym_hybrid --width 20 --n-coll 16384 --n-bnd 1024 --n-bnd-outer 1024
     --R0 0.5773502691896258 --rho-in 1.0 --rho-out 100.0 --inner-radius 1.0 --vtk-physical-inner 1.0
     --lam0 0.33333333333333337 --lam-inf 1.0 --ref-solution --ref-asymptotic 1.0
     --com-pin 1,1.5,3,10,30,100 --lam-bc-S2 -0.08333333333333333 --seed 0)

stage() {   # stage <name> <flags...>: runs it, prints the wall seconds
    local name=$1; shift
    local t0=$(date +%s)
    local small=(); [ "${SMOKE:-0}" = 1 ] && small=(--n-coll 256 --n-bnd 64 --n-bnd-outer 64 --steps 0 --lbfgs-steps 0)
    [ "${SMOKE:-0}" = 1 ] && [ "$name" = adam ] && small+=(--steps 20)
    [ "${SMOKE:-0}" = 1 ] && [ "$name" = ssb ] && small+=(--lbfgs-steps 20 --qn-block 10)
    "$PY" -u -m stationary.train --outdir "$OUT/$name" "${NUM[@]}" "$@" ${small[@]+"${small[@]}"} > "$OUT/$name.log" 2>&1
    echo $(( $(date +%s) - t0 ))
}
echo "bench on $HOST ($CHIP, $CORES)  $(date -u '+%F %T UTC')"
TA=$(stage adam --steps 1000 --plateau-min-iters -1 --lbfgs-steps 0)
echo "  adam 1000: ${TA} s   $(grep final_loss "$OUT/adam.log" | tail -1 | tr -s ' ')"
TS=$(stage ssb --steps 0 --qn-method ssbroyden --qn-linesearch wolfe --lbfgs-steps 1000 \
         --plateau-min-iters -1 --init-from "$OUT/adam/params.pkl")
echo "  ssbroyden 1000: ${TS} s   $(grep final_loss "$OUT/ssb.log" | tail -1 | tr -s ' ')"
echo "  total: $((TA + TS)) s"
[ "${SMOKE:-0}" = 1 ] || printf '%s\t%s\t%s\t%s\tadam=%ss\tssb=%ss\ttotal=%ss\n' "$(date -u '+%F %T')" "$HOST" "$CHIP" "$CORES" \
    "$TA" "$TS" "$((TA + TS))" >> logs/bench_machines.txt
rm -rf "$CACHE"
