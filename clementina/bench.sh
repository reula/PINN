#!/bin/bash
# ---------------------------------------------------------------------------
# DSGNAR benchmark on ONE Intel GPU, FLAT vs COMPOSITE tiles.
#
#     bash bench.sh gpu [flat|composite]
#
# CPU runs are deliberately NOT offered: this allocation has few CPU hours, and a
# production recipe would cost ~640 CPU-hours on 64 cores against ~20 GPU-hours.
# Everything here is GPU-only.
#
# Env knobs: NCOLLS (default "2201 16384"), STEPS (15), SKETCH (300),
#            STEP_TIMEOUT (900 s per case).
#
# The tiny run (512 points) was 2.9x faster on the CPU than on one tile: at that
# size the DSGNAR step is dominated by per-step Python/host-sync overhead, which
# a GPU makes worse.  This bench measures the production batch (2201) and a
# Stationary-sized batch (16384) to project the real GPU cost, and checks whether
# COMPOSITE (both tiles as one 128 GB device) beats FLAT.
# ---------------------------------------------------------------------------
set -uo pipefail

TAG="${1:-gpu}"
MODE="${2:-flat}"
CXXI_DIR="${CXXI_DIR:-/home/oreula/Julia/PINN/clementina}"
[ -f "$CXXI_DIR/env.sh" ] || { echo "FATAL: no $CXXI_DIR/env.sh" >&2; exit 1; }
# shellcheck source=env.sh
source "$CXXI_DIR/env.sh"

if [ "$TAG" != "gpu" ]; then
    echo "FATAL: this benchmark is GPU-only (CPU hours are scarce on this allocation)." >&2
    exit 1
fi

# The login shell's Lmod environment is exported into the job by Slurm; the pip
# oneAPI runtime needs none of it.
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
cxxi_pick_data

PY="$(cxxi_python)" || { echo "FATAL: no venv; run 01_setup_env.sh" >&2; exit 1; }
NCOLLS="${NCOLLS:-2201 16384}"
STEPS="${STEPS:-15}"
SKETCH="${SKETCH:-300}"
STEP_TIMEOUT="${STEP_TIMEOUT:-900}"
RUNROOT="${RUNROOT:-$CXXI_DATA/Evolution_try}"
PROJ="$CXXI_REPO/Evolution_try"
CSV="$CXXI_DATA/bench_gpu.csv"
cd "$PROJ" || { echo "FATAL: no $PROJ" >&2; exit 1; }
mkdir -p "$RUNROOT/runs"

case "$MODE" in
    composite) export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE; unset ZE_AFFINITY_MASK ;;
    flat)      unset ZE_FLAT_DEVICE_HIERARCHY ZE_AFFINITY_MASK ;;
    *) echo "FATAL: MODE must be flat or composite" >&2; exit 1 ;;
esac
cxxi_pin_gpu

echo "mode       $MODE"
echo "host       $(hostname)   $(date -u '+%F %T UTC')"
echo "job        ${SLURM_JOB_ID:-?}  ${SLURM_JOB_PARTITION:-?}  cpus ${SLURM_CPUS_PER_TASK:-?}"
echo "python     $PY"
echo "head       $(git rev-parse --short HEAD 2>/dev/null)"
echo "grid       n_coll='$NCOLLS'  steps=$STEPS  sketch=$SKETCH"
echo "csv        $CSV"
echo "ZE         ZE_FLAT_DEVICE_HIERARCHY=${ZE_FLAT_DEVICE_HIERARCHY:-<unset=FLAT>}  ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<unset>}"
"$PY" -c "import jax; print('devices', jax.devices(), '| backend', jax.default_backend())"
echo

[ -f "$CSV" ] || echo "mode,n_coll,steps,dsgnar_s,ms_per_iter,projected_10window_h,wall_s,rc" > "$CSV"

for NC in $NCOLLS; do
    LABEL="cxxi_bench_${MODE}_nc${NC}"
    OUT="$RUNROOT/runs/$LABEL"
    rm -rf "$OUT"
    mkdir -p "$OUT"
    echo "########## $(date -u '+%T')  $LABEL"
    t0=$(date +%s)
    timeout "$STEP_TIMEOUT" "$PY" -m wave_pinn.cli --label "$LABEL" --outdir "$OUT" \
        --set T=2 --set windows=1 --set ansatz=t2 \
        --set window_ic=hard --set w_ic=100 \
        --set n_coll="$NC" --set sampler=random \
        --set "snapshot_times=[0,1,2]" \
        --set optimizer=dsgnar --set dsgnar_steps="$STEPS" --set dsgnar_sketch="$SKETCH" \
        --set dsgnar_delta0=1.0 --set dsgnar_delta_min=1e-14 --set resample_rounds=1 \
        2>&1 | tee "$OUT/run.log"
    rc=${PIPESTATUS[0]}
    wall=$(( $(date +%s) - t0 ))

    line=$(grep -E 'DSGNAR: loss .* in [0-9]+ iterations \([0-9.]+s\)' "$OUT/run.log" | tail -1)
    its=$(printf '%s\n' "$line" | grep -oE 'in [0-9]+ iterations' | grep -oE '[0-9]+')
    secs=$(printf '%s\n' "$line" | grep -oE '\([0-9.]+s\)' | tr -d '()s')
    timing=$(grep -E 'DSGNAR: timing:' "$OUT/run.log" | tail -1)

    ms="NA"; proj="NA"
    if [ -n "${secs:-}" ] && [ -n "${its:-}" ] && [ "$its" -gt 0 ] 2>/dev/null; then
        read -r ms proj <<< "$("$PY" -c "
secs, its = float('$secs'), int('$its')
print(f'{1000*secs/its:.1f}', f'{secs/its*15000/3600:.1f}')")"
    fi
    echo
    echo "[bench] mode=$MODE n_coll=$NC steps=${its:-?} dsgnar=${secs:-?}s  ms/iter=$ms  -> ~${proj} GPU-h for the full 10-window recipe  wall=${wall}s  rc=$rc"
    echo "        ${timing:-no timing line}"
    echo "$MODE,$NC,${its:-NA},${secs:-NA},$ms,$proj,$wall,$rc" >> "$CSV"
    echo
done

echo "===== $CSV ====="
cat "$CSV"
echo
echo "projected_10window_h = ms/iter * 15000 steps (10 windows x 5 rounds x 300 steps) / 3600"
