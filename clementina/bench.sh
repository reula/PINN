#!/bin/bash
# ---------------------------------------------------------------------------
# DSGNAR benchmark: is one Intel GPU or 64 CPU cores the right place for these
# PINNs?  Called by 06_bench_gpu.sbatch and 07_bench_cpu.sbatch; both run the
# same grid, so the two CSVs are directly comparable.
#
#     bash bench.sh gpu|cpu
#
# Env knobs: NCOLLS (default "2201 16384"), STEPS (20), SKETCH (300).
#
# The tiny run (512 points) was 2.9x faster on the CPU: at that size the DSGNAR
# step is dominated by per-step Python/host-sync overhead, which the GPU makes
# worse, not better.  This bench asks whether the production batch (n_coll=2201)
# and a Stationary-sized batch (16384) change that.
# ---------------------------------------------------------------------------
set -uo pipefail

TAG="${1:-gpu}"
CXXI_DIR="${CXXI_DIR:-/home/oreula/Julia/PINN/clementina}"
[ -f "$CXXI_DIR/env.sh" ] || { echo "FATAL: no $CXXI_DIR/env.sh" >&2; exit 1; }
# shellcheck source=env.sh
source "$CXXI_DIR/env.sh"

# The login shell's Lmod environment is exported into the job by Slurm; the pip
# oneAPI runtime needs none of it.
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
cxxi_pick_data

PY="$(cxxi_python)" || { echo "FATAL: no venv; run 01_setup_env.sh" >&2; exit 1; }
NCOLLS="${NCOLLS:-2201 16384}"
STEPS="${STEPS:-20}"
SKETCH="${SKETCH:-300}"
RUNROOT="${RUNROOT:-$CXXI_DATA/Evolution_try}"
PROJ="$CXXI_REPO/Evolution_try"
CSV="$CXXI_DATA/bench_cxxi.csv"
cd "$PROJ" || { echo "FATAL: no $PROJ" >&2; exit 1; }
mkdir -p "$RUNROOT/runs"

if [ "$TAG" = "cpu" ]; then
    export JAX_PLATFORMS=cpu
else
    cxxi_pin_gpu
fi

echo "tag        $TAG"
echo "host       $(hostname)   $(date -u '+%F %T UTC')"
echo "job        ${SLURM_JOB_ID:-?}  ${SLURM_JOB_PARTITION:-?}  cpus ${SLURM_CPUS_PER_TASK:-?}"
echo "python     $PY"
echo "head       $(git rev-parse --short HEAD 2>/dev/null)"
echo "grid       n_coll='$NCOLLS'  steps=$STEPS  sketch=$SKETCH"
echo "csv        $CSV"
echo "ZE         ZE_FLAT_DEVICE_HIERARCHY=${ZE_FLAT_DEVICE_HIERARCHY:-<unset=FLAT>}  ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<unset>}"
"$PY" -c "import jax; print('devices', jax.devices(), '| backend', jax.default_backend())"
echo

[ -f "$CSV" ] || echo "tag,n_coll,steps,dsgnar_s,ms_per_iter,wall_s,rc" > "$CSV"

for NC in $NCOLLS; do
    LABEL="cxxi_bench_${TAG}_nc${NC}"
    OUT="$RUNROOT/runs/$LABEL"
    rm -rf "$OUT"
    mkdir -p "$OUT"
    echo "########## $(date -u '+%T')  $LABEL"
    t0=$(date +%s)
    "$PY" -m wave_pinn.cli --label "$LABEL" --outdir "$OUT" \
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

    ms="NA"
    if [ -n "${secs:-}" ] && [ -n "${its:-}" ] && [ "$its" -gt 0 ] 2>/dev/null; then
        ms=$("$PY" -c "print(f'{1000*float('$secs')/int('$its'):.1f}')")
    fi
    echo
    echo "[bench] $TAG n_coll=$NC steps=${its:-?} dsgnar=${secs:-?}s  ms/iter=$ms  wall=${wall}s  rc=$rc"
    echo "        ${timing:-no timing line}"
    echo "$TAG,$NC,${its:-NA},${secs:-NA},$ms,$wall,$rc" >> "$CSV"
    echo
done

echo "===== $CSV ====="
cat "$CSV"
echo
echo "Lower ms/iter is better; the row where the tag stops mattering is the crossover."
