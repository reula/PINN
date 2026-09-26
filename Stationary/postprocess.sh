#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Figures, the lambda-vs-rho profile and the text report of an EXISTING run.
#
#   ./postprocess.sh runs/<name>                     # params.pkl, or ckpt.pkl if absent
#   ./postprocess.sh runs/<name> .venv/bin/python    # interpreter
#   ./postprocess.sh runs/<name> .venv/bin/python report    # only this step
#
# Called automatically at the end of every run launched by run_hub.sh (the generated
# <outdir>/job.sh ends with this), and on demand by `./run_hub.sh --post [--outdir RUN]
# [--only STEPS]`. Nothing here touches the trained parameters, so it is safe to run
# repeatedly; a run that crashed part-way can still be processed from its last checkpoint.
#
# Writes into the run directory:
#   diagnostics.png, lambda_inner.png, lambda_multipoles_outer.png,
#   lambda_multipole_decay.png, lambda_vs_rho.png, report.txt
#
# Speed. Most of the wall time is XLA *compilation*, not execution: each module builds its
# own graphs (residuals, each boundary condition, the figures). Two things keep that from
# hurting, both on by default here:
#   * a persistent compilation cache (<repo>/.jaxcache), so the second and later
#     invocations -- including every `--post` on the same architecture -- reuse the
#     compiled executables instead of recompiling; disable with JAX_CACHE=0.
#   * a thread cap (POST_THREADS, default 4, inherited from TRAIN_THREADS). XLA's CPU
#     thread pool defaults to every core on the machine; on a 128-core node that is
#     pure overhead for graphs this small and it fights with any training run still
#     finishing on the same box.
# STEPS may be any of: evaluate, profile, report. Use `report` alone for the fastest
# possible text answer.
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?usage: postprocess.sh RUN_DIR [PYTHON] [STEPS]}"
PY="${2:-python}"
STEPS="${3:-evaluate,profile,report,vtk}"
PARAMS="params.pkl"

cd "$HERE"
if [ ! -f "$OUT/config.json" ]; then
    echo "[post] $OUT has no config.json -- not a run directory" >&2
    exit 1
fi
if [ ! -f "$OUT/$PARAMS" ]; then
    if [ -f "$OUT/ckpt.pkl" ]; then
        echo "[post] no params.pkl in the run dir -> using the last checkpoint ckpt.pkl"
        PARAMS="ckpt.pkl"      # a crashed run has only this
    else
        echo "[post] neither params.pkl nor ckpt.pkl in $OUT -- nothing to post-process" >&2
        exit 1
    fi
fi

# ------------------------------------------------------------------ environment
MPLCONFIGDIR="${MPLCONFIGDIR:-$HERE/.mplcache}"
mkdir -p "$MPLCONFIGDIR" 2>/dev/null || true
export MPLBACKEND=Agg MPLCONFIGDIR

# All cores by default is the wrong default here (see the header).
export OMP_NUM_THREADS="${POST_THREADS:-${TRAIN_THREADS:-4}}"

# Do not grab 75% of the GPU just to draw a few figures next to somebody else's job.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

# Persistent compilation cache: the difference between ~30 s and ~5 s per module on the
# second invocation (and it accumulates across runs of the same architecture).
if [ "${JAX_CACHE:-1}" != "0" ]; then
    export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HERE/.jaxcache}"
    export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
    mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || true
fi

echo
echo "[post] run dir  $OUT"
echo "[post] params   $PARAMS"
echo "[post] python   $PY"
echo "[post] threads  OMP_NUM_THREADS=$OMP_NUM_THREADS (override with POST_THREADS=n)"
if [ -n "${JAX_COMPILATION_CACHE_DIR:-}" ]; then
    echo "[post] cache    $JAX_COMPILATION_CACHE_DIR (JAX_CACHE=0 to disable)"
fi

# `-u`: without it the output is block-buffered when redirected to the log, so `tail -f`
# shows nothing for minutes and a working step looks like a hung one.
run_step() {
    local name="$1" what="$2" t0=$SECONDS rc
    echo "[post] $name: $what"
    "$PY" -u -m "stationary.$name" --outdir "$OUT" --params-file "$PARAMS" "${@:3}"
    rc=$?
    echo "[post] $name finished in $((SECONDS - t0))s (exit $rc)"
    [ $rc -eq 0 ] || echo "[post] $name FAILED (continuing)"
}

want() { case ",$STEPS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

want evaluate && run_step evaluate "diagnostics.png + lambda_inner.png + multipole figures"
want profile  && run_step profile  "lambda_vs_rho.png (table printed below)"
want report   && run_step report   "$OUT/report.txt" --out "$OUT/report.txt"
# VTK for VisIt, only when the run asked for it (--vtk): the shell-conforming spherical grid
# in physical coordinates (stationary.vtk's default --grid spherical, the same export the Weyl
# runs use).  Large and regenerable, so it is gitignored.
if want vtk; then
    if grep -q '"make_vtk": true' "$OUT/config.json" 2>/dev/null; then
        run_step vtk "$OUT/vtk/solution.vtk (VisIt: shell-conforming grid, physical coords)"
    else
        echo "[post] vtk: skipped (this run was not launched with --vtk)"
    fi
fi

echo
echo "[post] files now in $OUT:"
ls -1 "$OUT" 2>/dev/null | sed 's/^/  /'
