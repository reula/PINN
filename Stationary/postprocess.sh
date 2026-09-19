#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Figures, the lambda-vs-rho profile and the text report of an EXISTING run.
#
#   ./postprocess.sh runs/<name>            # params.pkl, or ckpt.pkl if absent
#   ./postprocess.sh runs/<name> .venv/bin/python
#
# Called automatically at the end of every run launched by run_hub.sh (the generated
# <outdir>/job.sh ends with this), and on demand by `./run_hub.sh --post [--outdir RUN]`.
# Nothing here touches the trained parameters, so it is safe to run repeatedly; a run
# that crashed part-way can still be processed from its last checkpoint.
#
# Writes into the run directory:
#   diagnostics.png, lambda_inner.png, lambda_multipoles_outer.png,
#   lambda_multipole_decay.png, lambda_vs_rho.png, report.txt
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?usage: postprocess.sh RUN_DIR [PYTHON]}"
PY="${2:-python}"
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

MPLCONFIGDIR="${MPLCONFIGDIR:-$HERE/.mplcache}"
mkdir -p "$MPLCONFIGDIR" 2>/dev/null || true
export MPLBACKEND=Agg MPLCONFIGDIR

echo
echo "[post] run dir  $OUT"
echo "[post] params   $PARAMS"

echo "[post] evaluate -> diagnostics.png + lambda_inner.png + multipole figures"
"$PY" -m stationary.evaluate --outdir "$OUT" --params-file "$PARAMS" \
    || echo "[post] evaluate FAILED (continuing)"

echo "[post] profile  -> lambda_vs_rho.png   (table printed below)"
"$PY" -m stationary.profile --outdir "$OUT" --params-file "$PARAMS" \
    || echo "[post] profile FAILED (continuing)"

echo "[post] report   -> $OUT/report.txt"
"$PY" -m stationary.report --outdir "$OUT" --params-file "$PARAMS" --out "$OUT/report.txt" \
    || echo "[post] report FAILED (continuing)"

echo
echo "[post] files now in $OUT:"
ls -1 "$OUT" 2>/dev/null | sed 's/^/  /'
