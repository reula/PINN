#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Launch a stationary-Einstein PINN run on a plain JupyterHub server.
#
# Why this exists: a JupyterHub single-user server is NOT a job scheduler. A
# process started in a JupyterLab terminal belongs to that terminal's session and
# a notebook kernel dies with the server. This launcher detaches the training into
# its own session, writes results to persistent ($HOME) storage, checkpoints every
# CKPT_EVERY steps, and leaves behind an exact command to resume the run.
#
# Usage
#   ./run_hub.sh --check                        # verify env + GPU, run tests, smoke
#   ./run_hub.sh --steps 20000 --n-coll 4096    # launch detached
#   ./run_hub.sh --outdir ~/runs/m3 --arch sym_hybrid --steps 20000
#
# Never launch a multi-hour run from a notebook cell.
#
# Environment overrides
#   OUTDIR         where results go (default <repo>/runs/<timestamp>, i.e. inside the
#                  checkout so that everything stays with the project)
#   LOGDIR         where the log goes            (default <repo>/logs)
#   CKPT_EVERY     checkpoint period, Adam steps (default 500; 0 disables)
#   TRAIN_THREADS  cap on CPU threads            (default 4)
#   PY             interpreter to use            (default python)
#   SMOKE          --check smoke-run output dir   (default <repo>/runs/_smoke)
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-python}"
CKPT_EVERY="${CKPT_EVERY:-500}"
TRAIN_THREADS="${TRAIN_THREADS:-4}"
LOGDIR="${LOGDIR:-$HERE/logs}"
OUTDIR="${OUTDIR:-}"
# Matplotlib must not try to build its font cache in an unwritable home/cache dir
# on every run; point it somewhere persistent and writable.
MPLCONFIGDIR="${MPLCONFIGDIR:-$HERE/.mplcache}"
mkdir -p "$MPLCONFIGDIR" 2>/dev/null || true

# JAX preallocates 75% of the visible GPU memory by default. On a shared hub that is
# antisocial and, when another user already holds the device, it fails outright with
# "cuBlas allocation failure" from gpublasCreate even for trivial ops (jit_add). This
# workload is tiny (13.8k parameters), so allocate on demand. Set
# XLA_PYTHON_CLIENT_MEM_FRACTION=<fraction> to cap it instead, and
# CUDA_VISIBLE_DEVICES=<n> to pick a free device when several are exposed.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
if [ -n "${XLA_PYTHON_CLIENT_MEM_FRACTION:-}" ]; then
    export XLA_PYTHON_CLIENT_MEM_FRACTION
fi

if ! command -v "$PY" >/dev/null 2>&1; then
    echo "error: interpreter '$PY' not found. Activate your environment or set PY=..." >&2
    exit 1
fi

# ---------------------------------------------------------------- device check
# JAX falls back to CPU silently when no accelerator is visible, which turns a
# 20-minute GPU run into a 2-hour CPU run without saying anything. So say it.
check_devices() {
    # How much GPU memory did we actually get? A JupyterHub can hand out a small
    # vGPU slice (e.g. 750 MiB of a 24 GB A30); the smoke run alone needs ~750 MiB,
    # so this is worth knowing before anything else.
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.total,memory.used \
                   --format=csv,noheader 2>/dev/null | sed 's/^/  gpu: /' || true
        TOT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
        if [ "${TOT:-0}" -gt 0 ] && [ "${TOT:-0}" -lt 4096 ]; then
            echo "WARNING: the visible GPU has only ${TOT} MiB. The --check smoke run alone" >&2
            echo "  needs ~750 MiB and a production run scales with --n-coll, so expect" >&2
            echo "  'cuBlas allocation failure' / HAMI OOM. Ask your admin for a larger GPU" >&2
            echo "  profile (GBs, ideally the whole card), or run with JAX_PLATFORMS=cpu." >&2
        fi
    fi
    "$PY" - <<'PYEOF'
import sys
import jax

devs = jax.devices()
print(f"jax {jax.__version__}  ->  {[str(d) for d in devs]}")
if {d.platform for d in devs} == {"cpu"}:
    print("WARNING: JAX sees CPU only. If this hub has GPUs, likely causes:", file=sys.stderr)
    print("  * the server was spawned without a GPU (JupyterHub spawner resource", file=sys.stderr)
    print("    request, not something you can fix from inside the container), or", file=sys.stderr)
    print("  * the CUDA plugin is missing from this environment (see requirements.txt).", file=sys.stderr)
    sys.exit(2)
PYEOF
}

# ------------------------------------------------------------------ check mode
if [ "${1:-}" = "--check" ]; then
    echo "== interpreter =="; "$PY" -V
    echo "== jax devices =="; check_devices || true
    echo "== imports =="
    "$PY" -c "import stationary.train, stationary.evaluate, stationary.multipoles; print('imports ok')"
    echo "== test suite (takes ~4-5 min) =="
    MPLBACKEND=Agg MPLCONFIGDIR="$MPLCONFIGDIR" "$PY" -m pytest tests/ -q
    echo "== smoke run (200 steps, must finish in seconds) =="
    SMOKE="${SMOKE:-$HERE/runs/_smoke}"
    # the shell creates the redirect target before python can create the outdir
    mkdir -p "$(dirname "$SMOKE")"
    # Smoke-test the PRODUCTION architecture, not the generic 3-D one: the 3-D model
    # emits 25 fields and its residual graph is by far the heaviest thing in the repo,
    # so a 3-D smoke run can exhaust a small GPU while the real (axisymmetric) runs fit.
    MPLBACKEND=Agg MPLCONFIGDIR="$MPLCONFIGDIR" "$PY" -m stationary.train \
        --steps 200 --n-coll 256 --n-bnd 64 --lbfgs-steps 0 --arch sym_hybrid \
        --no-figures --outdir "$SMOKE" > "$SMOKE.log" 2>&1 \
        && echo "smoke ok -> $SMOKE/report.json" \
        || { echo "smoke FAILED, last lines of $SMOKE.log:"; tail -20 "$SMOKE.log"; exit 1; }
    echo
    echo "Environment looks good. Launch for real with e.g.:"
    echo "  ./run_hub.sh --steps 20000 --n-coll 4096 --arch sym_hybrid"
    exit 0
fi

# ------------------------------------------------- split off flags we manage
# --outdir and --ckpt-every are extracted so the script can route the log, the
# pid file and resume.sh consistently; everything else is forwarded verbatim.
USER_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --outdir)    OUTDIR="$2"; shift 2 ;;
        --outdir=*)  OUTDIR="${1#*=}"; shift ;;
        --ckpt-every)   CKPT_EVERY="$2"; shift 2 ;;
        --ckpt-every=*) CKPT_EVERY="${1#*=}"; shift ;;
        *) USER_ARGS+=("$1"); shift ;;
    esac
done

if [ -z "$OUTDIR" ]; then
    OUTDIR="$HERE/runs/$(date +%Y%m%d-%H%M%S)"
fi

case "$OUTDIR" in
    /tmp/*|/var/tmp/*)
        echo "warning: OUTDIR is under $OUTDIR -- /tmp is usually wiped with the container." >&2
        echo "         Results may not survive a server restart." >&2
        ;;
esac

if [ -n "${JAX_ENABLE_X64:-}" ] && [ "${JAX_ENABLE_X64}" != "0" ]; then
    echo "WARNING: JAX_ENABLE_X64=${JAX_ENABLE_X64} is set." >&2
    echo "  Training runs in float32 by default and every result under runs/ was" >&2
    echo "  produced that way. Enabling x64 changes the trajectory (and is ~2x slower" >&2
    echo "  on GPU), so this run will not reproduce them." >&2
fi

mkdir -p "$OUTDIR" "$LOGDIR"
NAME="$(basename "$OUTDIR")"
LOG="$LOGDIR/$NAME.log"
PIDFILE="$OUTDIR/run.pid"
RESUME_SH="$OUTDIR/resume.sh"

ENVS=(MPLBACKEND=Agg OMP_NUM_THREADS="$TRAIN_THREADS" MPLCONFIGDIR="$MPLCONFIGDIR")
[ -n "${JAX_PLATFORMS:-}" ] && ENVS+=("JAX_PLATFORMS=$JAX_PLATFORMS")

TRAIN=(env "${ENVS[@]}" "$PY" -m stationary.train
       --outdir "$OUTDIR" --ckpt-every "$CKPT_EVERY" "${USER_ARGS[@]}")

# ------------------------------------------------------- exact resume command
RESUME_CMD=("${TRAIN[@]}" --resume auto)
{
    echo '#!/usr/bin/env bash'
    echo '# Generated by run_hub.sh -- continue this run from its last checkpoint.'
    echo '# Run it only after the previous process has stopped.'
    echo 'set -euo pipefail'
    echo "cd $(printf '%q' "$HERE")"
    printf 'exec'
    printf ' %q' "${RESUME_CMD[@]}"
    printf '\n'
} > "$RESUME_SH"
chmod +x "$RESUME_SH"

# ------------------------------------------------------------------ detach
if command -v setsid >/dev/null 2>&1; then
    setsid nohup bash -c 'echo $$ > "$1"; shift; exec "$@"' _ "$PIDFILE" "${TRAIN[@]}" \
        > "$LOG" 2>&1 < /dev/null &
else
    nohup "${TRAIN[@]}" > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
fi
disown 2>/dev/null || true

sleep 3
PID="$(cat "$PIDFILE" 2>/dev/null || true)"
if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    STATUS="running (pid $PID)"
else
    STATUS="NOT running -- check the log"
fi

cat <<EOF

outdir    $OUTDIR
log       $LOG
pid file  $PIDFILE
status    $STATUS
checkpoint every $CKPT_EVERY Adam steps

watch     tail -f "$LOG"
status    kill -0 \$(cat "$PIDFILE") && echo running || echo stopped
resume    "$RESUME_SH"        # after a crash/restart, continues from the checkpoint
figures   $PY -m stationary.evaluate --outdir "$OUTDIR"   # after it finishes

EOF

case "$STATUS" in
    NOT*) tail -20 "$LOG" ;;
esac
