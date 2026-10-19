# shellcheck shell=bash
# ---------------------------------------------------------------------------
# Shared settings for Clementina XXI.  Source it, never execute it:
#
#     source ~/Julia/PINN/clementina/env.sh
#
# Every value can be overridden from the environment before sourcing, e.g.
#
#     CXXI_ZE_MODE=FLAT sbatch 02_smoke.sbatch
#
# Account: pci_143 (IPAC PCI number 143, user oreula).
# ---------------------------------------------------------------------------

export CXXI_ACCOUNT="${CXXI_ACCOUNT:-pci_143}"
export CXXI_USER="${CXXI_USER:-oreula}"

# Where the checkout and the Python environment live (small, on /home).
export CXXI_REPO="${CXXI_REPO:-$HOME/Julia/PINN}"
export CXXI_PROJECT="${CXXI_PROJECT:-Evolution_try}"
export CXXI_VENV="${CXXI_VENV:-$HOME/venvs/pinn-oneapi}"

# Where runs/logs/caches go (large, on the 1 PB /data filesystem; /home is 500 GB).
export CXXI_DATA="${CXXI_DATA:-/data/contrib/${CXXI_ACCOUNT}/pinn}"

# ---------------------------------------------------------------------------
# Internet.  On Clementina, login and compute nodes reach the outside world only
# through the SMN proxy (wiki: FAQ/acceso_internet_externo).  Needed by uvicorn
# and git; JAX itself never phones home.
# ---------------------------------------------------------------------------
export http_proxy="${http_proxy:-http://172.28.3.3:3128}"
export https_proxy="${https_proxy:-http://172.28.3.3:3128}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
# Do not send the cluster's own addresses through the proxy.
export no_proxy="${no_proxy:-localhost,127.0.0.1,ssh.clementinaxxi.org.ar,172.28.0.0/16,172.29.0.0/16}"
export NO_PROXY="$no_proxy"

# ---------------------------------------------------------------------------
# Intel GPU visibility.  Two probe jobs (cn034 with --gpus=1, cn038 with
# --gpus=1 and the debug job) showed:
#
#   * Slurm does NOT set ZE_AFFINITY_MASK here, and the device nodes are NOT
#     filtered: on cn038 a one-GPU job could see three Max 1550s through
#     /dev/dri (xpu-smi listed 3).  CUDA_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES
#     do not influence Level Zero, so JAX enumerates every visible card -- two
#     tile-devices per card in the default FLAT hierarchy -- unless we mask.
#   * cn034 showed a single card, so how many cards are visible depends on what
#     else is running/present on the node, not on our --gpus request.
#
# Consequence: for a single-device program the default device is whichever card
# Level Zero enumerates first, which is *probably* the Slurm-allocated one but is
# not guaranteed.  Knobs:
#
#   CXXI_ZE_MODE=COMPOSITE   one 128 GB device per card (2 tiles each).
#   CXXI_ZE_MODE=FLAT        default; one 64 GB device per tile.
#   CXXI_ZE_MASK              set ZE_AFFINITY_MASK explicitly.  Syntax follows
#                            the hierarchy: FLAT lists tiles (0,1 = card 0;
#                            2,3 = card 1; ...), COMPOSITE lists device.tile
#                            pairs (0.0,0.1 = card 0).
#   CXXI_UNSET_ZE_MASK=1     drop a mask inherited from somewhere else.
# ---------------------------------------------------------------------------
if [ -n "${CXXI_ZE_MODE:-}" ]; then
    export ZE_FLAT_DEVICE_HIERARCHY="$CXXI_ZE_MODE"
fi
if [ "${CXXI_UNSET_ZE_MASK:-0}" = "1" ]; then
    unset ZE_AFFINITY_MASK
fi

# ---------------------------------------------------------------------------
# Pin Level Zero to the GPU Slurm gave us.
#
# /dev/dri is not filtered (cn038: a --gpus=1 job saw 3 cards = 6 tile-devices),
# so JAX enumerates every card on the node unless we mask.  Slurm sets no
# ZE_AFFINITY_MASK, and CUDA_VISIBLE_DEVICES does nothing for Level Zero.
#
# Device order matches Slurm's: in the debug job on cn038, xpu-smi topology gave
# GPU 0/0 the CPU affinity 0-7, exactly the cpuset Slurm allocated, while GPUs 1
# and 2 had none -- i.e. Slurm's GPU 0 is Level Zero device 0.
#
# FLAT lists tiles (2g, 2g+1 per card); COMPOSITE lists device.tile pairs.
# CXXI_ZE_MASK=off disables the pinning.
# ---------------------------------------------------------------------------
cxxi_pin_gpu() {
    [ "${CXXI_ZE_MASK:-on}" = "off" ] && return 0
    [ -n "${ZE_AFFINITY_MASK:-}" ] && return 0
    [ -n "${SLURM_JOB_GPUS:-}" ] || return 0
    local g out="" IFS=,
    for g in $SLURM_JOB_GPUS; do
        g="${g//[!0-9]/}"                       # "0" or "0,1" or gres syntax
        [ -n "$g" ] || continue
        if [ "${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}" = "COMPOSITE" ]; then
            out="${out:+$out,}$g.0,$g.1"
        else
            out="${out:+$out,}$((2 * g)),$((2 * g + 1))"
        fi
    done
    if [ -n "$out" ]; then
        export ZE_AFFINITY_MASK="$out"
        echo "pinned ZE_AFFINITY_MASK=$out  (Slurm GPUs: ${SLURM_JOB_GPUS}; CXXI_ZE_MASK=off to disable)"
    fi
}

# ---------------------------------------------------------------------------
# Caches.  XLA's compilation cache is worth a lot for these workloads (the DSGNAR
# trace is big and the same shape is recompiled across resample rounds).
# ---------------------------------------------------------------------------
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$CXXI_DATA/.cache/jax}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CXXI_DATA/.cache/mpl}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

# CUDA-only in JAX, harmless on the oneAPI backend; wave_pinn sets it itself.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

# Pick a writable run root: /data if the IPAC directory exists, else $HOME.
cxxi_pick_data() {
    if mkdir -p "$CXXI_DATA" 2>/dev/null && [ -w "$CXXI_DATA" ]; then
        :
    else
        CXXI_DATA="$HOME/pinn_runs"
        mkdir -p "$CXXI_DATA" 2>/dev/null || true
        echo "note: /data/contrib/$CXXI_ACCOUNT not writable; using $CXXI_DATA" >&2
    fi
    export CXXI_DATA
    mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$MPLCONFIGDIR" 2>/dev/null || true
}

# Resolve the interpreter: explicit $PY wins, then the oneAPI venv, then a CPU venv.
cxxi_python() {
    local py
    for py in "${PY:-}" "$CXXI_VENV/bin/python" "$HOME/venvs/pinn-cpu/bin/python"; do
        [ -n "$py" ] && [ -x "$py" ] || continue
        printf '%s\n' "$py"
        return 0
    done
    return 1
}

# Refuse to pretend: these runs take hours and a silent CPU fallback looks like a
# slow GPU rather than an environment problem.  Prints the backend either way.
cxxi_assert_accelerator() {
    local py="$1"
    "$py" - "$@" <<'PYEOF'
import sys
import jax

devs = jax.devices()
backend = jax.default_backend()
print(f"jax {jax.__version__} | backend {backend} | devices {devs}")
if devs:
    print(f"device_kind: {devs[0].device_kind}")
if backend == "cpu":
    print("FATAL: JAX fell back to CPU; the Intel oneAPI plugin did not load a GPU.", file=sys.stderr)
    sys.exit(3)
PYEOF
}
