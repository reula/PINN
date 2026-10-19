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
# Intel GPU visibility (wiki: Desarrollo/gpus).
#
# Slurm sets ZE_AFFINITY_MASK for the allocation, but the wiki documents that it
# is buggy: with --gpus=N it exposes only the first *tile* of each GPU, so each
# visible "device" is a 64 GB half-card rather than the 128 GB card.  Two knobs:
#
#   CXXI_ZE_MODE            unset (default): leave the Slurm allocation alone.
#                           COMPOSITE: 4 devices/node, each a full 2-tile 128 GB
#                                      GPU (best for one job per GPU).
#                           FLAT     : 8 devices/node, one per tile, 64 GB each.
#   CXXI_UNSET_ZE_MASK=1    unset Slurm's mask and expose every device on the node.
#                           ONLY safe when you own the whole node (--gpus=4).
#
# Do not switch modes blindly: the mask has different syntax in FLAT (tile
# indices 0..7) and COMPOSITE (device.tile pairs 0.0, 0.1, 1.0, ...), so a mask
# that is right in one mode can be wrong and even out of range in the other.
# Run 00_probe.sbatch first and look at what Slurm actually sets.
# ---------------------------------------------------------------------------
if [ -n "${CXXI_ZE_MODE:-}" ]; then
    export ZE_FLAT_DEVICE_HIERARCHY="$CXXI_ZE_MODE"
fi
if [ "${CXXI_UNSET_ZE_MASK:-0}" = "1" ]; then
    unset ZE_AFFINITY_MASK
fi

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
