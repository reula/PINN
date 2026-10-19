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
# Intel GPU visibility (wiki: Desarrollo/gpus).  Probe job 1547927 on cn034, a
# --gpus=1 allocation, showed:
#
#   * Slurm here does NOT set ZE_AFFINITY_MASK; the allocation is enforced one
#     level lower, in the cgroup: xpu-smi discovery listed only Device ID 0
#     (physical, /dev/dri/card1) and xpu-smi topology showed just GPU 0/0 and
#     0/1.  Nothing outside the allocation is visible to the job.
#   * with the default hierarchy (FLAT, unset) each *tile* is a device, so JAX
#     reports [OneapiDevice(0), OneapiDevice(1)]: two 64 GB devices.  Single
#     device code like wave_pinn uses device 0 and has 64 GB, which is 2-3x what
#     the A30 runs used.
#
# Knobs:
#   CXXI_ZE_MODE=COMPOSITE   one 128 GB device spanning both tiles (the wiki's
#                            recommendation for programs that want the whole
#                            card).  Worth a smoke-test measurement, not needed.
#   CXXI_ZE_MODE=FLAT        the default; one device per tile.
#   CXXI_UNSET_ZE_MASK=1     drop a ZE_AFFINITY_MASK set by Slurm.  There is none
#                            on CXXI today; kept for other clusters.
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
