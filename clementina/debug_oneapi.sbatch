#!/bin/bash
# ---------------------------------------------------------------------------
# Why did 02_smoke.sbatch sit on cn021 for 15 minutes without printing anything?
#
# Every step below is wrapped in `timeout`, so a hang is reported as
# "rc=124 (TIMED OUT)" and the next step still runs.  Fifteen bounded steps are
# cheaper than one more 15-minute blind submission.
#
#     cd ~/Julia/PINN
#     sbatch -p gpunode -t 00:20:00 clementina/debug_oneapi.sbatch
#     # then read clementina/debug_<jobid>.out
#
# The suspects, in order:
#   1. xpu-smi hanging on the assigned GPU (driver/ioctl) -- probe's cn034 was fine
#   2. python startup / `import numpy` on the GPFS venv
#   3. `import jax` initialising the oneAPI PJRT plugin (521 MB .so)
#   4. the plugin enumerating Level Zero devices with no ZE_AFFINITY_MASK set
# ---------------------------------------------------------------------------
#SBATCH --job-name=cxxi_debug
#SBATCH --account=pci_143
#SBATCH --partition=gpunode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/oreula/Julia/PINN/clementina/debug_%j.out
#SBATCH --error=/home/oreula/Julia/PINN/clementina/debug_%j.err

set -uo pipefail

CXXI_DIR="${CXXI_DIR:-/home/oreula/Julia/PINN/clementina}"
[ -f "$CXXI_DIR/env.sh" ] || { echo "FATAL: no $CXXI_DIR/env.sh" >&2; exit 1; }
# shellcheck source=env.sh
source "$CXXI_DIR/env.sh"
cxxi_pick_data
PY="$(cxxi_python)" || { echo "FATAL: no venv; run 01_setup_env.sh" >&2; exit 1; }

step() {   # step <timeout-s> <title> <command...>
    local t="$1" title="$2"; shift 2
    echo
    echo "############ $(date -u '+%T') ${title}   [timeout ${t}s]"
    timeout "$t" "$@"
    local rc=$?
    if [ "$rc" = 124 ]; then
        echo "############ rc=124  *** TIMED OUT: the hang is HERE ***"
    else
        echo "############ rc=$rc"
    fi
}

sstep() {  # sstep <timeout-s> <title> <shell-string>
    local t="$1" title="$2"; shift 2
    step "$t" "$title" bash -c "$*"
}

echo "host     $(hostname)   $(date -u '+%F %T UTC')"
echo "job      ${SLURM_JOB_ID:-?}  partition ${SLURM_JOB_PARTITION:-?}  account ${SLURM_JOB_ACCOUNT:-?}"
echo "head     $(git -C "$CXXI_REPO" rev-parse --short HEAD 2>/dev/null)"
echo "python   $PY"
echo "script   $CXXI_DIR/debug_oneapi.sbatch"
echo
echo "===== allocation and environment ====="
sstep 10 "ulimit"            'ulimit -a'
sstep 10 "cgroup CPUs"       'nproc; grep -E "^Cpus_allowed_list" /proc/self/status'
sstep 10 "memory"            'free -g'
sstep 10 "GPU device nodes"  'ls -l /dev/dri 2>&1; ls -l /dev/accel* 2>&1'
sstep 10 "visibility env"    'env | grep -iE "^(ZE_|ONEAPI_|SYCL_|CUDA_|ROCR_|SLURM_|OMP_|KMP_|MKL_|TBB_|PYTHON|LD_)" | sort'
sstep 10 "driver versions"   'cat /sys/module/i915/version 2>/dev/null; cat /sys/module/xe/version 2>/dev/null; ls /lib64/libze* 2>/dev/null'

echo
echo "===== suspect 1: xpu-smi / Level Zero ==="
sstep 20 "xpu-smi discovery"    'xpu-smi discovery'
sstep 20 "xpu-smi topology -m"  'xpu-smi topology -m'

echo
echo "===== suspect 2: python startup and numpy on the GPFS venv ==="
sstep 30 "python -V"            '"$PY" -V; readlink -f "$PY"'
sstep 60 "import sys only"      '"$PY" -u -c "import sys; print(\"sys ok\", sys.version.split()[0])"'
sstep 120 "import numpy"        '"$PY" -u -c "import numpy; print(\"numpy ok\", numpy.__version__)"'
sstep 60 "venv size/stat"       'du -sh "$(dirname "$(dirname "$PY")")" 2>/dev/null; df -h "$(dirname "$(dirname "$PY")")"'

echo
echo "===== suspect 3/4: the oneAPI plugin and JAX ==="
sstep 120 "jax on CPU (JAX_PLATFORMS=cpu)" \
      'JAX_PLATFORMS=cpu "$PY" -u -c "import jax; print(\"jax\", jax.__version__, jax.devices(), jax.default_backend())"'
sstep 60 "import jax_oneapi_plugin only" \
      '"$PY" -u -c "import jax_oneapi_plugin; print(\"plugin module ok\")"'
sstep 240 "import jax + devices() on oneapi" \
      '"$PY" -u -c "import jax; print(\"jax imported\", jax.__version__, flush=True); print(\"devices\", jax.devices(), flush=True); print(\"backend\", jax.default_backend(), flush=True)"'

echo
echo "===== strace of the last step, if strace exists (hang location) ====="
if command -v strace >/dev/null 2>&1; then
    sstep 90 "strace import jax + devices" \
        'strace -f -tt -e trace=futex,ioctl,connect,openat -o /tmp/strace_jax.txt "$PY" -u -c "import jax; print(jax.devices())" >/dev/null 2>&1; tail -40 /tmp/strace_jax.txt'
else
    echo "strace not installed"
fi

echo
echo "===== the real smoke test, unbuffered and time-boxed ====="
sstep 300 "smoke.py --skip-bench" '"$PY" -u "$CXXI_DIR/smoke.py" --skip-bench'

echo
echo "===== done $(date -u '+%F %T UTC') ====="
echo "Any step with rc=124 is where it hangs."
