#!/bin/bash
# ---------------------------------------------------------------------------
# Step 1: find out what the machine actually gives us.  Read the output before
# running anything else -- the rest of this directory is written against the
# answers, and a couple of them (partition names, the ZE mask, whether python3
# is new enough) are guessed from the wiki in advance.
#
# From the checkout root (~/Julia/PINN), on the login node:
#
#     cd ~/Julia/PINN/clementina
#     sbatch 00_probe.sbatch
#     tail -n +1 probe_*.out          # ~1-2 min: the testing queue is short
#
# The testing partition is capped at 20 minutes and is a shared courtesy queue,
# so this asks for one GPU, four cores, ten minutes.
# ---------------------------------------------------------------------------
#SBATCH --job-name=cxxi_probe
#SBATCH --account=pci_143
#SBATCH --partition=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=probe_%j.out
#SBATCH --error=probe_%j.err

set -u

section() { printf '\n\n===== %s =====\n' "$*"; }
run() {
    # Run a diagnostic, never let it kill the probe, indent its output.
    printf '\n--- %s\n' "$*"
    "$@" 2>&1 | sed 's/^/    /'
    printf -- '--- exit %s\n' "${PIPESTATUS[0]}"
}

section "identity"
run hostname
run whoami
run date -u
run bash -c 'echo "SLURM_JOB_ID=$SLURM_JOB_ID  SLURM_JOB_PARTITION=$SLURM_JOB_PARTITION  SLURM_JOB_ACCOUNT=$SLURM_JOB_ACCOUNT"'
run bash -c 'scontrol show job "$SLURM_JOB_ID" | head -40'

section "GPU allocation as Slurm sees it"
run bash -c 'env | grep -iE "^(ZE_|ONEAPI_|SYCL_|LIBOMPTARGET_|CUDA_|ROCR_|SLURM_.*GPU)" | sort'
run bash -c 'command -v xpu-smi && xpu-smi discovery' 
run bash -c 'command -v sycl-ls || echo "sycl-ls: not on PATH (needs module load oneapi)"'
run xpu-smi topology -m

section "level zero / sycl libraries (what the pip JAX plugin links against)"
run bash -c 'ldconfig -p | grep -iE "libze_|libze1|libsycl|libOpenCL|libur_loader"'
run bash -c 'ls -l /usr/lib64/libze* /usr/lib/libze* 2>/dev/null || echo "no /usr/lib64/libze*"'
run bash -c 'ls -d /opt/intel/oneapi 2>/dev/null && ls /opt/intel/oneapi | head -20'
run bash -c 'ls /opt/ohpc/pub/modulefiles 2>/dev/null'

section "modules (the wiki lists gnu12/intel/oneapi, not python or jax)"
run module list
run bash -c 'module avail 2>&1 | head -80'
run bash -c 'module spider oneapi 2>&1 | head -40'
run bash -c 'module spider python 2>&1 | head -30'
run bash -c 'module spider level-zero 2>&1 | head -20'

section "sycl-ls, with and without the Slurm mask"
run bash -c 'module load oneapi 2>/dev/null; sycl-ls 2>&1 | head -20'
run bash -c 'module load oneapi 2>/dev/null; unset ZE_AFFINITY_MASK; sycl-ls 2>&1 | head -20'
run bash -c 'module load oneapi 2>/dev/null; echo "ZE_FLAT_DEVICE_HIERARCHY=${ZE_FLAT_DEVICE_HIERARCHY:-<unset, = FLAT>}"; echo "ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<unset>}"'

section "python: the oneAPI JAX needs >= 3.12; Rocky 8 ships 3.6/3.9"
run bash -c 'for c in python3 python3.12 python3.11 python3.9 python3.6; do command -v $c >/dev/null && { printf "%s -> " $c; $c -V; }; done; echo done'
run bash -c 'python3 -c "import sys, platform; print(sys.executable, sys.version); print(platform.platform())"'
run bash -c 'python3 -m pip --version 2>&1 | head -3'
run bash -c 'command -v uv || echo "uv: absent"'
run bash -c 'python3 -c "import numpy; print(\"numpy\", numpy.__version__)" 2>&1 | head -2'

section "storage"
run bash -c 'df -h /home /data 2>/dev/null'
run bash -c 'ls -ld /data/contrib /data/contrib/pci_143 2>&1'
run bash -c 'echo HOME=$HOME; du -sh "$HOME" 2>/dev/null'

section "scheduler: real partition names, limits, and the account"
run bash -c 'sinfo -o "%.20P %.5a %.12l %.6D %.10N %f"'
run bash -c 'scontrol show partition gpunode 2>&1 | head -25'
run bash -c 'sacctmgr -n -P show assoc user="$USER" format=Account,Partition,QOS,MaxJobs 2>&1 | head -10'
run bash -c 'sshare -u "$USER" -A "$SLURM_JOB_ACCOUNT" 2>&1 | head -10'
run bash -c 'squeue -u "$USER" -o "%.10i %.30j %.8T %.10M %R" 2>&1 | head'

section "hardware"
run bash -c 'nproc; lscpu | grep -E "Model name|Socket|Core|Thread|NUMA node\(s\)"'
run bash -c 'free -g'
run bash -c 'lsblk 2>/dev/null | head -10'

section "internet through the SMN proxy"
run bash -c 'echo "http_proxy=${http_proxy:-<unset>} https_proxy=${https_proxy:-<unset>}"'
run bash -c 'curl -sS -o /dev/null -w "pypi without proxy: %{http_code} in %{time_total}s\n" --max-time 20 --noproxy "*" https://pypi.org/simple/jax/'
run bash -c 'curl -sS -o /dev/null -w "pypi with proxy   : %{http_code} in %{time_total}s\n" --max-time 20 https://pypi.org/simple/jax/'

section "done"
echo "Send probe_${SLURM_JOB_ID}.out (and .err) back.  The three things that"
echo "decide the next step are in 'GPU allocation', 'level zero / sycl libraries'"
echo "and 'python'."
