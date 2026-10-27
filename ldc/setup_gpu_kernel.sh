#!/usr/bin/env bash
# Register a Jupyter kernel from the CUDA environment, so notebooks run on the GPU (not CPU).
#
# Run this ONCE, in the JupyterLab terminal on the hub (the venv only exists there; from the
# serafin.ccad.unc.edu.ar login node this directory does not exist).  Then in JupyterLab:
#   Kernel -> Change Kernel -> "PINN (jax, CUDA)"
#
# Background: a kernel registered from ~/venvs/pinn has no CUDA jax, so `jax.devices()` returns
# ['cpu:0'] and everything runs on the CPU (HUB.md section 1: the CUDA environment is the
# checkout's own Stationary/.venv).  That is what the 2026-10-19 notebook run did.
set -euo pipefail

VENV="${VENV:-$HOME/serafin/Julia/PINN/Stationary/.venv}"
NAME="${NAME:-pinn-cuda}"
DISPLAY="${DISPLAY:-PINN (jax, CUDA)}"

if [ ! -x "$VENV/bin/python" ]; then
    echo "No python at $VENV" >&2
    echo "Set VENV=/path/to/Stationary/.venv and retry." >&2
    exit 1
fi

echo "--- 1/3  does $VENV see a GPU?"
"$VENV/bin/python" - <<'EOF'
import sys
import jax
print(f"python {sys.version.split()[0]}  jax {jax.__version__}  devices {jax.devices()}")
if jax.default_backend() != "gpu":
    sys.exit("This environment does NOT see a GPU (backend=%s). "
             "Fix the CUDA jaxlib install before registering a kernel, or set VENV to the "
             "CUDA environment." % jax.default_backend())
EOF

echo "--- 2/3  ipykernel in that environment"
"$VENV/bin/python" -m pip install -q --disable-pip-version-check ipykernel

echo "--- 3/3  register the kernel"
"$VENV/bin/python" -m ipykernel install --user --name "$NAME" --display-name "$DISPLAY"

"$VENV/bin/python" -m jupyter kernelspec list 2>/dev/null || true
cat <<EOF

Registered '$NAME'.  In JupyterLab: Kernel -> Change Kernel -> "$DISPLAY".
If the menu does not show it, refresh the page (or ask the hub to restart your server).
Check it worked: the notebook's first cell must print  backend gpu  and  GPU: <name>.
EOF
