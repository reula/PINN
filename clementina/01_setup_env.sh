#!/bin/bash
# ---------------------------------------------------------------------------
# Step 2: build the Python environment for JAX on the Intel GPUs.
#
# Run ON THE LOGIN NODE (it needs the internet, which the login node has through
# the proxy; the compute nodes are for jobs, not for downloading wheels):
#
#     cd ~/Julia/PINN/clementina
#     bash 01_setup_env.sh
#
# What it does, in order:
#   1. sanity-check the proxy;
#   2. get `uv` (a single static binary; no root, no conda);
#   3. let uv fetch a CPython 3.12 (the cluster's python3 is Rocky 8 vintage and
#      the oneAPI JAX wheels require >= 3.12);
#   4. create ~/venvs/pinn-oneapi and `uv pip install "jax[oneapi]==0.11.2"`;
#   5. verify the import and print what was installed.
#
# Size: the oneAPI PJRT plugin alone is a ~500 MB .so and the `with-oneapi` extra
# pulls the oneAPI runtime from PyPI (mkl, sycl-rt, tbb, ...); budget ~2 GB in
# $HOME.  Nothing here needs root.
#
# Why 0.11.2 and not the 0.11.1 the repo pins: the only released
# `jax-oneapi-plugin` is 0.11.2 (dev builds exist for 0.11.0/0.11.1), and
# PJRT/plugin/jaxlib/jax versions must match exactly.  The
# code is a patch release apart; step 4 in README.md is how to check that it
# still produces the same numbers.
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

FORCE="${FORCE:-0}"
JAX_PIN="${JAX_PIN:-0.11.2}"
PY_VER="${PY_VER:-3.12}"

say() { printf '\n== %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

mkdir -p "$(dirname "$CXXI_VENV")" "$CXXI_DATA" 2>/dev/null || true

say "proxy check (https://pypi.org through ${https_proxy})"
code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 25 https://pypi.org/simple/jax/ 2>/dev/null || true)
if [ "$code" != "200" ]; then
    code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 25 --noproxy '*' https://pypi.org/simple/jax/ 2>/dev/null || true)
    if [ "$code" = "200" ]; then
        say "no proxy needed; unsetting http(s)_proxy for this script"
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    else
        die "cannot reach PyPI (HTTP ${code:-none}).  Check the proxy: export https_proxy=http://172.28.3.3:3128"
    fi
else
    echo "ok: PyPI reachable (HTTP 200)"
fi

say "uv"
if ! have uv; then
    # The curl installer needs no python at all, which matters because the system
    # python3 may be 3.6 (Rocky 8): `pip install uv` does not work there.
    curl -LsSf --max-time 180 https://astral.sh/uv/install.sh | sh || true
    export PATH="$HOME/.local/bin:$PATH"
fi
if ! have uv; then
    python3 -m pip install --user --upgrade uv || die "could not install uv"
    export PATH="$HOME/.local/bin:$PATH"
fi
have uv || die "uv still not on PATH; look for it in ~/.local/bin and re-run with PATH set"
uv --version

say "CPython ${PY_VER} (managed by uv, no root)"
uv python install "$PY_VER" || die "uv could not fetch CPython ${PY_VER}"
uv python find "$PY_VER"

say "virtualenv: $CXXI_VENV"
if [ -x "$CXXI_VENV/bin/python" ] && [ "$FORCE" = "0" ]; then
    echo "already exists (FORCE=1 to rebuild)"
else
    rm -rf "$CXXI_VENV"
    uv venv --python "$PY_VER" "$CXXI_VENV" || die "uv venv failed"
fi
PY="$CXXI_VENV/bin/python"
[ -x "$PY" ] || die "no interpreter at $PY"
"$PY" -V

say "packages: jax[oneapi]==${JAX_PIN} + the project's dependencies"
# equinox/jaxtyping are what the vendored dsgnar_pinn package (used by the LDC
# notebook) imports; wave_pinn itself does not need them.
# jax first, so pip/uv resolves jaxlib and the Intel PJRT/plugin to the same
# version instead of pulling a CUDA-less jaxlib and then fighting about it.
uv pip install --python "$PY" "jax[oneapi]==${JAX_PIN}" || die "installing jax[oneapi] failed"
uv pip install --python "$PY" "numpy" "scipy" "optax" "matplotlib" "pytest" "tqdm" \
    "equinox" "jaxtyping" \
    || die "installing the remaining dependencies failed"

say "verify"
"$PY" - <<'PYEOF'
import jax, jaxlib
print("jax    ", jax.__version__)
print("jaxlib ", jaxlib.__version__)
try:
    import jax_oneapi_plugin
    print("plugin ", getattr(jax_oneapi_plugin, "__version__", "?"))
except Exception as exc:                                   # noqa: BLE001
    print("plugin  NOT IMPORTABLE:", exc)
try:
    print("devices", jax.devices())
except Exception as exc:                                   # noqa: BLE001
    # Expected on the login node: there is no GPU here.  The compute-node check
    # is 02_smoke.sbatch.
    print("devices (login node, no GPU expected):", type(exc).__name__, exc)
PYEOF

say "done"
cat <<EOF
Environment: $PY

Next, on a GPU node (the login node has no GPU, so this is where the real check is):

    cd ~/Julia/PINN/clementina && sbatch 02_smoke.sbatch

If you rebuild: FORCE=1 bash 01_setup_env.sh
EOF
