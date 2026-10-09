# shellcheck shell=bash
# ---------------------------------------------------------------------------
# Resolve a Python interpreter that has what this project needs.
#
# Sourced by every script here:
#     source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
#     PY="$(find_python)" || { echo "set PY=" >&2; exit 1; }
#
# Evolution_try is an independent project.  It vendors the Crunch SSBroyden it
# needs under vendor/, keeps its own runs/ and logs/, and can have a venv of its
# own; the only thing it borrows at run time is an interpreter.  The search is
# therefore ordered from most-local-to-this-project to whatever is on PATH, so
#
#   * on the hub, `Evolution_try/.venv` wins if it exists, otherwise the
#     Stationary venv next door (which has the CUDA jax) is used as a fallback --
#     convenient, but not a dependency: nothing in this project imports anything
#     from Stationary/, and the scripts that talk to both are the `rsync`-style
#     ones a person runs by hand;
#   * on the Mac, ~/jax_env is picked up without anyone exporting PY.
#
# `optax` and `scipy` are part of the test on purpose: an interpreter with jax but
# no optax looks fine until the first Adam phase, and trustregion.py imports a
# private scipy module.
# ---------------------------------------------------------------------------

ENV_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$ENV_SH_DIR/.." && pwd)"          # .../Evolution_try

find_python() {
    local c
    for c in "${PY:-}" \
             "$PROJECT_ROOT/.venv/bin/python" \
             "$HOME/jax_env/bin/python" \
             "$PROJECT_ROOT/../Stationary/.venv/bin/python" \
             "$HOME/venvs/pinn/bin/python" \
             "$(command -v python3 2>/dev/null || true)" \
             "$(command -v python  2>/dev/null || true)"; do
        [ -n "$c" ] && [ -x "$c" ] || continue
        if "$c" -c 'import jax, numpy, scipy, optax' >/dev/null 2>&1; then
            printf '%s\n' "$c"
            return 0
        fi
    done
    return 1
}

# Wall-clock MPL cache and headless backend: matplotlib must not try to build its
# font cache in an unwritable home on every run.
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.mplcache}"
mkdir -p "$MPLCONFIGDIR" 2>/dev/null || true
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
