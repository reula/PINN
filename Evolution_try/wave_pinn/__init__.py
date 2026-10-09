"""wave_pinn -- physics-informed neural networks for the 1+1 wave equation.

A small, scheme-oriented framework for solving

    u_tt - c^2 u_xx = 0        (``equation="wave2"``, the default)
    u_t  + c   u_x  = 0        (``equation="advection"``)

on ``x in [-L, L]`` with periodic boundary conditions, from initial data
``u(0, x) = u0(x)``, ``u_t(0, x) = v0(x) = -c u0'(x)`` that is hard-coded into
the ansatz, using a 6-layer x 20-neuron network and either SSBroyden or DSGNAR.

Everything is driven by :class:`wave_pinn.config.Config`; see ``README.md`` in
``Evolution_try/`` for the motivation, the results and the commands.
"""

import os as _os

# ---------------------------------------------------------------------------
# Must happen before the first JAX import anywhere in this package.
#
# JAX preallocates 75 % of the visible device when it initialises the backend.
# On a shared GPU that is antisocial at best and, when somebody else already
# holds part of the card, a hard failure -- measured on this project's hub as
# `Device 0 OOM 19334492688 / 19327352832`, i.e. 18.0 GiB requested against an
# 18.0 GiB A30 that already had 4.4 GiB in use.  The failure surfaces wherever
# the first device->host copy happens, which makes it look like a bug in the
# optimiser rather than an environment default.
#
# Doing it here rather than in the scripts means `python -m wave_pinn.cli ...`
# behaves on a shared card no matter how it was invoked.  Both are
# setdefault, so an explicit export still wins: raise
# XLA_PYTHON_CLIENT_MEM_FRACTION if you want a hard cap as well.
# ---------------------------------------------------------------------------
_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("MPLBACKEND", "Agg")
_os.environ.setdefault(
    "MPLCONFIGDIR",
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), ".mplcache"))

from .config import Config, apply_overrides, resolve_outdir   # noqa: F401

__all__ = ["Config", "apply_overrides", "resolve_outdir"]
__version__ = "0.1.0"
