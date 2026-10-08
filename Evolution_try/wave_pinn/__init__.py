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

from .config import Config, apply_overrides, resolve_outdir   # noqa: F401

__all__ = ["Config", "apply_overrides", "resolve_outdir"]
__version__ = "0.1.0"
