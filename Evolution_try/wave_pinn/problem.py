"""The physical problem: initial data, the exact solution, and the PDE residual.

Equation
--------
``equation="wave2"`` (the default, the *second-order* wave equation)

    u_tt - c^2 u_xx = 0,            x in [-L, L],  t in [0, T],

with periodic boundary conditions in ``x``.  ``equation="advection"`` selects
the first-order transport equation ``u_t + c u_x = 0`` instead, which shares the
same initial data and the same exact solution.

Initial data
------------
The network ansatz hard-codes

    u(t, x) = u0(x) + t v0(x) + t^2 N_theta(t, x),      v0(x) = -c u0'(x),

so ``u(0, x) = u0(x)`` and ``u_t(0, x) = v0(x)`` hold *exactly*, for every
parameter vector.  With ``v0 = -c u0'`` the exact solution is the pure right
mover ``u(t, x) = u0(x - c t)``: substituting the d'Alembert decomposition
``u = F(x - ct) + G(x + ct)`` and imposing both initial conditions gives
``G' = 0`` and ``F = u0``.

Periodisation
-------------
For periodic boundary conditions to hold *exactly*, the initial profile itself
must be 2L-periodic.  A Gaussian of width ``sigma = 0.2`` on ``[-1, 1]`` is not
(it is ``exp(-12.5) = 3.7e-6`` at the ends, and its slope there is not zero), so
``periodize_ic=True`` replaces ``u0`` by its smooth periodic sum

    u0_per(x) = sum_n u0_raw(x + 2 n L),

which is analytic and exactly periodic; on ``[-L, L]`` it differs from the raw
Gaussian by less than ``4e-6``.  Profiles that are already periodic (``sin``,
``sin2``) are not summed, since the sum would not converge.
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from .config import Config

# --------------------------------------------------------------------------
# initial profiles
# --------------------------------------------------------------------------
NATIVE_PERIODIC = ("sin", "sin2")


def _raw_profile(cfg: Config, x):
    """The profile as written down, without any periodisation."""
    a, L, s = cfg.u0_amp, cfg.L, cfg.u0_sigma
    if cfg.u0 == "gaussian":
        return a * jnp.exp(-(x * x) / (2.0 * s * s))
    if cfg.u0 == "sin":
        return a * jnp.sin(jnp.pi * x / L)
    if cfg.u0 == "sin2":
        return a * jnp.sin(2.0 * jnp.pi * x / L)
    if cfg.u0 == "sech2":
        return a / jnp.cosh(x / s) ** 2
    if cfg.u0 == "cosine_bump":
        return a * jnp.cos(0.5 * jnp.pi * x / L) ** 2
    if cfg.u0 == "poly_bump":
        u = x / L
        return a * jnp.clip(1.0 - u * u, 0.0, None) ** 3
    raise ValueError(f"unknown u0 profile {cfg.u0!r}")


def n_images(cfg: Config) -> int:
    """How many periodic images to add on each side (only for decaying profiles)."""
    half = cfg.L
    reach = 8.0 * max(cfg.u0_sigma, 1e-3) + 2.0 * half
    return int(min(64, max(3, math.ceil(reach / (2.0 * half)))))


def profile(cfg: Config, x):
    """The initial profile actually used: periodised when that is wanted."""
    if not (cfg.periodic and cfg.periodize_ic) or cfg.u0 in NATIVE_PERIODIC:
        return _raw_profile(cfg, x)
    n = n_images(cfg)
    total = _raw_profile(cfg, x)
    for k in range(1, n + 1):
        total = total + _raw_profile(cfg, x + 2.0 * k * cfg.L) + _raw_profile(cfg, x - 2.0 * k * cfg.L)
    return total


def dprofile(cfg: Config, x):
    """``u0'(x)``, for scalar or batched ``x`` (JAX does the differentiating, so a new
    profile needs no hand-written derivative)."""
    g = jax.grad(lambda z: profile(cfg, z))
    if jnp.ndim(x) == 0:
        return g(x)
    return jax.vmap(g)(x)


def initial_data(cfg: Config, x):
    """``(u0(x), v0(x))`` with ``v0 = -c u0'``."""
    return profile(cfg, x), -cfg.c * dprofile(cfg, x)


def exact_solution(cfg: Config, t, x):
    """The exact solution of the problem as posed.

    For ``wave2`` with ``v0 = -c u0'`` this is the right-moving wave
    ``u0(x - c t)``.  ``profile`` is periodic by construction, so evaluating it at
    ``x - c t`` is exactly the periodic solution.
    """
    return profile(cfg, x - cfg.c * t)


# --------------------------------------------------------------------------
# the PDE residual
# --------------------------------------------------------------------------
def _diag_hessian(f: Callable, t, x) -> Tuple:
    """``(d^2f/dt^2, d^2f/dx^2)`` at every point of the (t, x) batches."""
    pts = jnp.stack([t, x], axis=-1)
    hess = jax.vmap(jax.hessian(lambda p: f(p[0], p[1])))(pts)
    return hess[..., 0, 0], hess[..., 1, 1]


def pde_residual(cfg: Config, u_fn: Callable, t, x):
    """Pointwise residual of the selected equation at the collocation points.

    ``u_fn`` is a scalar-valued callable ``(t, x) -> u`` (the ansatz), traced by
    JAX; ``t`` and ``x`` are flat arrays.
    """
    if cfg.equation == "wave2":
        u_tt, u_xx = _diag_hessian(u_fn, t, x)
        return u_tt - cfg.c * cfg.c * u_xx
    if cfg.equation == "advection":
        u_t = jax.vmap(jax.grad(u_fn, 0))(t, x)
        u_x = jax.vmap(jax.grad(u_fn, 1))(t, x)
        return u_t + cfg.c * u_x
    raise ValueError(f"unknown equation {cfg.equation!r}")


def periodicity_defect(cfg: Config, u_fn: Callable, t, x_left, x_right):
    """``u(t, -L) - u(t, +L)`` (and the same for the x-derivative), per boundary point.

    Only used when ``cfg.w_periodic > 0``.  With a cos/sin feature map the
    network part is periodic by construction, so this penalises exactly the part
    of the ansatz that is *not*: the hard-coded ``u0 + t v0``.  It therefore acts
    as a diagnostic; the default weight is zero.
    """
    d_val = jax.vmap(u_fn, in_axes=(0, 0))(t, x_left) - jax.vmap(u_fn, in_axes=(0, 0))(t, x_right)
    dl = jax.vmap(jax.grad(u_fn, 1))(t, x_left)
    dr = jax.vmap(jax.grad(u_fn, 1))(t, x_right)
    return d_val, dl - dr
