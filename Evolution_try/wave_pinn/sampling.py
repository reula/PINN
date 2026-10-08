"""Collocation sampling on ``[-L, L] x [0, T]``.

Four samplers, selected by ``cfg.sampler``:

``uniform``       a structured tensor grid, ``x`` at cell centres (so the grid is
                  symmetric under the periodic shift) and ``t`` at ``linspace``.
                  DETERMINISTIC: it ignores the key, so it cannot be redrawn --
                  ``resample_every`` with this sampler is rejected by ``validate``.
``random``        i.i.d. uniform points, redrawn whenever a fresh batch is asked for.
``grid_random``   random ``x``, deterministic ``t`` grid -- the usual PINN recipe.
``lhs``           a Latin hypercube, which stratifies both axes with one point per
                  column and one per row.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import Config
from .problem import initial_data


class Batch(NamedTuple):
    """One collocation sample.

    ``t``/``x`` are the interior collocation points; ``t_b`` with ``x_left``/
    ``x_right`` are the periodic-boundary probes, used only when
    ``cfg.w_periodic > 0``.
    """

    t: jax.Array
    x: jax.Array
    t_b: jax.Array
    x_left: jax.Array
    x_right: jax.Array
    scale: jax.Array          # residual normalisation, shape (); see residual_scale()


def _aspect_grid(cfg: Config) -> tuple:
    """Split ``n_coll`` into ``(n_x, n_t)`` matching the rectangle's aspect ratio."""
    aspect = (2.0 * cfg.L) / cfg.T                     # x-extent per unit t
    n_x = int(round(np.sqrt(cfg.n_coll * aspect)))
    n_x = max(2, min(cfg.n_coll, n_x))
    n_t = max(1, cfg.n_coll // n_x)
    return n_x, n_t


def _uniform(cfg: Config, t0: float = 0.0, t1: float | None = None) -> tuple:
    t1 = cfg.T if t1 is None else t1
    n_x, n_t = _aspect_grid(cfg)
    x = (-cfg.L + (np.arange(n_x) + 0.5) * (2.0 * cfg.L / n_x)).astype(np.float64)
    t = np.linspace(t0, t1, n_t, dtype=np.float64)
    X, T = np.meshgrid(x, t, indexing="xy")
    return T.ravel(), X.ravel()


def _random(cfg: Config, key, t0: float = 0.0, t1: float | None = None) -> tuple:
    t1 = cfg.T if t1 is None else t1
    kx, kt = jax.random.split(key)
    x = jax.random.uniform(kx, (cfg.n_coll,), minval=-cfg.L, maxval=cfg.L)
    t = jax.random.uniform(kt, (cfg.n_coll,), minval=t0, maxval=t1)
    return t, x


def _grid_random(cfg: Config, key, t0: float = 0.0, t1: float | None = None) -> tuple:
    t1 = cfg.T if t1 is None else t1
    _, n_t = _aspect_grid(cfg)
    n_x = max(1, cfg.n_coll // n_t)
    x = jax.random.uniform(key, (n_t, n_x), minval=-cfg.L, maxval=cfg.L)
    t = jnp.linspace(t0, t1, n_t)[:, None] * jnp.ones((1, n_x))
    return t.ravel(), x.ravel()


def _lhs(cfg: Config, key, t0: float = 0.0, t1: float | None = None) -> tuple:
    t1 = cfg.T if t1 is None else t1
    n = cfg.n_coll
    k1, k2 = jax.random.split(key)
    ux = (jnp.arange(n) + jax.random.uniform(k1, (n,))) / n
    ut = (jnp.arange(n) + jax.random.uniform(k2, (n,))) / n
    return t0 + ut * (t1 - t0), -cfg.L + ux * 2.0 * cfg.L


def sample_interior(cfg: Config, key, t0: float = 0.0, t1: float | None = None) -> tuple:
    """Collocation points with ``t`` drawn from ``[t0, t1]`` (default ``[0, T]``).

    The window arguments are what make slab marching work: a slab's residual must
    be measured only where its ansatz is defined.
    """
    t1 = cfg.T if t1 is None else t1
    if cfg.sampler == "uniform":
        return _uniform(cfg, t0, t1)
    if cfg.sampler == "random":
        return _random(cfg, key, t0, t1)
    if cfg.sampler == "grid_random":
        return _grid_random(cfg, key, t0, t1)
    if cfg.sampler == "lhs":
        return _lhs(cfg, key, t0, t1)
    raise ValueError(f"unknown sampler {cfg.sampler!r}")


def residual_scale(cfg: Config, t, x, scale_override=None, ic=None, t0: float = 0.0):
    """A constant by which to divide the PDE residual before optimising.

    The raw residual of this problem is O(u0'') ~ O(1/sigma^2), i.e. of order a
    hundred for ``sigma = 0.2``, which makes every gradient tolerance and line
    search in the optimiser a statement about the units of the problem rather
    than about its accuracy.  Dividing by the RMS residual of the *hard-coded
    part* fixes the scale once and for all, at no cost to the physics.

    For the second-order equation the hard part's own residual has RMS ~ 84; for
    the first-order equation that residual vanishes identically (the ansatz is
    built to make it vanish), so the scale is taken from the size of the two
    terms that would appear in it.
    """
    if scale_override is not None:
        return jnp.asarray(scale_override, dtype=jnp.float64)
    if cfg.residual_norm == "none":
        return jnp.asarray(1.0)
    from .problem import pde_residual

    def u_hard(tt, xx):
        u0, v0 = initial_data(cfg, xx) if ic is None else ic(xx)
        return u0 + (tt - t0) * v0

    if cfg.equation == "advection":
        # For the hard part u_t = v0 and c u_x = c u0', and v0 = -c u0' makes their
        # sum vanish identically, so the size of the individual terms is used instead.
        _, v0 = initial_data(cfg, x)
        s = jnp.sqrt(jnp.mean(v0 * v0))
    else:
        r_hard = pde_residual(cfg, u_hard, t, x)
        s = jnp.sqrt(jnp.mean(r_hard * r_hard))
    return jnp.where(s > 0.0, s, 1.0)


def make_batch(cfg: Config, key, scale=None, t0: float = 0.0,
               t1: float | None = None, ic=None) -> Batch:
    """Draw one collocation batch, with ``t`` confined to ``[t0, t1]``."""
    t, x = sample_interior(cfg, key, t0=t0, t1=t1)
    if cfg.n_boundary > 0:
        tb = jnp.linspace(t0, cfg.T if t1 is None else t1, cfg.n_boundary)
        xl = -cfg.L * jnp.ones_like(tb)
        xr = cfg.L * jnp.ones_like(tb)
    else:
        tb = jnp.zeros((0,))
        xl = jnp.zeros((0,))
        xr = jnp.zeros((0,))
    scale = residual_scale(cfg, jnp.asarray(t), jnp.asarray(x), scale_override=scale,
                           ic=ic, t0=t0)
    return Batch(t=jnp.asarray(t), x=jnp.asarray(x), t_b=tb, x_left=xl, x_right=xr,
                 scale=scale)


def test_grid(cfg: Config, n_x: int | None = None, t: float | None = None) -> tuple:
    """A dense evaluation grid: all of ``[0, T]`` when ``t`` is None, else one time slice."""
    n_x = n_x or cfg.n_test_x
    x = jnp.linspace(-cfg.L, cfg.L, n_x)
    if t is None:
        times = jnp.asarray(cfg.snapshot_times)
        return times, x
    return jnp.full_like(x, float(t)), x
