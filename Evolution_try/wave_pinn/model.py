"""The network, and the hard-constrained ansatz built on top of it.

The network is a plain MLP written directly in JAX (a pytree of ``{"W", "b"}``
dicts) -- no Flax or Equinox dependency, so that flattening it for a dense
quasi-Newton inverse Hessian is trivial via ``jax.flatten_util.ravel_pytree``.

The ansatz is

    u_theta(t, x) = u0(x) + t v0(x) + t^2 N_theta(phi(t, x)),      v0 = -c u0',

with ``phi`` the feature map from :mod:`wave_pinn.features`.  The first two terms
carry the initial condition exactly; the network only has to learn the part of
the solution that grows like ``t^2``, and because ``u0`` and ``v0`` are both
exactly 2L-periodic (see :mod:`wave_pinn.problem`) and ``phi`` is built from
cos/sin of ``x``, the whole ansatz is periodic in ``x`` for every parameter
vector.  Neither the initial condition nor the boundary condition is ever
enforced with a penalty term.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List

import jax
import jax.numpy as jnp

from .config import Config
from .features import feature_dim, features
from .problem import initial_data


# --------------------------------------------------------------------------
# activations and weight initialisation
# --------------------------------------------------------------------------
def activate(cfg: Config, h):
    a = cfg.activation
    if a == "tanh":
        return jnp.tanh(h)
    if a == "sin":
        return jnp.sin(h)
    if a == "gelu":
        return jax.nn.gelu(h)
    if a == "relu":
        return jax.nn.relu(h)
    if a == "softplus":
        return jax.nn.softplus(h)
    raise ValueError(f"unknown activation {a!r}")


def _bound(cfg: Config, fan_in: int, fan_out: int, first: bool) -> float:
    if cfg.init == "glorot":
        return cfg.init_scale * math.sqrt(6.0 / (fan_in + fan_out))
    if cfg.init == "lecun":
        return cfg.init_scale * math.sqrt(3.0 / fan_in)
    if cfg.init == "siren":
        # SIREN's own scheme: the first layer is scaled by 1/w0 so that sin() sees
        # O(1) pre-activations, hidden layers keep unit-variance activations.
        w0 = 30.0
        return cfg.init_scale * (1.0 / fan_in if first else math.sqrt(6.0 / fan_in) / w0)
    if cfg.init == "uniform":
        return cfg.init_scale
    raise ValueError(f"unknown init {cfg.init!r}")


def init_params(cfg: Config, key) -> List[dict]:
    """Initialise the MLP weights.  ``cfg.n_layers`` hidden layers of ``cfg.n_neurons``."""
    dims = [feature_dim(cfg)] + [cfg.n_neurons] * cfg.n_layers + [1]
    layers = []
    for i in range(len(dims) - 1):
        fan_in, fan_out = dims[i], dims[i + 1]
        key, sub = jax.random.split(key)
        if cfg.init == "zeros":
            W = jnp.zeros((fan_in, fan_out))
        else:
            b = _bound(cfg, fan_in, fan_out, first=(i == 0))
            W = jax.random.uniform(sub, (fan_in, fan_out), minval=-b, maxval=b)
        layers.append({"W": W, "b": jnp.zeros((fan_out,))})
    return layers


def n_parameters(params) -> int:
    return int(sum(p.size for p in jax.tree_util.tree_leaves(params)))


def mlp(params: List[dict], feat, cfg: Config):
    """Forward pass.  ``feat`` has shape ``(..., d_in)``; the result is ``(...)``."""
    h = feat
    for layer in params[:-1]:
        h = activate(cfg, h @ layer["W"] + layer["b"])
    out = h @ params[-1]["W"] + params[-1]["b"]
    return out[..., 0]


# --------------------------------------------------------------------------
# the ansatz
# --------------------------------------------------------------------------
def ansatz_u(params, cfg: Config, t, x, ic=None, t0: float = 0.0, t_scale=None):
    """``u_theta(t, x)``.

    Works both for scalars (inside ``jax.hessian``, when the residual is built)
    and for flat batches (when the solution is evaluated), because every
    operation in it is elementwise apart from the MLP's ``matmul``, which
    broadcasts over a leading batch axis.

    ``ic`` and ``t0`` are what make the windowed (slab-marching) mode possible.
    ``ic(x)`` returns the pair ``(U, V)`` that the ansatz interpolates from, and
    ``t0`` is the time at which that pair holds:

        u(t, x) = U(x) + (t - t0) V(x) + factor(t - t0) * N_theta(t - t0, x).

    With ``ic=None`` and ``t0=0`` this is exactly the global problem, with
    ``U = u0`` and ``V = v0 = -c u0'``.  For a later window the caller passes the
    previous window's solution and its time derivative evaluated at ``t0``, which
    is frozen: it enters the residual as data, not as something to differentiate.
    The network sees the *local* time ``t - t0``, so the weights it must learn do
    not depend on how far along the chain the window sits.
    """
    if ic is None:
        u0, v0 = initial_data(cfg, x)
    else:
        u0, v0 = ic(x)
    t = t - t0
    if jnp.ndim(t) == 0 and jnp.ndim(x) > 0:
        # a slab's IC is evaluated at one time against a batch of x
        t = jnp.broadcast_to(t, jnp.shape(x))
    feat = features(cfg, t, x, u0, v0, t_scale=t_scale)
    net = mlp(params, feat, cfg)
    if cfg.ansatz == "t2":
        return u0 + t * v0 + t * t * net
    if cfg.ansatz == "t2sat":
        return u0 + t * v0 + saturation_factor(cfg, t) * net
    if cfg.ansatz == "t":
        return u0 + t * net
    raise ValueError(f"unknown ansatz {cfg.ansatz!r}")


def saturation_factor(cfg: Config, t):
    """``t^2 / (tau^2 + t^2)``: the ``t^2`` weight, bounded by 1.

    Two properties make it usable in place of ``t^2``.  It vanishes like ``t^2``
    at the origin and so do both of its first two derivatives vanish in the sense
    that matter here -- ``f(0) = 0`` and ``f'(0) = 0`` -- so the hard-coded
    initial condition ``u(0) = u0``, ``u_t(0) = v0`` survives untouched.  And it
    is bounded, so ``du/dtheta = f(t) * dN/dtheta`` has the *same* scale at
    ``t = 20`` as at ``t = 1``, where with a bare ``t^2`` it is 400 times larger.

    That is the whole point: the network output is no longer divided by ``t^2``,
    so it is not asked to carry an O(1) answer in a number of size
    ``1/t^2``; instead the cancellation of the growing hard part happens at unit
    gain.
    """
    return (t * t) / (cfg.ansatz_tau ** 2 + t * t)


def make_u_fn(params, cfg: Config, ic=None, t0: float = 0.0, t_scale=None) -> Callable:
    """Return the scalar-argument callable ``u(t, x)`` that the residual AD uses."""
    return lambda t, x: ansatz_u(params, cfg, t, x, ic=ic, t0=t0, t_scale=t_scale)
