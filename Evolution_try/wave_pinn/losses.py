"""The objective, written as a least-squares problem.

Everything is expressed through the residual *vector* ``r(theta)`` rather than
through its squared norm, because the two optimisers want different views of the
same thing:

* SSBroyden (a quasi-Newton method) only ever asks for ``L`` and ``grad L``;
* DSGNAR (a sketched Gauss-Newton method) needs ``r`` itself and the action of
  its Jacobian, since it builds a Gauss-Newton model of ``||r||^2``.

The loss is ``L = mean(r**2)`` over the concatenated residual vector, i.e. the
mean squared PDE residual, plus optional periodic-boundary and weight-decay
terms folded in as additional rows (so their weights enter as ``sqrt(w)``).
The initial condition never appears here: it is hard-coded in the ansatz.
"""

from __future__ import annotations

from typing import Callable, Dict, NamedTuple

import jax
import jax.flatten_util
import jax.numpy as jnp

from .config import Config
from .model import make_u_fn, n_parameters
from .problem import pde_residual, periodicity_defect
from .sampling import Batch


def raw_residual(params, cfg: Config, batch: Batch, ic=None, t0: float = 0.0):
    """The residual rows in physical units, before normalisation."""
    u = make_u_fn(params, cfg, ic=ic, t0=t0)
    rows = [pde_residual(cfg, u, batch.t, batch.x)]
    if cfg.w_periodic > 0.0 and batch.t_b.size:
        d_val, d_der = periodicity_defect(cfg, u, batch.t_b, batch.x_left, batch.x_right)
        w = jnp.sqrt(cfg.w_periodic)
        rows.append(w * d_val)
        rows.append(w * d_der)
    if cfg.w_l2 > 0.0:
        flat, _ = jax.flatten_util.ravel_pytree(params)
        rows.append(jnp.sqrt(cfg.w_l2) * flat)
    return rows[0] if len(rows) == 1 else jnp.concatenate(rows)


def residual_vector(params, cfg: Config, batch: Batch, ic=None, t0: float = 0.0):
    """The residual vector ``r(theta)`` handed to the optimisers.

    Only the PDE rows are normalised, by the fixed constant ``batch.scale``
    (see :func:`wave_pinn.sampling.residual_scale`); the optional penalty rows
    already carry their own weights.
    """
    u = make_u_fn(params, cfg, ic=ic, t0=t0)
    pde = pde_residual(cfg, u, batch.t, batch.x) / batch.scale
    rows = [pde]
    if cfg.w_periodic > 0.0 and batch.t_b.size:
        d_val, d_der = periodicity_defect(cfg, u, batch.t_b, batch.x_left, batch.x_right)
        w = jnp.sqrt(cfg.w_periodic)
        rows.append(w * d_val)
        rows.append(w * d_der)
    if cfg.w_l2 > 0.0:
        flat, _ = jax.flatten_util.ravel_pytree(params)
        rows.append(jnp.sqrt(cfg.w_l2) * flat)
    return rows[0] if len(rows) == 1 else jnp.concatenate(rows)


def loss_from_residual(r):
    return jnp.mean(r * r)


def pde_loss_only(params, cfg: Config, batch: Batch, ic=None, t0: float = 0.0):
    """The *unnormalised* mean squared PDE residual -- the physical training loss."""
    u = make_u_fn(params, cfg, ic=ic, t0=t0)
    r = pde_residual(cfg, u, batch.t, batch.x)
    return jnp.mean(r * r)


class Objective:
    """Jitted, flat-parameter view of the objective.

    ``flat`` is the network parameter vector produced by
    ``jax.flatten_util.ravel_pytree``; ``unflatten`` puts it back into the MLP
    pytree.  All the optimiser-facing code goes through this class so that the
    quasi-Newton methods see a plain ``R^n -> R`` function, as they require.
    """

    def __init__(self, cfg: Config, batch: Batch, like_params, ic=None, t0: float = 0.0):
        self.cfg = cfg
        self.batch = batch
        self.ic = ic
        self.t0 = float(t0)
        flat0, self.unflatten = jax.flatten_util.ravel_pytree(like_params)
        self.n = int(flat0.size)
        self.dtype = flat0.dtype
        self.n_params_network = n_parameters(like_params)

        @jax.jit
        def _loss(flat):
            return loss_from_residual(
                residual_vector(self.unflatten(flat), cfg, batch, ic=ic, t0=self.t0))

        @jax.jit
        def _pde_loss(flat):
            return pde_loss_only(self.unflatten(flat), cfg, batch, ic=ic, t0=self.t0)

        @jax.jit
        def _residual(flat):
            return residual_vector(self.unflatten(flat), cfg, batch, ic=ic, t0=self.t0)

        self._loss = _loss
        self._pde_loss = _pde_loss
        self._residual = _residual
        self.value_and_grad = jax.jit(jax.value_and_grad(_loss))
        self.grad = jax.jit(jax.grad(_loss))

    # convenience wrappers that return python floats
    def loss(self, flat) -> float:
        return float(self._loss(flat))

    def pde_loss(self, flat) -> float:
        return float(self._pde_loss(flat))

    def residual(self, flat):
        return self._residual(flat)

    def jacobian(self, flat):
        """The full residual Jacobian ``dr/dtheta`` (``M x n``) -- affordable here."""
        return jax.jacfwd(lambda f: residual_vector(self.unflatten(f), self.cfg, self.batch,
                                                    ic=self.ic, t0=self.t0))(flat)

    def jacobian_operator(self, flat) -> Callable:
        """``v -> J v`` without materialising ``J``; cheap for DSGNAR's sketches."""
        _, jvp = jax.linearize(
            lambda f: residual_vector(self.unflatten(f), self.cfg, self.batch,
                                      ic=self.ic, t0=self.t0), flat)
        return jvp

    def with_batch(self, batch: Batch) -> "Objective":
        """A new objective on a fresh sample, keeping ic, t0 and the parameter layout."""
        return Objective(self.cfg, batch, self.unflatten(jnp.zeros((self.n,), self.dtype)),
                         ic=self.ic, t0=self.t0)
