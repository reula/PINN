"""The flat-parameter, residual-vector view of the loss that a least-squares optimiser needs.

`train.py` already runs two phases on a `R^n -> R` view of the objective (Crunch's SSBroyden
and `optax.lbfgs`).  DSGNAR is a Gauss-Newton method, so it needs the same objective in a
different form: the RESIDUAL VECTOR, plus its Jacobian-vector products.  This class is the
adapter, and it is deliberately thin -- the objective itself is `losses.total_loss`, unchanged
(see `losses.residual_vector`).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .losses import residual_vector


class FlatObjective:
    """`R^n -> R^M` and `R^n -> R` views of `total_loss` for the DSGNAR phase.

    The residual is returned SCALED by `sqrt(2 M)` (`M` = number of rows) so that

        loss(flat) := 0.5 * mean(residual(flat)**2)

    is exactly `total_loss(state, ...)`.  That factor is not cosmetic: DSGNAR predicts a model
    decrease for `0.5 * ||r||^2` and multiplies it by `1/M` to compare it with the loss it was
    handed, so the loss it is handed has to be `0.5 * mean(r^2)` -- that is the convention its
    trust-region ratio is built on.  The number that reaches the history is then comparable
    with the other phases' losses.
    """

    def __init__(self, state, batch, cfg, model, exact_fields=None, weights=None,
                 pde_scale=1.0, lam_inf=None):
        self.cfg, self.batch, self.model = cfg, batch, model
        self.exact_fields, self.weights = exact_fields, weights
        self.pde_scale, self.lam_inf = pde_scale, lam_inf
        self.flat0, self.unflatten = jax.flatten_util.ravel_pytree(state)
        self.n = int(self.flat0.size)
        self.dtype = self.flat0.dtype

        @jax.jit
        def _residual(flat):
            r = residual_vector(self.unflatten(flat), batch, cfg, model, exact_fields,
                                weights=weights, pde_scale=pde_scale, lam_inf=lam_inf)
            return jnp.sqrt(2.0 * r.shape[0]) * r

        @jax.jit
        def _loss(flat):
            return 0.5 * jnp.mean(_residual(flat) ** 2)

        self._residual = _residual
        self._loss = _loss
        self.value_and_grad = jax.jit(jax.value_and_grad(_loss))
        self.grad = jax.jit(jax.grad(_loss))

    def residual(self, flat):
        return self._residual(jnp.asarray(flat))

    def loss(self, flat) -> float:
        return float(self._loss(jnp.asarray(flat)))
