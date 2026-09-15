"""Neural-network ansatz for the fields (h, Gamma, lambda) in the harmonic chart.

Inputs are the chart coordinates x = (x, y, z) of the shell; internally they are
turned into the gauge-adapted features (n, log rho), n = x/rho, which make the
radial power-law structure easy to represent, plus Fourier features in log rho.

Outputs (25 numbers):
    h_ij   = delta_ij + (6 packed components)     -- residual (near-flat) form
    G^i_jk = (18 packed components)               -- Gamma, zero for flat space
    lambda = exp(u)                               -- strictly positive
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from .geometry import Fields, christoffel, gamma3, sym3

I3 = jnp.eye(3)


class FieldNet(nn.Module):
    width: int = 64
    depth: int = 4
    fourier: int = 8
    rho_in: float = 2.0
    rho_out: float = 20.0
    lam_scale: float = 1.0
    out_std: float = 1e-2      # small final-layer init: start near flat space

    @nn.compact
    def __call__(self, x):
        x = jnp.atleast_2d(x)
        rho = jnp.linalg.norm(x, axis=-1, keepdims=True)
        n = x / rho
        t = jnp.log(rho / self.rho_in) / jnp.log(self.rho_out / self.rho_in)
        feats = [n, t]
        for i in range(1, self.fourier + 1):
            feats.append(jnp.sin(jnp.pi * i * t))
            feats.append(jnp.cos(jnp.pi * i * t))
        z = jnp.concatenate(feats, axis=-1)
        for _ in range(self.depth):
            z = jnp.tanh(nn.Dense(self.width)(z))
        out = nn.Dense(25, kernel_init=nn.initializers.normal(self.out_std),
                       bias_init=nn.initializers.zeros)(z)

        h = I3 + sym3(out[..., :6])
        G = gamma3(out[..., 6:24])
        lam = self.lam_scale * jnp.exp(out[..., 24])
        return Fields(h, G, lam.squeeze(-1) if lam.ndim > 1 else lam)


def point_fields(model, params):
    """Adapt the batched network to the single-point `fields` signature."""
    if isinstance(model, HybridNet):
        def h_of(y):
            return model.apply(params, y[None, :]).h[0]

        def fields(x):
            f = model.apply(params, x[None, :])
            return Fields(f.h[0], christoffel(h_of, x), f.lam[0])

        return fields

    def fields(x):
        f = model.apply(params, x[None, :])
        return Fields(f.h[0], f.G[0], f.lam[0] if f.lam.ndim else f.lam)

    return fields


def batch_fields(model, params):
    """Batched version: x (N,3) -> Fields with leading batch axis."""
    return lambda x: model.apply(params, x)


# --------------------------------------------------------------------------- sym
class SymFieldNet(nn.Module):
    """Spherically symmetric ansatz: six functions of rho, no angular freedom.

        h_ij          = alpha(rho) delta_ij + beta(rho) n_i n_j
        Gamma^i_{jk}  = a n_i n_j n_k + b (delta_ij n_k + delta_ik n_j) + c delta_jk n_i
        lambda        = exp(u(rho))

    This is the general rotationally invariant field content of the first-order
    formulation, so it exercises exactly the same residual/boundary machinery while
    making the one-dimensional structure easy for the optimiser.  alpha, beta and u
    are offset so that the initial state is flat space with constant lambda.
    """
    width: int = 64
    depth: int = 3
    fourier: int = 12
    rho_in: float = 2.0
    rho_out: float = 20.0
    out_std: float = 1e-2

    @nn.compact
    def __call__(self, x):
        x = jnp.atleast_2d(x)
        rho = jnp.linalg.norm(x, axis=-1, keepdims=True)
        n = x / rho
        t = jnp.log(rho / self.rho_in) / jnp.log(self.rho_out / self.rho_in)
        feats = [t]
        for i in range(1, self.fourier + 1):
            feats.append(jnp.sin(jnp.pi * i * t))
            feats.append(jnp.cos(jnp.pi * i * t))
        z = jnp.concatenate(feats, axis=-1)
        for _ in range(self.depth):
            z = jnp.tanh(nn.Dense(self.width)(z))
        out = nn.Dense(6, kernel_init=nn.initializers.normal(self.out_std),
                       bias_init=nn.initializers.zeros)(z)

        alpha = 1.0 + out[..., 0]
        beta = out[..., 1]
        a = out[..., 2]
        b = out[..., 3]
        c = out[..., 4]
        lam = jnp.exp(out[..., 5])

        d = jnp.eye(3)
        nn_ = jnp.einsum("ni,nj->nij", n, n)                       # n_i n_j
        h = alpha[:, None, None] * d + beta[:, None, None] * nn_
        # Gamma^i_{jk} = a n_i n_j n_k + b (delta_ij n_k + delta_ik n_j) + c delta_jk n_i
        n3 = jnp.einsum("ni,nj,nk->nijk", n, n, n)
        g2 = (jnp.einsum("ij,nk->nijk", d, n) + jnp.einsum("ik,nj->nijk", d, n))
        g3 = jnp.einsum("jk,ni->nijk", d, n)
        G = (a[:, None, None, None] * n3 + b[:, None, None, None] * g2
             + c[:, None, None, None] * g3)
        return Fields(h, G, lam)


class HybridNet(nn.Module):
    """Metric-only formulation: the network outputs h and lambda only.

    Gamma is the Christoffel symbol of h (autodiff), so the compatibility equation
    is satisfied identically and the residual set reduces to Ricci + gauge +
    lambda equation.  This removes the degenerate "flat metric with Gamma = 0"
    direction of the independent-connection formulation, at the price of taking
    second derivatives of the network output.
    """
    width: int = 64
    depth: int = 4
    fourier: int = 8
    rho_in: float = 2.0
    rho_out: float = 20.0
    out_std: float = 1e-2

    @nn.compact
    def __call__(self, x):
        x = jnp.atleast_2d(x)
        rho = jnp.linalg.norm(x, axis=-1, keepdims=True)
        n = x / rho
        t = jnp.log(rho / self.rho_in) / jnp.log(self.rho_out / self.rho_in)
        feats = [n, t]
        for i in range(1, self.fourier + 1):
            feats.append(jnp.sin(jnp.pi * i * t))
            feats.append(jnp.cos(jnp.pi * i * t))
        z = jnp.concatenate(feats, axis=-1)
        for _ in range(self.depth):
            z = jnp.tanh(nn.Dense(self.width)(z))
        out = nn.Dense(7, kernel_init=nn.initializers.normal(self.out_std),
                       bias_init=nn.initializers.zeros)(z)
        h = I3 + sym3(out[..., :6])
        lam = jnp.exp(out[..., 6])
        G = jnp.zeros(x.shape[:-1] + (3, 3, 3))
        return Fields(h, G, lam)
