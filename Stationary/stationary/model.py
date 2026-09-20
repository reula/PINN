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
import jax
import jax.numpy as jnp

from .geometry import Fields, christoffel, gamma3, sym3


def _rdtype():
    """float64 iff x64 is enabled.

    Flax defaults every layer to float32 whatever jax_enable_x64 says, so exporting
    JAX_ENABLE_X64=1 alone leaves the PARAMETERS in float32: the optimiser can then not
    refine them below ~1e-7 relative, which is exactly the headroom the higher-order
    Robin conditions need (the fourth-order operator amplifies the field's radial
    frequency content by (pi*fourier/log(rho_out/rho_in))^4 ~ 890 at fourier 8, 1.4e4 at
    fourier 16).  With this helper an x64 run is float64 from the parameters up.
    """
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def _dense(features, **kw):
    d = _rdtype()
    return nn.Dense(features, dtype=d, param_dtype=d, **kw)


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
            z = jnp.tanh(_dense(self.width)(z))
        out = _dense(25, kernel_init=nn.initializers.normal(self.out_std),
                       bias_init=nn.initializers.zeros)(z)

        h = I3 + sym3(out[..., :6])
        G = gamma3(out[..., 6:24])
        lam = self.lam_scale * jnp.exp(out[..., 24])
        return Fields(h, G, lam.squeeze(-1) if lam.ndim > 1 else lam)


def point_fields(model, params):
    """Adapt the batched network to the single-point `fields` signature."""
    if isinstance(model, (HybridNet, SymHybridNet, AxisymHybridNet)):
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
    decay: bool = False      # add rho_in/rho: the natural decay variable of the solution

    @nn.compact
    def __call__(self, x):
        x = jnp.atleast_2d(x)
        rho = jnp.linalg.norm(x, axis=-1, keepdims=True)
        n = x / rho
        t = jnp.log(rho / self.rho_in) / jnp.log(self.rho_out / self.rho_in)
        feats = [t]
        if self.decay:
            feats.append(self.rho_in / rho)
        for i in range(1, self.fourier + 1):
            feats.append(jnp.sin(jnp.pi * i * t))
            feats.append(jnp.cos(jnp.pi * i * t))
        z = jnp.concatenate(feats, axis=-1)
        for _ in range(self.depth):
            z = jnp.tanh(_dense(self.width)(z))
        out = _dense(6, kernel_init=nn.initializers.normal(self.out_std),
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
            z = jnp.tanh(_dense(self.width)(z))
        out = _dense(7, kernel_init=nn.initializers.normal(self.out_std),
                       bias_init=nn.initializers.zeros)(z)
        h = I3 + sym3(out[..., :6])
        lam = jnp.exp(out[..., 6])
        G = jnp.zeros(x.shape[:-1] + (3, 3, 3))
        return Fields(h, G, lam)


class SymHybridNet(nn.Module):
    """Symmetric metric-only ansatz: alpha, beta, u only.

        h_ij   = alpha(rho) delta_ij + beta(rho) n_i n_j
        lambda = exp(u(rho))

    Gamma is the Christoffel symbol of h (autodiff), so metric compatibility holds
    identically: the compat residual (which dominates the loss for the
    independent-Gamma ansatz on this problem) is removed, together with three of the
    six scalar outputs.  The pair (alpha, beta) is exactly the gauge-independent
    content of the harmonic-gauge spherically symmetric metric.
    """
    width: int = 64
    depth: int = 4
    fourier: int = 8
    rho_in: float = 2.0
    rho_out: float = 20.0
    out_std: float = 1e-2
    decay: bool = False

    @nn.compact
    def __call__(self, x):
        x = jnp.atleast_2d(x)
        rho = jnp.linalg.norm(x, axis=-1, keepdims=True)
        n = x / rho
        t = jnp.log(rho / self.rho_in) / jnp.log(self.rho_out / self.rho_in)
        feats = [t]
        if self.decay:
            feats.append(self.rho_in / rho)
        for i in range(1, self.fourier + 1):
            feats.append(jnp.sin(jnp.pi * i * t))
            feats.append(jnp.cos(jnp.pi * i * t))
        z = jnp.concatenate(feats, axis=-1)
        for _ in range(self.depth):
            z = jnp.tanh(_dense(self.width)(z))
        out = _dense(3, kernel_init=nn.initializers.normal(self.out_std),
                       bias_init=nn.initializers.zeros)(z)
        d = jnp.eye(3)
        nn_ = jnp.einsum("ni,nj->nij", n, n)
        h = (1.0 + out[..., 0])[:, None, None] * d + out[..., 1][:, None, None] * nn_
        G = jnp.zeros(x.shape[:-1] + (3, 3, 3))
        return Fields(h, G, jnp.exp(out[..., 2]))


class AxisymHybridNet(nn.Module):
    """Axisymmetric metric-only ansatz, Gamma derived from h.

    The data of interest (a z-dependent lambda on the inner sphere, S2 = 0) is
    invariant under rotations about the z axis, so by uniqueness the solution is
    axisymmetric and the general rotationally-covariant metric built from the unit
    radial vector n and the axis z is

        h_ij = (1 + a) delta_ij + b n_i n_j + c (n_i z_j + z_i n_j) + d z_i z_j

    with a, b, c, d functions of (rho, mu = n_z), and lambda = exp(u(rho, mu)).
    Five scalar functions of two variables instead of twenty-five of three: far more
    accurate than the generic 3-D ansatz for this class of data, and still able to
    represent every axisymmetric field content.  Gamma is the Christoffel symbol of h,
    so compatibility holds identically.
    """
    width: int = 64
    depth: int = 4
    fourier: int = 8
    rho_in: float = 1.0
    rho_out: float = 100.0
    out_std: float = 1e-2
    decay: bool = False

    @nn.compact
    def __call__(self, x):
        x = jnp.atleast_2d(x)
        rho = jnp.linalg.norm(x, axis=-1, keepdims=True)
        n = x / rho
        t = jnp.log(rho / self.rho_in) / jnp.log(self.rho_out / self.rho_in)
        mu = n[..., 2:3]
        feats = [t, mu]
        if self.decay:
            feats.append(self.rho_in / rho)
        for i in range(1, self.fourier + 1):
            feats.append(jnp.sin(jnp.pi * i * t))
            feats.append(jnp.cos(jnp.pi * i * t))
        for i in range(1, self.fourier + 1):
            feats.append(jnp.cos(jnp.pi * i * mu))
        z = jnp.concatenate(feats, axis=-1)
        for _ in range(self.depth):
            z = jnp.tanh(_dense(self.width)(z))
        out = _dense(5, kernel_init=nn.initializers.normal(self.out_std),
                       bias_init=nn.initializers.zeros)(z)

        d3 = jnp.eye(3)
        ez = jnp.array([0.0, 0.0, 1.0])
        nn_ = jnp.einsum("ni,nj->nij", n, n)
        nz = 0.5 * (jnp.einsum("ni,j->nij", n, ez) + jnp.einsum("i,nj->nij", ez, n))
        zz = jnp.einsum("i,j->ij", ez, ez)
        h = ((1.0 + out[..., 0])[:, None, None] * d3
             + out[..., 1][:, None, None] * nn_
             + out[..., 2][:, None, None] * nz
             + out[..., 3][:, None, None] * zz)
        G = jnp.zeros(x.shape[:-1] + (3, 3, 3))
        return Fields(h, G, jnp.exp(out[..., 4]))
