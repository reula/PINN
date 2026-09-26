"""Riemannian geometry and PDE residuals in the harmonic ('Cartesian-like') chart.

The computational chart is the one whose coordinate functions x^i are required to
be harmonic, x^i = rho n^i.  The harmonic gauge condition is then literally

    Gamma^i_{jk} h^{jk} = 0        (equivalently  Delta_h x^i = 0).

INDEX CONVENTIONS  (regression tested in tests/test_geometry.py -- do not change
casually: jax puts the INPUT axis of a jacobian LAST)

    J  = jacfwd(h)(x)          J[i,j,a]    = d_a h_ij
    JG = jacfwd(G)(x)          JG[i,j,k,a] = d_a G[i,j,k]
    G[i,j,k] = Gamma^i_{jk}, symmetric under j <-> k
    Hinv[i,j] = h^{ij}

RESIDUAL GROUPS of the stationary system
    Ricci(h)_ab = (1/(2 lambda^2)) grad_a lambda grad_b lambda
    h^{ab} grad_a grad_b lambda = (1/lambda) h^{ab} grad_a lambda grad_b lambda
    Gamma^a_{bc} h^{bc} = 0

    compat[a,b,c] = d_a h_bc - G^d_{ab} h_dc - G^d_{ac} h_bd          (18)
    ricci[i,j]    = R_ij(G) - (1/(2 lam^2)) d_i lam d_j lam           (6)
    gauge[i]      = G^i_{jk} h^{jk}                                   (3)
    lam_eq        = h^{ij}(d_i d_j lam - G^k_{ij} d_k lam)
                    - (1/lam) h^{ij} d_i lam d_j lam                  (1)
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

# --------------------------------------------------------------------- packing
PAIR = jnp.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])  # PAIR[j,k] -> slot, j <= k
SLOT_I = jnp.array([0, 0, 0, 1, 1, 2])
SLOT_J = jnp.array([0, 1, 2, 1, 2, 2])


def sym3(v):
    """(..., 6) -> (..., 3, 3) symmetric matrix."""
    return v[..., PAIR]


def pack_sym(H):
    """(..., 3, 3) -> (..., 6) (upper triangle, symmetric)."""
    return H[..., SLOT_I, SLOT_J]


def gamma3(v):
    """(..., 18) -> (..., 3, 3, 3) with Gamma^i_{jk} symmetric in (j, k)."""
    vv = v.reshape(v.shape[:-1] + (3, 6))
    return vv[..., :, PAIR]


def pack_gamma(G):
    """(..., 3, 3, 3) -> (..., 18) (18 independent components)."""
    out = G[..., jnp.arange(3)[:, None], SLOT_I, SLOT_J]
    return out.reshape(G.shape[:-3] + (18,))


class Fields(NamedTuple):
    """Fields of the first-order formulation at one or more points."""
    h: jnp.ndarray    # (..., 3, 3)
    G: jnp.ndarray    # (..., 3, 3, 3)   Gamma^i_{jk}
    lam: jnp.ndarray  # (...)


# ------------------------------------------------------------------ christoffel
def christoffel(h: Callable, x):
    """G[i,j,k] = Gamma^i_{jk} of the metric h (analytic reference path)."""
    H = h(x)
    J = jax.jacfwd(h)(x)                      # J[i,j,a] = d_a h_ij
    Hinv = jnp.linalg.inv(H)
    T = J.transpose(0, 2, 1) + J - J.transpose(2, 1, 0)   # T[l,j,k]
    return 0.5 * jnp.einsum("il,ljk->ijk", Hinv, T)


def laplacian(h: Callable, f: Callable, x):
    """Delta_h f = (1/sqrt h) d_a ( sqrt h h^{ab} d_b f ) -- independent route."""
    def V(y):
        H = h(y)
        return jnp.sqrt(jnp.linalg.det(H)) * jnp.einsum(
            "ab,b->a", jnp.linalg.inv(H), jax.jacfwd(f)(y))

    return jnp.trace(jax.jacfwd(V)(x)) / jnp.sqrt(jnp.linalg.det(h(x)))


def ricci_from_gamma(G, dG):
    """R_ij = d_k G^k_ij - d_i G^k_kj + G^k_kl G^l_ij - G^k_il G^l_kj."""
    return (jnp.einsum("kijk->ij", dG) - jnp.einsum("kkji->ij", dG)
            + jnp.einsum("kkl,lij->ij", G, G) - jnp.einsum("kil,lkj->ij", G, G))


# ------------------------------------------------------------------- residuals
def residuals_at(fields: Callable[[jnp.ndarray], Fields], x: jnp.ndarray,
                 gauge_src: Callable[[jnp.ndarray], jnp.ndarray] | None = None) -> dict:
    """All four residual groups at a single point x (no scaling applied).

    `fields` maps a point to (h, G, lam); the same code path is used by the
    network and by the tests (which feed the exact solution).

    `gauge_src(x)` is an INHOMOGENEOUS gauge source: the gauge residual becomes
    Gamma^i_{jk} h^{jk} - gauge_src^i instead of Gamma^i_{jk} h^{jk}.  The default None is
    the harmonic (de Donder) condition.  A source is what lets a chart that is *not*
    harmonic -- e.g. the Weyl chart of a two-black-hole solution -- be an exact solution
    of the whole system without a coordinate transformation, exactly as the manufactured
    Robin source does for the outer boundary.
    """
    h, G, lam = fields(x)
    dh, dG, dlam = jax.jacfwd(fields)(x)
    d2lam = jax.hessian(lambda y: fields(y)[2])(x)
    Hinv = jnp.linalg.inv(h)

    compat = (dh.transpose(2, 0, 1)
              - jnp.einsum("dab,dc->abc", G, h)
              - jnp.einsum("dac,bd->abc", G, h))
    ric = ricci_from_gamma(G, dG) - (1.0 / (2.0 * lam**2)) * jnp.outer(dlam, dlam)
    gauge = jnp.einsum("ijk,jk->i", G, Hinv)
    if gauge_src is not None:
        gauge = gauge - gauge_src(x)
    hess = d2lam - jnp.einsum("cij,c->ij", G, dlam)
    lam_eq = (jnp.einsum("ij,ij->", Hinv, hess)
              - (1.0 / lam) * jnp.einsum("ij,i,j->", Hinv, dlam, dlam))

    return dict(compat=compat, ricci=ric, gauge=gauge, lam_eq=lam_eq)


def residuals_batch(fields: Callable, xs: jnp.ndarray,
                    gauge_src: Callable[[jnp.ndarray], jnp.ndarray] | None = None) -> dict:
    """Vectorised residuals at many points."""
    return jax.vmap(lambda x: residuals_at(fields, x, gauge_src))(xs)


def scaled_residuals_batch(fields: Callable, xs: jnp.ndarray, exps: dict,
                           ref: float | None = None,
                           gauge_src: Callable[[jnp.ndarray], jnp.ndarray] | None = None) -> dict:
    """Residuals multiplied by a length**p so that the groups can be compared.

    Dimensionally compat, gauge ~ 1/length and ricci, lam_eq ~ 1/length^2, so
    exponents (1, 2, 1, 2) with a FIXED reference length give a loss measured in
    units of the reference scale (well conditioned).  With ref=None the local rho
    is used instead, which measures every residual relative to the size of its own
    terms but makes the far field dominate the loss by many orders of magnitude.
    """
    r = residuals_batch(fields, xs, gauge_src)
    if ref is None:
        base = jnp.linalg.norm(xs, axis=-1)
    else:
        base = jnp.full((xs.shape[0],), float(ref))
    out = {}
    for k, v in r.items():
        p = exps[k]
        out[k] = v * (base ** p)[(...,) + (None,) * (v.ndim - 1)]
    return out


def lam_eq_at(fields: Callable[[jnp.ndarray], Fields], x: jnp.ndarray) -> jnp.ndarray:
    """The lambda-equation residual at one point, unscaled.

    The `lam_eq` entry of `residuals_at`, on its own: `radial_derivative_residual_batch`
    differentiates it in x, and going through `residuals_at` would differentiate the Ricci
    and gauge groups (jax prunes unused outputs only partially, and the Ricci group is the
    expensive one) for nothing.
    """
    h, G, lam = fields(x)
    dlam = jax.jacfwd(lambda y: fields(y)[2])(x)
    d2lam = jax.hessian(lambda y: fields(y)[2])(x)
    Hinv = jnp.linalg.inv(h)
    hess = d2lam - jnp.einsum("cij,c->ij", G, dlam)
    return (jnp.einsum("ij,ij->", Hinv, hess)
            - (1.0 / lam) * jnp.einsum("ij,i,j->", Hinv, dlam, dlam))


def radial_derivative_residual_batch(fields: Callable, xs: jnp.ndarray, exps: dict,
                                     ref: float | None = None) -> jnp.ndarray:
    """d/drho of the raw lambda-equation residual, times length**(exp + 1).

    The lambda-equation is second order, so a loss built from it alone is blind to
    lambda''' -- and lambda''' at rho_out is exactly where the order-3 Robin condition lets a
    wrong far-field level hide (the operator's kernel is rho^-1, rho^-2, rho^-3, so its
    residual is a cancellation of terms of order 0.1 that leaves 1.3e-08; see the Config
    comment on the far-field pins).  One rho-derivative makes the loss see that content.

    The extra factor of length keeps the term dimensionless and scale-covariant, exactly as
    `scaled_residuals_batch` does for the groups themselves: the residual carries
    length^-exps['lam_eq'] = length^-2 and its radial derivative length^-3, so the product
    rho^3 d(rho)/d rho is invariant under rho -> rho/s with the fields relabelled.
    """
    p = float(exps.get("lam_eq", 2.0)) + 1.0

    def one(x):
        rho = jnp.linalg.norm(x)
        n = x / rho
        # jvp, not jacfwd + contraction: only the derivative ALONG n is wanted, and the full
        # Jacobian would build all three directional derivatives and throw two away.  Measured
        # on the production network at 16384 points this is the difference between 1.9x and
        # ~1.5x the plain PDE cost per value+gradient.
        d = jax.jvp(lambda y: lam_eq_at(fields, y), (x,), (n,))[1]
        base = rho if ref is None else jnp.asarray(float(ref))
        return d * base ** p

    return jax.vmap(one)(xs)
