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
# Whether the compatibility residual is FORMED at all.  None means "follow this default",
# which the program sets once from the model it built: a model that derives Gamma from h does
# not need it.  A module-level default rather than an argument at five call sites because the
# callers hold closures over the model, not the model -- and getting one of them wrong would
# silently drop the residual where it IS an equation.
_WANT_COMPAT = True

# The lambda source in the Ricci equation: Ricci_ab = (1/2 lambda^2) d_a lambda d_b lambda.  Set
# to 0 to solve Ricci(h) = 0 with the lambda data and equation otherwise untouched, so the metric
# relaxes to the VACUUM metric of the imposed data while the gauge still fixes the chart.
_RICCI_LAM_SOURCE = 1.0

# Which form of the lambda equation is imposed (Config.lam_eq_form, set once by the program):
#
#   "lambda"   Delta_h lam - (1/lam) |d lam|^2_h
#   "log"      the same equation written for phi = log lam,  Delta_h phi,
#
# related by Delta_h phi = (1/lam) [Delta_h lam - (1/lam)|d lam|^2_h].  The zeros are the same
# set, but the "lambda" residual is HOMOGENEOUS OF DEGREE ONE in lam: lam -> c lam scales it by
# c, so lam -> 0 is a free direction -- it costs nothing in the loss and solves nothing.  That
# is not academic: runs/pq_c100_vac6_fixref (fixed scale_ref = rho_in, i.e. the rho^d weights
# replaced by constants) collapsed lam from 0.25 at the inner sphere to 1e-7 by rho = 1.8 and
# still finished with a loss of 2.8e-13, because the residual it was minimising is that
# homogeneous one.  The "log" form divides by lam and charges for exactly that collapse; it is
# the same equation, differently weighted, not a different equation.  The division is guarded at
# the single point lam = 0 and the ratio is capped, so a network crossing zero gives a large but
# finite residual instead of inf/NaN.  What the guard must NOT do is clamp |lam| up to a floor:
# that would send base/lam -> base/floor, and since base is proportional to lam for a uniformly
# small field, the charged collapse would become CHEAP again below the floor -- the degenerate
# direction would just move.  Only an exact zero needs the guard.
_LAM_EQ_FORM = "lambda"
_LAM_EQ_FLOOR = 1e-12
_LAM_EQ_CAP = 1e12


def set_lam_eq_form(form: str, floor: float | None = None) -> None:
    global _LAM_EQ_FORM, _LAM_EQ_FLOOR
    if form not in ("lambda", "log"):
        raise ValueError(f"lam_eq_form must be 'lambda' or 'log', got {form!r}")
    if form == "log" and _RELATIVE_TERMS:
        raise ValueError(
            "lam_eq_form='log' and relative_terms are alternatives, not composable: "
            "Delta_h(log lam) is a single term, so its relative form is a sign")
    _LAM_EQ_FORM = form
    if floor is not None:
        _LAM_EQ_FLOOR = float(floor)


def lam_eq_of(Hinv, hess, dlam, lam):
    """The lambda-equation residual in the form `_LAM_EQ_FORM` selects.

    `hess[i,j] = d_i d_j lam - Gamma^k_{ij} d_k lam` is the covariant Hessian of lam, computed
    by the caller; the second term of the equation is |d lam|^2_h / lam.  The "log" form is
    `base / lam`, which is exactly Delta_h(log lam); it is invariant under lam -> c lam and
    therefore does not vanish as the field shrinks.
    """
    lam_safe = jnp.where(lam == 0.0, _LAM_EQ_FLOOR, lam)
    lap = jnp.einsum("ij,ij->", Hinv, hess)
    src = jnp.einsum("ij,i,j->", Hinv, dlam, dlam) / lam_safe
    base = lap - src
    if _RELATIVE_TERMS:
        return base / (jnp.abs(lap) + jnp.abs(src) + _REL_EPS)
    if _LAM_EQ_FORM == "lambda":
        return base
    return jnp.clip(base / lam_safe, -_LAM_EQ_CAP, _LAM_EQ_CAP)


def set_ricci_lam_source(v: float) -> None:
    global _RICCI_LAM_SOURCE
    _RICCI_LAM_SOURCE = float(v)


def set_want_compat(flag: bool) -> None:
    global _WANT_COMPAT
    _WANT_COMPAT = bool(flag)


# RELATIVE residuals (Config.relative_terms / --relative-terms).  Each group's residual is a
# signed sum of terms of the same dimension, so dividing the sum by the sum of the ABSOLUTE
# VALUES of those terms gives a dimensionless number in [-1, 1] that measures how much of the
# equation failed to cancel -- the local relative error, independent of the local scale.  Two
# consequences, both of which are the point:
#
#   * no length scale has to be chosen, and no rho^d weight is needed: the residual is already
#     dimensionless, so `scaled_residuals_batch` becomes a no-op in this mode.  In particular
#     the far field is not silently de-emphasised, which is what killed runs/pq_c100_vac6_fixref;
#   * a residual that is small only because ITS TERMS are small (the lambda equation is
#     homogeneous of degree one in lambda, so lambda -> 0 does that) is divided by those same
#     small terms and comes back O(1): the collapse direction is charged.
#
# The elementary terms are formed explicitly in `residuals_at` (they were already computed
# there, combined), so the denominator is the size of exactly what cancelled.
_RELATIVE_TERMS = False
_REL_EPS = 1e-300          # only guards 0/0; the ratio is bounded by construction


def set_relative_terms(flag: bool) -> None:
    """Turn the relative-residual losses on or off.

    Not composable with the log form of the lambda equation: Delta_h(log lam) is a SINGLE term,
    so its relative form would be a sign, not a size.  `--relative-terms` already charges the
    lambda -> 0 collapse by itself, so the two are alternatives; asking for both is a mistake
    and is rejected here rather than silently resolved.
    """
    global _RELATIVE_TERMS
    if flag and _LAM_EQ_FORM != "lambda":
        raise ValueError(
            "relative_terms and lam_eq_form='log' are alternatives, not composable: "
            "Delta_h(log lam) has one term, so its relative form is a sign")
    _RELATIVE_TERMS = bool(flag)


def _norm(t):
    """Frobenius norm of a term tensor (a scalar for a rank-0 term)."""
    return jnp.sqrt(jnp.sum(t * t))


def residuals_at(fields: Callable[[jnp.ndarray], Fields], x: jnp.ndarray,
                 gauge_src: Callable[[jnp.ndarray], jnp.ndarray] | None = None, want_compat: bool | None = None) -> dict:
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

    if want_compat is None:
        want_compat = _WANT_COMPAT
    # `compat` is NOT formed when the caller does not need it.  For a model that derives Gamma
    # from h it holds identically and is not in the loss, so computing it is pure cost; it is
    # kept for the first-order formulation, where it is a real equation.
    compat = None
    if want_compat:
        t1 = dh.transpose(2, 0, 1)                      # d_a h_bc
        t2 = jnp.einsum("dab,dc->abc", G, h)            # Gamma^d_{ab} h_dc
        t3 = jnp.einsum("dac,bd->abc", G, h)            # Gamma^d_{ac} h_bd
        compat = t1 - t2 - t3
        if _RELATIVE_TERMS:
            compat = compat / (_norm(t1) + _norm(t2) + _norm(t3) + _REL_EPS)

    if _RELATIVE_TERMS:
        # The four elementary pieces of R_ij, kept apart so their sizes are known.
        r1 = jnp.einsum("kijk->ij", dG)
        r2 = jnp.einsum("kkji->ij", dG)
        r3 = jnp.einsum("kkl,lij->ij", G, G)
        r4 = jnp.einsum("kil,lkj->ij", G, G)
        r5 = _RICCI_LAM_SOURCE * (1.0 / (2.0 * lam**2)) * jnp.outer(dlam, dlam)
        ric = (r1 - r2 + r3 - r4 - r5) / (_norm(r1) + _norm(r2) + _norm(r3) + _norm(r4)
                                          + _norm(r5) + _REL_EPS)
    else:
        ric = (ricci_from_gamma(G, dG)
               - _RICCI_LAM_SOURCE * (1.0 / (2.0 * lam**2)) * jnp.outer(dlam, dlam))

    # Gamma^i_{jk} h^{jk}: the elementary contributions are the nine products, so the
    # relative denominator is their absolute sum (per i).
    contrib = G * Hinv[None, :, :]
    gauge = jnp.sum(contrib, axis=(1, 2))
    gsrc = gauge_src(x) if gauge_src is not None else None
    if gsrc is not None:
        gauge = gauge - gsrc
    if _RELATIVE_TERMS:
        den = jnp.sum(jnp.abs(contrib), axis=(1, 2))
        if gsrc is not None:
            den = den + jnp.abs(gsrc)
        gauge = gauge / (den + _REL_EPS)

    hess = d2lam - jnp.einsum("cij,c->ij", G, dlam)
    lam_eq = lam_eq_of(Hinv, hess, dlam, lam)

    out = dict(ricci=ric, gauge=gauge, lam_eq=lam_eq)
    if want_compat:
        out["compat"] = compat
    return out


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

    In relative mode (`set_relative_terms(True)`) the residuals are already dimensionless
    ratios -- each one divided by the size of its own terms -- so no length is applied and the
    exponents are ignored.  Multiplying them by rho^p as well would double-count the
    dimensions and undo the point of the mode.
    """
    r = residuals_batch(fields, xs, gauge_src)
    if _RELATIVE_TERMS:
        return r
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
    return lam_eq_of(Hinv, hess, dlam, lam)


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

    In relative mode the residual itself is dimensionless (a ratio), so its radial derivative
    carries length^-1 and the factor is rho^1, not rho^3.
    """
    p = (0.0 if _RELATIVE_TERMS else float(exps.get("lam_eq", 2.0))) + 1.0

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
