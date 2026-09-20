"""Residuals, boundary conditions and the total loss.

Inner boundary (rho = rho_in):
    * lambda = lam0 + S1 z/rho_in + S2 (z^2-(x^2+y^2)/2)/rho_in^2   (see problem.py)
    * the induced metric on the sphere is the round metric of areal radius
      `inner_radius` (default rho_in)
    * optionally h_rr = const (off by default: it over-determines the radial gauge)

Outer boundary (rho = rho_out):
    * "dirichlet_exact": h and lambda from the exact solution (manufactured test)
    * "robin": higher-multipole decay condition.  With n = cfg.robin_order and the
      leading decay exponent k_f of each field,

          prod_{i=0}^{n-1} (rho d_rho + k_f + i) (field - field_inf) = 0 .

      The Euler operators (rho d_rho + a) commute and annihilate exactly rho^-a, so
      n = 1 is the familiar (field-field_inf)/rho + d_rho field = 0 and n = 2 lets the
      next multipole through, e.g. for lambda: rho^2 lam'' + 4 rho lam' + 2 (lam-1) = 0.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .geometry import pack_gamma, pack_sym, residuals_batch, scaled_residuals_batch
from .problem import lam_inner_bc, sample_sphere

I3 = jnp.eye(3)
OFFW = jnp.array([1.0, jnp.sqrt(2.0), jnp.sqrt(2.0), 1.0, jnp.sqrt(2.0), 1.0])


def robin_operator(field_fun, x, base, order, inf_val=0.0):
    """prod_{i<order} (rho d_rho + base+i) applied to (field - inf_val) at x.

    (rho d_rho + a) annihilates exactly rho^-a, and these Euler operators commute, so
    the product annihilates the powers rho^-base ... rho^-(base+order-1) and lets the
    next multipoles through.  `order=1` is the familiar first-order decay condition.
    """
    cur = lambda y: field_fun(y) - inf_val
    for i in range(order):
        k = base + i
        prev = cur

        def cur(y, prev=prev, k=k):
            r = jnp.linalg.norm(y)
            n = y / r
            d = jnp.einsum("a,...a->...", n, jax.jacfwd(prev)(y))       # d_rho prev
            return r * d + k * prev(y)

    return cur(x)


# ------------------------------------------------------------------ PDE terms
def pde_terms(point_fields, xs, cfg) -> dict:
    """Mean squared (rho-scaled) residuals of the four equation groups."""
    r = scaled_residuals_batch(point_fields, xs, cfg.scale_exps, cfg.scale_ref)
    return {k: jnp.mean(v ** 2) for k, v in r.items()}


# -------------------------------------------------------------- inner boundary
def inner_bc_terms(point_fields, xs, cfg) -> dict:
    def one(x):
        f = point_fields(x)
        rho = jnp.linalg.norm(x)
        n = x / rho
        nn = jnp.outer(n, n)
        h_rr = n @ f.h @ n
        c = cfg.inner_radius**2 / rho**2          # round metric of areal radius inner_radius
        M = f.h - h_rr * nn - c * (I3 - nn)
        hrr_ref = 1.0 if cfg.inner_h_rr is None else cfg.inner_h_rr
        return jnp.concatenate([
            jnp.array([f.lam - lam_inner_bc(x, cfg), h_rr - hrr_ref]),
            pack_sym(M) * OFFW,
        ])

    R = jax.vmap(one)(xs)
    out = {"lam": jnp.mean(R[:, 0] ** 2), "h_tan": jnp.mean(R[:, 2:] ** 2)}
    if cfg.inner_h_rr is not None:
        out["h_rr"] = jnp.mean(R[:, 1] ** 2)
    return out


def reference_consistency(point_fields, cfg, n: int = 64, seed: int = 7) -> dict:
    """How much the exact reference itself violates the imposed INNER boundary data.

    Zero means the inner data and the outer data (the Dirichlet values, or the
    manufactured Robin source) come from one and the same exact solution, so that
    solution is a genuine zero of the loss.  Anything else means they come from two
    different solutions: no metric can satisfy both, the run converges to a compromise,
    and the numbers it reports near the inner sphere do not mean what they look like.
    The usual cause is `inner_radius` not being the reference's own inner areal radius
    (see Config.__post_init__).
    """
    xs = sample_sphere(jax.random.PRNGKey(seed), n, cfg.rho_in)
    return {k: float(v) for k, v in inner_bc_terms(point_fields, xs, cfg).items()}


# -------------------------------------------------------------- outer boundary
def outer_bc_terms(point_fields, xs, cfg, exact_fields=None, lam_inf=None) -> dict:
    if cfg.outer_bc == "dirichlet_exact":
        if exact_fields is None:
            raise ValueError("dirichlet_exact outer BC needs exact_fields")

        def one(x):
            f = point_fields(x)
            e = exact_fields(x)
            return jnp.concatenate([
                pack_sym(f.h - e.h) * OFFW,
                jnp.array([f.lam - e.lam]),
            ])

        R = jax.vmap(one)(xs)
        return {"h": jnp.mean(R[:, :6] ** 2), "lam": jnp.mean(R[:, 6] ** 2)}

    if cfg.outer_bc == "robin":
        ph, pG, pl = cfg.robin_exps["h"], cfg.robin_exps["G"], cfg.robin_exps["lam"]
        orders = cfg.robin_orders or {k: cfg.robin_order for k in ("h", "G", "lam")}
        if lam_inf is None:
            lam_inf = cfg.lam_inf_init
        if cfg.robin_source and exact_fields is None:
            raise ValueError("robin_source needs exact_fields (set --ref-solution)")

        def robin(field_fun, x, base, inf_val, order):
            return robin_operator(field_fun, x, base, order, inf_val)

        def one(x):
            f = point_fields(x)
            rh = robin(lambda y: point_fields(y).h, x, ph, I3, orders["h"])
            rl = robin(lambda y: point_fields(y).lam, x, pl, lam_inf, orders["lam"])
            sh = sl = 0.0
            if cfg.robin_source:
                sh = robin(lambda y: exact_fields(y).h, x, ph, I3, orders["h"])
                sl = robin(lambda y: exact_fields(y).lam, x, pl, lam_inf, orders["lam"])
            if cfg.robin_include_G:
                rG = robin(lambda y: point_fields(y).G, x, pG, jnp.zeros((3, 3, 3)),
                           orders["G"])
                sG = 0.0
                if cfg.robin_source:
                    sG = robin(lambda y: exact_fields(y).G, x, pG, jnp.zeros((3, 3, 3)),
                               orders["G"])
                return jnp.concatenate([pack_sym(rh - sh) * OFFW,
                                        pack_gamma(rG - sG),
                                        jnp.array([rl - sl])])
            return jnp.concatenate([pack_sym(rh - sh) * OFFW, jnp.array([rl - sl])])

        R = jax.vmap(one)(xs)
        out = {"h": jnp.mean(R[:, :6] ** 2), "lam": jnp.mean(R[:, -1] ** 2)}
        if cfg.robin_include_G:
            out["G"] = jnp.mean(R[:, 6:24] ** 2)
        return out

    raise ValueError(f"unknown outer_bc {cfg.outer_bc!r}")


# ------------------------------------------------------------------ total loss
GROUP_KEYS = ("compat", "ricci", "gauge", "lam_eq", "inner", "outer")


def default_weights(cfg):
    w = {k: jnp.asarray(cfg.eq_weights[k]) for k in ("compat", "ricci", "gauge", "lam_eq")}
    w["inner"] = jnp.asarray(cfg.w_inner)
    w["outer"] = jnp.asarray(cfg.w_outer)
    return w


def group_terms(state, batch, cfg, model, exact_fields=None, lam_inf=None) -> dict:
    """Unweighted mean-squared residuals of the six loss groups.

    Used both for diagnostics and for gradient-norm adaptive weighting.
    """
    from .model import point_fields as make_point_fields

    pf = make_point_fields(model, state["net"])
    out = dict(pde_terms(pf, batch["coll"], cfg))
    out["inner"] = sum(inner_bc_terms(pf, batch["inner"], cfg).values())
    out["outer"] = sum(outer_bc_terms(pf, batch["outer"], cfg, exact_fields, lam_inf).values())
    return out


def total_loss(state, batch, cfg, model, exact_fields=None, pde_scale=1.0,
               weights=None, lam_inf=None):
    """state = {'net': params[, 'lam_inf': scalar]};  batch = dict of point arrays.

    `lam_inf` overrides the state leaf, which is how a FROZEN asymptotic value is
    imposed: leaving it optimisable lets the trivial (flat, lambda = const) branch
    re-select itself, since that branch satisfies every Robin condition whenever
    lambda_inf is allowed to equal lambda.
    """
    from .model import point_fields as make_point_fields

    pf = make_point_fields(model, state["net"])
    if lam_inf is None:
        lam_inf = state.get("lam_inf", None)
    if weights is None:
        weights = default_weights(cfg)

    parts = {}
    for k, v in pde_terms(pf, batch["coll"], cfg).items():
        parts[f"pde_{k}"] = v
    inner = inner_bc_terms(pf, batch["inner"], cfg)
    for k, v in inner.items():
        parts[f"inner_{k}"] = v
    outer = outer_bc_terms(pf, batch["outer"], cfg, exact_fields, lam_inf)
    for k, v in outer.items():
        parts[f"outer_{k}"] = v

    loss = 0.0
    for k in ("compat", "ricci", "gauge", "lam_eq"):
        loss = loss + pde_scale * weights[k] * parts[f"pde_{k}"]
    loss = loss + weights["inner"] * sum(inner.values())
    loss = loss + weights["outer"] * sum(outer.values())

    return loss, parts
