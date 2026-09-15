"""Residuals, boundary conditions and the total loss.

Inner boundary (rho = rho_in), as specified:
    * lambda = lambda_0
    * the induced metric on the sphere is the round metric of areal radius 2
    * h_rr = 1        (normal-normal component; fixes the remaining metric freedom)

Outer boundary (rho = rho_out):
    * "dirichlet_exact": h and lambda from the exact solution (milestone 1 -- a
       manufactured-solution test whose exact answer is known)
    * "robin":  n^i d_i field = -(field - field_inf)/rho_out  for h, Gamma, lambda
       with field_inf = delta_ij, 0, lambda_inf  (the decay condition)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .geometry import pack_gamma, pack_sym, residuals_batch, scaled_residuals_batch

I3 = jnp.eye(3)
OFFW = jnp.array([1.0, jnp.sqrt(2.0), jnp.sqrt(2.0), 1.0, jnp.sqrt(2.0), 1.0])


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
        c = 4.0 / rho**2
        M = f.h - h_rr * nn - c * (I3 - nn)
        return jnp.concatenate([
            jnp.array([f.lam - cfg.lam0, h_rr - 1.0]),
            pack_sym(M) * OFFW,
        ])

    R = jax.vmap(one)(xs)
    return {
        "lam": jnp.mean(R[:, 0] ** 2),
        "h_rr": jnp.mean(R[:, 1] ** 2),
        "h_tan": jnp.mean(R[:, 2:] ** 2),
    }


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
        r_out = cfg.rho_out

        def one(x):
            f = point_fields(x)
            dh, dG, dlam = jax.jacfwd(point_fields)(x)
            rho = jnp.linalg.norm(x)
            n = x / rho
            dr_h = jnp.einsum("a,ija->ij", n, dh)       # n^a d_a h_ij
            dr_G = jnp.einsum("a,ijka->ijk", n, dG)
            dr_l = jnp.einsum("a,a->", n, dlam)
            return jnp.concatenate([
                pack_sym(dr_h + (f.h - I3) / r_out) * OFFW,
                pack_gamma(dr_G + f.G / r_out),
                jnp.array([dr_l + (f.lam - lam_inf) / r_out]),
            ])

        R = jax.vmap(one)(xs)
        return {"h": jnp.mean(R[:, :6] ** 2), "G": jnp.mean(R[:, 6:24] ** 2),
                "lam": jnp.mean(R[:, 24] ** 2)}

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
               weights=None, parts_out=None):
    """state = {'net': params[, 'lam_inf': scalar]};  batch = dict of point arrays."""
    from .model import point_fields as make_point_fields

    pf = make_point_fields(model, state["net"])
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
