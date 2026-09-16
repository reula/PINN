"""Chart-independent check of a spherically symmetric PINN solution.

The harmonic gauge leaves the residual freedom of harmonic diffeomorphisms, so the
metric components h_ij(x) are not directly comparable between two solutions written
in different (both harmonic) charts.  Geometric scalars are:

    r_a(rho)  areal radius of the sphere through x        (sqrt of the induced area / 4pi)
    lambda    the scalar field
    R         the Ricci scalar

For the exact family h = d rho_c^2 + (rho_c^2-R0^2) dOmega^2,
lambda = k (rho_c-R0)/(rho_c+R0) one has the chart-independent relations

    R = 2 R0^2 / r_a^4
    lambda = k (sqrt(r_a^2+R0^2) - R0)/(sqrt(r_a^2+R0^2) + R0)

which allow (R0, k) to be read off from any solution and checked for consistency.
"""
from __future__ import annotations

import argparse
import json
import pickle

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from . import exact
from .geometry import christoffel, ricci_from_gamma
from .model import point_fields
from .problem import Config
from .train import make_model


def invariants_at(pf, rho, n=None):
    """(r_a, lambda, R) at the point rho*n for a spherically symmetric solution."""
    if n is None:
        n = jnp.array([1.0, 0.0, 0.0])
    x = rho * n
    f = pf(x)
    h = f.h
    # areal radius: for h = alpha delta + beta n n the tangential coefficient is alpha,
    # and r_a^2 = rho^2 * (coefficient of the round part)
    hrr = n @ h @ n
    alpha = (jnp.trace(h) - hrr) / 2.0
    r_a = rho * jnp.sqrt(alpha)
    # Ricci scalar from the network's Gamma
    dh, dG, _ = jax.jacfwd(pf)(x)
    R = jnp.einsum("ij,ij->", jnp.linalg.inv(h), ricci_from_gamma(f.G, dG))
    return float(r_a), float(f.lam), float(R)


def family_params_from_solution(pf, cfg, nrho=40, lo=None, hi=None):
    """Read (R0, k) off the solution using the two chart-independent relations."""
    lo = cfg.rho_in + 1e-6 if lo is None else lo
    hi = cfg.rho_out if hi is None else hi
    rhos = jnp.linspace(lo, hi, nrho)
    ra, lam, R = zip(*[invariants_at(pf, r) for r in rhos])
    ra = jnp.array(ra); lam = jnp.array(lam); R = jnp.array(R)
    # R = 2 R0^2 / r_a^4  ->  R0^2 = R r_a^4 / 2   (use the outer half, where R is small
    # but the relation is cleanest)
    R0sq = 0.5 * R * ra**4
    # lambda = k (sqrt(ra^2+R0^2)-R0)/(sqrt(ra^2+R0^2)+R0)
    R0 = float(jnp.median(jnp.sqrt(jnp.abs(R0sq))))
    g = (jnp.sqrt(ra**2 + R0**2) - R0) / (jnp.sqrt(ra**2 + R0**2) + R0)
    k = float(jnp.median(lam / g))
    resid_R = float(jnp.max(jnp.abs(R - 2 * R0**2 / ra**4)))
    resid_lam = float(jnp.max(jnp.abs(lam - k * (jnp.sqrt(ra**2 + R0**2) - R0)
                                      / (jnp.sqrt(ra**2 + R0**2) + R0))))
    return dict(R0=R0, k=k, max_resid_R=resid_R, max_resid_lambda=resid_lam,
                ra=[float(v) for v in ra], lam=[float(v) for v in lam],
                R=[float(v) for v in R], R0sq=[float(v) for v in R0sq])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--params-file", default="params.pkl")
    a = p.parse_args()
    with open(f"{a.run_dir}/config.json") as fh:
        cfg = Config(**json.load(fh))
    model = make_model(cfg)
    with open(f"{a.run_dir}/{a.params_file}", "rb") as fh:
        state = pickle.load(fh)
    pf = point_fields(model, state["net"])
    out = family_params_from_solution(pf, cfg)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}, indent=2))
    print("\n r_a        lambda      R            2 R0^2/r_a^4   lambda_family")
    ra, lam, R, R0sq = out["ra"], out["lam"], out["R"], out["R0sq"]
    k, R0 = out["k"], out["R0"]
    for i in range(0, len(ra), max(1, len(ra) // 12)):
        fam = k * (jnp.sqrt(ra[i] ** 2 + R0**2) - R0) / (jnp.sqrt(ra[i] ** 2 + R0**2) + R0)
        print(f"  {ra[i]:8.4f} {lam[i]:10.6f} {R[i]:12.5e} {2*R0**2/ra[i]**4:12.5e}  {float(fam):10.6f}")


if __name__ == "__main__":
    main()
