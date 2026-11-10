#!/usr/bin/env python
"""Check the Green identities behind the centre-of-mass and rotation fluxes, with a gauge source.

    .venv/bin/python scripts/green_identities.py runs/<run> [--params params.pkl] [--radii 1,1.5,3,6,10]

With Gamma^i = h^jk Gamma^i_jk (= S^i in the gauge Gamma^i = S^i) the divergence theorem on the shell
rho1 < |x| < rho2 gives, EXACTLY for any metric (d_l(sqrt h h^lm) = -sqrt h Gamma^m):

  rotation flux  F_k = eps_kij oint sqrt h x^i h^jl n_l dA          (= 8 pi r_k, losses.rot_flux)
      F(rho2) - F(rho1) = - int eps_kij x^i Gamma^j sqrt h d3x
  centre of mass D^i = oint sqrt h h^lm (x^i d_m U - U delta^i_m) n_l dA   (= 4 pi d^i, losses.com_flux)
      D(rho2) - D(rho1) = int ( x^i Delta_h U + U Gamma^i ) sqrt h d3x

The PHYSICAL identities replace Gamma by the gauge source S and Delta_h U by 0 (the lambda equation):
their mismatch is then the integral of eps x (Gamma - S) sqrt h, resp. x Delta_h U + U (Gamma - S),
i.e. of the gauge and lambda-equation residuals.  For each shell this prints the flux difference,
the volume term with the exact Gamma (checks the quadrature: must agree to its accuracy), the volume
term with S, and the mismatch, for the run's network AND for the exact solution of its data
(exact_asset; Weyl runs).  2026-11-10, Paper/NOTES.md "The rotation gauge condition".
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stationary.evaluate import load_run                  # noqa: E402
from stationary.losses import com_flux, gauge_source_of, rot_flux   # noqa: E402
from stationary.model import point_fields                 # noqa: E402
from stationary.train import exact_asset                  # noqa: E402


def sphere_quad(nmu, nphi):
    mu, w = np.polynomial.legendre.leggauss(nmu)
    phi = 2 * np.pi * np.arange(nphi) / nphi
    MU, PHI = np.meshgrid(mu, phi, indexing="ij")
    st = np.sqrt(1 - MU ** 2)
    n = np.stack([st * np.cos(PHI), st * np.sin(PHI), MU], -1).reshape(-1, 3)
    return n, np.outer(w, np.full(nphi, 2 * np.pi / nphi)).reshape(-1)


def check(name, pf, cfg, radii, nr, nmu, nphi):
    S_of = gauge_source_of(cfg, pf)

    def integrands(x):
        f = pf(x)
        h, G = f.h, f.G
        hinv = jnp.linalg.inv(h)
        sq = jnp.sqrt(jnp.linalg.det(h))
        Gam = jnp.einsum("jk,ijk->i", hinv, G)
        S = jnp.zeros(3) if S_of is None else S_of(x)
        Uf = lambda y: 0.5 * jnp.log(pf(y).lam)
        U = Uf(x)
        dU = jax.grad(Uf)(x)
        ddU = jax.hessian(Uf)(x)
        lapU = jnp.einsum("lm,lm->", hinv, ddU) - Gam @ dU
        return (-jnp.cross(x, Gam) * sq, -jnp.cross(x, S) * sq,
                (x * lapU + U * Gam) * sq, (U * S) * sq, (x * lapU + U * (Gam - S)) * sq)

    vint = jax.jit(jax.vmap(integrands))
    n, wn = sphere_quad(nmu, nphi)
    F = np.asarray(rot_flux(pf, cfg, n_mu=nmu, n_phi=nphi, radii=radii)).reshape(-1, 3) * 8 * np.pi
    cc = cfg.__class__(**{**cfg.__dict__, "com_radii": tuple(radii)})
    D = np.asarray(com_flux(pf, cc, n_mu=nmu, n_phi=nphi)).reshape(-1, 3) * 4 * np.pi
    out = []
    print(f"[green] {name}")
    for a in range(len(radii) - 1):
        r1, r2 = radii[a], radii[a + 1]
        t, wt = np.polynomial.legendre.leggauss(nr)
        rr = 0.5 * (r2 - r1) * t + 0.5 * (r2 + r1)
        wr = 0.5 * (r2 - r1) * wt * rr ** 2
        acc = [np.zeros(3) for _ in range(5)]
        for rk, wk in zip(rr, wr):
            vals = vint(jnp.asarray(rk * n))
            for i in range(5):
                acc[i] += wk * np.einsum("p,pk->k", wn, np.asarray(vals[i]))
        dF, dD = F[a + 1] - F[a], D[a + 1] - D[a]
        row = {"shell": [r1, r2], "dF": dF.tolist(), "rot_vol_Gamma": acc[0].tolist(),
               "rot_vol_S": acc[1].tolist(), "dD": dD.tolist(), "com_vol_Gamma": acc[2].tolist(),
               "com_vol_S": acc[3].tolist(), "com_vol_residual": acc[4].tolist()}
        out.append(row)
        nrm = lambda v: float(np.linalg.norm(v))
        print(f"[green]   shell [{r1:g},{r2:g}]  ROTATION |dF| {nrm(dF):.2e}  |dF - vol(Gamma)| {nrm(dF - acc[0]):.1e}"
              f" (quadrature)  |vol(S)| {nrm(acc[1]):.2e}  |dF - vol(S)| {nrm(dF - acc[1]):.2e}")
        print(f"[green]                    CENTRE OF MASS |dD| {nrm(dD):.2e}  |dD - vol(Gamma, Delta U)| "
              f"{nrm(dD - acc[2]):.1e} (quadrature)  |vol(U S)| {nrm(acc[3]):.2e}  |dD - vol(U S)| "
              f"{nrm(dD - acc[3]):.2e}  (residual term {nrm(acc[4]):.2e})")
    return {"F": F.tolist(), "D": D.tolist(), "shells": out}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--params", default="params.pkl")
    ap.add_argument("--radii", default=None, help="default: the run's com_radii")
    ap.add_argument("--nr", type=int, default=16)
    ap.add_argument("--nmu", type=int, default=48)
    ap.add_argument("--nphi", type=int, default=32)
    a = ap.parse_args(argv)
    cfg, model, state = load_run(a.run, a.params)[:3]
    radii = [float(v) for v in a.radii.split(",")] if a.radii else list(cfg.com_radii)
    pf = point_fields(model, state["net"])
    ex = exact_asset(cfg)
    res = {"run": a.run, "radii": radii, "gauge_source": cfg.gauge_source,
           "network": check("network", pf, cfg, radii, a.nr, a.nmu, a.nphi)}
    if ex is not None:
        res["exact"] = check("exact solution", ex, cfg, radii, a.nr, a.nmu, a.nphi)
    with open(os.path.join(a.run, "green_identities.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
