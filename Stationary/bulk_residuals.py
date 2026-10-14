"""Acceptance table for a run: the BULK residual, not just the value on the outer sphere.

The report's `PDE RESIDUALS (raw units)` table is evaluated on a sphere at `rho_out` only.  That is
fine when the field is a genuine solution, but it is gameable: a run can satisfy an outer boundary
condition inside a razor-thin layer at `rho_out` (measured: `|B_3[h - I_3]| ~ 5` in the bulk against
`1.8e-04` at `rho = 100.0` in `pq_r100_phihyb025_L2`) and the table then reports the layer, not the
solution.  This script prints both, side by side, for one or more run directories:

    bulk     the rms over a fresh collocation batch (what the loss actually integrates)
    outer    the rms on a sphere at rho_out (what report.txt quotes)
    h, lam   the l = 0 radial profiles at a few radii, which is how "the metric decays to the
             identity" is checked: `h_rr - 1` must fall with rho, not sit at a constant

    ./.venv/bin/python bulk_residuals.py runs/pq_b_U1 runs/pq_b_U2 ...
"""
from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def _means(batched, rhos, n_th=64):
    """Spherical-mean (l = 0) profiles of h_rr, h_tan, lambda for a BATCHED field function."""
    mu, w = np.polynomial.legendre.leggauss(n_th)
    mu_j = jnp.asarray(mu)
    sin_th = jnp.sqrt(1.0 - mu_j ** 2)
    rows = {}
    for r in rhos:
        x = jnp.stack([float(r) * sin_th, jnp.zeros_like(sin_th), float(r) * mu_j], axis=-1)
        f = batched(x)
        n = x / float(r)
        h_rr = jnp.einsum("ni,nij,nj->n", n, f.h, n)
        h_tan = 0.5 * (jnp.einsum("nii->n", f.h) - h_rr)
        # the l = 0 Legendre projection is (2*0+1)/2 * int dmu = (1/2) sum w .
        rows[r] = (0.5 * float(jnp.sum(w * h_rr)), 0.5 * float(jnp.sum(w * h_tan)),
                   0.5 * float(jnp.sum(w * f.lam.reshape(-1))))
    return rows


def profile(run_dir, rhos=(2.0, 10.0, 50.0, 99.0), n_th=64):
    """The run's own l = 0 profiles, and their departure from the SPHERICAL reference.

    Three cautions, all of which bit us at least once.

    (1) `h_rr` is NOT supposed to be 1 in the inner region: the Weyl metric in these variables is
        0.9155 at rho = 2 (the mass term) and only approaches 1 outward, so "h_rr - 1 = -8.6e-02"
        is the correct value, not an error.

    (2) That reference is SPHERICAL and these runs impose a QUADRUPOLE inner datum
        (lam_bc_S2 = -1/12), which the reference cannot carry -- `train.build` prints exactly that
        warning.  So the departure is NOT an error either: near the source it is dominated by the
        quadrupole's back-reaction (measured: 3.4e-03 in lambda at rho = 2, consistent with the
        second-order size S2^2 ~ 7e-03), and it is only in the far field, where that back-reaction
        has died away, that the numbers can be read as accuracy.

    (3) Therefore the accuracy statement for a run has to come from the EQUATIONS -- the bulk
        residual printed above -- and from the boundary data, not from this column.
    """
    from stationary.evaluate import load_run

    cfg, model, state = load_run(run_dir)
    net = _means(lambda x: model.apply(state["net"], x), rhos, n_th)
    try:
        from stationary.train import exact_asset
        ref = _means(jax.vmap(exact_asset(cfg)), rhos, n_th)   # spherical: no S2, so compare with care
    except Exception:
        ref = None
    return net, ref, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n-coll", type=int, default=0, help="0 = the run's own n_coll")
    args = ap.parse_args()

    from stationary.evaluate import load_run
    from stationary.geometry import residuals_batch
    from stationary.losses import outer_bc_terms, pde_raw
    from stationary.model import point_fields
    from stationary.problem import sample_sphere
    from stationary.train import exact_asset, make_batch

    keys = ("gauge", "lam_eq", "ricci")
    print("%-34s %-26s %-26s %s" % ("run", "BULK rms (collocation)", "OUTER rms (rho_out)",
                                    "l=0 profile:  rho : h_rr / h_tan / lam"))
    for run_dir in args.runs:
        try:
            cfg, model, state = load_run(run_dir)
        except Exception as exc:                      # a crashed run has no params
            print(f"{os.path.basename(run_dir.rstrip('/')):34s} skipped: {exc}")
            continue
        pf = point_fields(model, state["net"])
        if args.n_coll:
            cfg.n_coll = int(args.n_coll)
        xs = make_batch(jax.random.PRNGKey(int(cfg.seed)), cfg)["coll"]
        bulk = pde_raw(pf, xs, cfg)
        xs_out = sample_sphere(jax.random.PRNGKey(0), 2048, cfg.rho_out)
        outer = outer_bc_terms(pf, xs_out, cfg, exact_asset(cfg), lam_inf=1.0)
        raw_out = residuals_batch(pf, xs_out)
        rows, ref, cfg = profile(run_dir)
        b = "  ".join(f"{k}={float(jnp.sqrt(jnp.mean(bulk[k] ** 2))):.2e}" for k in keys)
        o = "  ".join(f"{k}={float(jnp.sqrt(jnp.mean(raw_out[k] ** 2))):.2e}" for k in keys)
        prof = "  ".join(f"{r:g}: {rows[r][0]:.5f}/{rows[r][1]:.5f}/{rows[r][2]:.5f}"
                         for r in (2.0, 50.0, 99.0))
        if ref is not None:
            err = "  ".join(f"{r:g}: {rows[r][0]-ref[r][0]:+.1e}/{rows[r][1]-ref[r][1]:+.1e}/"
                            f"{rows[r][2]-ref[r][2]:+.1e}" for r in (2.0, 50.0, 99.0))
        else:
            err = "no exact reference"
        print(f"{os.path.basename(run_dir.rstrip('/')):34s} {b:26s} {o:26s} {prof}")
        print(f"{'':34s} departure from the SPHERICAL reference (h_rr / h_tan / lam), "
              f"NOT an error -- the reference has no quadrupole:  {err}")
        print(f"{'':34s} outer BC rms: " + "  ".join(f"{k}={float(jnp.sqrt(jnp.mean(v ** 2))):.2e}"
                                                     for k, v in outer.items()))


if __name__ == "__main__":
    main()
