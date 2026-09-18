"""lambda (and the metric components) as a function of rho.

    python -m stationary.profile --outdir runs/<name>              # figure + table
    python -m stationary.profile --outdir runs/<name> --no-table   # figure only
    python -m stationary.profile --outdir runs/<name> --thetas 0,0.7,1.57

Writes <outdir>/lambda_vs_rho.png: lambda against rho along the chosen polar angles,
with the asymptotic value lambda_inf marked and -- when the run has a reference
(--ref-solution / --ref-asymptotic) -- the exact solution overlaid for comparison.
"""
from __future__ import annotations

import argparse
import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import exact
from .evaluate import load_run
from .model import point_fields


def directions(theta, n_phi=1):
    """Unit vectors at polar angle theta (and a few azimuths if wanted)."""
    th = jnp.asarray(theta)
    return jnp.stack([jnp.sin(th), jnp.zeros_like(th), jnp.cos(th)], axis=-1)


def profile(point_fields_fn, cfg, n_rho=240, thetas=(0.0,)):
    rhos = jnp.geomspace(cfg.rho_in, cfg.rho_out, n_rho)
    out = {}
    for th in thetas:
        n = directions(th)
        xs = rhos[:, None] * n[None, :]
        out[float(th)] = jax.vmap(lambda y: point_fields_fn(y).lam)(xs)
    return rhos, out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--params-file", default="params.pkl")
    p.add_argument("--thetas", default="0,0.7,1.5707963",
                   help="comma separated polar angles in radians (default 0,0.7,pi/2)")
    p.add_argument("--n-rho", type=int, default=240)
    p.add_argument("--no-table", action="store_true")
    a = p.parse_args()
    run_dir = a.outdir or a.run_dir
    if run_dir is None:
        p.error("give the run directory, as `--outdir RUN` or as the first argument")

    cfg, model, state = load_run(run_dir, a.params_file)
    pf = point_fields(model, state["net"])
    thetas = tuple(float(t) for t in a.thetas.split(","))
    rhos, curves = profile(pf, cfg, a.n_rho, thetas)

    # reference, when the run is one that has one
    ref = None
    if getattr(cfg, "ref_asymptotic", None) is not None:
        ref, _ = exact.reference_fields_asymptotic(
            cfg.R0, cfg.ref_asymptotic, cfg.rho_in, r_areal=cfg.inner_radius)
    elif cfg.outer_bc == "dirichlet_exact":
        ref = exact.exact_fields(cfg.R0, exact.k_from_lambda0(cfg.R0, cfg.lam0))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for th, lam in curves.items():
        ax[0].plot(rhos, lam, label=fr"$\theta={th:.2f}$")
    if ref is not None:
        lam_ref = profile(ref, cfg, a.n_rho, (thetas[0],))[1][thetas[0]]
        ax[0].plot(rhos, lam_ref, "k--", lw=1.5, label="exact reference")
    if ref is not None:
        # The reference is rebuilt from cfg.R0; if that R0 is not the one the run was
        # actually built with, its lambda on the inner sphere will not match lam0 and
        # the overlay (and the difference panel) would be meaningless.  Say so.
        lam_ref_inner = float(ref(cfg.rho_in * directions(thetas[0])).lam)
        if abs(lam_ref_inner - cfg.lam0) > 1e-3 * max(1.0, abs(cfg.lam0)):
            print(f"[profile] WARNING: the reference has lambda(rho_in) = {lam_ref_inner:.6f} "
                  f"but the run imposed lam0 = {cfg.lam0:.6f}.\n"
                  f"          cfg.R0 = {cfg.R0} is probably not the R0 this run was built with,"
                  f" so the overlay and the difference panel are NOT meaningful.")
            ref = None

    lam_inf = cfg.lam_inf if cfg.lam_inf is not None else cfg.lam_inf_init
    ax[0].axhline(lam_inf, color="grey", ls=":", label=fr"$\lambda_\infty={lam_inf:g}$")
    ax[0].axhline(cfg.lam0, color="grey", ls="-.", alpha=0.6, label=fr"$\lambda_0={cfg.lam0:g}$")
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$\rho$")
    ax[0].set_ylabel(r"$\lambda(\rho)$")
    ax[0].set_title(f"$\\lambda$ vs $\\rho$   ({os.path.basename(os.path.normpath(run_dir))})   "
                    f"$\\lambda_0={cfg.lam0:g}$, $S_1={cfg.lam_bc_S1:g}$, "
                    f"$S_2={cfg.lam_bc_S2:g}$")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    for th, lam in curves.items():
        if ref is not None:
            lam_ref = profile(ref, cfg, a.n_rho, (th,))[1][th]
            ax[1].semilogy(rhos, jnp.abs(lam - lam_ref) + 1e-18, label=fr"$\theta={th:.2f}$")
    if ref is None:
        ax[1].semilogy(rhos, jnp.abs(profile(pf, cfg, a.n_rho, (thetas[0],))[1][thetas[0]] - lam_inf),
                       label=fr"$|\lambda-\lambda_\infty|$")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$\rho$")
    ax[1].set_ylabel(r"$|\lambda-\lambda_{\rm ref}|$")
    ax[1].set_title("difference from the reference" if ref is not None
                    else fr"distance from $\lambda_\infty={lam_inf:g}$")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(run_dir, "lambda_vs_rho.png")
    fig.savefig(out, dpi=120)
    print(f"[profile] wrote {out}")

    if not a.no_table:
        print(f"\n{'rho':>10} " + " ".join(f"{'lam(th=%.2f)' % th:>14}" for th in curves))
        step = max(1, len(rhos) // 18)
        for i in range(0, len(rhos), step):
            print(f"{float(rhos[i]):10.4f} " +
                  " ".join(f"{float(curves[th][i]):14.7f}" for th in curves))


if __name__ == "__main__":
    main()
