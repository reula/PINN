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
from .problem import reference_is_departure_only


def directions(theta, n_phi=1):
    """Unit vectors at polar angle theta (and a few azimuths if wanted)."""
    th = jnp.asarray(theta)
    return jnp.stack([jnp.sin(th), jnp.zeros_like(th), jnp.cos(th)], axis=-1)


def profile(point_fields_fn, cfg, n_rho=240, thetas=(0.0,)):
    """lambda and h_rr along each polar angle.

    Returns (rhos, lam, h_rr).  The h_rr key was added because lambda alone cannot show the
    metric: `far-field VALUES` reports h_rr at rho_out as an rms over the sphere, and the
    averaged pins hold only its MEAN, so whether the pointwise value approaches 1 -- and at
    which angles it does not -- was not visible anywhere in the outputs.

    h_rr = n^T h n with n the radial unit vector at that point, the same projection the metric
    pins use.  h_rr_of in losses.py is a closure and cannot be imported, so it is written out.
    """
    rhos = jnp.geomspace(cfg.rho_in, cfg.rho_out, n_rho)
    out, hrr = {}, {}
    for th in thetas:
        n = directions(th)
        xs = rhos[:, None] * n[None, :]
        f = jax.vmap(point_fields_fn)(xs)
        out[float(th)] = f.lam
        hrr[float(th)] = jax.vmap(lambda m, y: m.T @ y @ y)(
            f.h, xs / jnp.linalg.norm(xs, axis=-1, keepdims=True))
    return rhos, out, hrr


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
    rhos, curves, hrr_curves = profile(pf, cfg, a.n_rho, thetas)

    # reference, when the run is one that has one
    ref = None
    if getattr(cfg, "ref_asymptotic", None) is not None:
        ref, _ = exact.reference_fields_asymptotic(
            cfg.R0, cfg.ref_asymptotic, cfg.rho_in, r_areal=cfg.inner_radius)
    elif cfg.outer_bc == "dirichlet_exact":
        ref = exact.exact_fields(cfg.R0, exact.k_from_lambda0(cfg.R0, cfg.lam0))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    # A spherically symmetric reference cannot carry angular inner data (S1, S2), so for
    # those runs it is not a solution of this problem: label it, and the second panel, as
    # DEPARTURE.  The curves stay -- where the solution leaves the spherical one is the point
    # of the angular data -- but nothing may read as an error.
    departure_only = reference_is_departure_only(cfg)
    ref_label = ("spherical reference (departure, not a target)" if departure_only
                 else "exact reference")
    for th, lam in curves.items():
        ax[0].plot(rhos, lam, label=fr"$\theta={th:.2f}$")
    if ref is not None:
        lam_ref = profile(ref, cfg, a.n_rho, (thetas[0],))[1][thetas[0]]
        ax[0].plot(rhos, lam_ref, "k--", lw=1.5, label=ref_label)
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
    ax[1].set_ylabel(r"$|\lambda-\lambda_{\rm sph}|$ (departure)" if departure_only
                     else r"$|\lambda-\lambda_{\rm ref}|$")
    if ref is None:
        title = fr"distance from $\lambda_\infty={lam_inf:g}$"
    elif departure_only:
        title = ("DEPARTURE from the spherical reference, not error\n"
                 "(no exact solution exists for angular inner data)")
    else:
        title = "difference from the reference"
    ax[1].set_title(title)
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(run_dir, "lambda_vs_rho.png")
    fig.savefig(out, dpi=120)
    print(f"[profile] wrote {out}")

    # the metric, at the same angles: h_rr must approach 1 at rho_out, and the averaged pin
    # holds only its mean, so any angular departure shows here
    # ONE figure, two panels: lambda and h_rr are the same structure over the same angles, and
    # two files for them was duplication -- the same objection applies to lambda_vs_rho.png
    # below, which shows the left panel alone and predates this.
    fig3, ax3 = plt.subplots(1, 2, figsize=(12, 4.6))
    for th in thetas:
        ax3[0].semilogx(rhos, curves[th], label=fr"$\lambda(\theta={th:.2f})$")
        ax3[1].semilogx(rhos, hrr_curves[th], label=fr"$h_{{\rho\rho}}(\theta={th:.2f})$")
    ax3[1].axhline(1.0, color="grey", ls=":", lw=1)
    ax3[0].set_ylabel(r"$\lambda$"); ax3[1].set_ylabel(r"$h_{\rho\rho}$")
    ax3[0].set_title("lambda against rho"); ax3[1].set_title("h_rr against rho")
    # `axp`, NOT `a`: `a` is the argparse Namespace, and rebinding it here made every later
    # `a.no_table` an AttributeError on an Axes object -- masked in every check I ran because
    # `... | tail` reports tail's exit status, not python's.
    for axp in ax3:
        axp.set_xlabel(r"$\rho$"); axp.legend(fontsize=8); axp.grid(alpha=0.3)
    out3 = os.path.join(run_dir, "profiles_vs_rho.png")
    fig3.tight_layout(); fig3.savefig(out3, dpi=120)
    print(f"[profile] wrote {out3}")
    print(f"\n{'rho':>10} " + " ".join(f"{'h_rr(th=%.2f)' % th:>15}" for th in thetas))
    step3 = max(1, len(rhos) // 12)
    for i in range(0, len(rhos), step3):
        print(f"{float(rhos[i]):10.4f} " +
              " ".join(f"{float(hrr_curves[th][i]):15.7f}" for th in thetas))

    if not a.no_table:
        print(f"\n{'rho':>10} " + " ".join(f"{'lam(th=%.2f)' % th:>14}" for th in curves))
        step = max(1, len(rhos) // 18)
        for i in range(0, len(rhos), step):
            print(f"{float(rhos[i]):10.4f} " +
                  " ".join(f"{float(curves[th][i]):14.7f}" for th in curves))


if __name__ == "__main__":
    main()
