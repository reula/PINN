"""Evaluate a trained checkpoint: residual diagnostics, exact-solution comparison, figures.

    .venv/bin/python -m stationary.evaluate --outdir runs/m1_sym
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import diagnostics, exact
from .losses import inner_bc_terms, outer_bc_terms, pde_terms
from .model import point_fields
from .train import make_model
from .problem import Config, sample_shell, sample_sphere


def load_run(run_dir: str, params_file: str = "params.pkl"):
    with open(os.path.join(run_dir, "config.json")) as fh:
        cfg = Config(**json.load(fh))
    model = make_model(cfg)
    with open(os.path.join(run_dir, params_file), "rb") as fh:
        state = pickle.load(fh)
    # params.pkl holds the parameter state itself; ckpt.pkl holds a training payload with
    # the state under "state" (and is what a crashed run has to offer). Accept both.
    if isinstance(state, dict) and "state" in state and "net" not in state:
        state = state["state"]
    return cfg, model, state


def evaluate(run_dir: str, params_file: str = "params.pkl", make_plots: bool = True):
    cfg, model, state = load_run(run_dir, params_file)
    pf = point_fields(model, state["net"])
    report = {}

    exact_fields = None
    if cfg.outer_bc == "dirichlet_exact":
        exact_fields = exact.exact_fields(cfg.R0, exact.k_from_lambda0(cfg.R0, cfg.lam0))

    report.update(diagnostics.residual_report(pf, cfg))
    report.update(diagnostics.inner_boundary_report(pf, cfg))

    key = jax.random.PRNGKey(4242)
    report["bc_inner"] = {k: float(v) for k, v in
                          inner_bc_terms(pf, sample_sphere(key, 512, cfg.rho_in), cfg).items()}
    if cfg.outer_bc == "dirichlet_exact":
        report["bc_outer"] = {k: float(v) for k, v in
                              outer_bc_terms(pf, sample_sphere(key, 512, cfg.rho_out), cfg,
                                             exact_fields).items()}
    if exact_fields is not None:
        report.update(diagnostics.exact_comparison(pf, exact_fields, cfg))

    print(json.dumps(report, indent=2, default=float))

    if make_plots:
        _plots(run_dir, cfg, pf, exact_fields, report)
        try:                       # multipole figures: same set a training run writes
            from .multipoles import make_figures
            report["figures"] = make_figures(pf, cfg, run_dir)
        except Exception as exc:
            print(f"[evaluate] multipole figures skipped: {exc}")
    with open(os.path.join(run_dir, "report_eval.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    return report


def _plots(run_dir, cfg, pf, exact_fields, report):
    """Six self-describing panels; every axis is labelled and every curve is in a legend."""
    import matplotlib.pyplot as plt

    from .geometry import residuals_batch

    name = os.path.basename(os.path.normpath(run_dir))
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))

    # ---------------------------------------------------------------- 1. loss history
    hist_path = os.path.join(run_dir, "history.json")
    labels = {"loss": "total", "pde_compat": "compatibility $\\partial h=\\Gamma h$",
              "pde_ricci": "Ricci", "pde_gauge": "harmonic gauge",
              "pde_lam_eq": "$\\lambda$ equation"}
    if os.path.exists(hist_path):
        with open(hist_path) as fh:
            hist = json.load(fh)
        # Not every row carries every group: the quasi-Newton phase logs its first row with
        # `loss` only, so requiring the key in hist[0] is not enough (it used to raise
        # KeyError: 'pde_compat' on every run that went through SSBroyden).  Plot the rows
        # that have the key, and keep the run's own numbering on the x axis.
        for key in ("loss", "pde_compat", "pde_ricci", "pde_gauge", "pde_lam_eq"):
            xs = [h["step"] for h in hist if key in h]
            ys = [h[key] for h in hist if key in h]
            if len(xs) > 1:
                ax[0, 0].semilogy(xs, ys, label=labels[key], lw=2 if key == "loss" else 1)
        ax[0, 0].legend(fontsize=8)
    else:
        ax[0, 0].text(0.5, 0.5, "no history.json", ha="center", va="center",
                      transform=ax[0, 0].transAxes)
    ax[0, 0].set_title("loss history (weighted)")
    ax[0, 0].set_xlabel("step")
    ax[0, 0].set_ylabel("loss group (mean square)")

    # ------------------------------------------------------------- radial profiles
    rhos = jnp.geomspace(cfg.rho_in, cfg.rho_out, 80)
    xs = jnp.stack([rhos, jnp.zeros_like(rhos), jnp.zeros_like(rhos)], axis=-1)
    nn = xs / jnp.linalg.norm(xs, axis=-1, keepdims=True)
    f = jax.vmap(pf)(xs)
    h_rr = jnp.einsum("nij,ni,nj->n", f.h, nn, nn)
    tang = (jnp.einsum("nii->n", f.h) - h_rr) / 2.0 / rhos**2      # areal radius^2 / rho^2

    has_ref = exact_fields is not None
    ref = jax.vmap(exact_fields)(xs) if has_ref else None
    r_hrr = jnp.einsum("nij,ni,nj->n", ref.h, nn, nn) if has_ref else None
    r_tang = ((jnp.einsum("nii->n", ref.h) - r_hrr) / 2.0 / rhos**2) if has_ref else None

    lam_inf = cfg.lam_inf if cfg.lam_inf is not None else cfg.lam_inf_init
    ref_name = "exact/reference solution" if has_ref else "reference (none for this run)"

    ax[0, 1].plot(rhos, f.lam, "C0-", lw=2, label="PINN")
    if has_ref:
        ax[0, 1].plot(rhos, ref.lam, "k--", label=ref_name)
    ax[0, 1].axhline(lam_inf, color="grey", ls=":", label=fr"$\lambda_\infty={lam_inf:g}$")
    ax[0, 1].axhline(cfg.lam0, color="grey", ls="-.", alpha=0.6, label=fr"$\lambda_0={cfg.lam0:g}$")
    ax[0, 1].set_title(r"$\lambda$ along $\theta=0$")
    ax[0, 1].set_xlabel(r"$\rho$")
    ax[0, 1].set_ylabel(r"$\lambda$")
    ax[0, 1].legend(fontsize=8)

    ax[0, 2].plot(rhos, h_rr, "C0-", lw=2, label="PINN")
    if has_ref:
        ax[0, 2].plot(rhos, r_hrr, "k--", label=ref_name)
    ax[0, 2].set_title(r"$h_{\rho\rho}=\lambda$-independent normal component")
    ax[0, 2].set_xlabel(r"$\rho$")
    ax[0, 2].set_ylabel(r"$h_{rr}$")
    ax[0, 2].legend(fontsize=8)

    ax[1, 0].plot(rhos, tang, "C0-", lw=2, label="PINN")
    if has_ref:
        ax[1, 0].plot(rhos, r_tang, "k--", label=ref_name)
    ax[1, 0].axvline(cfg.rho_in, color="grey", alpha=0.4)
    ax[1, 0].set_title(r"tangential metric: (areal radius)$^2/\rho^2$")
    ax[1, 0].set_xlabel(r"$\rho$")
    ax[1, 0].set_ylabel(r"$\alpha$  (1 = flat, $\rho_{in}^2$ at the inner sphere)")
    ax[1, 0].legend(fontsize=8)

    # ------------------------------------------------------------ 5. error vs rho
    if has_ref:
        ax[1, 1].loglog(rhos, jnp.abs(f.lam - ref.lam) + 1e-18, label=r"$|\Delta\lambda|$")
        ax[1, 1].loglog(rhos, jnp.abs(h_rr - r_hrr) + 1e-18, label=r"$|\Delta h_{rr}|$")
        ax[1, 1].loglog(rhos, jnp.abs(tang - r_tang) + 1e-18, label=r"$|\Delta\alpha|$")
        ax[1, 1].set_title("pointwise difference from the reference")
    else:
        ax[1, 1].loglog(rhos, jnp.abs(f.lam - lam_inf) + 1e-18,
                        label=fr"$|\lambda-\lambda_\infty|$")
        ax[1, 1].set_title(r"distance from $\lambda_\infty$ (no reference for this run)")
    ax[1, 1].set_xlabel(r"$\rho$")
    ax[1, 1].set_ylabel("absolute difference")
    ax[1, 1].legend(fontsize=8)

    # -------------------------------------------------------- 6. residuals vs rho
    res = jax.vmap(lambda x: residuals_batch(pf, x[None, :]))(xs)
    res = jax.tree.map(lambda a: a[:, 0], res)
    for key, lab in (("compat", "compatibility"), ("ricci", "Ricci"),
                     ("gauge", "harmonic gauge"), ("lam_eq", r"$\lambda$ equation")):
        ax[1, 2].loglog(rhos, jnp.max(jnp.abs(res[key]), axis=tuple(range(1, res[key].ndim)))
                        + 1e-18, label=lab)
    ax[1, 2].set_title("PDE residuals (max over components), raw units")
    ax[1, 2].set_xlabel(r"$\rho$")
    ax[1, 2].set_ylabel("residual")
    ax[1, 2].legend(fontsize=8)

    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle(f"{name}   (arch={cfg.arch}, "
                 fr"$\rho\in[{cfg.rho_in:g},{cfg.rho_out:g}]$, "
                 fr"$\lambda_0={cfg.lam0:g}$, $\lambda_\infty={lam_inf:g}$, "
                 fr"$\bf S_1={cfg.lam_bc_S1:g}$, $S_2={cfg.lam_bc_S2:g}$, "
                 f"Robin order {cfg.robin_orders or cfg.robin_order})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(run_dir, "diagnostics.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[evaluate] wrote {out}")


def main():
    p = argparse.ArgumentParser(description="Diagnostics and figures for a finished run.")
    p.add_argument("run_dir", nargs="?", default=None,
                   help="run directory (same as --outdir)")
    p.add_argument("--outdir", default=None,
                   help="run directory, same spelling as train.py uses")
    p.add_argument("--params-file", default="params.pkl",
                   help="checkpoint inside the run dir (default params.pkl)")
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()
    run_dir = a.outdir or a.run_dir
    if run_dir is None:
        p.error("give the run directory, as `--outdir RUN` or as the first argument")
    evaluate(run_dir, a.params_file, make_plots=not a.no_plots)


if __name__ == "__main__":
    main()
