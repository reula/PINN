"""Evaluate a trained checkpoint: residual diagnostics, exact-solution comparison, figures.

    .venv/bin/python -m stationary.evaluate runs/m1_a
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
    with open(os.path.join(run_dir, "report_eval.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    return report


def _plots(run_dir, cfg, pf, exact_fields, report):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # 1. loss history
    hist_path = os.path.join(run_dir, "history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as fh:
            hist = json.load(fh)
        steps = [h["step"] for h in hist]
        axes[0, 0].semilogy(steps, [h["loss"] for h in hist], label="total")
        for k in ("pde_compat", "pde_ricci", "pde_gauge", "pde_lam_eq"):
            if k in hist[0]:
                axes[0, 0].semilogy(steps, [h[k] for h in hist], "--", label=k)
        axes[0, 0].set_title("loss history")
        axes[0, 0].set_xlabel("step")
        axes[0, 0].legend(fontsize=7)

    # radial profiles
    nrho = 60
    rhos = jnp.linspace(cfg.rho_in, cfg.rho_out, nrho)
    xs = jnp.stack([rhos, jnp.zeros_like(rhos), jnp.zeros_like(rhos)], axis=-1)
    nn = xs / jnp.linalg.norm(xs, axis=-1, keepdims=True)

    def fields_at(x):
        f = pf(x)
        return f.h, f.G, f.lam

    h, G, lam = jax.vmap(fields_at)(xs)
    h_rr = jnp.einsum("ni,nij,nj->n", nn, h, nn)
    # tangential coefficient: h_theta_theta / rho^2  (round-sphere piece)
    tang = (jnp.einsum("nii->n", h) - h_rr) / 2.0 / rhos**2

    axes[0, 1].plot(rhos, lam, label="PINN")
    axes[0, 2].plot(rhos, h_rr, label="PINN $h_{rr}$")
    axes[1, 0].plot(rhos, tang, label="PINN tang. coeff")

    if exact_fields is not None:
        eh, eG, elam = jax.vmap(lambda x: exact_fields(x))(xs)
        e_hrr = jnp.einsum("ni,nij,nj->n", nn, eh, nn)
        e_tang = (jnp.einsum("nii->n", eh) - e_hrr) / 2.0 / rhos**2
        axes[0, 1].plot(rhos, elam, "--", label="exact")
        axes[0, 2].plot(rhos, e_hrr, "--", label="exact")
        axes[1, 0].plot(rhos, e_tang, "--", label="exact")
        axes[1, 1].semilogy(rhos, jnp.abs(lam - elam) + 1e-16, label="$|\\Delta\\lambda|$")
        axes[1, 1].semilogy(rhos, jnp.abs(h_rr - e_hrr) + 1e-16, label="$|\\Delta h_{rr}|$")
        axes[1, 1].set_title("errors vs exact")
        axes[1, 1].legend(fontsize=8)

    for ax, title in zip(axes.ravel()[:5],
                         ["loss history", "$\\lambda(\\rho)$", "$h_{rr}(\\rho)$",
                          "tangential coefficient", "errors vs exact"]):
        ax.set_title(title)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[1:5]:
        ax.set_xlabel("$\\rho$")
        if not ax.get_legend_handles_labels()[0]:
            pass
        else:
            ax.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(run_dir, "diagnostics.png")
    fig.savefig(out, dpi=120)
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
