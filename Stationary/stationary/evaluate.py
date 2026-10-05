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
import matplotlib.ticker as mticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import diagnostics, exact
from .losses import inner_bc_terms, outer_bc_terms, pde_terms
from .model import point_fields
from .train import make_model
from .problem import Config, reference_is_departure_only, sample_shell, sample_sphere


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

    # One asset, built in ONE place.  This used to build a reference only for a
    # dirichlet_exact run, which left every Robin run without one: the boundary residuals
    # then either raised (inner_bc="reference" needs one) or went unreported, and a Weyl run
    # would have been compared against the spherical solution.  train.exact_asset covers
    # dirichlet_exact, ref_solution, robin_source and weyl alike.
    from .train import exact_asset
    exact_fields = exact_asset(cfg)

    report.update(diagnostics.residual_report(pf, cfg))
    report.update(diagnostics.inner_boundary_report(pf, cfg))

    key = jax.random.PRNGKey(4242)
    report["bc_inner"] = {k: float(v) for k, v in
                          inner_bc_terms(pf, sample_sphere(key, 512, cfg.rho_in), cfg,
                                         exact_fields).items()}
    if cfg.outer_bc in ("dirichlet_exact", "robin"):
        report["bc_outer"] = {k: float(v) for k, v in
                              outer_bc_terms(pf, sample_sphere(key, 512, cfg.rho_out), cfg,
                                             exact_fields, cfg.lam_inf).items()}
    if exact_fields is not None:
        report.update(diagnostics.exact_comparison(pf, exact_fields, cfg))

    print(json.dumps(report, indent=2, default=float))

    if make_plots:
        _plots(run_dir, cfg, pf, exact_fields, report, model=model)
        try:                       # multipole figures: same set a training run writes
            from .multipoles import make_figures
            report["figures"] = make_figures(pf, cfg, run_dir, exact_fields=exact_fields)
        except Exception as exc:
            print(f"[evaluate] multipole figures skipped: {exc}")
    with open(os.path.join(run_dir, "report_eval.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    return report


def pde_plot_keys(model) -> tuple:
    """The per-group curves the residual panels should carry, in plot order.

    The formulation's own groups (`losses.equation_keys`) and not a hard-coded four: for a
    metric-only model compatibility is not in the loss, and its 3.5e-36 value would otherwise
    draw a straight line 36 decades below everything else and flatten the log axis until the
    curves that matter are indistinguishable.  That is a plotting failure, not a small
    residual -- the term is not being minimised at all.
    """
    from .losses import equation_keys
    return tuple(f"pde_{k}" for k in equation_keys(model))


def _plots(run_dir, cfg, pf, exact_fields, report, model=None):
    """Six self-describing panels; every axis is labelled and every curve is in a legend."""
    import matplotlib.pyplot as plt

    from .geometry import residuals_batch

    name = os.path.basename(os.path.normpath(run_dir))
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    pde_keys = pde_plot_keys(model)
    departure_only = reference_is_departure_only(cfg)

    # ---------------------------------------------------------------- 1. loss history
    hist_path = os.path.join(run_dir, "history.json")
    # NO "pde_compat" entry: the user asked for it not to be plotted, and for a model that
    # derives Gamma from h it is not in the loss either.  Keeping a label for a group that is
    # deliberately absent only invites it back into the panel.
    labels = {"loss": "total", "pde_ricci": "Ricci", "pde_gauge": "harmonic gauge",
              "pde_lam_eq": "$\\lambda$ equation"}
    hist, hist_src = None, None
    if os.path.exists(hist_path):
        with open(hist_path) as fh:
            hist = json.load(fh)
        hist_src = "history.json"
    else:
        # The CHECKPOINT carries the SAME trajectory: `save_checkpoint(..., history, ...)`
        # stores exactly what `write_progress` appends.  A run whose history.json is missing --
        # killed before its first quasi-Newton block, or a directory rebuilt around a surviving
        # checkpoint -- therefore still has its loss history on disk, and this panel used to
        # print "no history.json" and stop, which reads as "no loss history was recorded".
        _ck = os.path.join(run_dir, "ckpt.pkl")
        if os.path.exists(_ck):
            try:
                with open(_ck, "rb") as fh:
                    _h = pickle.load(fh).get("history")
                if _h:
                    hist, hist_src = _h, "ckpt.pkl"
            except Exception:
                pass
        # Not every row carries every group: the quasi-Newton phase logs its first row with
        # `loss` only, so requiring the key in hist[0] is not enough (it used to raise
        # KeyError: 'pde_compat' on every run that went through SSBroyden).  Plot the rows
        # that have the key, and keep the run's own numbering on the x axis.
        for key in ("loss",) + pde_keys:
            xs = [h["step"] for h in hist if key in h]
            ys = [h[key] for h in hist if key in h]
            # A group that is identically machine-zero must NOT be drawn on the same log axis.
            # `pde_compat` is exactly that for every architecture whose Gamma is derived from h,
            # and history.json's first row carries 7.77e-37: the axis then spans 37 decades and
            # every other curve lies flat against the top edge.  The panel LOOKED empty while
            # plotting 17 rows with every key present.  A group that is zero is reported as
            # zero, not scaled into the picture.
            if ys and max(abs(v) for v in ys) < 1e-25:
                continue
            if len(xs) > 1:
                ax[0, 0].semilogy(xs, ys, label=labels[key], lw=2 if key == "loss" else 1)
            elif len(xs) == 1:
                # ONE row is still information; dropping it silently is what made an empty
                # frame possible in the first place.
                ax[0, 0].semilogy(xs, ys, "o", label=labels[key], ms=4)
        ax[0, 0].legend(fontsize=8)
    if hist:
        # FIXED Y-LIMITS from the loss's own range.  pq_c200_vac converged to 2.1e-14 with groups
        # at 1e-16..1e-21, so auto-scaling spanned twenty-one decades and every curve collapsed
        # into a near-vertical line at the left edge: a frame, a title, and an empty interior.
        # The groups are still drawn; they are simply not allowed to set the range.
        _ls = [h["loss"] for h in hist if "loss" in h and h["loss"] > 0]
        if _ls:
            ax[0, 0].set_ylim(min(_ls) * 0.5, max(_ls) * 2.0)
        ax[0, 0].set_title(f"loss history (weighted)   "
                           f"[{len(hist)} rows from {hist_src}]")
    else:
        ax[0, 0].text(0.5, 0.5, "no history: neither history.json\nnor a checkpoint with one",
                      ha="center", va="center", transform=ax[0, 0].transAxes)
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
    # `departure_only`: the reference is spherically symmetric and this run's inner data are
    # not, so it is not a solution of this problem and the differences are departure, not
    # error.  The curves stay -- seeing where the solution leaves the spherical one is the
    # point of the angular data -- but nothing is titled or labelled as an error.
    ref_name = ("exact/reference solution" if not departure_only else
                "spherical reference (departure, not a target)") if has_ref else \
               "reference (none for this run)"

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
    if has_ref and not departure_only:
        ax[1, 1].loglog(rhos, jnp.abs(f.lam - ref.lam) + 1e-18, label=r"$|\Delta\lambda|$")
        ax[1, 1].loglog(rhos, jnp.abs(h_rr - r_hrr) + 1e-18, label=r"$|\Delta h_{rr}|$")
        ax[1, 1].loglog(rhos, jnp.abs(tang - r_tang) + 1e-18, label=r"$|\Delta\alpha|$")
        ax[1, 1].set_title("pointwise error against the exact reference")
    elif has_ref:
        # Not an error panel.  This run's inner data are angular and the reference is
        # symmetric, so it is not a solution of this problem: the curves say how far the
        # solution has left the spherical one, and a reader must not take them for accuracy.
        ax[1, 1].loglog(rhos, jnp.abs(f.lam - ref.lam) + 1e-18,
                        label=r"departure $|\lambda-\lambda_{sph}|$")
        ax[1, 1].loglog(rhos, jnp.abs(h_rr - r_hrr) + 1e-18,
                        label=r"departure $|\Delta h_{rr}|$")
        ax[1, 1].loglog(rhos, jnp.abs(tang - r_tang) + 1e-18,
                        label=r"departure $|\Delta\alpha|$")
        ax[1, 1].loglog(rhos, jnp.abs(f.lam - lam_inf) + 1e-18,
                        label=r"$|\lambda-\lambda_\infty|$")
        ax[1, 1].set_title("DEPARTURE from the spherical reference, not error\n"
                           "(no exact solution exists for angular inner data)")
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
    for key, lab in ((k[4:], labels.get(k, k[4:])) for k in pde_keys):
        ax[1, 2].loglog(rhos, jnp.max(jnp.abs(res[key]), axis=tuple(range(1, res[key].ndim)))
                        + 1e-18, label=lab)
    ax[1, 2].set_title("PDE residuals (max over components), raw units")
    ax[1, 2].set_xlabel(r"$\rho$")
    ax[1, 2].set_ylabel("residual")
    ax[1, 2].legend(fontsize=8)

    # ------------------------------------------------- radial axes, where the action is
    # The shell spans rho_out/rho_in, a factor of hundreds, so on a linear axis the first
    # decade -- where the field varies fastest and where the inner data are imposed -- is a
    # sliver a few pixels wide.  Log axes are the whole fix.
    #
    # There WAS an inset on the lambda profile zoomed to that decade, and it is gone.  It sat
    # in the upper right of the panel, which is exactly where a lambda profile that has risen
    # off its inner value passes as rho grows, so it covered the curves it was there to show:
    # an inset that hides the plot it is inset into.  The log axis already resolves the region
    # and a reader can zoom.
    for a in (ax[0, 1], ax[0, 2], ax[1, 0]):
        a.set_xscale("log")

    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle(f"{name}   (arch={cfg.arch}, "
                 fr"$\rho\in[{cfg.rho_in:g},{cfg.rho_out:g}]$, "
                 fr"$\lambda_0={cfg.lam0:g}$, $\lambda_\infty={lam_inf:g}$, "
                 fr"$\bf S_1={cfg.lam_bc_S1:g}$, $S_2={cfg.lam_bc_S2:g}$, "
                 f"Robin order {cfg.robin_orders or cfg.robin_order})"
                 # said on the figure itself, because the panel is the thing people paste
                 + ("\nangular inner data: the spherical reference is NOT a solution of this "
                    "problem -- differences from it are departure, not error"
                    if departure_only else ""), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(run_dir, "diagnostics.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[evaluate] wrote {out}")


def main():
    # The STRUCTURAL check lives here, not in the algorithm: a model that derives Gamma
    # from h holds compatibility identically and the training path does not form it, but
    # these diagnostics report it by design.  Ask for it explicitly, or every reader of
    # the residual dict fails with KeyError on those models.
    from .geometry import set_want_compat
    set_want_compat(True)
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
