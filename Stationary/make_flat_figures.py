"""Figures for the flat-Laplacian dipole study.

Reads `plots/flat_results.json` (from `collect_flat_results.py`) and the production DSGNAR
histories, and writes the report's figures into `plots/`:

    flat_orders.png          max|err| per Robin order, pure quadrupole vs the shipped datum
    flat_sectors.png         the growing rho^l coefficient per angular sector, l = 1 vs l = 2
    flat_sampling.png        the hybrid radial law, and the exact field along the meridian
    flat_error_map.png       the solved error field (order 1, S1 = 0): even in z, no dipole
    production_convergence.png   DSGNAR loss vs iteration on the c100 runs, with the baselines
    flat_optimizers.png      SSBroyden vs DSGNAR on the flat problem

    ./.venv/bin/python make_flat_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")
SHIPPED_ORDER1 = 3.45e-03        # Laplace_Robin.md Sec. 6.3, reproduced by lap_quad.py
SHIPPED_L1_GROW = 3.07e-03       # the same run's l = 1 growing coefficient (A)
BASELINES = {"vac3 (SSBroyden, 6019 it)": 8.65e-13,
             "vacE (SSBroyden, 13519 it)": 8.85e-14,
             "vacF (SSBroyden, 30000 it)": 1.33e-14}


def load(path=os.path.join(PLOTS, "flat_results.json")):
    return json.load(open(path))


def fig_orders(data):
    cases = [c for c in data["flat"] if c["optimizer"] == "ssbroyden" and c["s1"] == 0.0]
    orders = sorted({c["order"] for c in cases})
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    width = 0.35
    for i, c in enumerate(sorted(cases, key=lambda c: c["order"])):
        errs = [b["max_err"] for b in c["batches"]]
        ax.bar(i, np.mean(errs), width, color="tab:blue", alpha=0.75,
               label="pure quadrupole $S_1 = 0$" if i == 0 else None)
        ax.scatter([i] * len(errs), errs, color="k", zorder=3, s=18)
    ax.axhline(SHIPPED_ORDER1, color="tab:red", ls="--", lw=1.4,
               label=r"$S_1 = 0.1$ (shipped datum), order 1")
    ax.set_yscale("log")
    ax.set_xticks(range(len(orders)))
    ax.set_xticklabels([f"order {o}" for o in orders])
    ax.set_ylabel(r"$\max|\lambda_{\rm solved} - \lambda_{\rm exact}|$  (fresh points)")
    ax.set_title("Flat Laplacian, ratio-4 shell: the error falls with the Robin order\n"
                 "and removing the datum's DIPOLE removes the dipole-sized error")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "flat_orders.png"), dpi=150)
    print("wrote flat_orders.png")


def sectors_of(batch):
    """`{l: {...}}` with int keys: JSON has stringified them."""
    return {int(k): v for k, v in batch["sectors"].items()}


def fig_sectors(data):
    cases = [c for c in data["flat"] if c["optimizer"] == "ssbroyden" and c["s1"] == 0.0]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    labels, grow1, grow2 = [], [], []
    for c in sorted(cases, key=lambda c: c["order"]):
        labels.append(f"order {c['order']}")
        grow1.append(np.mean([abs(sectors_of(b).get(1, {}).get("grow", np.nan)) for b in c["batches"]]))
        grow2.append(np.mean([abs(sectors_of(b).get(2, {}).get("grow", np.nan)) for b in c["batches"]]))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, grow1, 0.4, label=r"$l = 1$  (dipole)", color="tab:orange")
    ax.bar(x + 0.2, grow2, 0.4, label=r"$l = 2$  (quadrupole)", color="tab:green")
    ax.axhline(SHIPPED_L1_GROW, color="tab:red", ls="--", lw=1.4,
               label=r"$l = 1$ with the shipped datum ($S_1 = 0.1$)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$|$coefficient of the growing $\rho^l$ mode$|$")
    ax.set_title("Per-sector growing modes, pure quadrupole datum ($S_1 = 0$)\n"
                 "the dipole sits at the solver's floor in every batch; the quadrupole is real")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "flat_sectors.png"), dpi=150)
    print("wrote flat_sectors.png")


def fig_sampling():
    from laplace_probe import load_problem
    P = load_problem()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    bins = np.linspace(P.cfg_ratio4().rho_in, 1.0, 40)
    for radial, frac, color in (("log", 0.0, "tab:blue"), ("volume", 0.0, "tab:green"),
                                ("hybrid", 0.4, "tab:red")):
        cfg = P.cfg_ratio4()
        cfg.radial, cfg.radial_log_frac = radial, frac
        xs = P.sample_shell(jax.random.PRNGKey(0), 20000, cfg)
        r = np.asarray(jnp.linalg.norm(xs, axis=1))
        axes[0].hist(r, bins=bins, histtype="step", lw=1.6, color=color,
                     label=f"{radial}" + (f" (log frac {frac})" if radial == "hybrid" else ""),
                     density=True)
    axes[0].set_xlabel(r"$\rho$")
    axes[0].set_ylabel("density")
    axes[0].set_title("radial sampling laws of the shell")
    axes[0].legend(fontsize=9)
    th = np.linspace(0, np.pi, 200)
    r = np.linspace(P.cfg_ratio4().rho_in, 1.0, 200)
    R, T = np.meshgrid(r, th)
    X = np.stack([R * np.sin(T), np.zeros_like(R), R * np.cos(T)], -1)
    lam = np.asarray(jax.vmap(jax.vmap(P.lam_exact_ratio4))(X))
    m = axes[1].pcolormesh(R * np.sin(T), R * np.cos(T), lam, shading="auto", cmap="viridis")
    fig.colorbar(m, ax=axes[1], label=r"$\lambda_{\rm exact}$")
    axes[1].set_aspect("equal")
    axes[1].set_xlabel(r"$R=\rho\sin\theta$")
    axes[1].set_ylabel(r"$z=\rho\cos\theta$")
    axes[1].set_title(r"exact solution, $S_1 = 0$, $S_2 = 0.05$ (even in $z$)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "flat_sampling.png"), dpi=150)
    print("wrote flat_sampling.png")


def fig_error_map():
    """One order-1 pure-quadrupole solve, drawn: the error must be EVEN in z (no dipole)."""
    from laplace_probe import load_problem, draw_batch
    P = load_problem()
    P.S1, P.S2 = 0.0, 0.05
    cfg = P.cfg_ratio4()
    cfg.radial, cfg.radial_log_frac = "hybrid", 0.4
    coll, inner, outer = draw_batch(P, cfg, 0, P.RATIO4_COLL, P.RATIO4_BND)
    model = P.LapNet()
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, 3)))

    def loss(p):
        lam = lambda y: model.apply(p, y[None, :])[0]
        lap = jax.vmap(lambda y: jnp.trace(jax.hessian(lam)(y)))(coll)
        r_in = jax.vmap(lambda y: lam(y) - P.lam_inner_bc(y, cfg))(inner)
        r_out = jax.vmap(lambda y: P.robin_operator(lam, y, P.BASE, 1, P.LAM0))(outer)
        return (jnp.mean((jnp.sum(coll * coll, axis=1) * lap) ** 2)
                + P.W_BC * jnp.mean(r_in ** 2) + P.W_BC * jnp.mean(r_out ** 2))

    flat0, unflatten = jax.flatten_util.ravel_pytree(params)
    res = P.crunch_minimize(lambda flat: loss(unflatten(flat)), flat0, args=(), method="BFGS",
                            options={"maxiter": P.QN_MAXITER, "gtol": P.RATIO4_GTOL,
                                     "initial_H": jnp.eye(flat0.size),
                                     "update_method": "ssbroyden2", "initial_scale": False})
    th = np.linspace(0.02, np.pi - 0.02, 220)
    r = np.linspace(cfg.rho_in, 1.0, 220)
    R, T = np.meshgrid(r, th)
    X = np.stack([R * np.sin(T), np.zeros_like(R), R * np.cos(T)], -1)
    got = np.asarray(jax.vmap(jax.vmap(lambda y: model.apply(unflatten(res.x), y[None, :])[0]))(X))
    ex = np.asarray(jax.vmap(jax.vmap(P.lam_exact_ratio4))(X))
    err = got - ex
    lim = np.percentile(np.abs(err), 99.5)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for ax, val, name in ((axes[0], err, r"$\lambda_{\rm net} - \lambda_{\rm exact}$"),
                          (axes[1], lam_sector_split(P, X, err),
                           "l = 1 part of the error (the dipole)")):
        m = ax.pcolormesh(R * np.sin(T), R * np.cos(T), val, shading="auto", cmap="coolwarm",
                          vmin=-lim, vmax=lim)
        fig.colorbar(m, ax=ax, label=name)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$R=\rho\sin\theta$")
        ax.set_ylabel(r"$z=\rho\cos\theta$")
    axes[0].set_title(f"error field, order 1, $S_1 = 0$  (max {np.max(np.abs(err)):.1e})")
    axes[1].set_title("its $l = 1$ content: antisymmetric by construction")
    fig.suptitle("The error is EVEN in z (a pure quadrupole): with no dipole in the datum the "
                 "flat problem has none", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(PLOTS, "flat_error_map.png"), dpi=150)
    print("wrote flat_error_map.png")


def lam_sector_split(P, X, err):
    """The l = 1 part of the error: project onto P_1(n_z) at each radius, put it back."""
    r = np.linalg.norm(X, axis=-1)
    nz = X[..., 2] / r
    c = 1.5 * np.trapezoid(err * nz, nz, axis=0)
    return c[None, :] * nz


def fig_production(data):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for name, h in data["production"].items():
        ax.plot(h["step"], h["loss"], lw=1.6, label=f"{name.split('_')[-1]} ({len(h['step'])} pts)")
    for label, val in BASELINES.items():
        ax.axhline(val, ls=":", lw=1.2, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("DSGNAR iteration")
    ax.set_ylabel("loss")
    ax.set_title("c100 hybrid recipe: DSGNAR from F's Adam end, against the SSBroyden baselines")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "production_convergence.png"), dpi=150)
    print("wrote production_convergence.png")


def fig_optimizers(data):
    cases = [c for c in data["flat"] if c["s1"] == 0.0 and c["order"] == 1]
    if not cases:
        print("flat_optimizers.png skipped: no flat cases yet")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels, means, pts = [], [], []
    for c in sorted(cases, key=lambda c: (c["optimizer"], -c.get("n_batches", 0))):
        errs = [b["max_err"] for b in c["batches"]]
        labels.append(f"{c['optimizer']}\n" + (c.get("settings") or "") +
                      f"\n{len(errs)} batch(es), {c['batches'][0]['iterations']} it")
        means.append(np.mean(errs))
        pts.append(errs)
    ax.bar(range(len(labels)), means, 0.5, color=["tab:blue" if "ssbroyden" in l else "tab:purple"
                                                  for l in labels], alpha=0.8)
    for i, e in enumerate(pts):
        ax.scatter([i] * len(e), e, color="k", zorder=3, s=18)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"$\max|\lambda_{\rm net} - \lambda_{\rm exact}|$")
    ax.set_title("Flat problem, order 1, pure quadrupole: both optimisers\n"
                 "(the flat residual is third-derivative in the network, so a sketch costs more here)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "flat_optimizers.png"), dpi=150)
    print("wrote flat_optimizers.png")


def main():
    data = load()
    fig_orders(data)
    fig_sectors(data)
    fig_sampling()
    fig_error_map()
    fig_production(data)
    fig_optimizers(data)


if __name__ == "__main__":
    main()
