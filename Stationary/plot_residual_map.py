"""Per-point residual maps of a run: where the equations are NOT satisfied.

    ./.venv/bin/python plot_residual_map.py --run runs/<name> [--group lam_eq] [--points 27768]

Draws the residual of one equation group at the very collocation points where the loss evaluates
it, in the meridian plane (R = rho sin(theta), z = rho cos(theta)) with a colourbar.  Two panels:
the RAW residual (what the report's rms lines quote) and the rho-SCALED one (the units the loss
actually minimises), because on the c100 shells they tell different stories -- the raw residual
sits in a thin ring at the inner sphere while the scaled one is dominated by a large-scale
antisymmetric (dipole) pattern in the far field.

Reads the run's own config.json through `evaluate.load_run`, so the residual switches are the ones
the run trained with (see the provenance note in report.py).
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory (config.json + params.pkl)")
    ap.add_argument("--params", default="params.pkl")
    ap.add_argument("--group", default="lam_eq", help="lam_eq (default), gauge, ricci, compat")
    ap.add_argument("--points", type=int, default=0, help="override n_coll (0 = as configured)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from stationary.evaluate import load_run
    from stationary.geometry import residuals_batch
    from stationary.losses import pde_raw
    from stationary.model import point_fields
    from stationary.train import make_batch

    cfg, model, state = load_run(args.run, args.params)
    if args.points:
        cfg.n_coll = int(args.points)
    pf = point_fields(model, state["net"])
    xs = make_batch(jax.random.PRNGKey(int(cfg.seed)), cfg)["coll"]
    def per_point(v):
        """A group can have several components per point (gauge has three): colour by the norm."""
        v = np.asarray(v)
        return np.linalg.norm(v, axis=-1) if v.ndim > 1 else v

    raw = per_point(residuals_batch(pf, xs)[args.group])
    sca = per_point(pde_raw(pf, xs, cfg)[args.group])
    X = np.asarray(xs)
    R, Z = np.hypot(X[:, 0], X[:, 1]), X[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    for ax, vals, name in ((axes[0], raw, "raw"), (axes[1], sca, r"$\rho$-scaled (loss units)")):
        lim = np.percentile(np.abs(vals), 99.0)
        s = ax.scatter(R, Z, c=vals, s=2.5, cmap="coolwarm", vmin=-lim, vmax=lim, linewidths=0)
        fig.colorbar(s, ax=ax, fraction=0.046, pad=0.03).set_label(
            f"{args.group} residual ({name})", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$R=\rho\,\sin\theta$")
        ax.set_ylabel(r"$z=\rho\,\cos\theta$")
        ax.set_title(f"{name}:  rms {np.sqrt(np.mean(vals ** 2)):.2e},  "
                     f"max {np.abs(vals).max():.2e}", fontsize=10)
        ax.plot([0, 0], [-cfg.rho_out, cfg.rho_out], color="0.6", lw=0.6, ls=":")
    fig.suptitle(f"{args.group} residual at the {len(R)} collocation points where the loss "
                 f"evaluates it\n{cfg.outdir}: rho_out = {cfg.rho_out:g}, "
                 f"lam_eq form {cfg.lam_eq_form!r}, {cfg.radial} sampling", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = args.out or f"plots/{args.group}_residual_{cfg.outdir.split('/')[-1]}.png"
    fig.savefig(out, dpi=150)
    print(f"raw rms {np.sqrt(np.mean(raw ** 2)):.3e} (max {np.abs(raw).max():.3e}) | "
          f"scaled rms {np.sqrt(np.mean(sca ** 2)):.3e} (max {np.abs(sca).max():.3e})\nwrote {out}")


if __name__ == "__main__":
    main()
