"""Where the collocation and boundary points actually are.

    python -m stationary.plot_sampling --rho-out 8 --n-coll 32768 --n-bnd 2048 --n-bnd-outer 4096

Three panels, because "a plot of all the points" is not one picture:

    meridional   every point projected onto the (rho_cyl, z) plane, coloured by group.  For an
                 axisymmetric problem this is the honest view: it shows the radial grading, the
                 two spheres and how the shell is filled, with nothing hidden.
    radial       the number of points per logarithmic rho bin, with the boundary counts as
                 spikes at rho_in and rho_out.  This is the panel that answers "is the inner
                 region sampled enough", and it is the one that changes when --n-coll or the
                 shell ratio changes.
    3D           a subsample of the same points in space, so the sphere fill is visible
                 directly.  Subsample only: 40 000 markers overplot into a solid ball.

Every count is printed as well as drawn, and the numbers come from `make_batch`, the same
function the trainer uses -- not from a re-implementation, so the figure cannot describe a
sampling the run does not have.
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
import numpy as np

from .problem import Config
from .train import make_batch, parse_args


def sample(cfg: Config, seed: int = 0):
    """The trainer's own batch, plus the same points split by group."""
    b = make_batch(jax.random.PRNGKey(seed), cfg)
    return {k: np.asarray(v) for k, v in b.items()}


def main(argv=None):
    a = parse_args(argv)
    cfg = a if isinstance(a, Config) else a
    b = sample(cfg, seed=a.seed if hasattr(a, "seed") else 0)
    coll, inner, outer = b["coll"], b["inner"], b["outer"]

    n_coll, n_in, n_out = len(coll), len(inner), len(outer)
    print(f"[sampling] rho in [{cfg.rho_in:g}, {cfg.rho_out:g}]   radial {cfg.radial}")
    print(f"[sampling] collocation {n_coll}   inner sphere {n_in}   outer sphere {n_out}"
          f"   (boundary {100 * (n_in + n_out) / (n_coll + n_in + n_out):.1f}% of all points)")
    rho = np.linalg.norm(coll, axis=1)
    print(f"[sampling] collocation rho: min {rho.min():.4g}  median {np.median(rho):.4g}  "
          f"max {rho.max():.4g}")
    # how much of the sample sits in the first decade from the inner sphere, which is where
    # the field varies fastest and where the log-uniform measure puts a fixed fraction
    first = float(np.mean(rho <= cfg.rho_in * 10))
    print(f"[sampling] {100 * first:.1f}% of the collocation points lie within a factor 10 of "
          f"rho_in (log-uniform would give {100 * np.log(10) / np.log(cfg.rho_out / cfg.rho_in):.1f}%)")

    fig = plt.figure(figsize=(16, 5.4))

    # ---------------------------------------------------------------- meridional
    ax = fig.add_subplot(1, 3, 1)
    for pts, name, col, size in ((coll, "collocation", "tab:blue", 1.0),
                                 (inner, "inner sphere", "crimson", 3.0),
                                 (outer, "outer sphere", "black", 3.0)):
        ax.scatter(pts[:, 0], pts[:, 2], s=size, c=col, alpha=0.35 if size < 2 else 0.7,
                   linewidths=0, label=f"{name} ({len(pts)})")
    th = np.linspace(0, 2 * np.pi, 400)
    for r, ls in ((cfg.rho_in, "--"), (cfg.rho_out, ":")):
        ax.plot(r * np.cos(th), r * np.sin(th), "k" + ls, lw=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\rho_{cyl}$")
    ax.set_ylabel("$z$")
    ax.set_title("meridional projection (every point)")
    ax.legend(fontsize=8, markerscale=4, loc="upper right")
    ax.grid(alpha=0.25)

    # ---------------------------------------------------------------- radial density
    ax = fig.add_subplot(1, 3, 2)
    bins = np.geomspace(cfg.rho_in, cfg.rho_out, 61)
    ax.hist(rho, bins=bins, color="tab:blue", alpha=0.75, label="collocation")
    ax.axvline(cfg.rho_in, color="crimson", ls="--", lw=1.2,
               label=f"inner sphere ({n_in})")
    ax.axvline(cfg.rho_out, color="black", ls=":", lw=1.2, label=f"outer sphere ({n_out})")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("points per log bin")
    ax.set_title(f"radial density ({cfg.radial} in $\\rho$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ---------------------------------------------------------------- 3D
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    for pts, name, col in ((coll, "collocation", "tab:blue"),
                           (inner, "inner sphere", "crimson"),
                           (outer, "outer sphere", "black")):
        k = min(len(pts), 4000)
        idx = np.linspace(0, len(pts) - 1, k).astype(int)
        ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2], s=1.0, c=col, alpha=0.35,
                   linewidths=0, label=name)
    # a cut-away, so the inner sphere is not hidden inside the shell
    ax.set_xlim(-cfg.rho_out, cfg.rho_out)
    ax.set_ylim(-cfg.rho_out, cfg.rho_out)
    ax.set_zlim(-cfg.rho_out, cfg.rho_out)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    ax.set_title("all groups in space (subsampled)")
    ax.legend(fontsize=8, markerscale=8)

    fig.suptitle(f"sampling of {cfg.outdir}   rho in [{cfg.rho_in:g}, {cfg.rho_out:g}]   "
                 f"n_coll {n_coll}   n_bnd {n_in} / n_bnd_outer {n_out}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(cfg.outdir, "sampling.png")
    os.makedirs(cfg.outdir, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"[sampling] wrote {out}")
    return out


if __name__ == "__main__":
    main()
