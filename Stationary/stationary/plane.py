"""2-D maps of an axisymmetric run: the (rho_cyl, z) half-plane.

The configurations of §8.9-8.10 are axisymmetric -- the rods sit on the z axis and nothing
depends on phi -- so their entire field content lives in the half-plane (rho_cyl, z) with
rho_cyl >= 0.  A colour map of that half-plane therefore shows everything a 3-D VTK file
would, with the holes and the two spheres drawn where they actually are, at a fraction of
the size and of the evaluation cost.

    python -m stationary.plane --outdir runs/weyl_unequal
    python -m stationary.plane --outdir runs/weyl_prod --n-rho 240 --n-z 480

Only points inside the shell are evaluated; the rest is masked, so the figure shows the
domain the run was actually given.  Refuses to run on a non-axisymmetric architecture
unless `--force` is given, since the half-plane would then be a slice and not the solution.

Writes into <outdir>/plane/: `plane.png` (four panels) and `plane.json` (the numbers).

    lambda          the computed field, so its structure is visible
    lambda_err      lambda(computed) - lambda(exact), signed, diverging colour map
    h_err           max_ij |h(computed) - h(exact)|, log scale
    curvature       R_ab R^ab of the EXACT solution, log scale -- how curved the domain
                    really is, which is what decides whether a shell is a strong-field
                    test at all.  At rho_in = 7.5 for the two-rod solution this is 2.8e-06,
                    against ~0.19 at the horizon of a mass-1 hole.

The evaluation is dominated by the exact reference (a quadrature per point), so the default
grid is modest; `--n-rho/--n-z` trade cost for resolution.
"""
from __future__ import annotations

import argparse
import json
import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from .evaluate import load_run
from .geometry import ricci_from_gamma
from .model import point_fields

AXISYMMETRIC = ("sym", "sym_hybrid", "axisym_hybrid")
I3 = jnp.eye(3)


def half_plane(n_r: int, n_theta: int, rho_out: float, rho_in: float):
    """Nodes of a shell-conforming (r, theta) grid, drawn in the (rho_cyl, z) half-plane.

    NOT a uniform rectangle in (rho_cyl, z): that spends almost all its points on the far
    field and leaves the inner sphere a handful of cells -- hopeless for a run whose chart is
    small, or whose shell spans two decades, and it is exactly where the field varies
    fastest.  Radial levels are geometric, so both spheres are hit exactly and the cells grow
    by a constant factor outwards; theta is uniform.  Every node is inside the shell, so
    nothing needs masking, and the cells are drawn as the curved quadrilaterals they are.

    Returns the flattened points, an all-True mask (kept so callers need no special case),
    and the node coordinates R = r sin(theta), Z = r cos(theta) for plotting.
    """
    r = rho_in * (rho_out / rho_in) ** jnp.linspace(0.0, 1.0, n_r)
    th = jnp.linspace(0.0, jnp.pi, n_theta)
    R = jnp.outer(r, jnp.sin(th))
    Z = jnp.outer(r, jnp.cos(th))
    pts = jnp.stack([R, jnp.zeros_like(R), Z], axis=-1)
    inside = jnp.ones(R.shape, dtype=bool)
    return pts.reshape(-1, 3), inside.reshape(-1), R, Z


def maps(pf, ref, pts, inside, shape):
    """The four panels, evaluated at the grid points and masked outside the shell.

    Returned with the grid's shape (n_rho, n_z), NaN outside the shell, ready to plot.
    """
    def one(x):
        f = pf(x)
        e = ref(x)
        _, dG, _ = jax.jacfwd(ref)(x)
        ric = ricci_from_gamma(e.G, dG)
        hinv = jnp.linalg.inv(e.h)
        return (f.lam, f.lam - e.lam,
                jnp.max(jnp.abs(f.h - e.h)),
                jnp.trace(ric @ hinv @ ric @ hinv))

    lam, dlam, dh, curv = jax.vmap(one)(pts)
    keep = inside
    out = {}
    for name, v in (("lambda", lam), ("lambda_err", dlam), ("h_err", dh), ("curvature", curv)):
        out[name] = jnp.where(keep, v, jnp.nan).reshape(shape)
    return out


def figure(cfg, data, R, Z, out_png, factor=1.0):
    """Four panels with the rods, the two spheres and equal aspect.

    `factor` converts the run's chart coordinates to the units the figure is drawn in:
    the fields themselves are invariant under that rescaling (lambda and h - I are), so
    only the axes, the rods and the two spheres move.  Use it to present a run that was
    trained in a scaled chart in the standard sizes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (("lambda", data["lambda"], "viridis", r"$\lambda$ (computed)"),
              ("lambda_err", data["lambda_err"], "RdBu_r",
               r"$\lambda_{net} - \lambda_{exact}$"),
              ("h_err", jnp.log10(data["h_err"]), "magma",
               r"$\log_{10}\ \max_{ij}|h_{net}-h_{exact}|$"),
              ("curvature", jnp.log10(data["curvature"]), "cividis",
               r"$\log_{10}\ R_{ab}R^{ab}$ (exact, in the units of the axes)"))
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    # R_ab R^ab has dimension 1/length^4, so drawing the axes in other units means converting
    # it as well.  lambda, lambda_err and h_err are invariant and need nothing.
    data = dict(data)
    data["curvature"] = jnp.asarray(data["curvature"]) / factor**4
    rho_max = float(R[-1, 0]) * factor
    for a, (name, arr, cmap, title) in zip(ax.reshape(-1), panels):
        v = jnp.asarray(arr)
        finite = v[jnp.isfinite(v)]
        kw = {}
        if name in ("h_err", "curvature"):
            kw = dict(vmin=float(jnp.min(finite)), vmax=float(jnp.max(finite)))
        elif name == "lambda_err":
            m = float(jnp.max(jnp.abs(finite)))
            kw = dict(vmin=-m, vmax=m)
        im = a.pcolormesh(jnp.asarray(R) * factor, jnp.asarray(Z) * factor, v, cmap=cmap,
                          shading="gouraud", **kw)
        # the horizons: rods on the axis, and the two spheres
        for (z0, z1) in rods_spans(cfg):
            a.plot([0, 0], [z0 * factor, z1 * factor], "k-", lw=6, solid_capstyle="butt")
        th = jnp.linspace(0, 2 * jnp.pi, 400)
        for r, style in ((cfg.rho_in, "w--"), (cfg.rho_out, "w:")):
            a.plot(r * factor * jnp.cos(th), r * factor * jnp.sin(th), style, lw=1.2)
        a.set_title(title, fontsize=12)
        a.set_xlabel(r"$\rho_{cyl}$")
        a.set_ylabel("$z$")
        a.set_aspect("equal")
        a.set_xlim(0, rho_max)
        a.set_ylim(-rho_max, rho_max)
        fig.colorbar(im, ax=a, shrink=0.85)
    fig.suptitle(f"{cfg.outdir}   arch {cfg.arch}   rho in "
                 f"[{cfg.rho_in * factor:g}, {cfg.rho_out * factor:g}]"
                 + ("" if factor == 1.0 else f"   (chart values x {factor:g})")
                 + "   rods on the axis (black)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def rods_spans(cfg):
    """The rod spans of a Weyl run, or nothing for other references."""
    if not getattr(cfg, "weyl", False):
        return ()
    from .weyl import Rods
    return Rods.pair(cfg.weyl_half_length, cfg.weyl_half_length_b, cfg.weyl_half_gap).spans


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--params-file", default="params.pkl")
    p.add_argument("--n-rho", type=int, default=200, help="radial levels (geometric)")
    p.add_argument("--n-z", type=int, default=400, help="angular nodes")
    p.add_argument("--force", action="store_true",
                   help="map the half-plane even for a non-axisymmetric architecture")
    p.add_argument("--physical-inner", type=float, default=None,
                   help="draw the axes in the units in which the inner sphere sits at this "
                        "radius, the same flag and meaning as stationary.vtk.  The fields are "
                        "invariant under the rescaling, so only the axes, the rods and the "
                        "spheres move -- which is how a run trained in a scaled chart is "
                        "presented in the standard sizes")
    a = p.parse_args(argv)
    run_dir = a.outdir or a.run_dir
    if run_dir is None:
        p.error("give the run directory, as `--outdir RUN` or as the first argument")

    cfg, model, state = load_run(run_dir, a.params_file)
    if cfg.arch not in AXISYMMETRIC and not a.force:
        print(f"[plane] {cfg.arch} is not an axisymmetric architecture: the half-plane would "
              f"be a slice of a genuinely 3-D solution, not the whole thing.  Nothing "
              f"written (use --force to override).")
        return 0

    from .train import exact_asset
    ref = exact_asset(cfg)
    if ref is None:
        raise SystemExit("[plane] this run has no exact reference to compare against")
    pf = point_fields(model, state["net"])

    pts, inside, R, Z = half_plane(a.n_rho, a.n_z, cfg.rho_out, cfg.rho_in)
    print(f"[plane] {run_dir}: {int(inside.sum())} of {inside.size} grid points inside the "
          f"shell [{cfg.rho_in:g}, {cfg.rho_out:g}], arch {cfg.arch}")
    data = maps(pf, ref, pts, inside, (a.n_rho, a.n_z))
    r_in_phys = float(a.physical_inner or cfg.vtk_physical_inner)
    factor = r_in_phys / cfg.rho_in

    out_dir = os.path.join(run_dir, "plane")
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "plane.png")
    figure(cfg, data, R, Z, png, factor=factor)

    # a one-line summary of each panel, on the grid that was actually evaluated
    summary = {"n_rho": a.n_rho, "n_z": a.n_z, "n_points": int(inside.sum()),
               "rho_in": cfg.rho_in, "rho_out": cfg.rho_out,
               "physical_inner": r_in_phys, "factor": factor,
               "rho_in_drawn": cfg.rho_in * factor, "rho_out_drawn": cfg.rho_out * factor}
    for name in ("lambda", "lambda_err", "h_err", "curvature"):
        v = jnp.asarray(data[name])[inside.reshape(a.n_rho, a.n_z)]
        if name == "curvature":
            v = v / factor**4        # report it in the units the axes are drawn in
        summary[name] = {"min": float(jnp.min(v)), "max": float(jnp.max(v)),
                         "rms": float(jnp.sqrt(jnp.mean(v**2)))}
    summary["curvature_horizon_scale"] = 12.0     # 12 m^2/r^6 at r = 2m for a mass-1 hole
    with open(os.path.join(out_dir, "plane.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[plane] lambda_err in [{summary['lambda_err']['min']:.3e}, "
          f"{summary['lambda_err']['max']:.3e}] (rms {summary['lambda_err']['rms']:.3e}), "
          f"h_err max {summary['h_err']['max']:.3e}")
    print(f"[plane] R_ab R^ab of the exact solution: max "
          f"{summary['curvature']['max']:.3e} in these units, against ~0.19 at the horizon of "
          f"a mass-1 hole")
    print(f"[plane] axes in units where the inner sphere sits at {cfg.rho_in * factor:g} "
          f"(factor {factor:g} from the run's chart)")
    print(f"[plane] wrote {png} and plane.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
