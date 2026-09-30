"""Figures of the gauge-invariant geometry of a run, next to its exact solution.

    python -m stationary.geometry_figures --outdir runs/weyl_rot45_hub

Writes two PNGs into the run directory:

    geometry_slice.png     the (rho, z) slice about the run's own axis, six panels:
                           the Kretschmann scalar of the network and of the exact solution,
                           the gauge-invariant error |K_net/K_exact - 1|, the vacuum defect
                           |R_ab|/sqrt|K|, the algebraic speciality |S - 1|, and the
                           Pontryagin density (identically zero while the solution is static)
    hawking_profile.png    m_H(rho) of the network and of the exact solution, m_lapse, and the
                           total mass the configuration must have

The sphere integrals (the profile) use the reference in the code's CARTESIAN-like chart, because
`sphere_geometry` builds the sphere as x = r n and takes its normal in that chart -- the chart
fields below are for the curvature panels, where any chart will do because the scalars are
invariant.

WHY THE EXACT SOLUTION IS EVALUATED IN THE CHART.  Differentiating the Cartesian components of
an axisymmetric metric near its axis is hopeless (`h_cart` carries a 1/rho^2 second-derivative
structure there, and each evaluation runs a quadrature), so the exact panels are computed from
the Weyl metric written analytically in the chart (rho, phi, z) about the RODS' axis:
h = diag(e^{2k}, rho^2, e^{2k}), lambda = e^{2U}.  That is exactly the same space-time, it is
cheap, and it is valid for the rotated configurations too -- rotating the rods is only a
relabeling of which direction the axis points, so the chart functions are the unrotated ones.

The network panels use the Cartesian components, because for a network those are smooth; nodes
within `rho_min` of the run's axis are evaluated at that distance instead (the same side limit
as the VTK export).
"""
from __future__ import annotations

import argparse
import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from .evaluate import load_run
from .geometry_invariants import (READINGS, axis_frame, axis_from_config, curvature_at,
                                  geometry_report, sphere_geometry)
from .model import point_fields


# ------------------------------------------------------------------ exact, in the chart
def rods_of_config(cfg):
    from .geometry_invariants import weyl_rods
    return weyl_rods(cfg)


def exact_chart_fields(cfg, n_quad: int = 200):
    from .geometry_invariants import weyl_chart_fields
    return weyl_chart_fields(cfg, n_quad)


def horizon_table(cfg):
    from .geometry_invariants import weyl_horizon_table
    return weyl_horizon_table(cfg)


# ------------------------------------------------------------------------ the slice
def slice_grid(cfg, n_r=48, n_theta=64, rho_floor=None):
    """The shell-conforming POLAR grid of the plane through the run's axis.

    Same recipe as `plane.half_plane`, and for the same reason (its docstring): a uniform
    rectangle in (rho_cyl, z) spends nearly all its nodes in the far field and leaves the inner
    sphere -- where the field varies fastest -- a handful of cells.  Radial levels are geometric
    so both spheres are hit exactly and the cells grow by a constant factor outward; theta is
    uniform.  Every node is inside the shell, so nothing needs masking.

    The coordinate radius is floored at `rho_floor`, default 0.1 * rho_in.  That is where the
    two errors cross: the chart's Christoffel symbols lose ~1/rho^2 digits in forming a
    four-index contraction (so the value at the floor is good to ~1% there), while below it no
    frame helps because the loss happens before the contraction.  The axis column is therefore
    the limit from rho_floor; no node is ever evaluated exactly on the axis.

    Returns R, Z (for plotting), the CHART coordinates y = (rho, 0, z) to evaluate, and a mask
    that is True everywhere except in the (empty) region below the floor.
    """
    r_out = float(cfg.rho_out)
    r_in = float(cfg.rho_in)
    floor = float(rho_floor if rho_floor is not None else 0.1 * r_in)
    r = r_in * (r_out / r_in) ** jnp.linspace(0.0, 1.0, n_r)
    th = jnp.linspace(0.0, jnp.pi, n_theta)
    R = jnp.outer(r, jnp.sin(th))
    Z = jnp.outer(r, jnp.cos(th))
    # floor rho by moving ALONG the same sphere (theta -> theta_min), not by pushing the node
    # outwards: that keeps every node exactly at its radial level and inside the shell
    s = jnp.clip(R / r[:, None], 0.0, 1.0)
    s = jnp.maximum(s, floor / r[:, None])
    Rc = r[:, None] * s
    Zc = jnp.sign(Z) * r[:, None] * jnp.sqrt(jnp.maximum(1.0 - s**2, 0.0))
    y = jnp.stack([Rc, jnp.zeros_like(Rc), Zc], axis=-1)
    mask = jnp.ones(R.shape, dtype=bool)
    return np.asarray(Rc), np.asarray(Zc), y, np.asarray(mask)


def _panels(h, lam, ys, reading="vacuum", frame="orthonormal"):
    """The six scalars on the grid, in the chart (rho, phi, z), flattened and masked by caller.

    `h`, `lam` must already be chart fields -- `cylindrical_metric` for a network, the analytic
    chart metric for the exact Weyl solution -- and `ys` are the chart coordinates.
    """
    yf = ys.reshape(-1, 3)

    def one(y):
        c = curvature_at(h, lam, y, reading, frame=frame)
        return (c["K"], c["C2"], c["CdotC"], jnp.linalg.norm(c["Ric"]),
                jnp.abs(c["S"] - 1.0), c["petrov_code"])
    K, C2, CC, ric, sdev, code = jax.vmap(one)(yf)
    return dict(K=np.asarray(K), C2=np.asarray(C2), pontryagin=np.asarray(CC),
                ricci=np.asarray(ric), speciality=np.asarray(sdev),
                type_D=np.asarray(code == 1, dtype=float),
                vacuum_defect=np.asarray(ric) / np.sqrt(np.abs(np.asarray(K))))


# ------------------------------------------------------------------------ figures
def geometry_slice_figure(run_dir, out_png=None, n_r=48, n_theta=64, n_quad=200,
                          params_file="params.pkl"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .geometry_invariants import cylindrical_metric

    cfg, model, state = load_run(run_dir, params_file)
    pf = point_fields(model, state["net"])
    axis = axis_from_config(cfg)
    R, Z, ys, mask = slice_grid(cfg, n_r=n_r, n_theta=n_theta)

    # BOTH panels are chart fields about the run's own axis, contracted in the orthonormal
    # frame: the network's through the general change of coordinates, the exact solution's
    # analytically.  This is the same device plane.py uses for R_ab R^ab, applied to the whole
    # curvature, and it is what keeps the panels meaningful where the axis sits.
    print(f"[fig] network panels in the chart about the axis ({mask.sum()} nodes) ...")
    hnet, lnet = cylindrical_metric(lambda x: pf(x).h, lambda x: pf(x).lam, "vacuum", axis)
    net = _panels(hnet, lnet, ys)
    print("[fig] exact panels (analytic chart metric of the rods) ...")
    hc, lc = exact_chart_fields(cfg, n_quad=n_quad)
    ex = _panels(hc, lc, ys)

    def put(arr):
        a = np.where(mask, arr.reshape(mask.shape), np.nan)
        return a

    K_net, K_ex = put(net["K"]), put(ex["K"])
    err = np.abs(K_net / K_ex - 1.0)

    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.2), constrained_layout=True)
    pairs = [
        (K_net, "Kretschmann  K_net  (log)", "viridis", True),
        (K_ex, "Kretschmann  K_exact  (log)", "viridis", True),
        (err, "gauge-invariant error  |K_net/K_exact - 1|", "magma", True),
        (put(net["vacuum_defect"]), "vacuum defect  |R_ab|/sqrt|K|", "cividis", True),
        (put(net["speciality"]), "algebraic speciality  |S - 1|   (0 = type D)", "magma", True),
        (put(np.abs(net["pontryagin"])), "Pontryagin  |C.C~|   (0 = static)", "magma", True),
    ]
    for a, (data, title, cmap, lg) in zip(ax.ravel(), pairs):
        d = np.log10(np.maximum(data, 1e-30)) if lg else data
        im = a.pcolormesh(R, Z, d, cmap=cmap, shading="auto")
        a.set_title(title, fontsize=10)
        a.set_xlabel("rho (about the run's axis)")
        a.set_ylabel("z")
        a.set_aspect("equal")
        fig.colorbar(im, ax=a, shrink=0.85, label="log10" if lg else "")
        for _m, (z1, z2), _area, _kappa in horizon_table(cfg):
            a.plot([0, 0], [z1, z2], color="w", lw=4)
            a.plot([0, 0], [z1, z2], color="r", lw=1.8)
    ht = horizon_table(cfg)
    htxt = ";  ".join(f"m={m:.5f}: A={area:.2e}, kappa={kappa:.1f}" for m, _sp, area, kappa in ht)
    fig.suptitle(f"rotated Weyl, gauge-invariant geometry: {run_dir}\n"
                 f"red segments on the axis = the two horizons ({htxt});  chart about the axis "
                 f"rotated {getattr(cfg, 'weyl_rotate_deg', 0):g} deg", fontsize=10)
    out = out_png or os.path.join(run_dir, "geometry_slice.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] wrote {out}")
    return out


def hawking_figure(run_dir, out_png=None, n_radii=16, params_file="params.pkl"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg, model, state = load_run(run_dir, params_file)
    pf = point_fields(model, state["net"])
    radii = np.geomspace(float(cfg.rho_in), float(cfg.rho_out) * 0.98, n_radii)
    # The EXACT profile needs the reference in the code's Cartesian-like chart: sphere_geometry
    # builds the sphere as x = r n and takes the normal in that chart, which is only the sphere
    # we mean when the metric is expressed in Cartesian components.  The chart fields above are
    # for the curvature panels, where any chart is allowed because the scalars are invariant.
    from .train import exact_asset
    ref = exact_asset(cfg, n_quad=48)
    hr, lr = (lambda x: ref(x).h), (lambda x: ref(x).lam)
    print(f"[fig] Hawking profile on {n_radii} spheres (network + exact) ...")
    prof = []
    for r in radii:
        a = sphere_geometry(lambda x: pf(x).h, lambda x: pf(x).lam, float(r), "vacuum",
                            n_mu=10, n_phi=8)
        b = sphere_geometry(hr, lr, float(r), "vacuum", n_mu=10, n_phi=8)
        prof.append([float(a["hawking_mass"]), float(b["hawking_mass"]),
                     float(a["lam_mean"]), float(a["r_areal"])])
    p = np.asarray(prof)
    m_exp = cfg.weyl_half_length + (cfg.weyl_half_length_b or cfg.weyl_half_length)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    ax[0].plot(radii, p[:, 0], "o-", label="m_H  network")
    ax[0].plot(radii, p[:, 1], "s--", label="m_H  exact (reference)")
    ax[0].plot(radii, 0.5 * p[:, 3] * (1 - p[:, 2]), "^-", label="m_lapse = r_a(1-lam)/2")
    ax[0].axhline(m_exp, color="k", lw=1, ls=":", label=f"sum of rod masses = {m_exp:.6f}")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("coordinate radius rho")
    ax[0].set_ylabel("mass")
    ax[0].set_title("Hawking mass profile (vacuum reading)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(radii, np.abs(p[:, 0] / m_exp - 1), "o-", label="|m_H(net)/M - 1|")
    ax[1].plot(radii, np.abs(p[:, 1] / m_exp - 1), "s--", label="|m_H(exact)/M - 1|")
    ax[1].plot(radii, np.abs(0.5 * p[:, 3] * (1 - p[:, 2]) / m_exp - 1), "^-",
               label="|m_lapse/M - 1|")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("coordinate radius rho")
    ax[1].set_ylabel("relative deviation")
    ax[1].set_title("approach to the total mass")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    for a in ax:
        a.axvline(float(cfg.rho_in), color="grey", lw=0.8)
        a.axvline(float(cfg.rho_out), color="grey", lw=0.8)
    fig.suptitle(f"{run_dir}: Hawking mass (0 in flat space, M for Schwarzschild, "
                 f"sum of rod masses for Israel-Khan)", fontsize=11)
    out = out_png or os.path.join(run_dir, "hawking_profile.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] wrote {out}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--params-file", default="params.pkl")
    p.add_argument("--n-r", type=int, default=48, help="radial levels (geometric)")
    p.add_argument("--n-theta", type=int, default=64, help="polar nodes")
    p.add_argument("--n-quad", type=int, default=200)
    p.add_argument("--n-radii", type=int, default=16)
    p.add_argument("--only", choices=("slice", "hawking"), default=None)
    a = p.parse_args()
    run_dir = a.outdir or a.run_dir
    if run_dir is None:
        p.error("give the run directory, as `--outdir RUN` or as the first argument")
    if a.only != "hawking":
        geometry_slice_figure(run_dir, n_r=a.n_r, n_theta=a.n_theta, n_quad=a.n_quad,
                              params_file=a.params_file)
    if a.only != "slice":
        hawking_figure(run_dir, n_radii=a.n_radii, params_file=a.params_file)


if __name__ == "__main__":
    main()
