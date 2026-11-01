#!/usr/bin/env python
"""Gauge-invariant geometry of a quadrupole run next to its spectral reference.

    .venv/bin/python scripts/geometry_quadrupole.py runs/<run> --ref runs/<r>/specref_nonlinear_<tag>.npz

Names: "spectral reference" is the spectral solution of the quadrupolar problem (--ref);
"spherical reference" (also called the Schwarzschild reference) is the spherical solution it
perturbs (train.exact_asset).  stationary/geometry_figures.py does this for the Weyl runs, whose
comparison is an exact solution; the quadrupolar problem has none.  The network and both references
are evaluated in the cylindrical chart about the axis (geometry_invariants.cylindrical_metric)
and contracted in its orthonormal frame, as there.  Writes into the run directory:

    geometry_slice.png     (rho, z) slice, six panels, all of the NETWORK: its Kretschmann K; its
    geometry_slice_zoom.png  change from the spherical reference |K/K_sph - 1|; its algebraic
                           speciality |S - 1| (0 = type D; the spherical reference is type D, the
                           quadrupole is not); its 4-D Ricci |R^(4)_ab|/sqrt|K| (the Ricci tensor of
                           the static vacuum completion g = -lam dt^2 + h/lam, over the curvature
                           scale; 0 for an exact solution); its error |K_net/K_spec - 1| against the
                           spectral reference (the only panel that uses it); |1 - lambda|
    hawking_profile.png    m_H(rho) and the lapse mass r_a(1 - lambda)/2 of the network and its M_0,
                           with the spectral and the spherical references as grey overlays
    geometry_profiles.png  the same curvature quantities along rays (--only profiles)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stationary.evaluate import load_run                                    # noqa: E402
from stationary.geometry_figures import _panels, slice_grid                 # noqa: E402
from stationary.geometry_invariants import (axis_from_config, cylindrical_metric,  # noqa: E402
                                            sphere_geometry)
from stationary.model import point_fields                                   # noqa: E402
from stationary.specref import SpectralAxisym                               # noqa: E402
from stationary.train import exact_asset                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("--ref", required=True, help="the spectral reference, specref_nonlinear_<tag>.npz")
    ap.add_argument("--out", default=None, help="output directory (default: the run directory)")
    ap.add_argument("--n-r", type=int, default=36)
    ap.add_argument("--n-theta", type=int, default=48)
    ap.add_argument("--n-radii", type=int, default=14)
    ap.add_argument("--only", choices=("slice", "hawking", "profiles"), default=None)
    ap.add_argument("--angles", default="0,45,90",
                    help="polar angles in degrees of the rays of geometry_profiles.png (0 = the "
                         "symmetry axis, taken at cylindrical radius 0.1 rho_in as in the slice)")
    ap.add_argument("--n-rays", type=int, default=80, help="radii on each ray")
    ap.add_argument("--replot-hawking", action="store_true",
                    help="redraw hawking_profile.png from hawking_profile.npz (no evaluation)")
    ap.add_argument("--zoom", type=float, default=6.0,
                    help="half-width of the second, zoomed slice (geometry_slice_zoom.png)")
    ap.add_argument("--moments", default=None,
                    help="gh_moments.json holding the spectral reference's M_0 (default: the run's own, "
                         "which scripts/gh_moments_runs.py wrote against --ref; the spectral reference's "
                         "directory may hold another variant's)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg, model, state = load_run(a.run)
    pf = point_fields(model, state["net"])
    sph = exact_asset(cfg, n_quad=48)
    z = np.load(a.ref)
    spec = SpectralAxisym(cfg, sph, ls=tuple(int(v) for v in z["ls"]), n_radial=int(z["n_radial"]))
    rf = spec.fields_of(jnp.asarray(z["c"]))
    axis = axis_from_config(cfg)
    out = a.out or a.run
    name = os.path.basename(os.path.normpath(a.run))
    M0 = json.load(open(a.moments or os.path.join(a.run, "gh_moments.json")))["ref"]["M_ernst"][0]
    print(f"[geom] M_0 of the spectral reference {M0:.6f} (from {a.moments or os.path.join(a.run, 'gh_moments.json')})")

    if a.only == "profiles":
        angles = [float(v) for v in a.angles.split(",")]
        r = float(cfg.rho_in) * (float(cfg.rho_out) / float(cfg.rho_in)) ** np.linspace(0, 1, a.n_rays)
        floor = 0.1 * float(cfg.rho_in)
        ys = []
        for deg in angles:
            s = np.maximum(np.sin(np.radians(deg)), floor / r)
            ys.append(np.stack([r * s, np.zeros_like(r), r * np.sqrt(np.maximum(1 - s**2, 0.0))], -1))
        ys = jnp.asarray(np.stack(ys, 1))                                   # (n_rays, n_angles, 3)
        print(f"[geom] {ys.shape[0] * ys.shape[1]} nodes on {len(angles)} rays ...", flush=True)
        net = _panels(*cylindrical_metric(lambda x: pf(x).h, lambda x: pf(x).lam, "vacuum", axis), ys)
        ref = _panels(*cylindrical_metric(lambda x: rf(x).h, lambda x: rf(x).lam, "vacuum", axis), ys)
        bg = _panels(*cylindrical_metric(lambda x: sph(x).h, lambda x: sph(x).lam, "vacuum", axis), ys)
        sh = (ys.shape[0], ys.shape[1])
        g = lambda d, k: np.asarray(d[k]).reshape(sh)                       # noqa: E731
        np.savez(os.path.join(out, "geometry_profiles.npz"), r=r, angles=angles,
                 K_net=g(net, "K"), K_spec=g(ref, "K"), K_sph=g(bg, "K"),
                 S_spec=g(ref, "speciality"), S_net=g(net, "speciality"),
                 ric_net=g(net, "vacuum_defect"))
        cols = ["#2a78d6", "#eb6834", "#2a9d5c", "#8e5bd0"]
        lab = lambda d: "axis" if d == 0 else ("equator" if d == 90 else f"{d:g}°")  # noqa: E731
        fig, ax = plt.subplots(2, 2, figsize=(9.5, 7.2), constrained_layout=True)
        for j, deg in enumerate(angles):
            c = cols[j % len(cols)]
            ax[0, 0].loglog(r, g(net, "K")[:, j], color=c, label=f"network, {lab(deg)}")
            ax[0, 0].loglog(r, g(bg, "K")[:, j], "--", color=c, lw=1.0)
            ax[0, 1].semilogx(r, g(net, "K")[:, j] / g(bg, "K")[:, j], color=c, label=lab(deg))
            ax[1, 0].loglog(r, np.maximum(g(net, "speciality")[:, j], 1e-30), color=c, label=lab(deg))
            ax[1, 1].loglog(r, np.maximum(np.abs(g(net, "K")[:, j] / g(ref, "K")[:, j] - 1), 1e-30),
                            color=c, label=f"|K_net/K_spec - 1|, {lab(deg)}")
            ax[1, 1].loglog(r, np.maximum(g(net, "vacuum_defect")[:, j], 1e-30), ":", color=c, lw=1.2)
        ax[0, 0].loglog(r, 48 * float(cfg.R0) ** 2 / (r + float(cfg.R0)) ** 6, ":", color="k", lw=1.0,
                        label="Schwarzschild reference 48 M²/r⁶ (M = R0, r = rho + R0)")
        ax[0, 0].set_ylabel("network Kretschmann K\n(dashed: spherical reference)")
        ax[0, 1].set_ylabel("network K / K_sph\n(K_sph: spherical reference)"); ax[0, 1].axhline(1, color="grey", lw=0.8)
        ax[1, 0].set_ylabel("network speciality |S - 1|\n(0 = type D)")
        ax[1, 1].set_ylabel("network: solid |K_net/K_spec - 1|\n(spectral reference); dotted |R⁽⁴⁾_ab|/√K")
        for p, t in zip(ax.ravel(), "abcd"):
            p.set_xlabel("coordinate radius rho"); p.grid(alpha=0.3, which="both")
            p.text(0.03, 0.92, f"({t})", transform=p.transAxes)
            p.axvline(float(cfg.rho_in), color="grey", lw=0.8)
        for p in (ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]):
            p.legend(fontsize=7, frameon=False)
        fig.suptitle(f"{name}: the network's geometry, S2 = {cfg.lam_bc_S2:g}, along rays at "
                     f"{', '.join(lab(d) for d in angles)} from the axis", fontsize=10)
        png = os.path.join(out, "geometry_profiles.png")
        fig.savefig(png, dpi=140); plt.close(fig)
        print(f"[geom] wrote {png}", flush=True)
        return

    if a.only != "hawking":
        R, Z, ys, mask = slice_grid(cfg, n_r=a.n_r, n_theta=a.n_theta)
        print(f"[geom] {mask.size} nodes: network, spectral reference, spherical reference ...", flush=True)
        net = _panels(*cylindrical_metric(lambda x: pf(x).h, lambda x: pf(x).lam, "vacuum", axis), ys)
        ref = _panels(*cylindrical_metric(lambda x: rf(x).h, lambda x: rf(x).lam, "vacuum", axis), ys)
        bg = _panels(*cylindrical_metric(lambda x: sph(x).h, lambda x: sph(x).lam, "vacuum", axis), ys)
        put = lambda v: np.asarray(v).reshape(mask.shape)                       # noqa: E731
        Kn, Kr, Kb = put(net["K"]), put(ref["K"]), put(bg["K"])
        # the network is the subject; the spectral reference enters only as the yardstick of panel 5
        lam_net = np.asarray(jax.vmap(lambda y: pf(y).lam)(ys.reshape(-1, 3))).reshape(mask.shape)
        panels = [
            (Kn, "network: Kretschmann  K", "viridis"),
            (np.abs(Kn / Kb - 1.0), "network: change from the spherical reference  |K/K_sph - 1|", "viridis"),
            (put(net["speciality"]), "network: speciality  |S - 1|  (0 = type D)", "magma"),
            (put(net["vacuum_defect"]), r"network: 4-D Ricci  $|R^{(4)}_{ab}|/\sqrt{|K|}$", "cividis"),
            (np.abs(Kn / Kr - 1.0), "network error against the spectral reference  |K_net/K_spec - 1|", "magma"),
            (np.abs(1.0 - lam_net), "network: |1 - lambda|  (lambda = -xi.xi)", "viridis"),
        ]
        np.savez(os.path.join(out, "geometry_slice.npz"), R=R, Z=Z,
                 **{t: np.asarray(d) for t, (d, _, _) in zip("abcdef", panels)})
        for zoom, fname in ((None, "geometry_slice.png"), (a.zoom, "geometry_slice_zoom.png")):
            fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.2) if zoom is None else (15.5, 5.6),
                                   constrained_layout=True)
            for p, (d, title, cmap) in zip(ax.ravel(), panels):
                v = np.log10(np.maximum(d, 1e-30))
                if zoom is not None:                         # colour range of the zoomed region only
                    inside = (R < zoom) & (np.abs(Z) < zoom)
                    kw = dict(vmin=np.nanmin(v[inside]), vmax=np.nanmax(v[inside]))
                else:
                    kw = {}
                im = p.pcolormesh(R, Z, v, cmap=cmap, shading="auto", **kw)
                p.set_title(title, fontsize=10)
                p.set_xlabel("rho"); p.set_ylabel("z"); p.set_aspect("equal")
                if zoom is not None:
                    p.set_xlim(0, zoom); p.set_ylim(-zoom, zoom)
                fig.colorbar(im, ax=p, shrink=0.85, label="log10")
            fig.suptitle(f"{name}: gauge-invariant geometry, S2 = {cfg.lam_bc_S2:g} "
                         f"(lambda on the sphere from {cfg.lam0 + cfg.lam_bc_S2:.3f} to "
                         f"{cfg.lam0 - 0.5 * cfg.lam_bc_S2:.3f});  spectral reference "
                         f"{os.path.basename(a.ref)}" + ("" if zoom is None else f";  zoom rho < {zoom:g}"),
                         fontsize=10)
            png = os.path.join(out, fname)
            fig.savefig(png, dpi=130); plt.close(fig)
            print(f"[geom] wrote {png}", flush=True)

    if a.only != "slice":
        if a.replot_hawking:
            d = np.load(os.path.join(out, "hawking_profile.npz"))
            radii, P, M0 = d["radii"], d["P"], float(d["M0"])
            Pn = d["Pn"] if "Pn" in d.files else None
        else:
            radii = np.geomspace(float(cfg.rho_in), float(cfg.rho_out) * 0.98, a.n_radii)
            rows = []
            for r in radii:
                g = [sphere_geometry(lambda x, f=f: f(x).h, lambda x, f=f: f(x).lam, float(r),
                                     "vacuum", n_mu=10, n_phi=8) for f in (pf, rf, sph)]
                rows.append([float(g[0]["hawking_mass"]), float(g[1]["hawking_mass"]),
                             float(g[2]["hawking_mass"]), float(g[1]["lam_mean"]), float(g[1]["r_areal"])])
            P = np.asarray(rows)
            Pn = None
        if Pn is None:                              # the network's own mean lambda and areal radius (cheap)
            Pn = np.asarray([[float(g["lam_mean"]), float(g["r_areal"])] for g in (
                sphere_geometry(lambda x: pf(x).h, lambda x: pf(x).lam, float(r), "vacuum",
                                n_mu=10, n_phi=8) for r in radii)])
            np.savez(os.path.join(out, "hawking_profile.npz"), radii=radii, P=P, M0=M0, Pn=Pn)
        M0n = json.load(open(a.moments or os.path.join(a.run, "gh_moments.json")))["net"]["M_ernst"][0]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
        ax[0].plot(radii, P[:, 0], "o-", label="network: m_H")
        ax[0].plot(radii, 0.5 * Pn[:, 1] * (1 - Pn[:, 0]), "^-", label="network: m_lapse = r_a(1-lam)/2")
        ax[0].plot(radii, P[:, 1], "s--", lw=0.9, ms=3, color="grey",
                   label="m_H of the spectral reference")
        ax[0].plot(radii, P[:, 2], ":", color="grey", label="m_H of the spherical (Schwarzschild) reference")
        ax[0].axhline(M0n, color="k", lw=1, ls=":", label=f"network: M_0 = {M0n:.5f}")
        ax[0].set_xscale("log"); ax[0].set_xlabel("coordinate radius rho"); ax[0].set_ylabel("mass")
        ax[0].set_title("Hawking mass profile (vacuum reading)"); ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
        ax[1].plot(radii, np.abs(P[:, 0] / M0n - 1), "o-", label="network: |m_H/M_0 - 1|")
        ax[1].plot(radii, np.abs(0.5 * Pn[:, 1] * (1 - Pn[:, 0]) / M0n - 1), "^-",
                   label="network: |m_lapse/M_0 - 1|")
        ax[1].plot(radii, np.abs(P[:, 0] - P[:, 1]) / M0n, "s--", lw=0.9, ms=3, color="grey",
                   label="network error: |m_H(net) - m_H(spec)| / M_0  (spectral reference)")
        ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("coordinate radius rho")
        ax[1].set_title("approach of the network to its mass M_0, and its error"); ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3, which="both")
        for p in ax:
            p.axvline(float(cfg.rho_in), color="grey", lw=0.8); p.axvline(float(cfg.rho_out), color="grey", lw=0.8)
        fig.suptitle(f"{name}: the network's Hawking mass, S2 = {cfg.lam_bc_S2:g}", fontsize=11)
        png = os.path.join(out, "hawking_profile.png")
        fig.savefig(png, dpi=130); plt.close(fig)
        print(f"[geom] wrote {png}", flush=True)


if __name__ == "__main__":
    main()
