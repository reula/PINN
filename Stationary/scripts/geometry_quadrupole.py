#!/usr/bin/env python
"""Gauge-invariant geometry of a quadrupole run next to its spectral reference.

    .venv/bin/python scripts/geometry_quadrupole.py runs/<run> --ref runs/<r>/specref_nonlinear_<tag>.npz

stationary/geometry_figures.py does this for the Weyl runs, whose comparison is an exact solution;
the quadrupolar problem has none, so the comparison is the spectral reference (and the spherical
background it perturbs).  Both the network and the reference are evaluated in the cylindrical
chart about the axis (geometry_invariants.cylindrical_metric) and contracted in its orthonormal
frame, as there.  Writes into the run directory:

    geometry_slice.png     (rho, z) slice, six panels: Kretschmann of the reference; its change
                           from the spherical background |K_ref/K_sph - 1| (what the quadrupole
                           does to the curvature); the gauge-invariant error |K_net/K_ref - 1|;
                           the network's 4-D Ricci |R^(4)_ab|/sqrt|K| (the Ricci tensor of the static
                           vacuum completion g = -lam dt^2 + h/lam, over the curvature scale; 0 for an
                           exact solution); the algebraic speciality
                           |S - 1| of the reference and of the network (0 = type D; the sphere
                           is type D, the quadrupole is not)
    hawking_profile.png    m_H(rho) of the network, of the reference and of the spherical solution,
                           the lapse mass r_a(1 - lambda)/2, and the reference's M_0
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
    ap.add_argument("--ref", required=True, help="specref_nonlinear_<tag>.npz")
    ap.add_argument("--out", default=None, help="output directory (default: the run directory)")
    ap.add_argument("--n-r", type=int, default=36)
    ap.add_argument("--n-theta", type=int, default=48)
    ap.add_argument("--n-radii", type=int, default=14)
    ap.add_argument("--only", choices=("slice", "hawking"), default=None)
    ap.add_argument("--zoom", type=float, default=6.0,
                    help="half-width of the second, zoomed slice (geometry_slice_zoom.png)")
    ap.add_argument("--moments", default=None,
                    help="gh_moments.json holding the reference's M_0 (default: the run's own, "
                         "which scripts/gh_moments_runs.py wrote against --ref; the reference's "
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
    print(f"[geom] M_0 of the reference {M0:.6f} (from {a.moments or os.path.join(a.run, 'gh_moments.json')})")

    if a.only != "hawking":
        R, Z, ys, mask = slice_grid(cfg, n_r=a.n_r, n_theta=a.n_theta)
        print(f"[geom] {mask.size} nodes: network, reference, spherical background ...", flush=True)
        net = _panels(*cylindrical_metric(lambda x: pf(x).h, lambda x: pf(x).lam, "vacuum", axis), ys)
        ref = _panels(*cylindrical_metric(lambda x: rf(x).h, lambda x: rf(x).lam, "vacuum", axis), ys)
        bg = _panels(*cylindrical_metric(lambda x: sph(x).h, lambda x: sph(x).lam, "vacuum", axis), ys)
        put = lambda v: np.asarray(v).reshape(mask.shape)                       # noqa: E731
        Kn, Kr, Kb = put(net["K"]), put(ref["K"]), put(bg["K"])
        panels = [
            (Kr, "Kretschmann of the reference  K_ref", "viridis"),
            (np.abs(Kr / Kb - 1.0), "change from the sphere  |K_ref/K_sph - 1|", "viridis"),
            (np.abs(Kn / Kr - 1.0), "gauge-invariant error  |K_net/K_ref - 1|", "magma"),
            (put(net["vacuum_defect"]), r"network: 4-D Ricci  $|R^{(4)}_{ab}|/\sqrt{|K|}$", "cividis"),
            (put(ref["speciality"]), "speciality  |S - 1|, reference (0 = type D)", "magma"),
            (put(net["speciality"]), "speciality  |S - 1|, network", "magma"),
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
                         f"{cfg.lam0 - 0.5 * cfg.lam_bc_S2:.3f});  reference "
                         f"{os.path.basename(a.ref)}" + ("" if zoom is None else f";  zoom rho < {zoom:g}"),
                         fontsize=10)
            png = os.path.join(out, fname)
            fig.savefig(png, dpi=130); plt.close(fig)
            print(f"[geom] wrote {png}", flush=True)

    if a.only != "slice":
        radii = np.geomspace(float(cfg.rho_in), float(cfg.rho_out) * 0.98, a.n_radii)
        rows = []
        for r in radii:
            g = [sphere_geometry(lambda x, f=f: f(x).h, lambda x, f=f: f(x).lam, float(r),
                                 "vacuum", n_mu=10, n_phi=8) for f in (pf, rf, sph)]
            rows.append([float(g[0]["hawking_mass"]), float(g[1]["hawking_mass"]),
                         float(g[2]["hawking_mass"]), float(g[1]["lam_mean"]), float(g[1]["r_areal"])])
        P = np.asarray(rows)
        np.savez(os.path.join(out, "hawking_profile.npz"), radii=radii, P=P, M0=M0)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
        ax[0].plot(radii, P[:, 0], "o-", label="m_H  network")
        ax[0].plot(radii, P[:, 1], "s--", label="m_H  reference")
        ax[0].plot(radii, P[:, 2], ":", color="grey", label="m_H  spherical solution")
        ax[0].plot(radii, 0.5 * P[:, 4] * (1 - P[:, 3]), "^-", label="m_lapse = r_a(1-lam)/2 (reference)")
        ax[0].axhline(M0, color="k", lw=1, ls=":", label=f"M_0 of the reference = {M0:.5f}")
        ax[0].set_xscale("log"); ax[0].set_xlabel("coordinate radius rho"); ax[0].set_ylabel("mass")
        ax[0].set_title("Hawking mass profile (vacuum reading)"); ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
        ax[1].plot(radii, np.abs(P[:, 0] - P[:, 1]) / M0, "o-", label="|m_H(net) - m_H(ref)| / M_0")
        ax[1].plot(radii, np.abs(P[:, 1] / M0 - 1), "s--", label="|m_H(ref)/M_0 - 1|")
        ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("coordinate radius rho")
        ax[1].set_title("network against reference, and approach to M_0"); ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3, which="both")
        for p in ax:
            p.axvline(float(cfg.rho_in), color="grey", lw=0.8); p.axvline(float(cfg.rho_out), color="grey", lw=0.8)
        fig.suptitle(f"{name}: Hawking mass, S2 = {cfg.lam_bc_S2:g}", fontsize=11)
        png = os.path.join(out, "hawking_profile.png")
        fig.savefig(png, dpi=130); plt.close(fig)
        print(f"[geom] wrote {png}", flush=True)


if __name__ == "__main__":
    main()
