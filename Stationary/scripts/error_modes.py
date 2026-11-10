#!/usr/bin/env python
"""Spherical-harmonic content of a network's error against the (rotated) spectral reference.

    .venv/bin/python scripts/error_modes.py runs/<run> --ref <specref .npz> [--params params.pkl]
        [--radii 1,1.2,1.5,2,3,5,10,20,50,100] [--lmax 8] [--tag final]

Works in the DATUM's frame (z along cfg.lam_bc_axis, the axis the reference is axisymmetric about):
    dh(y)   = R^T h_net(R y) R - h_ref(y),     dlambda(y) = lambda_net(R y) - lambda_ref(y).
On each sphere |y| = rho the error is split into the spherical-frame components
    rr, r_theta, r_phi, T = theta_theta + phi_phi, D = theta_theta - phi_phi, theta_phi, and lambda,
and each component is projected on real Y_lm (Gauss-Legendre in cos theta x uniform phi).  rr, T and
lambda are scalars on the sphere, so their (l, m) content is exact; for the others the m content is
still the azimuthal content about the datum axis (m = 0 axisymmetric, |m| = 1 a tilt-like pattern),
while l is only indicative (they are components of vector and tensor fields).
Prints, per radius and component, the rms and the leading (l, |m|) shares; writes
<run>/error_modes[_<tag>].json with every amplitude, and the power-law slope in rho of the leading
modes over the radii.  2026-11-10, Paper/NOTES.md "The off-axis error".
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import sph_harm_y

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grade_rotated import rotation_to                      # noqa: E402
from stationary.evaluate import load_run                  # noqa: E402
from stationary.model import point_fields                 # noqa: E402
from stationary.specref import SpectralAxisym             # noqa: E402
from stationary.train import exact_asset                  # noqa: E402

COMPS = ("rr", "r_theta", "r_phi", "T", "D", "theta_phi", "lambda")


def real_ylm(lmax, th, ph):
    """{(l, m): Y_lm(th, ph)} real, orthonormal on the unit sphere."""
    out = {}
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            y = sph_harm_y(l, abs(m), th, ph)
            if m > 0:
                out[(l, m)] = np.sqrt(2.0) * (-1) ** m * y.real
            elif m < 0:
                out[(l, m)] = np.sqrt(2.0) * (-1) ** m * y.imag
            else:
                out[(l, m)] = y.real
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--params", default="params.pkl")
    ap.add_argument("--radii", default="1,1.2,1.5,2,3,5,10,20,50,100")
    ap.add_argument("--lmax", type=int, default=8)
    ap.add_argument("--nmu", type=int, default=32)
    ap.add_argument("--nphi", type=int, default=64)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    cfg, model, state = load_run(a.run, a.params)[:3]
    pf = point_fields(model, state["net"])
    ex = exact_asset(cfg)
    R = np.asarray(rotation_to(cfg.lam_bc_axis if cfg.lam_bc_axis is not None else (0, 0, 1)))
    z = np.load(a.ref)
    ref0 = SpectralAxisym(cfg, ex, ls=tuple(int(v) for v in z["ls"]),
                          n_radial=int(z["n_radial"])).fields_of(jnp.asarray(z["c"]))
    Rj = jnp.asarray(R)

    def err(y):
        fn, fr = pf(Rj @ y), ref0(y)
        return Rj.T @ fn.h @ Rj - fr.h, fn.lam - fr.lam

    err_v = jax.jit(jax.vmap(err))

    mu, wmu = np.polynomial.legendre.leggauss(a.nmu)
    phi = 2 * np.pi * np.arange(a.nphi) / a.nphi
    TH, PH = np.meshgrid(np.arccos(mu), phi, indexing="ij")
    W = np.outer(wmu, np.full(a.nphi, 2 * np.pi / a.nphi))            # sums to 4 pi
    Y = real_ylm(a.lmax, TH, PH)
    st, ct, sp, cp = np.sin(TH), np.cos(TH), np.sin(PH), np.cos(PH)
    n = np.stack([st * cp, st * sp, ct], -1)
    et = np.stack([ct * cp, ct * sp, -st], -1)
    ep = np.stack([-sp, cp, np.zeros_like(sp)], -1)

    radii = [float(v) for v in a.radii.split(",")]
    res = {"run": a.run, "ref": a.ref, "params": a.params, "radii": radii, "lmax": a.lmax,
           "axis": list(map(float, R[:, 2])), "modes": {}}
    print(f"[modes] {a.run} ({a.params}) against {os.path.basename(a.ref)}, datum axis {R[:, 2].round(4)}")
    for rho in radii:
        Yp = (rho * n).reshape(-1, 3)
        dh, dl = err_v(jnp.asarray(Yp))
        dh = np.asarray(dh).reshape(a.nmu, a.nphi, 3, 3)
        dl = np.asarray(dl).reshape(a.nmu, a.nphi)
        c = lambda u, v: np.einsum("...i,...ij,...j->...", u, dh, v)
        tt, pp = c(et, et), c(ep, ep)
        fields = {"rr": c(n, n), "r_theta": c(n, et), "r_phi": c(n, ep), "T": tt + pp,
                  "D": tt - pp, "theta_phi": c(et, ep), "lambda": dl}
        row = {}
        line = []
        for k in COMPS:
            f = fields[k]
            rms = float(np.sqrt((W * f * f).sum() / (4 * np.pi)))
            coef = {lm: float((W * f * y).sum()) for lm, y in Y.items()}
            amp = {}
            for (l, m), v in coef.items():
                amp[(l, abs(m))] = amp.get((l, abs(m)), 0.0) + v * v
            amp = {lm: np.sqrt(v / (4 * np.pi)) for lm, v in amp.items()}        # rms of that part
            top = sorted(amp.items(), key=lambda kv: -kv[1])[:3]
            row[k] = {"rms": rms, "amp": {f"{l},{m}": v for (l, m), v in amp.items()}}
            line.append(f"{k} {rms:.1e} [" + " ".join(f"l{l}m{m}:{v / max(rms, 1e-300):.2f}"
                                                         for (l, m), v in top) + "]")
        # The axial l = 1 harmonic gauge mode xi = c x y / rho^3 (Laplace-harmonic, so it keeps the
        # harmonic gauge): in flat space dh = -3 (n (c x n) + (c x n) n) / rho^3, purely radial-
        # tangential; it leaves lambda, h_rr and the induced metric of every sphere unchanged to
        # first order.  Fit c by least squares on this sphere and say how much of dh it explains.
        cxn = lambda cv: np.cross(np.broadcast_to(cv, n.shape), n)
        basis = []
        for cv in np.eye(3):
            v = cxn(cv)
            basis.append((-3.0 / rho ** 3) * (n[..., :, None] * v[..., None, :] + v[..., :, None] * n[..., None, :]))
        Wf = np.sqrt(W)[..., None, None]
        A = np.stack([(Wf * b).ravel() for b in basis], 1)
        bvec = (Wf * dh).ravel()
        cfit, *_ = np.linalg.lstsq(A, bvec, rcond=None)
        resid = bvec - A @ cfit
        frac = float(np.linalg.norm(resid) / max(np.linalg.norm(bvec), 1e-300))
        row["axial_l1"] = {"c": cfit.tolist(), "residual_fraction": frac}
        res["modes"][str(rho)] = row
        print(f"[modes]            axial l=1 gauge mode xi = c x y/rho^3: c = ({cfit[0]:+.3e}, {cfit[1]:+.3e}, "
              f"{cfit[2]:+.3e}) (datum frame, z = datum axis); unexplained fraction of |dh| {frac:.3f}")
        print(f"[modes] rho {rho:6.2f}  " + "  ".join(line))

    # power-law slope of each component's leading mode (at the innermost radius) over all radii
    print("[modes] slope d log(amp) / d log(rho) of each component's leading mode:")
    slopes = {}
    r0 = str(radii[0])
    lr = np.log(radii)
    for k in COMPS:
        lead = max(res["modes"][r0][k]["amp"].items(), key=lambda kv: kv[1])[0]
        amps = np.array([res["modes"][str(r)][k]["amp"][lead] for r in radii])
        s_in = np.polyfit(lr[: len(lr) // 2 + 1], np.log(amps[: len(lr) // 2 + 1]), 1)[0]
        s_out = np.polyfit(lr[len(lr) // 2:], np.log(amps[len(lr) // 2:]), 1)[0]
        slopes[k] = {"mode": lead, "inner": float(s_in), "outer": float(s_out)}
        print(f"[modes]   {k:9s} (l,|m|) = ({lead})  inner radii {s_in:+.2f}   outer radii {s_out:+.2f}")
    res["slopes"] = slopes
    out = os.path.join(a.run, f"error_modes_{a.tag}.json" if a.tag else "error_modes.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[modes] wrote {out}")


if __name__ == "__main__":
    main()
