#!/usr/bin/env python
"""Error against the (rotated) spectral reference vs training loss, at several points of a run.

    .venv/bin/python scripts/error_vs_loss.py --ref <specref .npz> --out <json>
        <label>=<run dir>:<params file>:<loss> [...]

For each point: the rms of dh and dlambda over the shell (specref.compare, as grade_rotated.py), the
axial l = 1 harmonic gauge mode xi = c x y / rho^3 fitted on the sphere rho = 2 (|c|, see
error_modes.py), and the rms of dh on the spheres rho = 1.5, 3, 10 before and after removing that
mode (one c fitted over the three spheres together).  The loss is given on the command line (the
stage's report.json final_loss, or a history row), so that snapshots and stages are treated alike.
2026-11-10, Paper/NOTES.md "The off-axis error".
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grade_rotated import rotation_to                      # noqa: E402
from stationary.evaluate import load_run                  # noqa: E402
from stationary.model import Fields, point_fields         # noqa: E402
from stationary.specref import SpectralAxisym, compare    # noqa: E402
from stationary.train import exact_asset                  # noqa: E402

RADII = (1.5, 3.0, 10.0)


def sphere(nmu=24, nphi=48):
    mu, wmu = np.polynomial.legendre.leggauss(nmu)
    phi = 2 * np.pi * np.arange(nphi) / nphi
    th, ph = np.meshgrid(np.arccos(mu), phi, indexing="ij")
    w = np.outer(wmu, np.full(nphi, 2 * np.pi / nphi)).ravel()
    n = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1).reshape(-1, 3)
    return n, w


def one(run, params, ref_npz):
    cfg, model, state = load_run(run, params)[:3]
    pf = point_fields(model, state["net"])
    ex = exact_asset(cfg)
    R = jnp.asarray(rotation_to(cfg.lam_bc_axis if cfg.lam_bc_axis is not None else (0, 0, 1)))
    z = np.load(ref_npz)
    ref0 = SpectralAxisym(cfg, ex, ls=tuple(int(v) for v in z["ls"]),
                          n_radial=int(z["n_radial"])).fields_of(jnp.asarray(z["c"]))

    def ref(x):
        f = ref0(R.T @ x)
        return Fields(R @ f.h @ R.T, None, f.lam)

    cm = compare(pf, ref, ex, cfg, modes=False)
    err = jax.jit(jax.vmap(lambda y: R.T @ pf(R @ y).h @ R - ref0(y).h))
    n, w = sphere()
    A_rows, b_rows, raw = [], [], {}
    for rho in RADII:
        dh = np.asarray(err(jnp.asarray(rho * n)))
        sw = np.sqrt(w)[:, None, None]
        basis = []
        for cv in np.eye(3):
            v = np.cross(np.broadcast_to(cv, n.shape), n)
            basis.append((-3.0 / rho ** 3) * (n[:, :, None] * v[:, None, :] + v[:, :, None] * n[:, None, :]))
        A_rows.append(np.stack([(sw * b).ravel() for b in basis], 1))
        b_rows.append((sw * dh).ravel())
        raw[rho] = float(np.linalg.norm((sw * dh).ravel()) / np.sqrt(4 * np.pi * 9))
    A, b = np.concatenate(A_rows), np.concatenate(b_rows)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = b - A @ c
    out, k = {}, 0
    for i, rho in enumerate(RADII):
        m = len(b_rows[i])
        out[rho] = {"raw": raw[rho], "gauge_removed": float(np.linalg.norm(r[k:k + m]) / np.sqrt(4 * np.pi * 9))}
        k += m
    return {"rms_dh": cm["rms_dh"], "rms_dlam": cm["rms_dlam"], "max_dh": cm["max_dh"],
            "max_dlam": cm["max_dlam"], "c": c.tolist(), "abs_c": float(np.linalg.norm(c)),
            "spheres": {str(k): v for k, v in out.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("points", nargs="+", help="<label>=<run dir>:<params file>:<loss>")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = []
    for p in a.points:
        label, rest = p.split("=", 1)
        run, params, loss = rest.rsplit(":", 2)
        r = one(run, params, a.ref)
        r.update(label=label, run=run, params=params, loss=float(loss))
        res.append(r)
        s = r["spheres"]
        print(f"[evl] {label:24s} loss {r['loss']:.2e}  rms dh {r['rms_dh']:.2e}  rms dlam {r['rms_dlam']:.2e}"
              f"  |c| {r['abs_c']:.2e}  dh at rho 1.5/3/10: "
              + " ".join(f"{s[k]['raw']:.1e}->{s[k]['gauge_removed']:.1e}" for k in s), flush=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
