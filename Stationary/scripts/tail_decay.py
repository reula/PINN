#!/usr/bin/env python
"""Do the multipoles of a solution decay at the order they must?  The tail check on the GH potential.
    .venv/bin/python scripts/tail_decay.py runs/<run> [...] [--ref <specref .npz>] [--lmax 4]

The Legendre coefficients c_l(rho) of P = (1 - lam^2)/(4 lam) (multipoles.legendre_spectrum)
about the solution's own symmetry axis (cfg.lam_bc_axis, R z for rotated Weyl, else z) on spheres
rho = 2 .. rho_out, and their local power d ln|c_l| / d ln rho between consecutive radii.  A pure
tail of order l decays as rho^-(l+1), so c_l rho^(l+1) is constant; for the quadrupole data the
odd l must vanish (reflection symmetry), so their size is the error.  These are harmonic tail
coefficients, chart dependent beyond linear order -- the moments are gh_moments_runs.py's.
Writes <run>/tail_decay.json.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gh_moments_runs import run_axis                       # noqa: E402
from grade_rotated import rotation_to                      # noqa: E402
from stationary.evaluate import load_run                  # noqa: E402
from stationary.model import point_fields                 # noqa: E402
from stationary.multipoles import gh_fields, legendre_spectrum  # noqa: E402


def spectra(fields, R, rhos, lmax):
    Rj = jnp.asarray(R)
    f = gh_fields(lambda y: fields(Rj @ y))          # P, in the frame whose z is the symmetry axis
    return np.array([legendre_spectrum(f, r, lmax=lmax)[0] for r in rhos])


def table(name, C, rhos, lmax):
    print(f"[tail] {name}: c_l * rho^(l+1)   (local power between radii in brackets)")
    for l in range(lmax + 1):
        cl = C[:, l]
        pw = np.diff(np.log(np.abs(cl) + 1e-300)) / np.diff(np.log(rhos))
        row = "  ".join(f"{c * r ** (l + 1):+.4e}" for c, r in zip(cl, rhos))
        pws = " ".join(f"{p:+.2f}" for p in pw)
        print(f"[tail]   l={l}: {row}   [{pws}]  expected {-(l + 1)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--lmax", type=int, default=4)
    a = ap.parse_args(argv)
    for run in a.runs:
        cfg, model, state = load_run(run, "params.pkl")[:3]
        axis = run_axis(cfg)
        R = rotation_to(axis)
        rhos = np.array([r for r in (2, 3, 5, 10, 20, 30, 50, 70, 100) if cfg.rho_in < r <= cfg.rho_out]
                        or np.geomspace(1.2 * cfg.rho_in, cfg.rho_out, 8), dtype=float)
        if cfg.rho_out < 10:                         # a scaled chart (rotated Weyl): same ratios
            rhos = cfg.rho_in * np.array([1.5, 2, 3, 5, 7, 10.0])
        out = {"run": run, "axis": axis.tolist(), "rhos": rhos.tolist()}
        C = spectra(point_fields(model, state["net"]), R, rhos, a.lmax)
        out["net"] = C.tolist()
        table(f"{run} network", C, rhos, a.lmax)
        if a.ref and not getattr(cfg, "weyl", False):
            from fit_reference import RotatedReference
            from stationary.specref import SpectralAxisym
            from stationary.train import exact_asset
            z = np.load(a.ref)
            spec = SpectralAxisym(cfg, exact_asset(cfg), ls=tuple(int(v) for v in z["ls"]),
                                  n_radial=int(z["n_radial"]))
            Cr = spectra(RotatedReference(spec, R).fields_of(jnp.asarray(z["c"])), R, rhos, a.lmax)
            out["ref"] = Cr.tolist()
            table(f"{run} reference", Cr, rhos, a.lmax)
        with open(os.path.join(run, "tail_decay.json"), "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
