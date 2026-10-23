#!/usr/bin/env python
"""Geroch-Hansen mass moments of trained runs and of the spectral reference (stationary/gh_moments.py).
    .venv/bin/python scripts/gh_moments_runs.py runs/<run> [runs/<run> ...] [--ref <specref .npz>]

For each run: the mass-centred moments M0..M5 (Ernst route) and M0..M3 (FHPS route) from the axis,
with the median and spread over fit windows rho_lo in {1, 1.25, 1.5} x rho_in and orders {6, 8}; the
symmetry axis is cfg.lam_bc_axis (rotated data) or R z (rotated Weyl), else z.  Weyl runs are
compared with the closed-form rod moments (in chart units); quadrupole runs with the reference
(--ref, rotated onto the run's axis).  The shift returned by mass centring is the run's centre of
mass along the axis in its own chart.  Writes <run>/gh_moments.json.
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

from stationary import gh_moments as G                    # noqa: E402
from stationary.evaluate import load_run                  # noqa: E402
from stationary.model import point_fields                 # noqa: E402


def run_axis(cfg):
    if getattr(cfg, "lam_bc_axis", None) is not None:
        return np.asarray(cfg.lam_bc_axis, float) / np.linalg.norm(cfg.lam_bc_axis)
    if getattr(cfg, "weyl", False) and getattr(cfg, "weyl_rotate_deg", None):
        from stationary.weyl import rotation_matrix
        return np.asarray(rotation_matrix(cfg.weyl_rotate_deg)) @ np.array([0.0, 0.0, 1.0])
    return np.array([0.0, 0.0, 1.0])


def summarise(fields, axis, rho_in, rho_out):
    # windows from the inner sphere outwards and orders 6-8: on the exact Weyl rods these return
    # M0..M3 to ~1e-6 (lower orders bias M2, higher ones and windows starting far out are unstable)
    los = tuple(f * rho_in for f in (1.0, 1.25, 1.5))
    sc = G.scan(fields, axis, rho_los=los, orders=(6, 8), rho_hi=rho_out, n=3000)
    best = [r for r in sc["rows"] if r["rho_lo"] == los[0] and r["order"] == 8][0]
    return {"M_ernst": sc["M_ernst_median"], "M_ernst_spread": sc["M_ernst_spread"],
            "M_fhps": best["M_fhps"], "shift": best["shift"], "gap": best["gap"],
            "fit_rms_xi": best["fit_rms_xi"], "rows": sc["rows"]}


def fmt(v):
    return "[" + ", ".join(f"{x:+.6e}" for x in v) + "]"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", default=None)
    a = ap.parse_args(argv)
    for run in a.runs:
        cfg, model, state = load_run(run, "params.pkl")[:3]
        axis = run_axis(cfg)
        pf = point_fields(model, state["net"])
        res = {"run": run, "axis": axis.tolist(), "arch": cfg.arch}
        res["net"] = summarise(pf, axis, cfg.rho_in, cfg.rho_out)
        print(f"[gh] {run}  ({cfg.arch}, axis {np.round(axis, 4).tolist()}, "
              f"rho [{cfg.rho_in:g}, {cfg.rho_out:g}])", flush=True)
        print(f"[gh]   network   M0..M5 {fmt(res['net']['M_ernst'])}", flush=True)
        print(f"[gh]             spread {fmt(res['net']['M_ernst_spread'])}   "
              f"FHPS M0..M3 {fmt(res['net']['M_fhps'])}   centre shift {res['net']['shift']:+.3e}",
              flush=True)
        if getattr(cfg, "weyl", False):
            from stationary.weyl import Rods
            rods = Rods.pair(cfg.weyl_half_length, cfg.weyl_half_length_b, cfg.weyl_half_gap)
            ex, zc = G.rods_moments(rods.spans)
            res["exact_rods"] = {"M_fhps": ex, "centre": zc}
            print(f"[gh]   exact rods M0..M3 {fmt(ex)}   centre {zc:+.6e}", flush=True)
        if a.ref and not getattr(cfg, "weyl", False):
            from fit_reference import RotatedReference
            from grade_rotated import rotation_to
            from stationary.specref import SpectralAxisym
            from stationary.train import exact_asset
            z = np.load(a.ref)
            spec = SpectralAxisym(cfg, exact_asset(cfg), ls=tuple(int(v) for v in z["ls"]),
                                  n_radial=int(z["n_radial"]))
            rref = RotatedReference(spec, rotation_to(axis))
            res["ref"] = summarise(rref.fields_of(jnp.asarray(z["c"])), axis, cfg.rho_in, cfg.rho_out)
            d = np.array(res["net"]["M_ernst"]) - np.array(res["ref"]["M_ernst"])
            print(f"[gh]   reference M0..M5 {fmt(res['ref']['M_ernst'])}", flush=True)
            print(f"[gh]             spread {fmt(res['ref']['M_ernst_spread'])}", flush=True)
            print(f"[gh]   net - ref        {fmt(d)}", flush=True)
        with open(os.path.join(run, "gh_moments.json"), "w") as fh:
            json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
