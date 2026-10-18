#!/usr/bin/env python
"""Grade a run with OFF-AXIS data against the axisymmetric spectral reference, rotated.

    .venv/bin/python scripts/grade_rotated.py runs/<run> --ref <specref .npz> [--params params.pkl]

The run's inner data are the S1/S2 data about the axis e = cfg.lam_bc_axis.  Harmonic coordinates are
rotation-covariant and every other condition is rotation invariant, so the solution is the
axisymmetric (axis z) solution rotated by R, R z = e:

    h'(x) = R h(R^T x) R^T,   lambda'(x) = lambda(R^T x).

The reference must have been solved for the same lam0, S1, S2 about z (e.g. the fine reference for
S1 = 0).  Prints the error per radial region next to the perturbation's size, and the centre-of-mass
flux (all three components) on four spheres; writes <run>/grade_rotated.json.
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

from stationary.evaluate import load_run                  # noqa: E402
from stationary.losses import com_flux                    # noqa: E402
from stationary.model import Fields, point_fields         # noqa: E402
from stationary.specref import SpectralAxisym, compare    # noqa: E402
from stationary.train import exact_asset                  # noqa: E402


def rotation_to(e):
    """The rotation R with R z = e (about the axis z x e)."""
    e = np.asarray(e, dtype=float) / np.linalg.norm(e)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(z, e), float(z @ e)
    if np.linalg.norm(v) < 1e-14:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--params", default="params.pkl")
    a = ap.parse_args(argv)
    cfg, model, state = load_run(a.run, a.params)[:3]
    pf = point_fields(model, state["net"])
    ex = exact_asset(cfg)
    R = jnp.asarray(rotation_to(cfg.lam_bc_axis if cfg.lam_bc_axis is not None else (0, 0, 1)))
    z = np.load(a.ref)
    ref0 = SpectralAxisym(cfg, ex, ls=tuple(int(v) for v in z["ls"]),
                          n_radial=int(z["n_radial"])).fields_of(jnp.asarray(z["c"]))

    def ref(x):
        f = ref0(R.T @ x)
        return Fields(R @ f.h @ R.T, None, f.lam)

    cm = compare(pf, ref, ex, cfg, modes=False)
    cc = cfg.__class__(**{**cfg.__dict__, "com_radii": (1.0, 2.0, 10.0, 100.0)})
    d = np.asarray(com_flux(pf, cc)).reshape(-1, 3)
    print(f"[grade] {a.run}: data axis {cfg.lam_bc_axis}, reference {os.path.basename(a.ref)} rotated")
    print(f"[grade] over the shell: rms |dlambda| {cm['rms_dlam']:.2e} (max {cm['max_dlam']:.2e}), "
          f"rms |dh| {cm['rms_dh']:.2e} (max {cm['max_dh']:.2e}); perturbation "
          f"{cm['perturbation_rms_lam']:.2e} / {cm['perturbation_rms_h']:.2e}")
    print("[grade] by region      rms|dlambda| (perturb.)   rms|dh| (perturb.)")
    for b in cm["bins"]:
        print(f"[grade]   rho {b['rho_lo']:5.1f}-{b['rho_hi']:5.1f}  {b['rms_dlam']:.2e} ({b['perturbation_rms_lam']:.2e})"
              f"   {b['rms_dh']:.2e} ({b['perturbation_rms_h']:.2e})")
    print("[grade] centre-of-mass flux d on spheres 1, 2, 10, 100: "
          + "  ".join("(" + ", ".join(f"{v:+.1e}" for v in row) + ")" for row in d))
    with open(os.path.join(a.run, "grade_rotated.json"), "w") as fh:
        json.dump({"compare": cm, "d": d.tolist()}, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
