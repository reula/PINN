#!/usr/bin/env python
"""The difference between two spectral references of the same problem: how accurate is the yardstick?

    .venv/bin/python scripts/compare_refs.py runs/<run> --a <specref .npz> --b <specref .npz> [--out NAME]

Both references are evaluated through SpectralAxisym on the run's own configuration (same data, same
formulation) and compared with stationary.specref.compare, exactly as a network is graded by
scripts/grade_rotated.py: rms and max of |dlambda| and |dh| over the shell and by radial region, next
to the perturbation's size.  If a network's error against reference A is comparable to |A - B|, that
error is the reference's, not the network's.  Writes <run>/compare_refs[_NAME].json.  Added
2026-11-06 for the paper's -1/12 reference (l <= 10, 36 radial against l <= 14, 48 radial).
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
from stationary.specref import SpectralAxisym, compare    # noqa: E402
from stationary.train import exact_asset                  # noqa: E402


def load_ref(cfg, ex, path):
    z = np.load(path)
    return SpectralAxisym(cfg, ex, ls=tuple(int(v) for v in z["ls"]),
                          n_radial=int(z["n_radial"])).fields_of(jnp.asarray(z["c"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--a", required=True, help="the reference being checked")
    ap.add_argument("--b", required=True, help="the larger-basis reference")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    cfg = load_run(a.run)[0]
    ex = exact_asset(cfg)
    A, B = load_ref(cfg, ex, a.a), load_ref(cfg, ex, a.b)
    cm = compare(A, B, ex, cfg, modes=False)
    print(f"[refs] {a.run}: {os.path.basename(a.a)} - {os.path.basename(a.b)}")
    print(f"[refs] over the shell: rms |dlambda| {cm['rms_dlam']:.2e} (max {cm['max_dlam']:.2e}), "
          f"rms |dh| {cm['rms_dh']:.2e} (max {cm['max_dh']:.2e}); perturbation "
          f"{cm['perturbation_rms_lam']:.2e} / {cm['perturbation_rms_h']:.2e}")
    for b in cm["bins"]:
        print(f"[refs]   rho {b['rho_lo']:5.1f}-{b['rho_hi']:5.1f}  {b['rms_dlam']:.2e}   {b['rms_dh']:.2e}")
    name = f"compare_refs_{a.out}.json" if a.out else "compare_refs.json"
    with open(os.path.join(a.run, name), "w") as fh:
        json.dump({"a": a.a, "b": a.b, "compare": cm}, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
