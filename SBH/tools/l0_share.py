"""The l = 0 share of the sphere-sampled mass error at the peak of the outgoing deficit (2026-10-26).

    python tools/l0_share.py RUN.npz [RUN.npz ...] [--times 4 6 8 10] [--rmin 2]

RUN.npz from tools/sphere_errors.py, which samples the Misner-Sharp error e = m/M - 1 along `--ndir`
Fibonacci directions on each sphere and stores its mean and rms. For a quadrature that integrates the
spherical harmonics exactly enough, mean^2 / mean(e^2) is the fraction of the error's power on the sphere
in l = 0 (the rest is l >= 1). At each time the peak of the band is the most negative sphere mean over
r > --rmin.
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("npz", nargs="+")
ap.add_argument("--times", type=float, nargs="+", default=[4, 6, 8, 10])
ap.add_argument("--rmin", type=float, default=2.0)
a = ap.parse_args()
for path in a.npz:
    z = np.load(path)
    r = z["r"]
    print(path)
    for tt in a.times:
        k = int(np.argmin(np.abs(z["t"] - tt)))
        mean, rms = z["mass_mean"][k], z["mass_rms"][k]
        sel = np.where(r > a.rmin)[0]
        i = sel[np.argmin(mean[sel])]
        print(f"   t = {z['t'][k]:5.2f}: peak at r = {r[i]:5.2f}, mean {100 * mean[i]:+6.1f} %, "
              f"rms {100 * rms[i]:5.1f} %, l = 0 share {mean[i] ** 2 / rms[i] ** 2:4.0%}")
