"""Gauge-invariant accuracy of vacuum 1D runs: Misner-Sharp mass, (grad R)^2 and the horizon mass. (2026-11-13)

    python tools/plot_invariants.py OUT.png REF.npz LABEL=RUN_DIR [LABEL=RUN_DIR ...]

The user (13 Nov): if what we see is gauge, judge the accuracy with gauge-invariant geometrical quantities.
In spherical symmetry the areal radius R and the Misner-Sharp mass m = (R/2)(1 - (grad R)^2) are spacetime
scalars; in vacuum (Birkhoff) the solution is Schwarzschild iff m = M everywhere, so m/M - 1 is the complete
gauge-invariant error (it is also the relative error of the curvature, Psi_2 = -m/R^3). It is ill-conditioned far
out: (grad R)^2 = 1 - 2m/R, so an error eps of the metric gives m/M - 1 ~ (R/2M) eps. The well-conditioned
invariant is (grad R)^2 itself: its error is (2M/R)(m/M - 1).
Top row: m/M - 1 against the areal radius at t = 2, 4, 6, 8; bottom row: error of (grad R)^2. Prints the horizon
mass M_AH (area of the apparent horizon; = M in Schwarzschild for any spherical slicing) and the maxima.
REF.npz: the vacuum 1D reference (for the areal radius of the report grid and the horizon finder).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference_1d"))
from sph_reference_scalar import diagnostics                       # noqa: E402

out, fref = sys.argv[1:3]
runs = [s.split("=", 1) for s in sys.argv[3:]]
Z = np.load(fref)
M = float(Z["mass"]) if "mass" in Z else 0.5
pad = lambda u: np.concatenate([u, np.zeros((12 - u.shape[0], u.shape[1]))]) if u.shape[0] < 12 else u
cm = plt.get_cmap("Blues")
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(2, len(runs), figsize=(5.3 * len(runs), 7), squeeze=False, sharex=True)
for k, (lab, d) in enumerate(runs):
    R = np.load(os.path.join(d, "report.npz"))
    r, t, err, mass = R["r"], R["t"], R["err"], R["mass"]
    sel = r >= 0.2
    print(lab)
    for T in (2, 4, 6, 8):
        j = int(np.argmin(abs(t - T))); jr = int(np.argmin(abs(Z["t"] - T)))
        ref = np.stack([np.interp(r, Z["r"], Z["U"][jr, i]) for i in range(10)])
        Rar = r * np.sqrt(ref[0] / ref[4])
        gR2 = 2 * M / Rar * mass[j]                                  # error of (grad R)^2, sign of the mass error
        ah_n = diagnostics(r, pad(ref + err[j].T))[1]; ah_r = diagnostics(r, pad(ref))[1]
        c = cm(0.3 + 0.7 * T / 8)
        ax[0, k].plot(Rar[sel], mass[j, sel], c=c, lw=1.6, label=f"t = {T}")
        ax[1, k].plot(Rar[sel], gR2[sel], c=c, lw=1.6)
        b = lambda lo, hi, v: np.abs(v[(r >= lo) & (r < hi)]).max()
        print(f"  t = {T}: max |m/M - 1| in r = 0.2-1 / 1-5 / 5-12 / 12-20: " + " ".join(f"{b(lo, hi, mass[j]):.1e}" for lo, hi in ((0.2, 1), (1, 5), (5, 12), (12, 20.1)))
              + " | max |error of (grad R)^2| same bands: " + " ".join(f"{b(lo, hi, gR2):.1e}" for lo, hi in ((0.2, 1), (1, 5), (5, 12), (12, 20.1)))
              + f" | M_AH/M - 1 = {ah_n / M - 1:+.1e} (reference on the same grid {ah_r / M - 1:+.1e})")
    ax[0, k].set_title(f"{lab}: m/M - 1"); ax[1, k].set_title(f"{lab}: error of (grad R)^2 = (2M/R)(m/M - 1)")
    ax[1, k].set_xlabel("areal radius R")
    for A, lim in ((ax[0, k], 6e-3), (ax[1, k], 6e-4)):
        A.set_xscale("log"); A.set_ylim(-lim, lim); A.axhline(0, c="0.6", lw=0.6); A.grid(alpha=0.3)
ax[0, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(out, dpi=110)
print("wrote", out)
