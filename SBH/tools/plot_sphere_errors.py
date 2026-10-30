"""Radius-time maps of the sphere-averaged errors (2026-10-26).

    python tools/plot_sphere_errors.py OUT.png NAME=RUN.npz [NAME=RUN.npz ...] [--linear_r]

RUN.npz from tools/sphere_errors.py. One column per run; rows: the sphere-averaged Misner-Sharp
error <m/M - 1> (signed), and the rms over each sphere of alpha, Khat, chi - the 1D reference and of Theta
(Khat and chi rows added 2026-10-30).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import numpy as np

linear_r = "--linear_r" in sys.argv          # radius axis linear instead of logarithmic
argv = [v for v in sys.argv[1:] if v != "--linear_r"]
out, runs = argv[0], [s.rsplit("=", 1) for s in argv[1:]]
MUTED = "#6b6a65"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titleweight": "bold", "axes.titlesize": 9.5, "axes.titlelocation": "left"})
ROWS = [("mass_mean", "⟨m/M − 1⟩ on spheres", "RdBu_r", SymLogNorm(linthresh=3e-3, vmin=-0.3, vmax=0.3, base=10)),
        ("alpha_rms", "rms of α − α_ref on spheres", "magma_r", LogNorm(1e-4, 1e-1)),
        ("Khat_rms", "rms of K̂ − K̂_ref on spheres", "magma_r", LogNorm(1e-4, 1e-1)),
        ("chi_rms", "rms of χ − χ_ref on spheres", "magma_r", LogNorm(1e-5, 1e-2)),
        ("Theta_rms", "rms of Θ on spheres", "magma_r", LogNorm(1e-5, 1e0))]
fig, axs = plt.subplots(len(ROWS), len(runs), figsize=(3.6 * len(runs) + 0.9, 2.7 * len(ROWS)),
                        sharex=True, sharey=True, squeeze=False)
for j, (name, path) in enumerate(runs):
    z = np.load(path)
    t, r = z["t"], z["r"]
    for i, (key, lab, cmap, norm) in enumerate(ROWS):
        ax = axs[i, j]
        v = z[key].T
        if key != "mass_mean":
            v = np.clip(v, norm.vmin, None)
        pc = ax.pcolormesh(t, r, v, cmap=cmap, norm=norm, shading="nearest", rasterized=True)
        if not linear_r:
            ax.set_yscale("log")
        if i == 0:
            ax.set_title(name)
        if j == 0:
            ax.set_ylabel("r / M")
        if i == len(ROWS) - 1:
            ax.set_xlabel("t / M")
        if j == len(runs) - 1:
            cb = fig.colorbar(pc, ax=axs[i, :], fraction=0.025, pad=0.015)
            cb.set_label(lab, fontsize=8.5)
            if key == "mass_mean":
                cb.set_ticks([-0.3, -0.1, -0.03, -0.01, 0, 0.01, 0.03, 0.1, 0.3])
                cb.set_ticklabels(["−30 %", "−10 %", "−3 %", "−1 %", "0", "+1 %", "+3 %", "+10 %", "+30 %"])
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
