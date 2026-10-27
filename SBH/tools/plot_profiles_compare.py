"""Sphere-averaged mass error and lapse error against radius at a few times, for several runs (2026-10-27).

    python tools/plot_profiles_compare.py OUT.png NAME=RUN.npz [NAME=RUN.npz ...] [--times 6 8 10] [--linear_r]

RUN.npz from tools/sphere_errors.py (with or without --stage2). Top row: <m/M - 1> on spheres; bottom row:
rms over the sphere of alpha - alpha_ref. One column per time.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("runs", nargs="+")
ap.add_argument("--times", type=float, nargs="+", default=[6, 8, 10])
ap.add_argument("--linear_r", action="store_true", help="radius axis linear instead of logarithmic")
a = ap.parse_args()
runs = [s.rsplit("=", 1) for s in a.runs]
COLORS = ["#6b6a65", "#2a78d6", "#eb6834", "#1baf7a"]
STYLES = ["-", "--", "-", ":"]
MUTED = "#6b6a65"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e4e3de", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "axes.titleweight": "bold", "axes.titlesize": 9.5, "axes.titlelocation": "left"})
fig, axs = plt.subplots(2, len(a.times), figsize=(3.9 * len(a.times), 6.2), sharex=True, squeeze=False)
for (name, path), c, ls in zip(runs, COLORS, STYLES):
    z = np.load(path)
    r = z["r"]
    for j, tt in enumerate(a.times):
        k = int(np.argmin(np.abs(z["t"] - tt)))
        axs[0, j].plot(r, 100 * z["mass_mean"][k], color=c, ls=ls, lw=1.6, label=name)
        axs[1, j].plot(r, z["alpha_rms"][k], color=c, ls=ls, lw=1.6, label=name)
for j, tt in enumerate(a.times):
    axs[0, j].set_title(f"mass error, t = {tt:g}")
    axs[0, j].axhline(0, color=MUTED, lw=0.8)
    axs[1, j].set_title(f"lapse error, t = {tt:g}")
    axs[1, j].set_yscale("log")
    axs[1, j].set_xlabel("r / M")
    for i in range(2):
        if a.linear_r:
            axs[i, j].set_xlim(0, 20)
        else:
            axs[i, j].set_xscale("log")
            axs[i, j].set_xlim(0.2, 20)
axs[0, 0].set_ylabel("⟨m/M − 1⟩ on spheres (%)")
axs[1, 0].set_ylabel("rms of α − α_ref on spheres")
axs[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(a.out, dpi=140)
print("wrote", a.out)
