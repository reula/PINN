"""Signed error profiles at the template seam against the cutoff functions. (2026-10-31)

    python tools/plot_seam_profile.py DATA_DIR [OUT.png]      (default plots/2026-10-31/seam_profile.png)

Sphere-averaged (signed) error of alpha, Khat, chi against the 1D reference at t = 0.25 and 0.5, for the
baseline (long_con10_w8.npz), sing_bl_hard and sing_bl_inner75 (sing_pinn.py --eval_spheres), with the
template's weight W = exp(-(r/r_c)^4), 1 - W and rho = r^2/(r^2 + r_c^2) for the sweep's r_c = 0.1, 0.2, 0.3 drawn on a twin axis (grey).
If the error follows (1 - W) or W' the seam structure itself is the source; if it is smooth across the
seam, the error only lives there.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots",
                                                         "2026-11-01", "seam_profile.png")
RUNS = [("w8_qn_con10 (baseline)", "long_con10_w8.npz", "#222222"), ("sing_bl_hard", "sing_bl_hard_sph.npz", "#d1495b"),
        ("sing_bl_inner (r_in 0.75, w0)", "sing_bl_inner75_sph.npz", "#2a6fdb"),
        ("r_c = 0.1 (w0)", "sing_w0_rc01_sph.npz", "#7a5195"), ("r_c = 0.3 (w0)", "sing_w0_rc03_sph.npz", "#1b9e77"),
        ("rho_power 4 (w0)", "sing_w0_rho4_sph.npz", "#e07b00"),
        ("lifted radial network (w0)", "sing_w0_radial_sph.npz", "#17a2b8")]
Z = {n: np.load(os.path.join(D, f)) for n, f, _ in RUNS if os.path.exists(os.path.join(D, f))}
R_C = 0.2
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.25})
fig, axs = plt.subplots(3, 2, figsize=(13, 10.5), sharex=True, constrained_layout=True)
for i, (q, lab) in enumerate((("alpha_mean", "α"), ("Khat_mean", "K̂"), ("chi_mean", "χ"))):
    for j, t in enumerate((0.25, 0.5)):
        ax = axs[i, j]
        for n, _, c in RUNS:
            if n not in Z:
                continue
            z = Z[n]
            k = np.flatnonzero(np.isclose(z["t"], t))
            if not k.size:
                continue
            ax.plot(z["r"], z[q][k[0]], color=c, lw=1.6, label=n)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlim(0.05, 20)
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_title(f"sphere mean of {lab} − {lab}_ref, t = {t:g}")
        r = np.geomspace(0.05, 20, 400)
        tw = ax.twinx()
        for rc, ls in ((0.1, "-."), (0.2, "--"), (0.3, ":")):                   # W for the three r_c of the sweep
            tw.plot(r, np.exp(-(r / rc) ** 4), color="0.45", lw=0.9, ls=ls, label=f"W, r_c = {rc}")
        tw.set_ylim(-0.05, 1.05)
        tw.grid(False)
        if i == 0 and j == 1:
            tw.legend(fontsize=8, loc="center right")
        if i == 2:
            ax.set_xlabel("r")
axs[0, 0].legend(fontsize=8, loc="lower left")
fig.suptitle("Signed error profiles across the template seam (symlog, linear below 1e-4); grey: W for r_c = 0.1 (-.), 0.2 (--), 0.3 (:)",
             fontsize=10.5)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
