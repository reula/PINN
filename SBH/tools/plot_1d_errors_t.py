"""All the errors of several sing_1d.py runs against time. (2026-11-10, the user: "plot the mass error and all the errors")

    python tools/plot_1d_errors_t.py OUT_PREFIX LABEL=RUN_DIR [LABEL=RUN_DIR ...]      (no '=' inside a label)

Writes OUT_PREFIX_mass.png and OUT_PREFIX_fields.png from RUN_DIR/report.npz (t, r, err (n_t, n_r, 10) signed f - f_ref for the 10
radial functions a, b, c, d, chi, Khat, Theta, g, alpha, B; mass (n_t, n_r) m/M - 1).
mass: worst |m/M - 1| over r >= 0.2 at each t, m/M - 1 at r = 1, 3, 10, and the profile m/M - 1 (r) at t = 4 and 8.
fields: rms error of each of the 10 functions against t, in r = 0.2-1 (solid) and 3-10 (dashed). The first run is black.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

pre, runs = sys.argv[1], [s.split("=", 1) for s in sys.argv[2:]]
COLS = ["#222222", "#e8a33d", "#d1495b", "#2a6fdb", "#1b9e77", "#7a5195", "#8c8c8c", "#17a2b8"]
NAMES = ["a", "b", "c", "d", "χ", "K̂", "Θ", "g (Γ̂ʳ)", "α", "B (βʳ)"]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.grid": True, "grid.alpha": 0.25})
Z = [(n, np.load(os.path.join(d, "report.npz"))) for n, d in runs]

fig, axs = plt.subplots(2, 3, figsize=(17, 8.6), constrained_layout=True)
for (n, z), c in zip(Z, COLS):
    t, r, m = z["t"], z["r"], z["mass"]
    s = r >= 0.2
    axs[0, 0].semilogy(t, np.abs(m[:, s]).max(axis=1), "-", color=c, lw=1.4, label=n)
    for ax, r0 in zip((axs[0, 1], axs[0, 2], axs[1, 0]), (1.0, 3.0, 10.0)):
        ax.plot(t, [np.interp(r0, r, row) for row in m], "-", color=c, lw=1.3)
    for ax, tt in zip((axs[1, 1], axs[1, 2]), (4.0, 8.0)):
        ax.semilogx(r[s], m[np.argmin(abs(t - tt))][s], "-", color=c, lw=1.2)
axs[0, 0].set_title("worst |m/M − 1| over r ≥ 0.2, at each t")
for ax, r0 in zip((axs[0, 1], axs[0, 2], axs[1, 0]), (1.0, 3.0, 10.0)):
    ax.set_yscale("symlog", linthresh=1e-4); ax.axhline(0, color="0.5", lw=0.8)
    ax.set_title(f"m/M − 1 at r = {r0:g} (symlog, linear below 1e-4)"); ax.set_xlabel("t")
for ax, tt in zip((axs[1, 1], axs[1, 2]), (4, 8)):
    ax.set_yscale("symlog", linthresh=1e-4); ax.axhline(0, color="0.5", lw=0.8)
    ax.set_title(f"m/M − 1 against r at t = {tt}"); ax.set_xlabel("r")
axs[0, 0].set_xlabel("t"); axs[0, 0].legend(fontsize=7.5)
fig.suptitle("Misner–Sharp mass error against the 1D finite-difference reference", fontsize=11)
fig.savefig(pre + "_mass.png", dpi=125); print("wrote", pre + "_mass.png")

fig, axs = plt.subplots(2, 5, figsize=(21, 7.6), constrained_layout=True)
for (n, z), c in zip(Z, COLS):
    t, r, E = z["t"], z["r"], z["err"]
    for i, ax in enumerate(axs.ravel()):
        for lo, hi, ls in ((0.2, 1.0, "-"), (3.0, 10.0, "--")):
            s = (r >= lo) & (r < hi)
            ax.semilogy(t, np.sqrt((E[:, s, i] ** 2).mean(axis=(1))) , ls, color=c, lw=1.2, label=n if (i == 0 and ls == "-") else None)
for i, ax in enumerate(axs.ravel()):
    ax.set_title(f"rms error of {NAMES[i]}"); ax.set_xlabel("t")
axs[0, 0].legend(fontsize=7)
fig.suptitle("rms error of the 10 radial functions against t: r = 0.2–1 (solid), 3–10 (dashed)", fontsize=11)
fig.savefig(pre + "_fields.png", dpi=110); print("wrote", pre + "_fields.png")
