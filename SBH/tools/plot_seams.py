"""Window hand-offs of a finished 1D run: how much each seam adds, and whether the seams add up to the error. (2026-11-07)

    python tools/plot_seams.py SEAMS.npz OUT.png

SEAMS.npz from `sing_1d.py --eval_seams RUN_DIR --eval_seams_out SEAMS.npz` (31 seams for 32 windows). Top row, for
alpha, chi and Khat: the signed jump f_new - f_prev summed over all seams (the error the hand-offs alone would leave
if every window evolved exactly) against the run's own error f - f_ref at the last seam (t = 7.75), on log r.
Bottom row: band rms of the jump at each seam against t (r = 0.2-1, 1-3, 3-10), the previous window's error there
(dotted), so the jump per seam can be compared with the error it sits on.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

z = np.load(sys.argv[1])
r, t = z["r"], z["t"]
FIELDS = (("alpha", 8), ("chi", 4), ("Khat", 5))
BANDS = (((0.2, 1.0), "#2a78d6"), ((1.0, 3.0), "#eb6834"), ((3.0, 10.0), "#1baf7a"))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.edgecolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK})
fig, ax = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
for j, (name, i) in enumerate(FIELDS):
    a = ax[0, j]
    S = z["jump"][:, :, i].sum(0)
    E = z["err_prev"][-1, :, i]
    a.plot(r, E, "-", color="#2a78d6", lw=1.8, label="error at t = %g" % t[-1])
    a.plot(r, S, "-", color="#eb6834", lw=1.8, label="sum of the %d seam jumps" % len(t))
    a.axhline(0, color=INK2, lw=0.6)
    a.set_xscale("log")
    a.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    a.set_xlabel("r")
    a.set_title(f"{name}: summed hand-off jumps vs the run's error")
    a.grid(True, color=GRID, lw=0.6)
    a.legend(frameon=False, fontsize=8)
    b = ax[1, j]
    for (lo, hi), c in BANDS:
        m = (r >= lo) & (r < hi)
        b.plot(t, np.sqrt((z["jump"][:, m, i] ** 2).mean(1)), "-o", color=c, ms=3, lw=1.3, label=f"jump, r = {lo:g}-{hi:g}")
        b.plot(t, np.sqrt((z["err_prev"][:, m, i] ** 2).mean(1)), ":", color=c, lw=1.4)
    b.set_yscale("log")
    b.set_xlabel("seam time t")
    b.set_title(f"{name}: rms jump per seam (solid) and error there (dotted)")
    b.grid(True, color=GRID, lw=0.6)
    b.legend(frameon=False, fontsize=8)
fig.savefig(sys.argv[2], dpi=130)
print("wrote", sys.argv[2])
