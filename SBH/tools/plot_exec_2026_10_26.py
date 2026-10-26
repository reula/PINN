"""Figures for docs/exec_2026-10-26 (physics summary: constraint penalties to t = 10, 2026-10-26).

    python tools/plot_exec_2026_10_26.py DATA_DIR OUT_DIR

DATA_DIR: cmpL_<run>_errors.csv, cmpL_<run>_puncture.csv (reference_1d/compare_pinn.py) and
se_<run>.npz (tools/sphere_errors.py: mass error averaged over spheres) for single_a, long_con10,
long_bia3. The second figure of the summary is tools/plot_sphere_errors.py's radius-time map.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3de"
RUN = {"single_a": ("#eb6834", "--", "no penalty, default settings"),
       "long_con10": ("#2a78d6", "-", "Θ/Zⁱ penalty (W = 10)"),
       "long_bia3": ("#1baf7a", ":", "Bianchi penalty (W = 1e-3)")}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.dpi": 150, "axes.titleweight": "bold",
                     "axes.titlesize": 9.5, "axes.titlelocation": "left"})


def rows(path, field=None):
    out = [r for r in csv.DictReader(open(path)) if r["side"] in ("start", "end")]
    if field:
        out = [r for r in out if r["field"] == field]
    return out


def mass(run):
    z = np.load(os.path.join(D, f"se_{run}.npz"))
    return z["r"], z["t"], z["mass_mean"]


M = {r: mass(r) for r in RUN}

# Figure 1: the headline -- lapse at the puncture, the hole's mass, the mass profile at t = 10
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.3))
ax = axs[0]
for i, (run, (c, ls, lab)) in enumerate(RUN.items()):
    P = rows(os.path.join(D, f"cmpL_{run}_puncture.csv"))
    t = [float(p["t"]) for p in P]
    if i == 0:
        ax.plot(t, [float(p["alpha_ref"]) for p in P], color=INK, lw=2.6, alpha=0.35, label="1D reference")
    ax.plot(t, [float(p["alpha_pinn"]) for p in P], color=c, ls=ls, label=lab)
ax.set_xlabel("t / M"); ax.set_ylabel("α at r = 0.03")
ax.set_title("(a) lapse at the puncture"); ax.legend(fontsize=7.5)
ax = axs[1]
for run, (c, ls, lab) in RUN.items():
    r, t, m = M[run]
    i = int(np.argmin(np.abs(r - 1.0)))
    ax.plot(t, 100 * m[:, i], color=c, ls=ls, label=lab)
ax.axhline(0, color=MUTED, lw=0.8)
ax.axhspan(7, 8, color=MUTED, alpha=0.18, lw=0)
ax.text(0.3, 8.6, "10–19 Oct final settings at t = 10: +7–8 %", fontsize=7.2, color=MUTED)
ax.set_xlabel("t / M"); ax.set_ylabel("m/M − 1  (%)")
ax.set_title("(b) the hole's mass (r = 1, sphere average)")
ax = axs[2]
for run, (c, ls, lab) in RUN.items():
    r, t, m = M[run]
    ax.plot(r, 100 * m[-1], color=c, ls=ls, label=lab)
ax.axhline(0, color=MUTED, lw=0.8); ax.set_xscale("log"); ax.set_xlim(0.2, 20)
ax.set_xlabel("r / M"); ax.set_ylabel("m/M − 1  (%)")
ax.set_title("(c) mass error at t = 10, sphere average")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "overview.png")); plt.close(fig)

print("wrote", OUT)
