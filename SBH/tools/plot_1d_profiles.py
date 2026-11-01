"""Signed error profiles of sing_1d.py runs against the 1D reference, with the template weight. (2026-11-01)

    python tools/plot_1d_profiles.py OUT.png NAME=RUN_DIR [NAME=RUN_DIR ...] [--t T]

RUN_DIR/report.npz from sing_1d.py (r, t, err (n_t, n_r, 10), mass (n_t, n_r)). Panels: alpha, Khat, chi,
a = gt_yy signed errors and m/M - 1 vs r at the window end nearest --t (default: the first window end), plus
the held-out loss histories. W = exp(-(r/r_c)^w_power) of each run on a twin axis (grey, dashed).
"""
import ast
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

argv = sys.argv[1:]
T = None
if "--t" in argv:
    i = argv.index("--t")
    T = float(argv[i + 1])
    argv = argv[:i] + argv[i + 2:]
out, runs = argv[0], [s.rsplit("=", 1) for s in argv[1:]]
COLS = ["#d1495b", "#2a6fdb", "#1b9e77", "#e07b00", "#7a5195", "#222222"]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.25})
PANELS = [(8, "α"), (5, "K̂"), (4, "χ"), (0, "a = γ̃_yy")]
fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
axs = axs.ravel()
for (n, d), c in zip(runs, COLS):
    z = np.load(os.path.join(d, "report.npz"))
    cfg = ast.literal_eval(str(z["config"]))
    k = 0 if T is None else int(np.argmin(np.abs(z["t"] - T)))
    r, e, m, t = z["r"], z["err"][k], z["mass"][k], float(z["t"][k])
    for ax, (i, lab) in zip(axs, PANELS):
        ax.plot(r, e[:, i], color=c, lw=1.6, label=f"{n} (t = {t:g})")
        tw = ax.twinx()
        tw.plot(r, np.exp(-(r / cfg["r_c"]) ** cfg["w_power"]), color="0.5", lw=0.8, ls="--")
        tw.set_ylim(-0.05, 1.05)
        tw.grid(False)
        tw.set_yticks([])
    axs[4].plot(r, m, color=c, lw=1.6, label=n)
    rows = list(csv.DictReader(open(os.path.join(d, "history.csv"))))
    axs[5].semilogy([int(x["iteration"]) + int(x["window"]) * cfg["iters"] for x in rows], [float(x["held"]) for x in rows],
                    color=c, lw=1.3, label=n)
for ax, (i, lab) in zip(axs, PANELS):
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_title(f"{lab} − reference (signed); grey dashed: W")
    ax.set_xlabel("r")
axs[0].legend(fontsize=8, loc="lower left")
axs[4].axhline(0, color="0.5", lw=0.8)
axs[4].set_xscale("log")
axs[4].set_yscale("symlog", linthresh=1e-5)
axs[4].set_title("Misner-Sharp m/M − 1")
axs[4].set_xlabel("r")
axs[5].set_title("held-out loss")
axs[5].set_xlabel("iteration (windows end to end)")
axs[5].legend(fontsize=8)
fig.suptitle("sing_1d.py: the template + correction ansatz in spherical symmetry, against the fine 1D reference", fontsize=10.5)
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
fig.savefig(out, dpi=130)
print("wrote", out)
