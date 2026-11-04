"""Several sing_1d.py runs against time: mass at r = 1, 3, 10 and field errors in two bands. (2026-11-04)

    python tools/plot_1d_runs_t.py OUT.png NAME=RUN_DIR [NAME=RUN_DIR ...]

RUN_DIR/report.npz from sing_1d.py (t, r, err (n_t, n_r, 10) signed f - f_ref, mass (n_t, n_r) m/M - 1). Top row:
m/M - 1 at r = 1, 3, 10 (symlog). Bottom row: rms of the alpha, chi and Khat errors in r = 0.2-1 (solid) and
3-10 (dashed). The first run is drawn black (the control).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out, runs = sys.argv[1], [s.rsplit("=", 1) for s in sys.argv[2:]]
COLS = ["#222222", "#d1495b", "#2a6fdb", "#1b9e77", "#e07b00", "#7a5195", "#8c8c8c", "#17a2b8", "#b5651d"]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.25})
fig, axs = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
for (name, d), c in zip(runs, COLS):
    z = np.load(os.path.join(d, "report.npz"))
    t, r = z["t"], z["r"]
    for ax, r0 in zip(axs[0], (1.0, 3.0, 10.0)):
        ax.plot(t, [np.interp(r0, r, m) for m in z["mass"]], "o-", color=c, ms=2.5, lw=1.3, label=name)
    for ax, (i, lab) in zip(axs[1], ((8, "α"), (4, "χ"), (5, "K̂"))):
        for lo, hi, ls in ((0.2, 1.0, "-"), (3.0, 10.0, "--")):
            s = (r >= lo) & (r < hi)
            ax.semilogy(t, [np.sqrt((e[s, i] ** 2).mean()) for e in z["err"]], ls, color=c, lw=1.2)
        ax.set_title(f"rms of the {lab} error: r = 0.2-1 (solid), 3-10 (dashed)")
for ax, r0 in zip(axs[0], (1.0, 3.0, 10.0)):
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_title(f"⟨m/M − 1⟩ at r = {r0:g} (symlog, linear below 1e-4)")
for ax in axs.ravel():
    ax.set_xlabel("t")
axs[0, 0].legend(fontsize=8)
fig.suptitle("sing_1d.py runs to t = 8, one change each against the control (black)", fontsize=11)
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
fig.savefig(out, dpi=130)
print("wrote", out)
