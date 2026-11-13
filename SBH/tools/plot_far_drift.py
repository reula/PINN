"""The far-zone slicing drift of the 1D runs: error of alpha and Khat against r at successive times. (2026-11-13)

    python tools/plot_far_drift.py OUT.png LABEL=RUN_DIR [LABEL=RUN_DIR ...]

One column per run (report.npz: network - finite-difference reference). Top: alpha error(r) at t = 1 ... 8 (light to
dark); bottom: Khat error(r). The true solution is static for r >~ 1.4 t + 1 (nothing has arrived), so an error
there is produced locally. Companion of tools/plot_gauge_error_map.py.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = sys.argv[1]
runs = [s.split("=", 1) for s in sys.argv[2:]]
TS = (1, 2, 3, 4, 5, 6, 7, 8)
cm = plt.get_cmap("Blues")
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(2, len(runs), figsize=(5.3 * len(runs), 7), squeeze=False, sharex=True)
for k, (lab, d) in enumerate(runs):
    R = np.load(os.path.join(d, "report.npz"))
    r, t, e = R["r"], R["t"], R["err"]
    m = r >= 0.3
    for T in TS:
        j = int(np.argmin(abs(t - T)))
        c = cm(0.25 + 0.75 * T / 8)
        ax[0, k].plot(r[m], e[j, m, 8], c=c, lw=1.6, label=f"t = {T}")
        ax[1, k].plot(r[m], e[j, m, 5], c=c, lw=1.6)
    ax[0, k].set_title(f"{lab}: error of alpha"); ax[1, k].set_title(f"{lab}: error of Khat")
    ax[1, k].set_xlabel("r")
    for A, lim in ((ax[0, k], (-5e-4, 3e-3)), (ax[1, k], (-6e-4, 2e-4))):
        A.set_xscale("log"); A.set_ylim(*lim); A.axhline(0, c="0.6", lw=0.6); A.grid(alpha=0.3)
ax[0, 0].legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(out, dpi=110)
print("wrote", out)
