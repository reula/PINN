"""The colleagues' diagnostic: signed Hamiltonian constraint and the mass it implies. (2026-11-06)

    python tools/plot_H_mass.py OUT.png T NAME=NPZ [NAME=NPZ ...]

NPZ from `sing_1d.py --eval_H RUN_DIR --eval_H_out NPZ` (r, t, H (n_t, n_r), Rareal, mass m/M - 1). At the window end
closest to T: left, R^2 H / 4 = dm/dR_areal (in 1D H is its own sphere mean), symlog; right, the measured m/M - 1
(solid) and m/M - 1 rebuilt as its value at r = 1 plus the integral of R^2 H / 4 dR / M from r = 1 (dotted).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out, T = sys.argv[1], float(sys.argv[2])
runs = [s.split("=", 1) for s in sys.argv[3:]]
COLS = ["#999999", "#222222", "#d1495b", "#2a6fdb", "#1b9e77"]
M = 0.5
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for (lab, f), c in zip(runs, COLS):
    z = np.load(f)
    k = int(np.argmin(np.abs(z["t"] - T)))
    r, H, Ra, m = z["r"], z["H"][k], z["Rareal"][k], z["mass"][k]
    dmdR = Ra ** 2 * H / 4
    a1.plot(r, dmdR, "-", color=c, lw=1.4, label=f"{lab} (t = {z['t'][k]:g})")
    i1 = np.searchsorted(r, 1.0)
    rec = m[i1] + np.concatenate([[0], np.cumsum(0.5 * (dmdR[i1 + 1:] + dmdR[i1:-1]) * np.diff(Ra[i1:]))]) / M
    a2.plot(r, m, "-", color=c, lw=1.6, label=f"{lab} measured")
    a2.plot(r[i1:], rec, ":", color=c, lw=2.0)
for a in (a1, a2):
    a.set_xscale("log")
    a.set_yscale("symlog", linthresh=1e-6)
    a.axhline(0, color="#999999", lw=0.6)
    a.set_xlabel("r")
    a.grid(True, alpha=0.25)
a1.set_title("dm/dR = R$^2$ H / 4 (signed; symlog, linear below 1e-6)")
a2.set_title("m/M - 1: measured (solid), rebuilt from H from r = 1 (dotted)")
a1.legend(frameon=False, fontsize=8)
a2.legend(frameon=False, fontsize=8)
fig.savefig(out, dpi=130)
print("wrote", out)
