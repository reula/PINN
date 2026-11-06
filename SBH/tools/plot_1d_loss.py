"""Loss histories of sing_1d.py runs. (2026-11-06)

    python tools/plot_1d_loss.py OUT.png NAME=RUN_DIR [NAME=RUN_DIR ...]

RUN_DIR/history.csv (window, iteration, seconds, train, held, held_<part>; one row every --n_change iterations).
Left: window 0, train (solid) and held (dashed) loss against iteration. Middle: window 0's held parts Einstein
(solid) and end (dotted). Right: the whole run, train loss against the cumulative iteration, window edges as
faint lines. The first run is drawn black.
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = sys.argv[1]
runs = [s.split("=", 1) for s in sys.argv[2:]]
COLS = ["#222222", "#e8a33d", "#2a6fdb", "#d1495b", "#1b9e77"]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5})
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
for (lab, d), c in zip(runs, COLS):
    rows = list(csv.DictReader(open(f"{d}/history.csv")))
    H = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    w0 = H["window"] == 0
    a1.semilogy(H["iteration"][w0], H["train"][w0], "-", color=c, lw=1.6, label=f"{lab} train")
    a1.semilogy(H["iteration"][w0], H["held"][w0], "--", color=c, lw=1.2, label=f"{lab} held")
    a2.semilogy(H["iteration"][w0], H["held_einstein"][w0], "-", color=c, lw=1.6, label=f"{lab} Einstein")
    a2.semilogy(H["iteration"][w0], H["held_end"][w0], ":", color=c, lw=1.6, label=f"{lab} end")
    # cumulative iteration: windows one after another
    cum, off, last_w, last_it = [], 0, None, 0
    for w, it in zip(H["window"], H["iteration"]):
        if last_w is not None and w != last_w:
            off += last_it
        cum.append(off + it)
        last_w, last_it = w, it
    a3.semilogy(cum, H["train"], "-", color=c, lw=1.0, label=lab)
for a, t in ((a1, "window 0 (t = 0 - 0.25): loss"), (a2, "window 0: held Einstein (solid) and end (dotted) parts"),
             (a3, "all windows: train loss, cumulative iteration")):
    a.set_title(t)
    a.set_xlabel("iteration")
    a.grid(True, alpha=0.25)
    a.legend(frameon=False, fontsize=8)
a1.set_ylabel("loss")
fig.savefig(out, dpi=130)
print("wrote", out)
