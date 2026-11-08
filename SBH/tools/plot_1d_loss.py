"""Loss histories of sing_1d.py runs. (2026-11-06)

    python tools/plot_1d_loss.py OUT.png NAME=RUN_DIR [NAME=RUN_DIR ...]

RUN_DIR/history.csv (window, iteration, seconds, train, held, held_<part>; one row every --n_change iterations).
Left: window 0, train (solid) and held (dashed) loss against iteration. Middle: window 0's held parts Einstein
(solid) and end (dotted). Third: the whole run, train loss against the cumulative iteration. Fourth: the held
loss (the fresh-points loss at every check) against the cumulative iteration, with the best held value per window
as dots; non-finite values are left out. The first run is drawn black. (Fourth panel 2026-11-08: the user asks for
the loss against iteration of every run.)
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
fig, (a1, a2, a3, a4) = plt.subplots(1, 4, figsize=(18.5, 4.2), constrained_layout=True)
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
    cum, held, ok = np.array(cum), H["held"], np.isfinite(H["held"])
    a4.semilogy(cum[ok], held[ok], "-", color=c, lw=0.8, alpha=0.6)
    best = [(cum[(H["window"] == w) & ok][np.argmin(held[(H["window"] == w) & ok])], held[(H["window"] == w) & ok].min())
            for w in np.unique(H["window"]) if ok[H["window"] == w].any()]
    a4.semilogy(*zip(*best), "o", color=c, ms=3.5, label=f"{lab}: best held per window")
for a, t in ((a1, "window 0 (t = 0 - 0.25): loss"), (a2, "window 0: held Einstein (solid) and end (dotted) parts"),
             (a3, "all windows: train loss, cumulative iteration"),
             (a4, "all windows: held loss (lines) and its best per window (dots)")):
    a.set_title(t)
    a.set_xlabel("iteration")
    a.grid(True, alpha=0.25)
    a.legend(frameon=False, fontsize=8)
a1.set_ylabel("loss")
fig.savefig(out, dpi=130)
print("wrote", out)
