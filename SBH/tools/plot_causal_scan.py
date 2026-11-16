"""Scan of the causal-weight strength eps of sing_1d.py (--causal_eps): the first windows of several runs. (2026-11-16)

    python tools/plot_causal_scan.py OUT.png N_WINDOWS OFF_LABEL=RUN_DIR[,OFF_LABEL=RUN_DIR...] EPS=RUN_DIR [EPS=RUN_DIR ...]

RUN_DIR/history.csv from sing_1d.py. The third argument lists the runs without causal weights (drawn black and
grey), the rest the runs with weights, each named by its eps (a number, e.g. 1e7), coloured light to dark with
eps. Only the first N_WINDOWS windows of every run are used, so full runs and short ones compare. Panels:
  1. held loss (fresh points, never weighted: the like-for-like loss) against the cumulative iteration;
  2. train loss (for the runs with weights this is the weighted loss the optimizer sees);
  3. the causal weight of the window-end term t = t1 (column causal_w_end) on the batch of each segment:
     0 = the end of the window is not being trained, 1 = it is trained as without weights;
  4. held loss at the end of each window against eps, with the runs without weights as horizontal lines.
Also prints the table of panel 4.
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out, nw = sys.argv[1], int(sys.argv[2])
offs = [s.split("=", 1) for s in sys.argv[3].split(",")]
scan = sorted(((float(s.split("=", 1)[0]), s.split("=", 1)[1]) for s in sys.argv[4:]))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OFFC = (INK, "#8d8c88", "#b9b8b3")
RAMP = plt.get_cmap("YlOrRd")(np.linspace(0.3, 1.0, len(scan)))  # one direction, light to dark with eps
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.edgecolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK})


def load(d):
    rows = [r for r in csv.DictReader(open(f"{d}/history.csv")) if int(r["window"]) < nw]
    H = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    per = int(H["iteration"].max())
    H["cum"] = H["window"] * per + H["iteration"]
    return H, per


fig, ax = plt.subplots(1, 4, figsize=(19, 4.6), constrained_layout=True)
ends = {}
for (lab, d), c in zip(offs, OFFC):
    H, per = load(d)
    ax[0].semilogy(H["cum"], H["held"], "-", color=c, lw=2, label=lab)
    ax[1].semilogy(H["cum"], H["train"], "-", color=c, lw=2, label=lab)
    ends[lab] = [H["held"][H["window"] == w][-1] for w in range(nw)]
for (eps, d), c in zip(scan, RAMP):
    H, per = load(d)
    lab = "eps = %.0e" % eps
    ax[0].semilogy(H["cum"], H["held"], "-", color=c, lw=2, label=lab)
    ax[1].semilogy(H["cum"], H["train"], "-", color=c, lw=2, label=lab)
    ax[2].plot(H["cum"], np.minimum(H["causal_w_end"], 1.0), "-", color=c, lw=2, label=lab)
    ends[eps] = [H["held"][H["window"] == w][-1] for w in range(nw)]
MK = ("o", "s", "^", "D")
for w in range(nw):
    ax[3].plot([e for e, _ in scan], [ends[e][w] for e, _ in scan], MK[w % 4], color=INK, ms=7,
               mfc=INK if w == 0 else "white", mew=1.5, ls="none", label="with weights, window %d" % w)
    for (lab, _), c in zip(offs, OFFC):
        ax[3].axhline(ends[lab][w], color=c, lw=1.6, ls="-" if w == 0 else "--", label="%s, window %d" % (lab, w))
ax[3].set_xscale("log")
ax[3].set_yscale("log")
for a in ax[:3]:
    for w in range(1, nw):
        a.axvline(w * per, color=INK2, lw=0.8, ls=":")
    a.set_xlabel("iteration (windows one after another)")
for a, t in zip(ax, ("held loss (never weighted)", "train loss (weighted where eps > 0)",
                     "causal weight of the window-end term", "held loss at the end of each window")):
    a.set_title(t)
    a.grid(True, color=GRID, lw=0.6)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
ax[0].set_ylabel("loss")
ax[2].set_ylim(-0.03, 1.03)
ax[3].set_xlabel("eps")
ax[0].legend(frameon=False, fontsize=8, ncol=2)
ax[2].legend(frameon=False, fontsize=8, loc="center left")
ax[3].legend(frameon=False, fontsize=7.5)
fig.suptitle("Causal weights w_j = exp(-eps (L_1 + ... + L_(j-1))) inside each window: scan of eps, first %d windows"
             % nw, fontsize=11)
fig.savefig(out, dpi=130)
print("wrote", out)
for k, v in ends.items():
    print(("%-12s" % k if isinstance(k, str) else "eps %-8.0e" % k), " ".join("%.2e" % x for x in v))
