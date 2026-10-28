"""Training losses of both stages (2026-10-28).

    python tools/plot_losses_stages.py OUT.png RUN_DIR NAME=stage2_history.csv [NAME=stage2_history.csv ...]

(a) Stage 1: every window's loss history (RUN_DIR/pinn_checkpoint_wNNNN.pkl, "loss_history"), drawn over
    its own time span [t_start, t_end] so the 40 windows line up along t: total and its parts (Einstein,
    gauge, initial = the hand-off from the previous window, Theta/Z penalty unweighted).
(b) Stage 2: held-out loss of u1 + correction over u1's own, against iteration (stage2.py's CSV), and
    (c) the same for the ADM part, mean(chi^2 (H^2 + |M|^2)), logged for every version.
Stage-1 losses are on each window's training points (core-heavy); stage-2 ones on points spread evenly
in the ball, so the two scales are not comparable; (b), (c) are ratios to u1 on the same points.
"""
import csv
import glob
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out, run = sys.argv[1], sys.argv[2]
hists = [s.rsplit("=", 1) for s in sys.argv[3:]]
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3de"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "axes.titleweight": "bold", "axes.titlesize": 9.5, "axes.titlelocation": "left"})
PARTS = [("loss_total", "total", INK, 1.5), ("loss_einstein", "Einstein (CCZ4 evolution)", COLORS[0], 1.1),
         ("loss_gauge", "gauge", COLORS[2], 1.1), ("loss_initial", "initial (hand-off)", COLORS[1], 1.1),
         ("loss_constraint", "Θ/Zⁱ (unweighted)", COLORS[3], 1.1)]

fig = plt.figure(figsize=(12, 6.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.38, wspace=0.18)
ax = fig.add_subplot(gs[0, :])
for k, f in enumerate(sorted(glob.glob(os.path.join(run, "pinn_checkpoint_w*.pkl")))):
    st = pickle.load(open(f, "rb"))
    H = st["loss_history"]
    steps = np.array([h["step"] for h in H], float)
    t = st["t_start"] + (st["t_end"] - st["t_start"]) * steps / max(steps.max(), 1)
    for key, lab, c, lw in PARTS:
        if key in H[0]:
            ax.plot(t, [h[key] for h in H], color=c, lw=lw, label=lab if k == 0 else None)
ax.set_yscale("log"); ax.set_xlim(0, 10)
ax.set_xlabel("t / M  (each window's 24 000 QN iterations drawn across its 0.25 M)")
ax.set_ylabel("training loss")
ax.set_title(f"(a) stage 1: the windowed run {os.path.basename(run.rstrip('/'))}, 40 windows")
ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17))

for j, (col, col_u1, title) in enumerate((("held_u", "held_u1", "(b) stage 2: held-out loss of u₁ + correction / u₁"),
                                          ("held_adm", "held_u1_adm", "(c) stage 2: its H/M part / u₁'s"))):
    ax = fig.add_subplot(gs[1, j])
    for (name, path), c in zip(hists, COLORS):
        R = list(csv.DictReader(open(path)))
        it = [int(r["iteration"]) for r in R]
        ax.plot(it, [float(r[col]) / float(r[col_u1]) for r in R], color=c, lw=1.6, label=name)
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_xlabel("QN iteration"); ax.set_title(title)
    ax.set_ylim(0, 1.15)
    if j == 0:
        ax.set_ylabel("ratio to the run alone")
        ax.legend(fontsize=8, loc="lower left")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
