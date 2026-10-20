"""Figures for docs/exec_2026-10-20 (executive summary of the SBH work, 17-20 Oct 2026).

    python tools/plot_exec_summary.py DATA_DIR OUT_DIR

DATA_DIR: <run>/loss_window*.csv for single_a, drift_*, fork15_*; mf_<run>.npz
(tools/mass_function.py); heldout15.csv (tools/eval_heldout.py); probe.txt (tools/puncture_probe.py).
"""
import csv
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3de"
RUN = {"single_a": (BLUE, "-", "moving χ bg + tracking (single_a)"),
       "drift_phys": (ORANGE, "--", "tracking only"),
       "drift_fixed": (AQUA, ":", "fixed centre")}
BR = {"fork15_qn": (BLUE, "-", "QN, resampled (baseline)"),
      "fork15_fixed": (ORANGE, "--", "QN, one fixed sample"),
      "fork15_dsg": (AQUA, ":", "QN → DSGNAR")}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.dpi": 150, "axes.titleweight": "bold",
                     "axes.titlesize": 9.5})
W = 0.25
H = list(csv.DictReader(open(os.path.join(D, "heldout15.csv"))))
mf = {r: np.load(os.path.join(D, f"mf_{r}.npz")) for r in RUN}


def probe(path):
    out, cur = {}, None
    for l in open(path):
        if l.startswith("=="):
            cur = l.split()[1]; out[cur] = []
        elif re.match(r"\s+\d", l):
            v = [float(x) for x in l.split()]
            out[cur].append(v)
    return {k: np.array(v) for k, v in out.items()}


PB = probe(os.path.join(D, "probe.txt"))


def mass_err(z):
    sel = (z["r"] >= 0.5) & (z["r"] <= 10.0)
    M = float(z["M"])
    return np.maximum(np.max(np.abs(z["m_plus"][:, sel] / M - 1), 1), np.max(np.abs(z["m_minus"][:, sel] / M - 1), 1))


# ---------------------------------------------------------------- overview row
fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.6))
ax = axs[0]
for run, (c, ls, lab) in BR.items():
    rr = sorted((float(r["t_end"]), float(r["loss_total"])) for r in H if r["run"] == run)
    ax.plot(*zip(*rr), color=c, ls=ls, marker="o", ms=3, label=lab)
ax.set_yscale("log"); ax.set_xlabel("t (M)"); ax.set_ylabel("loss on fresh points")
ax.set_title("(a) Optimisers at equal GPU time", loc="left"); ax.legend(fontsize=7, loc="center right")
ax = axs[1]
for run, (c, ls, lab) in RUN.items():
    ax.plot(mf[run]["t"], mass_err(mf[run]), color=c, ls=ls, label=lab)
ax.set_yscale("log"); ax.set_xlabel("t (M)"); ax.set_ylabel("max |m/M − 1|, r ∈ [0.5, 10]")
ax.set_title("(b) Misner–Sharp mass error", loc="left"); ax.legend(fontsize=7, loc="lower right")
ax = axs[2]
for run, (c, ls, lab) in RUN.items():
    p = PB[run]
    ax.plot(p[:, 0], np.hypot(p[:, 1], p[:, 2]), color=c, ls=ls, label=lab)
ax.set_yscale("log"); ax.set_xlabel("t (M)"); ax.set_ylabel("|β| at the origin")
ax.set_title("(c) Spurious push on the puncture", loc="left"); ax.legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "overview.png")); plt.close(fig)

# ---------------------------------------------------------------- loss by optimizer (fork15_dsg vs baseline)
OPT = {"bfgs": (BLUE, "QN (SSBroyden)"), "dsgnar": (AQUA, "DSGNAR"), "adam": (YELLOW, "Adam")}
fig, ax = plt.subplots(figsize=(10.5, 2.3))
for f in sorted(glob.glob(os.path.join(D, "fork15_qn", "loss_window*.csv"))):
    w = int(f.rsplit("window", 1)[1].split(".")[0])
    rows = list(csv.DictReader(open(f)))
    st = np.array([float(r["step"]) for r in rows]); lo = np.array([float(r["loss_total"]) for r in rows])
    ax.plot(w * W + W * st / st.max(), lo, color=MUTED, lw=0.8, alpha=0.6,
            label="baseline QN (48 restarts)" if w == 15 else None)
seen = set()
for f in sorted(glob.glob(os.path.join(D, "fork15_dsg", "loss_window*.csv"))):
    w = int(f.rsplit("window", 1)[1].split(".")[0])
    rows = list(csv.DictReader(open(f)))
    st = np.array([float(r["step"]) for r in rows]); lo = np.array([float(r["loss_total"]) for r in rows])
    opt = [r["optimizer"] for r in rows]
    # x proportional to WALL time: in this branch the 24 QN restarts took ~52 % of a window
    # (10:40 min) and the 12 DSGNAR iterations ~48 % (586 s incl. compilation), job 1180624
    qs = np.array([x for x, o in zip(st, opt) if o == "bfgs"]); q_max = qs.max() if qs.size else 1.0
    nd = max(1, sum(o == "dsgnar" for o in opt))
    frac, kd = [], 0
    for x_, o in zip(st, opt):
        if o == "dsgnar":
            kd += 1; frac.append(0.52 + 0.48 * kd / nd)
        else:
            frac.append(0.52 * x_ / q_max)
    tt = w * W + W * np.array(frac)
    for k in range(1, len(rows)):
        o = opt[k]
        col, name = OPT.get(o, (MUTED, o))
        ax.plot(tt[k - 1:k + 1], lo[k - 1:k + 1], color=col, lw=1.8,
                label=f"QN → DSGNAR branch: {name} part" if o not in seen else None)
        seen.add(o)
ax.set_yscale("log"); ax.set_xlabel("t (M); within each 0.25 M slab, x is proportional to wall time")
ax.set_ylabel("training loss (own samples)")
ax.set_title("Loss by optimiser, windows 16–25: per window, 24 QN restarts (blue, 52 % of the time) then 12 DSGNAR its (green, 48 %)", loc="left")
ax.legend(fontsize=7, loc="upper right", ncol=3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "loss_by_optimizer.png")); plt.close(fig)

# ---------------------------------------------------------------- mass profiles
fig, axs = plt.subplots(1, 2, figsize=(10.5, 2.45))
z = mf["single_a"]; M = float(z["M"]); r = z["r"]
ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
ax = axs[0]
for t, col in zip([0.0, 2.0, 4.0, 7.0, 10.0], ramp):
    i = int(np.argmin(np.abs(z["t"] - t)))
    ax.plot(r, z["m_plus"][i] / M - 1, color=col, label=f"t = {z['t'][i]:g} M")
    ax.plot(r, z["m_minus"][i] / M - 1, color=col, ls="--", lw=1.0)
ax.axhline(0, color=MUTED, lw=0.8); ax.set_xscale("log"); ax.set_ylim(-0.4, 0.45); ax.set_xlim(0.1, 20)
ax.set_xlabel("r from the tracked puncture"); ax.set_ylabel("m(r)/M − 1")
ax.set_title("(a) single_a, along +x (solid) and −x (dashed)", loc="left"); ax.legend(fontsize=7, ncol=2, loc="lower left")
ax = axs[1]
for run, (c, ls, lab) in RUN.items():
    zz = mf[run]
    th = np.sqrt(0.5 * (np.mean(zz["z_plus"] ** 2, 1) + np.mean(zz["z_minus"] ** 2, 1)))
    ax.plot(zz["t"], th, color=c, ls=ls, label=lab)
ax.set_yscale("log"); ax.set_xlabel("t (M)"); ax.set_ylabel("rms |Zⁱ| on the x-axis cut")
ax.set_title("(b) Momentum-constraint monitor", loc="left"); ax.legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "mass_and_constraints.png")); plt.close(fig)
print("wrote", OUT)
