"""The three report figures of 2026-10-20 (SBH/README.md section 3-4 results).

    python tools/plot_report_figures.py DATA_DIR OUT_DIR

DATA_DIR holds, per run, <run>/loss_window*.csv (copied from runs/), mf_<run>.npz
(tools/mass_function.py) and heldout15.csv (tools/eval_heldout.py, the window-15 branches).

  loss_history.png    training loss over the whole run, window by window (single_a, drift_phys, drift_fixed)
  errors.png          (a) held-out loss of the window-15 branches vs the baseline's logged loss,
                      (b) rms Theta, (c) rms |Z^i| on the x-axis through the puncture
  mass_function.png   (a) m(r)/M - 1 of single_a at several times, (b) max |m/M - 1| on r in [0.5, 10] vs t
"""
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# validated categorical order (dataviz reference palette, light), fixed per entity
C = {"single_a": "#2a78d6", "drift_phys": "#eb6834", "drift_fixed": "#1baf7a"}
LS = {"single_a": "-", "drift_phys": "--", "drift_fixed": ":"}
LABEL = {"single_a": "single_a: moving χ bg + tracking",
         "drift_phys": "drift_phys: tracking only",
         "drift_fixed": "drift_fixed: fixed centre"}
BR = {"fork15_qn": ("#2a78d6", "-", "QN, resampled (baseline)"),
      "fork15_fixed": ("#eb6834", "--", "QN, fixed sample"),
      "fork15_dsg": ("#1baf7a", ":", "QN → DSGNAR")}
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3de"
W = 0.25          # window width
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.dpi": 130})


def loss_history(run):
    """(t, loss) with each window's steps mapped onto [t_start, t_end], and per-window final loss."""
    ts, ls, ends = [], [], []
    for f in sorted(glob.glob(os.path.join(D, run, "loss_window*.csv")),
                    key=lambda p: int(p.rsplit("window", 1)[1].split(".")[0])):
        w = int(f.rsplit("window", 1)[1].split(".")[0])
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        st = np.array([float(r["step"]) for r in rows]); lo = np.array([float(r["loss_total"]) for r in rows])
        ts.append(w * W + W * st / max(st.max(), 1.0)); ls.append(lo)
        ends.append(((w + 1) * W, lo[-1]))
    return ts, ls, np.array(ends)


# ------------------------------------------------------------------ 1. loss history
fig, ax = plt.subplots(figsize=(9, 4.2))
for run in ("single_a", "drift_phys", "drift_fixed"):
    ts, ls, ends = loss_history(run)
    for i, (t, l) in enumerate(zip(ts, ls)):
        ax.plot(t, l, color=C[run], lw=0.8, alpha=0.35)
    ax.plot(ends[:, 0], ends[:, 1], color=C[run], ls=LS[run], marker="o", ms=3.5, label=LABEL[run])
ax.set_yscale("log"); ax.set_xlabel("t  (each window's QN history mapped onto its 0.25-wide time slab)")
ax.set_ylabel("training loss (on the window's own samples)")
ax.set_title("Loss history: thin = within each window, markers = end of window", loc="left", color=INK)
ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "loss_history.png")); plt.close(fig)

# ------------------------------------------------------------------ 2. errors
H = list(csv.DictReader(open(os.path.join(D, "heldout15.csv"))))
fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
ax = axs[0]
for run, (col, ls_, lab) in BR.items():
    rr = sorted((int(r["window"]), float(r["t_end"]), float(r["loss_total"])) for r in H if r["run"] == run)
    ax.plot([x[1] for x in rr], [x[2] for x in rr], color=col, ls=ls_, marker="o", ms=4, label=lab)
_, _, ends = loss_history("fork15_qn")
sel = ends[:, 0] >= 3.75 - 1e-9
ax.plot(ends[sel, 0], ends[sel, 1], color=MUTED, lw=1.0, ls="-.", label="baseline, logged (own samples)")
ax.set_yscale("log"); ax.set_xlabel("t (window end)"); ax.set_ylabel("loss on fresh points (held out)")
ax.set_title("(a) Window-15 branches, held-out loss", loc="left", color=INK)
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=2)
mf = {run: np.load(os.path.join(D, f"mf_{run}.npz")) for run in ("single_a", "drift_phys", "drift_fixed")}
for k, (key, title) in enumerate((("theta", "(b) rms Θ on the x-axis cut"), ("z", "(c) rms |Zⁱ| on the x-axis cut"))):
    ax = axs[k + 1]
    for run, z in mf.items():
        v = np.sqrt(0.5 * (np.mean(z[f"{key}_plus"] ** 2, axis=1) + np.mean(z[f"{key}_minus"] ** 2, axis=1)))
        ax.plot(z["t"], v, color=C[run], ls=LS[run], label=LABEL[run])
    ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_title(title, loc="left", color=INK)
axs[1].set_ylabel("rms over r ∈ [0.05, 20], both directions")
axs[2].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "errors.png")); plt.close(fig)

# ------------------------------------------------------------------ 3. mass function
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
z = mf["single_a"]; M = float(z["M"]); r = z["r"]
times = [0.0, 2.0, 4.0, 7.0, 10.0]
ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]     # sequential blue: later = darker
ax = axs[0]
for t, col in zip(times, ramp):
    i = int(np.argmin(np.abs(z["t"] - t)))
    ax.plot(r, z["m_plus"][i] / M - 1, color=col, ls="-", label=f"t = {z['t'][i]:g}  (+x)")
    ax.plot(r, z["m_minus"][i] / M - 1, color=col, ls="--", lw=1.1)
ax.axhline(0, color=MUTED, lw=0.8)
ax.axvline(M / 2, color=MUTED, lw=0.8, ls=":")
ax.text(M / 2 * 1.08, -0.5, "horizon at t = 0\n(isotropic r = M/2)", color=MUTED, fontsize=8, va="bottom")
ax.set_xscale("log"); ax.set_xlabel("coordinate distance r from the tracked puncture")
ax.set_ylabel("m(r)/M − 1   (exact: 0)")
ax.set_ylim(-0.6, 0.6)
ax.set_title("(a) single_a: Misner–Sharp mass; solid +x, dashed −x", loc="left", color=INK)
ax.legend(fontsize=8, loc="lower right")
ax = axs[1]
for run, zz in mf.items():
    sel = (zz["r"] >= 0.5) & (zz["r"] <= 10.0)
    e = np.maximum(np.max(np.abs(zz["m_plus"][:, sel] / M - 1), axis=1),
                   np.max(np.abs(zz["m_minus"][:, sel] / M - 1), axis=1))
    ax.plot(zz["t"], e, color=C[run], ls=LS[run], label=LABEL[run])
ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel("max |m/M − 1| over r ∈ [0.5, 10]")
ax.set_title("(b) Mass-function error vs time", loc="left", color=INK); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "mass_function.png")); plt.close(fig)
print("wrote", OUT)
