"""Mass error and loss histories of the sing_pinn.py runs against the w8_qn_con10 baseline. (2026-10-29)

    python tools/plot_sing_mass.py DATA_DIR [OUT_DIR]      (default plots/2026-10-29)

DATA_DIR holds, per run NAME, NAME_sph.npz (sing_pinn.py --eval_spheres) and NAME/sing_history.csv, plus the
baseline long_con10_w8.npz (tools/sphere_errors.py on long_con10's first 8 windows = w8_qn_con10, t <= 2).
Writes:
  sing_mass_profiles.png   <m/M - 1> on spheres vs r at t = 0, 1, 2 (sphere average, about the trained centre)
  sing_mass_time.png       <m/M - 1> vs t at r = 0.3, 1, 3, 10
  sing_losses.png          held-out loss and its parts vs iteration (all windows end to end), and |a(t1)|
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots",
                                                          "2026-10-29")
os.makedirs(OUT, exist_ok=True)
RUNS = [("w8_qn_con10 (baseline)", "long_con10_w8.npz", "#222222"),
        ("sing_bl", "sing_bl_sph.npz", "#2a6fdb"),
        ("sing_bl_pre", "sing_bl_pre_sph.npz", "#e07b00"),
        ("sing_tr (t = 10-12)", "sing_tr_sph.npz", "#1b9e77"),
        ("sing_bl_hard", "sing_bl_hard_sph.npz", "#d1495b")]
RUNS = [r for r in RUNS if os.path.exists(os.path.join(D, r[1]))]
Z = {n: np.load(os.path.join(D, f)) for n, f, _ in RUNS}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.25})


def symlog(ax, lin=1e-4):
    ax.set_yscale("symlog", linthresh=lin)
    ax.axhline(0, color="0.5", lw=0.8)


# ---- mass error vs r -------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True, constrained_layout=True)
for ax, tk in zip(axs, (0.0, 1.0, 2.0)):
    for n, _, c in RUNS:
        z = Z[n]
        k = np.flatnonzero(np.isclose(z["t"], tk))
        if k.size:                                         # runs still going lack the later times
            ax.plot(z["r"], z["mass_mean"][k[0]], color=c, lw=1.6, label=n)
    ax.set_xscale("log")
    symlog(ax)
    ax.axvline(1.0, color="0.6", lw=0.7, ls=":")          # horizon r ~ M/2 = 0.25 in isotropic, ~1 for trumpet
    ax.set_title(f"t = {tk:g}")
    ax.set_xlabel("r (coordinate distance from the centre)")
axs[0].set_ylabel("⟨m/M − 1⟩ on spheres (symlog, linear below 1e-4)")
axs[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Misner-Sharp mass error by radius: template + correction runs vs the network-only baseline", fontsize=11)
fig.savefig(os.path.join(OUT, "sing_mass_profiles.png"), dpi=130)
plt.close(fig)

# ---- mass error vs t -------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(17, 4.0), sharey=True, constrained_layout=True)
for ax, r0 in zip(axs, (0.3, 1.0, 3.0, 10.0)):
    for n, _, c in RUNS:
        z = Z[n]
        ax.plot(z["t"], [np.interp(r0, z["r"], m) for m in z["mass_mean"]], "o-", color=c, ms=3.5, lw=1.5, label=n)
    symlog(ax)
    ax.set_title(f"r = {r0:g}")
    ax.set_xlabel("t (run time)")
axs[0].set_ylabel("⟨m/M − 1⟩ on spheres")
axs[0].legend(fontsize=8)
fig.suptitle("Misner-Sharp mass error at window ends", fontsize=11)
fig.savefig(os.path.join(OUT, "sing_mass_time.png"), dpi=130)
plt.close(fig)

# ---- loss histories --------------------------------------------------------------------------------
PARTS = [("held_einstein", "Einstein"), ("held_gauge", "gauge"), ("held_end", "end (t1)"), ("held_bc", "boundary"),
         ("held_ic", "initial data"), ("held_constraint", "Θ, Z")]
PCOL = ["#2a6fdb", "#7a5195", "#e07b00", "#8c8c8c", "#d1495b", "#1b9e77"]
H = {}
for n, _, _ in RUNS[1:]:
    rows = list(csv.DictReader(open(os.path.join(D, n.split()[0], "sing_history.csv"))))
    keep = []                                              # a resumed window restarts: drop its first attempt
    for r in rows:
        while keep and keep[-1]["window"] == r["window"] and int(keep[-1]["iteration"]) >= int(r["iteration"]):
            keep.pop()
        keep.append(r)
    H[n] = keep
nr = len(RUNS) - 1
fig, axs = plt.subplots(2, max(nr, 3), figsize=(5.7 * max(nr, 3), 8), constrained_layout=True)
for ax, (n, _, c) in zip(axs[0], RUNS[1:]):
    rows = H[n]
    x = np.arange(1, len(rows) + 1) * 250
    ax.semilogy(x, [float(r["held"]) for r in rows], color="k", lw=1.8, label="held-out total")
    for (k, lab), pc in zip(PARTS, PCOL):
        ax.semilogy(x, [max(float(r[k]), 1e-12) for r in rows], color=pc, lw=1.0, label=lab)
    pre = [i for i, r in enumerate(rows) if r.get("phase") == "pre"]
    for i in pre:
        ax.axvspan(x[i] - 250, x[i], color="#e07b00", alpha=0.08, lw=0)
    inner = [i for i, r in enumerate(rows) if r.get("phase") == "inner"]      # --r_in runs: N_in-only blocks
    for i in inner:
        ax.axvspan(x[i] - 250, x[i], color="#2a6fdb", alpha=0.08, lw=0)
    for i in range(1, len(rows)):
        if rows[i]["window"] != rows[i - 1]["window"]:
            ax.axvline(x[i - 1], color="0.6", lw=0.6)
    ax.set_ylim(1e-8, 1)
    ax.set_title(n + (" (orange: IC + boundary-only phase)" if pre else "") + (" (blue: inner-network blocks)" if inner else ""))
    ax.set_xlabel("iteration (8 windows end to end; grey lines = window hand-offs)")
axs[0, 0].legend(fontsize=7.5, ncol=2, loc="upper right")
ax = axs[1, 0]
for n, _, c in RUNS[1:]:
    rows = H[n]
    ax.semilogy(np.arange(1, len(rows) + 1) * 250, [float(r["held"]) for r in rows], color=c, lw=1.4, label=n)
ax.set_title("held-out total, all runs")
ax.set_xlabel("iteration")
ax.legend(fontsize=8)
ax = axs[1, 1]
for n, _, c in RUNS[1:]:
    rows = H[n]
    W = sorted({int(r["window"]) for r in rows})
    for key, ls, lab in (("held_d0.01-0.05", "--", "0.01 ≤ d < 0.05"), ("held_d0.05-0.5", "-", "0.05 ≤ d < 0.5"),
                         ("held_d>0.5", ":", "d ≥ 0.5")):
        last = [float([r for r in rows if int(r["window"]) == w][-1][key]) for w in W]
        ax.semilogy(np.array(W) * 0.25 + 0.25, last, ls, color=c, marker="o", ms=3, lw=1.3,
                    label=f"{n}: {lab}" if n == RUNS[1][0] else None)
ax.set_title("Einstein + gauge loss by distance d to the puncture, at window ends\n(solid 0.05-0.5, dashed 0.01-0.05, dotted > 0.5)")
ax.set_xlabel("t (window end)")
ax = axs[1, 2]
for n, _, c in RUNS[1:]:
    z = Z[n]
    ax.plot(z["t"], np.linalg.norm(z["center"], axis=1), "o-", color=c, ms=3.5, lw=1.4, label=n)
ax.set_yscale("log")
ax.set_ylim(1e-6, 1e-1)
ax.set_title("|a(t)|: drift of the trained centre (true value 0)")
ax.set_xlabel("t (window end)")
ax.legend(fontsize=8)
for ax in axs[1, 3:]:
    ax.set_visible(False)
fig.suptitle("sing_pinn.py runs: held-out loss histories (8000 held-out interior points per window) and centre drift",
             fontsize=11)
fig.savefig(os.path.join(OUT, "sing_losses.png"), dpi=130)
plt.close(fig)
print("wrote", OUT)
