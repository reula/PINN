"""The 1D twin, the 3D radial network and the baseline over t <= 8, errors vs t. (2026-11-03)

    python tools/plot_T8_compare.py DATA_DIR [OUT.png]      (default plots/2026-11-03/T8_compare.png)

DATA_DIR (SBH/eval_data) holds 1d_T8/report.npz and 1d_T8/history.csv (sing_1d.py), sing_radial_notpl_sph.npz and
sing_radial_notpl/sing_history.csv (sing_pinn.py --radial_net --no_template --eval_spheres), long_con10_w32.npz
(tools/sphere_errors.py on the baseline's first 32 windows). Top row: rms of alpha, Khat, chi against the 1D
reference in the bands 0.2-1 (solid) and 1-3 (dashed), and the held-out loss at the window ends. Bottom row:
<m/M - 1> at r = 1, 3, 10, and |delta (1 - |grad R|^2)| = 2M |m/M - 1| / R, the field-level quantity the mass error
at radius R stands for (the mass amplifies it by R / 2M, 10x at r = 10).
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
                                                         "2026-11-03", "T8_compare.png")
M = 0.5
z1 = np.load(os.path.join(D, "1d_T8", "report.npz"))
N = np.load(os.path.join(D, "sing_radial_notpl_sph.npz"))
B = np.load(os.path.join(D, "long_con10_w32.npz"))
FIELD_1D = {"alpha": 8, "Khat": 5, "chi": 4}


def band(r, v, lo, hi):
    return np.sqrt((v[(r >= lo) & (r < hi)] ** 2).mean())


def rms_t(q, lo, hi):
    """(t, value) of the band rms for the three runs: 1D (signed err in report.npz), 3D radial, baseline."""
    i = FIELD_1D[q]
    return [(z1["t"], [band(z1["r"], e[:, i], lo, hi) for e in z1["err"]]),
            (N["t"], [band(N["r"], v, lo, hi) for v in N[q + "_rms"]]),
            (B["t"], [band(B["r"], v, lo, hi) for v in B[q + "_rms"]])]


def mass_t(r0):
    return [(z1["t"], [np.interp(r0, z1["r"], m) for m in z1["mass"]]),
            (N["t"], [np.interp(r0, N["r"], m) for m in N["mass_mean"]]),
            (B["t"], [np.interp(r0, B["r"], m) for m in B["mass_mean"]])]


def held(path):
    last = {}
    for row in csv.DictReader(open(path)):
        last[int(row["window"])] = float(row["held"])
    w = sorted(last)
    return 0.25 * (np.array(w) + 1), [last[k] for k in w]


RUNS = [("1D twin (sing_1d.py)", "#e07b00"), ("3D radial net, no template", "#7a5195"), ("baseline (long_con10)", "#222222")]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.25})
fig, axs = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
for ax, q, lab in zip(axs[0], ("alpha", "Khat", "chi"), ("α", "K̂", "χ")):
    for lo, hi, ls in ((0.2, 1.0, "-"), (1.0, 3.0, "--")):
        for (t, v), (n, c) in zip(rms_t(q, lo, hi), RUNS):
            ax.semilogy(t, v, ls, color=c, lw=1.4, label=f"{n}" if ls == "-" else None)
    ax.set_title(f"rms of {lab} error: r in 0.2-1 (solid), 1-3 (dashed)")
    ax.set_xlabel("t")
axs[0, 0].legend(fontsize=8)
ax = axs[0, 3]
for (path, (n, c)) in ((os.path.join(D, "1d_T8", "history.csv"), RUNS[0]),
                       (os.path.join(D, "sing_radial_notpl", "sing_history.csv"), RUNS[1])):
    t, v = held(path)
    ax.semilogy(t, v, "o-", color=c, ms=3, lw=1.2, label=n)
ax.axhline(1e-4, color=RUNS[2][1], lw=1.0, ls=":", label="baseline (~1e-4, its own loss)")
ax.set_title("held-out loss at the window ends")
ax.set_xlabel("t (window end)")
ax.legend(fontsize=8)
for ax, r0 in zip(axs[1, :3], (1.0, 3.0, 10.0)):
    for (t, v), (n, c) in zip(mass_t(r0), RUNS):
        ax.plot(t, v, "o-", color=c, ms=3, lw=1.3, label=n)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_title(f"⟨m/M − 1⟩ at r = {r0:g}")
    ax.set_xlabel("t")
ax = axs[1, 3]
for r0, ls in ((1.0, ":"), (3.0, "--"), (10.0, "-")):
    for (t, v), (n, c) in zip(mass_t(r0), RUNS):
        ax.semilogy(t, 2 * M * np.abs(np.asarray(v)) / r0, ls, color=c, lw=1.2)
ax.set_title("2M |m/M − 1| / r at r = 1 (:), 3 (--), 10 (—)\nthe field-level error behind the mass error")
ax.set_xlabel("t")
fig.suptitle("To t = 8: the 1D twin (same ansatz, scheme, loss), the 3D radial network and the baseline", fontsize=11)
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
