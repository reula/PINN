"""Window hand-offs of several 1D runs compared, by radial band and field. (2026-11-13)

    python tools/plot_seams_compare.py OUT.png LABEL=SEAMS.npz [LABEL=SEAMS.npz ...]

SEAMS.npz from `sing_1d.py [the run's own flags] --eval_seams RUN_DIR --eval_seams_out SEAMS.npz`. One column per
field (alpha, shift B, Khat, chi), one row per radial band. Solid: rms over the band of the jump
f_new(t_k) - f_prev(t_k) at each hand-off t_k. Dotted, same colour: rms of the error the previous window carries
there against the 1D finite-difference reference, so a jump can be read against the error it sits on. Also prints
the table of the jumps and, per band, the regression slope of the jump on the previous window's error
(slope -1: the new window starts on the 1D reference, not on the previous window's end).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = sys.argv[1]
runs = [(s.split("=", 1)[0], np.load(s.split("=", 1)[1])) for s in sys.argv[2:]]
FIELDS = (("alpha", 8), ("shift B", 9), ("Khat", 5), ("chi", 4))
BANDS = ((0.225, 0.5), (0.5, 1.0), (1.0, 3.0), (3.0, 10.0))
COL = ("#2a78d6", "#eb6834", "#1baf7a", "#8a5cd0")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.edgecolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK})


def rms(x, m):
    return np.sqrt((x[:, m] ** 2).mean(1))


fig, ax = plt.subplots(len(BANDS), len(FIELDS), figsize=(16, 12), sharex=True, constrained_layout=True)
for i, (lo, hi) in enumerate(BANDS):
    for j, (name, f) in enumerate(FIELDS):
        a = ax[i, j]
        for c, (lab, z) in zip(COL, runs):
            m = (z["r"] >= lo) & (z["r"] < hi)
            a.plot(z["t"], rms(z["jump"][:, :, f], m), "-", color=c, lw=1.8, label=lab + ": jump")
            a.plot(z["t"], rms(z["err_prev"][:, :, f], m), ":", color=c, lw=1.4, label=lab + ": error carried")
        a.set_yscale("log")
        a.grid(True, color=GRID, lw=0.6)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.set_title("%s, r = %g to %g" % (name, lo, hi))
        if i == len(BANDS) - 1:
            a.set_xlabel("hand-off time t")
        if j == 0:
            a.set_ylabel("rms over the band")
ax[0, 0].legend(fontsize=7.5, frameon=False, ncol=1)
fig.suptitle("Window hand-offs: jump f_new - f_prev at each seam (solid) against the error the previous window "
             "carries (dotted)", fontsize=11)
fig.savefig(out, dpi=130)
print("wrote", out)

SEAMS = (7, 15, 23, 27, 30)
for lab, z in runs:
    print("==", lab, "| rms jump at t =", " ".join("%.2f" % z["t"][k] for k in SEAMS))
    for name, f in FIELDS:
        for lo, hi in BANDS:
            m = (z["r"] >= lo) & (z["r"] < hi)
            J, E = z["jump"][:, m, f], z["err_prev"][:, m, f]
            sl = [(J[k] * E[k]).sum() / (E[k] ** 2).sum() for k in SEAMS]
            print("  %-8s r %5.3f-%-4g  jump %s | slope on carried error %s" % (
                name, lo, hi, " ".join("%.1e" % np.sqrt((J[k] ** 2).mean()) for k in SEAMS),
                " ".join("%+.2f" % s for s in sl)))
