"""Loss against error during one window's training (the user's experiment, 2026-11-06).

    python tools/plot_loss_vs_error.py RUN_DIR OUT.png

RUN_DIR holds loss_orders.npz and history.csv from `sing_1d.py --loss_orders N` (one window). At every loss check
(every --n_change iterations) the network's errors at the window end against the 1D reference were recorded;
snapshots (saved parameters) are where the held loss first fell another order of magnitude below its value
before training, L0. Panels: (a) errors against the held-out loss, (b) the same against the training loss of the
last draw, (c) both losses against the iteration. Errors: rms over r in [0.05, 24] (the report radii) of alpha,
chi and Khat, and max |m/M - 1| over r >= 0.2. The dashed grey line has slope 1/2 (error ~ sqrt(loss)).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

run, out = sys.argv[1], sys.argv[2]
z = np.load(f"{run}/loss_orders.npz")
h = np.genfromtxt(f"{run}/history.csv", delimiter=",", names=True)
it, held, snap, order = z["iteration"], z["held"], z["snapshot"].astype(bool), z["order"]
train = np.interp(it, h["iteration"], h["train"])
r = z["r"]
err = z["err"]                                          # (n, n_r, 10), signed f - f_ref at the window end
rms = lambda i: np.sqrt((err[:, :, i] ** 2).mean(axis=1))
mass = np.abs(z["mass"][:, r >= 0.2]).max(axis=1)
SER = [("alpha rms", rms(8), "#2a78d6"), ("chi rms", rms(4), "#eb6834"), ("Khat rms", rms(5), "#1baf7a"),
       ("max |m/M - 1|", mass, "#eda100")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
for ax, x, xl in ((a1, held, "held-out loss"), (a2, train, "training loss (last draw)")):
    for lab, y, c in SER:
        ax.plot(x, y, "-", color=c, lw=0.6, alpha=0.35)
        ax.plot(x, y, "o", color=c, ms=3.5, mec="white", mew=0.6, label=lab)
        ax.plot(x[snap], y[snap], "o", color=c, ms=9, mec="white", mew=2.0)
    lo, hi = x.min(), x.max()
    xs = np.array([lo, hi])
    y0 = np.median(rms(8)) / np.sqrt(np.median(x))
    ax.plot(xs, y0 * np.sqrt(xs), "--", color="#8a8984", lw=1.0, label="slope 1/2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xl)
    ax.set_ylabel("error at the window end (t = %g)" % float(z["err"].shape[0] and 0.25))
    ax.grid(True, color=GRID, lw=0.6)
for i in np.where(snap)[0]:
    a1.annotate(f"L0 x 1e-{order[i]}\nit {it[i]}", (held[i], rms(8)[i]), textcoords="offset points",
                xytext=(6, 8), fontsize=7.5, color=INK2)
a1.set_title("(a) errors against the held-out loss (large dots: snapshots)")
a2.set_title("(b) errors against the training loss")
a1.legend(frameon=False, fontsize=8, loc="lower right")
a3.plot(h["iteration"], h["held"], "-o", color="#2a78d6", ms=3, lw=1.2, label="held-out")
a3.plot(h["iteration"], h["train"], "-o", color="#eb6834", ms=3, lw=1.2, label="training (last draw)")
a3.plot(it[snap], held[snap], "o", color="#2a78d6", ms=9, mec="white", mew=2.0)
L0 = float(z["L0"][0])
for k in range(3, 7):
    a3.axhline(L0 * 10.0 ** -k, color=GRID, lw=1.0, ls=":")
    a3.text(h["iteration"].max(), L0 * 10.0 ** -k, f" L0 x 1e-{k}", va="center", fontsize=7.5, color=INK2)
a3.set_yscale("log")
a3.set_xlabel("iteration")
a3.set_ylabel("loss")
a3.set_title(f"(c) losses (L0 = {L0:.2e}, the held loss before training)")
a3.grid(True, color=GRID, lw=0.6)
a3.legend(frameon=False, fontsize=8)
fig.savefig(out, dpi=130)
print("wrote", out)
for i in range(len(it)):
    print(f"it {it[i]:6d} held {held[i]:.2e} train {train[i]:.2e} | alpha {rms(8)[i]:.2e} chi {rms(4)[i]:.2e} "
          f"Khat {rms(5)[i]:.2e} mass {mass[i]:.2e}" + ("  <- snapshot" if snap[i] else ""))
