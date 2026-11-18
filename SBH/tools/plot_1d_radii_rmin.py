"""The radial point distribution of sing_1d.py runs with an exact core (--r_in / --r_min). (2026-11-18)

    python tools/plot_1d_radii_rmin.py OUT.png [R_MIN [N]]        (defaults 0.225, 20000)

Re-draws radii() of sing_1d.py with its default clouds (inner 0.125 / sigma 0.05, core 0.375 / 0.5, focus 0.25 / 2,
the rest uniform in the 3D ball r <= R = 24) and the --r_min rule: every radius below R_MIN is redrawn log-uniform
on [R_MIN, R]. Left: points per decade of r for N points, stacked by where the point comes from (the three clouds
that reach outside R_MIN, and the redrawn points); the black line is the distribution as coded, before the
redraw. Right: points per radial band. The shapes come from 200 x N draws, scaled to N. The interior, end-slice,
hand-off and held sets are all drawn this way. Also prints the table.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = sys.argv[1]
RMIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.225
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
R = 24.0
CLOUDS = [("inner Gaussian, sigma 0.05", 0.125, 0.05), ("core Gaussian, sigma 0.5", 0.375, 0.5),
          ("focus Gaussian, sigma 2", 0.25, 2.0), ("uniform in the ball r <= 24", 0.25, None)]
COL = {"inner Gaussian, sigma 0.05": "#d1495b", "core Gaussian, sigma 0.5": "#2a6fdb", "focus Gaussian, sigma 2": "#1b9e77",
       "uniform in the ball r <= 24": "#8d8c88", "redrawn": "#e8a33d"}
BANDS = ((RMIN, 0.5), (0.5, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, R))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
rng = np.random.default_rng(0)
M = 200 * N
raw = []
for name, frac, s in CLOUDS:
    m = int(frac * M)
    r = s * np.linalg.norm(rng.normal(size=(m, 3)), axis=1) if s else R * rng.uniform(size=m) ** (1 / 3)
    raw.append((name, np.minimum(r, R)))
n_in = sum(int((r < RMIN).sum()) for _, r in raw)
kept = [(name, r[r >= RMIN]) for name, r in raw if (r >= RMIN).mean() > 1e-3]      # a cloud with nothing outside is left out
lab_re = "redrawn log-uniform outside the core (%.0f%% of the points)" % (100 * n_in / M)
kept.append((lab_re, RMIN * (R / RMIN) ** rng.uniform(size=n_in)))

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.edgecolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.6, 1]})
edges = np.logspace(-2.3, np.log10(R), 71)
dec = np.diff(np.log10(edges))
base = np.zeros(len(edges) - 1)
for name, r in kept:
    h = np.histogram(r, edges)[0] / M * N / dec
    a1.fill_between(edges[:-1], base, base + h, step="post", color=COL.get(name, COL["redrawn"]), lw=0, label=name)
    a1.step(edges[:-1], base + h, where="post", color="white", lw=0.8)
    base += h
hd = np.histogram(np.concatenate([r for _, r in raw]), edges)[0] / M * N / dec
a1.step(edges[:-1], hd, where="post", color=INK, lw=1.4, label="as coded, before the redraw")
a1.axvline(RMIN, color=INK2, lw=1, ls="--")
a1.text(RMIN * 0.93, base.max() * 0.50, "exact core, r <= %g:\nno points" % RMIN, ha="right", va="top", fontsize=8, color=INK2)
if RMIN > 0.1:
    a1.text(0.085, hd[np.searchsorted(edges, 0.085)] * 1.25, "inner cloud, sigma 0.05:\nall of it inside, redrawn", ha="center",
            va="bottom", fontsize=8, color=INK2)
a1.set_xscale("log")
a1.set_xlim(edges[0], R)
a1.set_ylim(0, base.max() * 1.08)
a1.set_xlabel("r")
a1.set_ylabel("points per decade of r (of %d)" % N)
a1.set_title("Where the points are: stacked by origin")
a1.legend(frameon=False, fontsize=8, loc="upper left")
tot = np.zeros(len(BANDS))
left = np.zeros(len(BANDS))
names = ["%g to %g" % b for b in BANDS]
for name, r in kept:
    h = np.array([((r >= lo) & (r < hi + 1e-9)).sum() for lo, hi in BANDS]) / M * N
    a2.barh(names, h, left=left, color=COL.get(name, COL["redrawn"]), edgecolor="white", lw=1.2, height=0.62)
    left += h
    tot += h
for i, v in enumerate(tot):
    a2.text(v + tot.max() * 0.015, i, "%d (%.0f%%)" % (round(v), 100 * v / N), va="center", fontsize=9)
a2.invert_yaxis()
a2.set_xlim(0, tot.max() * 1.25)
a2.set_xlabel("points per draw (of %d)" % N)
a2.set_ylabel("radial band, r")
a2.set_title("Points per radial band")
for a in (a1, a2):
    a.grid(True, color=GRID, lw=0.6, axis="y" if a is a1 else "x")
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.suptitle("sing_1d.py with an exact core (--r_min %g): radial distribution of one draw of %d points "
             "(interior, end-slice, hand-off and held sets alike)" % (RMIN, N), fontsize=11)
fig.savefig(out, dpi=130)
print("wrote", out)
print("redrawn: %.1f%% of all points" % (100 * n_in / M))
for (lo, hi), v in zip(BANDS, tot):
    print("r = %g to %g: %6.0f points (%.1f%%)" % (lo, hi, v, 100 * v / N))
