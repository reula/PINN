"""The radial point distribution of sing_1d.py, with and without --log_frac. (2026-11-06)

    python tools/plot_1d_radii.py OUT.png [LOG_FRAC [LOG_RMIN [N]]]        (defaults 0.25, 1e-3, 2000)

Re-draws radii() of sing_1d.py with its default clouds (inner 0.125 / sigma 0.05, core 0.375 / 0.5, focus 0.25 / 2,
the rest uniform in the 3D ball r <= R = 24; a share LOG_FRAC log-uniform on [LOG_RMIN, R] with the cloud shares
scaled by 1 - LOG_FRAC). Left: points per decade of r for N interior points, stacked by component, for the
--log_frac run; the default distribution as a black line. Right: share of the points below r, both distributions.
The shapes come from 400 x N draws, scaled to N.
Also OUT_hist.png: a plain histogram of ONE draw of N points (counts in 40 log-spaced bins, linear axis), default
and --log_frac side by side, stacked by cloud.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = sys.argv[1]
F = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
RMIN = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
N = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
R = 24.0
CLOUDS = [("inner Gaussian, sigma 0.05", 0.125, 0.05), ("core Gaussian, sigma 0.5", 0.375, 0.5),
          ("focus Gaussian, sigma 2", 0.25, 2.0)]
COLS = ["#d1495b", "#2a6fdb", "#1b9e77", "#999999", "#e8a33d"]
rng = np.random.default_rng(0)
M = 400 * N


def parts(f):
    n_log = int(f * M)
    out = []
    for name, frac, s in CLOUDS:
        m = int(frac * (M - n_log))
        out.append((name, s * np.linalg.norm(rng.normal(size=(m, 3)), axis=1)))
    out.append(("uniform in the ball r <= 24", R * rng.uniform(size=M - n_log - sum(len(p) for _, p in out)) ** (1 / 3)))
    if n_log:
        out.append((f"log-uniform on [{RMIN:g}, 24]", RMIN * (R / RMIN) ** rng.uniform(size=n_log)))
    return [(nm, np.minimum(p, R)) for nm, p in out]


edges = np.logspace(-4, np.log10(R), 61)
dec = np.diff(np.log10(edges))
P_log, P_def = parts(F), parts(0.0)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
base = np.zeros(len(edges) - 1)
for (nm, p), c in zip(P_log, COLS):
    h = np.histogram(p, edges)[0] / M * N / dec
    a1.fill_between(edges[:-1], base, base + h, step="post", color=c, alpha=0.8, lw=0, label=nm)
    base += h
hd = np.histogram(np.concatenate([p for _, p in P_def]), edges)[0] / M * N / dec
a1.step(edges[:-1], hd, where="post", color="#222222", lw=1.5, label="default (no --log_frac), total")
a1.set_xscale("log")
a1.set_yscale("log")
a1.set_ylim(1e-2, None)
a1.set_xlabel("r")
a1.set_ylabel(f"points per decade of r (of {N} interior points)")
a1.set_title(f"--log_frac {F:g} --log_rmin {RMIN:g}: stacked by cloud")
a1.legend(frameon=False, fontsize=8, loc="upper left")
a1.grid(True, alpha=0.25)
rs = np.logspace(-4, np.log10(R), 400)
for P, lab, ls, c in ((P_def, "default", "-", "#222222"), (P_log, f"--log_frac {F:g}", "--", "#e8a33d")):
    allr = np.sort(np.concatenate([p for _, p in P]))
    a2.loglog(rs, np.searchsorted(allr, rs) / M, ls, color=c, lw=1.8, label=lab)
for r0 in (1e-3, 1e-2, 5e-2):
    a2.axvline(r0, color="#999999", lw=0.8, ls=":")
a2.set_xlabel("r")
a2.set_ylabel("share of the points below r")
a2.set_ylim(1e-7, 1.5)
a2.set_title("cumulative share")
a2.legend(frameon=False)
a2.grid(True, alpha=0.25)
fig.suptitle("sing_1d.py interior points in r (t uniform in the window); the t1, IC and held sets are drawn the same way")
fig.savefig(out, dpi=130)
print("wrote", out)

# ---- plain histogram of one draw of N points --------------------------------------------------------
M = N
D_def, D_log = parts(0.0), parts(F)
be = np.logspace(-3.5, np.log10(R), 41)
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True, constrained_layout=True)
for ax, P, title in ((axs[0], D_def, "default"), (axs[1], D_log, f"--log_frac {F:g} --log_rmin {RMIN:g}")):
    names = [nm for nm, _ in P]
    cols = [COLS[[n for n, _, _ in CLOUDS].index(nm)] if nm in [n for n, _, _ in CLOUDS]
            else (COLS[3] if nm.startswith("uniform") else COLS[4]) for nm in names]
    ax.hist([p for _, p in P], bins=be, stacked=True, color=cols, label=names, edgecolor="white", lw=0.4)
    ax.set_xscale("log")
    n01 = sum((p < 0.01).sum() for _, p in P)
    n05 = sum((p < 0.05).sum() for _, p in P)
    ax.set_title(f"{title}\n{n01} points below r = 0.01, {n05} below r = 0.05 (of {N})")
    ax.set_xlabel("r")
    ax.grid(True, alpha=0.25, axis="y")
axs[0].set_ylabel("points per bin (40 bins, log-spaced in r)")
axs[1].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle(f"One draw of the {N} interior points of sing_1d.py, by radius")
o2 = out.replace(".png", "_hist.png")
fig.savefig(o2, dpi=130)
print("wrote", o2)
