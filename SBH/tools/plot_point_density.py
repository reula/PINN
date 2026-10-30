"""Collocation-point density averaged on spheres about the puncture: baseline vs sing_pinn.py. (2026-10-30)

    python tools/plot_point_density.py [OUT.png]          (default plots/2026-10-30/point_density.png)

Both samplers are mixtures of isotropic Gaussian clouds about the puncture plus a uniform ball r <= R = 24,
so the sphere average is the radial density itself. Curves: n(r) = sum_k N_k (2 pi s_k^2)^(-3/2)
exp(-r^2 / 2 s_k^2) + N_u / (4/3 pi R^3); dots: a Monte Carlo draw of the same mixture binned in shells,
as a check. Mixtures (window 0, puncture at the origin):
  baseline w8_qn_con10 (EEccz4_BH_puncture.py, long_con10's log): interior Nint 80000 = 60% Gaussian
    s = 2 (int_focus) + 40% uniform, plus core_points 48000 at s = 0.5; points with r < 0.02
    (min_puncture_dist) are redrawn. IC: 8000 = 60% s = 2 + 40% uniform, plus core_ic_points 4800 at s = 0.5.
  sing_pinn.py (defaults = sing_bl_hard): every set (interior 128000, IC 12800) is 12.5% s = 0.05 (inner),
    37.5% s = 0.5 (core), 25% s = 2 (focus), the rest uniform; no exclusion.
  log-uniform (a candidate, not implemented in either script): all points uniform in ln r for
    0.02 <= r <= 24, i.e. the same count per e-fold, n ~ 1 / r^3.
The time sampling (baseline: exponential in t) does not change the spatial density.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = 24.0
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots",
                                                         "2026-10-30", "point_density.png")


LOG_R = (0.02, R)


def mixture(n_main, n_core, focus_frac, inner_frac=0.0, core_frac=0.0, sing=False):
    """[(N, sigma)] Gaussians and the uniform count (a negative count = that many log-uniform points)."""
    if sing:
        n = n_main
        g = [(2 * int(f * n / 2), s) for f, s in ((inner_frac, 0.05), (core_frac, 0.5), (focus_frac, 2.0))]
        return g, n - sum(m for m, _ in g)
    nf = int(n_main * focus_frac)
    return [(nf, 2.0), (n_core, 0.5)], n_main - nf


SETS = {"baseline (w8_qn_con10)": {"interior": mixture(80000, 48000, 0.6), "IC (t = 0)": mixture(8000, 4800, 0.6)},
        "sing_pinn.py (sing_bl_hard)": {"interior": mixture(128000, 0, 0.25, 0.125, 0.375, True),
                                        "IC (t = 0)": mixture(12800, 0, 0.25, 0.125, 0.375, True)},
        "log-uniform in r, 0.02-24": {"interior": ([], -128000), "IC (t = 0)": ([], -12800)}}
RMIN_EXCL = {"baseline (w8_qn_con10)": 0.02, "sing_pinn.py (sing_bl_hard)": 0.0, "log-uniform in r, 0.02-24": 0.0}
COL = {"baseline (w8_qn_con10)": "#222222", "sing_pinn.py (sing_bl_hard)": "#d1495b",
       "log-uniform in r, 0.02-24": "#2a6fdb"}


def density(r, gauss, nu, rmin):
    if nu < 0:
        return np.where((r >= LOG_R[0]) & (r <= LOG_R[1]), -nu / (4 * np.pi * r ** 3 * np.log(LOG_R[1] / LOG_R[0])), 0.0)
    n = sum(N * (2 * np.pi * s ** 2) ** -1.5 * np.exp(-r ** 2 / (2 * s ** 2)) for N, s in gauss) + nu / (4 / 3 * np.pi * R ** 3)
    if rmin > 0:                                           # redrawn points: zero inside, total count unchanged
        rq = np.geomspace(1e-6, R, 20001)
        inside = np.trapezoid(4 * np.pi * rq ** 2 * np.where(rq < rmin, density(rq, gauss, nu, 0.0), 0.0), rq)
        n = np.where(r < rmin, 0.0, n * (sum(N for N, _ in gauss) + nu) / (sum(N for N, _ in gauss) + nu - inside))
    return n


def sample(gauss, nu, rmin, rng):
    if nu < 0:
        return LOG_R[0] * (LOG_R[1] / LOG_R[0]) ** rng.uniform(size=-nu)
    X = [s * rng.standard_normal((N, 3)) for N, s in gauss]
    u = rng.standard_normal((nu, 3))
    X.append(u / np.linalg.norm(u, axis=1, keepdims=True) * R * rng.uniform(size=(nu, 1)) ** (1 / 3))
    r = np.linalg.norm(np.concatenate(X), axis=1)
    r = np.minimum(r, R)
    return r[r >= rmin]


rng = np.random.default_rng(0)
r = np.geomspace(5e-3, R, 400)
edges = np.geomspace(5e-3, R, 41)
mid = np.sqrt(edges[1:] * edges[:-1])
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.25})
fig, axs = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
for j, part in enumerate(("interior", "IC (t = 0)")):
    for name, sets in SETS.items():
        g, nu = sets[part]
        n = density(r, g, nu, RMIN_EXCL[name])
        tot = sum(N for N, _ in g) + abs(nu)
        rs = sample(g, nu, RMIN_EXCL[name], rng)
        h = np.histogram(rs, edges)[0] * tot / rs.size     # redrawn points keep the total
        vol = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
        axs[0, j].loglog(r, n, color=COL[name], lw=1.7, label=f"{name}: {tot} points")
        axs[0, j].loglog(mid, np.where(h > 0, h / vol, np.nan), "o", color=COL[name], ms=3, alpha=0.6)
        axs[1, j].semilogx(r, 4 * np.pi * r ** 3 * n, color=COL[name], lw=1.7, label=name)
        axs[1, j].semilogx(mid, h / np.log(edges[1:] / edges[:-1]), "o", color=COL[name], ms=3, alpha=0.6)
    for ax in axs[:, j]:
        ax.axvspan(1.0, 10.0, color="#e07b00", alpha=0.08, lw=0)
        ax.set_xlabel("r (distance to the puncture)")
    axs[0, j].set_title(f"{part}: points per unit volume, averaged on spheres")
    axs[1, j].set_title(f"{part}: points per e-fold of r, dN/d ln r = 4π r³ n(r)")
    axs[0, j].legend(fontsize=8, loc="lower left")
    axs[0, j].set_ylim(1e-1, 1e9)
axs[0, 0].set_ylabel("n(r)")
axs[1, 0].set_ylabel("dN / d ln r")
fig.suptitle("Collocation-point density (curves: the sampling mixture; dots: one draw binned in shells). "
             "Shaded: 1 ≤ r ≤ 10, where sing_bl_hard loses mass. Log-uniform: a candidate", fontsize=10.5)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
for name, sets in SETS.items():
    g, nu = sets["interior"]
    rq = np.geomspace(1e-6, R, 40001)
    dN = 4 * np.pi * rq ** 2 * density(rq, g, nu, RMIN_EXCL[name])
    band = lambda lo, hi: int(np.trapezoid(np.where((rq >= lo) & (rq < hi), dN, 0), rq))
    print(f"{name:30s} interior points  r < 0.1: {band(0, 0.1):6d}  0.1-1: {band(0.1, 1):6d}  1-10: {band(1, 10):6d}"
          f"  10-24: {band(10, R + 1):6d}")
