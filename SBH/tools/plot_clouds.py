"""Picture of the interior collocation points around a single puncture: the default random
focused/core clouds vs --symmetric_focus (point-reflected pairs), with the production sampler.

    JAX_ENABLE_X64=1 python tools/plot_clouds.py OUT.png
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_defs import load
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
base = ["--n_stars", "1", "--BL_masses", "0.5", "--BL_momenta", "0", "--BL_momenta_y", "0",
        "--core_points", "48000", "--core_ic_points", "4800", "--core_radius", "0.5"]
INK, MUTED, BLUE, ORANGE = "#1f1f1e", "#6b6a65", "#2a78d6", "#eb6834"
fig, axs = plt.subplots(1, 3, figsize=(14, 4.9), gridspec_kw={"width_ratios": [1, 1, 1.05]})
stats = {}
for k, (flags, title) in enumerate([([], "default: random cloud"), (["--symmetric_focus"], "--symmetric_focus: mirrored pairs")]):
    ns = load(base + flags, outdir=tempfile.mkdtemp(prefix="cl_"))
    jax, jnp = ns["jax"], ns["jnp"]
    X, _ = ns["generate_interior"](int(ns["Nint"]), jax.random.PRNGKey(11), 0.0, 0.25, centers=[(0.0, 0.0, 0.0)])
    X = np.asarray(X); x, y, z = X[:, 1], X[:, 2], X[:, 3]
    r = np.sqrt(x * x + y * y + z * z)
    inner = r < 1.0
    cen = np.array([x[inner].mean(), y[inner].mean(), z[inner].mean()])
    stats[title] = (len(X), int(inner.sum()), cen)
    ax = axs[k]
    sl = (np.abs(z) < 0.05) & (np.abs(x) < 1.5) & (np.abs(y) < 1.5)
    ax.scatter(x[sl], y[sl], s=1.2, color=BLUE if k == 0 else ORANGE, alpha=0.45, lw=0)
    ax.plot(0, 0, marker="+", ms=14, mew=1.6, color=INK)
    ax.plot(cen[0], cen[1], marker="o", ms=7, mfc="none", mew=1.6, color=INK)
    ax.annotate(f"centroid of the {inner.sum():,} points with r < 1:\n({cen[0]:+.1e}, {cen[1]:+.1e}, {cen[2]:+.1e})",
                xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, color=INK,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=MUTED, lw=0.6))
    ax.set_aspect("equal"); ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"{title}\nslab |z| < 0.05 ({sl.sum():,} of {len(X):,} points); + puncture, o centroid",
                 loc="left", fontsize=9, color=INK)
    if k == 0:
        rr = r
ax = axs[2]
bins = np.geomspace(0.01, 25, 60)
h, e = np.histogram(rr, bins=bins)
ax.stairs(h / np.diff(e), e, color=BLUE, lw=1.6)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("r (distance from the puncture)")
ax.set_ylabel("points per unit r"); ax.axvline(0.5, color=MUTED, ls=":", lw=1)
ax.text(0.53, ax.get_ylim()[0] * 3, "core radius 0.5", color=MUTED, fontsize=8)
ax.set_title(f"radial distribution of all {len(rr):,} interior points\n(focus cloud + 48k core cloud + uniform bulk to R = 24)",
             loc="left", fontsize=9, color=INK)
for a in axs:
    a.grid(True, color="#e4e3de", lw=0.6)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(out, dpi=140)
for t, (n, ni, c) in stats.items():
    print(f"{t}: {n} points, {ni} within r < 1, centroid {c}")
