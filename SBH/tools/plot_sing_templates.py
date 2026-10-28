"""r-t maps of the 1D single-puncture solutions behind sing_pinn.py's templates, near the centre. (2026-10-28)

    python tools/plot_sing_templates.py [OUT_DIR]      (default plots/2026-10-28)

bl_rt.png:          the evolution from Brill-Lindquist data, t = 0..10 (reference_1d/sing_bl_m0.5_dense.npz)
trumpet_rt.png:     the same evolution continued, t = 10..20 (reference_1d/ref_single_m0.5_dr0.005_t20.npz);
                    the trumpet run starts from the t = 10 slice, so this is what it should follow
trumpet_drift.png:  that continuation minus the frozen t = 10 slice (the frozen trumpet template's error,
                    i.e. what the network has to supply inside the cutoff)
Same colour scale per field in the first two (chi and alpha on a log scale). Radius on a log axis, 0.0025 (the first grid point) to 1;
dashed lines at r_c = 0.2 and 2 r_c (template weight W = e^-1, e^-16).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "reference_1d")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "plots", "2026-10-28")
os.makedirs(OUT, exist_ok=True)
R_MAX, R_C = 1.0, 0.2

# (index in the 10 radial functions, label, signed?)
PANELS = [(4, r"$\chi$", False), (8, r"$\alpha$", False), (5, r"$\hat K$", True), (9, r"$\beta^r = B$", True),
          (1, r"$\tilde\gamma$: $b$ (coef. of $n_in_j$)", True), (3, r"$\tilde A$: $d$ (coef. of $n_in_j$)", True),
          (7, r"$\hat\Gamma^r = g$", True), (6, r"$\Theta$", True)]


def load(name, t_lo, t_hi):
    D = np.load(os.path.join(REF, name))
    r, t, U = D["r"], D["t"], D["U"]
    mr, mt = r <= R_MAX, (t >= t_lo - 1e-9) & (t <= t_hi + 1e-9)
    return r[mr], t[mt], U[mt][:, :, mr]


bl = load("sing_bl_m0.5_dense.npz", 0.0, 10.0)
tr = load("ref_single_m0.5_dr0.005_t20.npz", 10.0, 20.0)


def limits(i, signed, *sets):
    v = np.concatenate([U[:, i].ravel() for _, _, U in sets])
    lo, hi = np.percentile(v, [0.5, 99.5])
    if signed:
        m = max(abs(lo), abs(hi))
        return -m, m
    return lo, hi


def rt_figure(sets, title, fname, diff=False):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2), constrained_layout=True)
    for ax, (i, lab, signed) in zip(axes.ravel(), PANELS):
        r, t, U = sets[0]
        Z = U[:, i]
        if diff:
            Z = Z - Z[0][None]
            signed = True
            m = np.percentile(np.abs(Z), 99.5) or 1e-12
            lo, hi = -m, m
        else:
            lo, hi = limits(i, signed, bl, tr)
        if signed:
            norm, cmap = TwoSlopeNorm(0.0, lo, hi), "RdBu_r"
        else:                          # chi, alpha: log scale, so their zero at the puncture shows
            norm, cmap = LogNorm(max(lo, 1e-6), hi), "viridis"
        pc = ax.pcolormesh(r, t, Z, norm=norm, cmap=cmap, shading="auto", rasterized=True)
        ax.set_xscale("log")
        ax.set_xlim(r[0], R_MAX)
        for rr in (R_C, 2 * R_C):
            ax.axvline(rr, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(lab if not diff else lab + r"$(t) - (t{=}10)$", fontsize=11)
        ax.set_xlabel("r")
        ax.set_ylabel("t")
        fig.colorbar(pc, ax=ax, shrink=0.85)
    fig.suptitle(title, fontsize=13)
    fig.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close(fig)
    print("wrote", os.path.join(OUT, fname))


rt_figure([bl], "Brill-Lindquist puncture, m = 0.5: 1D evolution t = 0-10 (template --template bl)", "bl_rt.png")
rt_figure([tr], "The same evolution, t = 10-20: the trumpet (--template trumpet starts from the t = 10 slice)",
          "trumpet_rt.png")
rt_figure([tr], "Trumpet run: departure from the frozen t = 10 slice (supplied by the network inside the cutoff)",
          "trumpet_drift.png", diff=True)

# Radial profiles at a few times: BL t = 0, 0.5, 1, 2, 5, 10 and the trumpet slice (t = 10) on one plot per field.
fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
times = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
cols = plt.cm.viridis(np.linspace(0.0, 0.9, len(times)))
for ax, (i, lab, _) in zip(axes.ravel(), PANELS):
    r, t, U = bl
    for tt, c in zip(times, cols):
        j = int(np.argmin(np.abs(t - tt)))
        ax.plot(r, U[j, i], color=c, lw=1.6, label=f"t = {t[j]:g}")
    r2, t2, U2 = tr
    ax.plot(r2, U2[-1, i], color="k", lw=1.2, ls=":", label="t = 20")
    ax.set_xscale("log")
    if i in (4, 8):
        ax.set_yscale("log")
    ax.set_xlim(r[0], R_MAX)
    for rr in (R_C, 2 * R_C):
        ax.axvline(rr, color="0.5", lw=0.8, ls="--")
    ax.set_title(lab, fontsize=11)
    ax.set_xlabel("r")
    ax.grid(alpha=0.25)
axes[0, 0].legend(fontsize=9)
fig.suptitle("Radial profiles near the puncture (t = 0 is Brill-Lindquist; t = 10 is the trumpet template)", fontsize=13)
fig.savefig(os.path.join(OUT, "profiles.png"), dpi=130)
print("wrote", os.path.join(OUT, "profiles.png"))
