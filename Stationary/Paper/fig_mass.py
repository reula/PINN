"""Fig. mass: the mass shift M0 - R0 and the quadrupole moment M2 of the spectral approximations
against the datum's amplitude S2 (Paper/hub_mass.sh; Sec. IV.A, tab:mass), with the network
masses where a network was trained.  Reads <run>/gh_moments.json and config.json.
    .venv/bin/python Paper/fig_mass.py [out.png]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "Paper/figs/mass_s2.png"
RUNS = ["ms_s2h_w2", "ab_none_w2", "ms_s2t_cfg", "ms_s2d_w2", "ms_s2p_cfg"]
NET = {"ms_s2h_w2", "ab_none_w2", "ms_s2d_w2"}           # the others hold only a reference
R0 = 1 / np.sqrt(3)
REF, NETC, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#898781", "#e1e0d9"

s2, M0, M2, M0n = [], [], [], []
for r in RUNS:
    g = json.load(open(f"runs/{r}/gh_moments.json"))
    s2.append(json.load(open(f"runs/{r}/config.json"))["lam_bc_S2"])
    M0.append(g["ref"]["M_ernst"][0]); M2.append(g["ref"]["M_ernst"][2])
    M0n.append(g["net"]["M_ernst"][0] if r in NET else np.nan)
s2, M0, M2, M0n = map(np.array, (s2, M0, M2, M0n))
dM = M0 - R0
A = np.stack([s2 ** k for k in (2, 3, 4)], 1)
a, b, c = np.linalg.lstsq(A, dM, rcond=None)[0]
print(f"[fig_mass] fit M0 - R0 = {a:.3f} S2^2 {b:+.3f} S2^3 {c:+.2f} S2^4, "
      f"max residual {np.abs(dM - A @ [a, b, c]).max():.1e}")
x = np.linspace(-0.18, 0.1, 300)

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})
fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.9))
ax[0].plot(x, a * x**2 + b * x**3 + c * x**4, color=INK, lw=1.0, label="fit")
ax[0].plot(x, a * x**2, "--", color=MUTED, lw=1.0, label=rf"${a:.2f}\,S_2^2$")
ax[0].plot(s2, dM, "o", color=REF, ms=4.5, label="spectral")
m = ~np.isnan(M0n)
ax[0].plot(s2[m], M0n[m] - R0, "+", color=NETC, ms=7, mew=1.2, label="network")
ax[0].set_ylabel(r"$M_0 - R_0$")
ax[1].plot(x, -2 * x, "--", color=MUTED, lw=1.0, label=r"$-2\,S_2$")
ax[1].plot(s2, M2, "o", color=REF, ms=4.5)
ax[1].set_ylabel(r"$M_2$")
for p, tag in zip(ax, "ab"):
    p.set_xlabel(r"$S_2$")
    p.axhline(0, color=GRID, lw=0.6); p.axvline(0, color=GRID, lw=0.6)
    p.grid(True, which="major", color=GRID, lw=0.6)
    for s in ("top", "right"):
        p.spines[s].set_visible(False)
    p.text(0.5, 0.92, f"({tag})", transform=p.transAxes, fontsize=8, color=INK, ha="center")
    p.set_xticks([-1/6, -1/12, 0, 1/12]); p.set_xticklabels([r"$-\frac{1}{6}$", r"$-\frac{1}{12}$", "0", r"$\frac{1}{12}$"])
ax[0].legend(frameon=False, loc="upper right", handlelength=1.6)
ax[1].legend(frameon=False, loc="upper right", handlelength=1.6)
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(out, dpi=300)
fig.savefig(out.rsplit(".", 1)[0] + ".pdf")
print("[fig_mass] wrote", out)
