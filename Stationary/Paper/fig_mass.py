"""Fig. mass: the mass shift M_ADM - M_0 (M_0 = R_0) and the quadrupole moment M2 against the datum's amplitude S2
(Paper/hub_mass.sh, hub_hard*.sh, hub_bigref.sh; Sec. IV.A, tab:mass).  The NETWORK masses are the
points (the study object); the spectral reference's mass is the grey marker beside each, the
yardstick.  The curve is the fit of the spectral references with |S2| <= 1/6 and continues dashed
beyond, where it stops describing them.  Reads the gh_moments*.json files named below.
    .venv/bin/python Paper/fig_mass.py [out.png]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "Paper/figs/mass_s2.png"
# S2, run holding the spectral reference's moments, its file, run and file of the best NETWORK (or None)
# (at -1/4 the reference moments come from the network run's file, graded against specref_nonlinear_damp:
#  ms_s2_025_w2/gh_moments.json holds an older, uncorrected reference with M0 = 0.537)
SRC = [
    (-0.30,  "ms_s2_030_w2", "gh_moments_big_l20.json", "hs_030_n65k_w3",   "gh_moments_big_l20.json"),
    (-0.25,  "hs_025_n65k_w3", "gh_moments_damp.json",  "hs_025_n65k_w3",   "gh_moments_damp.json"),
    (-1 / 6, "ms_s2d_w2",    "gh_moments.json",         "ms_s2d_w2",        "gh_moments.json"),
    (-1 / 8, "ms_s2t_cfg",   "gh_moments.json",         None,               None),
    (-1 / 12, "ab_none_w2",  "gh_moments.json",         "ab_none_w2",       "gh_moments.json"),
    (-1 / 24, "ms_s2h_w2",   "gh_moments.json",         "ms_s2h_w2",        "gh_moments.json"),
    (1 / 12, "ms_s2p_cfg",   "gh_moments.json",         None,               None),
]
R0 = 1 / np.sqrt(3)
REFC, NETC, INK, MUTED, GRID = "#898781", "#2a78d6", "#1f1f1e", "#898781", "#e1e0d9"

s2, M0, M2, e0, e2, sn, Mn0, Mn2 = [], [], [], [], [], [], [], []
for s, rd, rf, nd, nf in SRC:
    g = json.load(open(f"runs/{rd}/{rf}"))["ref"]
    s2.append(s); M0.append(g["M_ernst"][0]); M2.append(g["M_ernst"][2])
    e0.append(g["M_ernst_spread"][0]); e2.append(g["M_ernst_spread"][2])
    if nd:
        n = json.load(open(f"runs/{nd}/{nf}"))["net"]
        sn.append(s); Mn0.append(n["M_ernst"][0]); Mn2.append(n["M_ernst"][2])
s2, M0, M2, e0, e2, sn, Mn0, Mn2 = map(np.array, (s2, M0, M2, e0, e2, sn, Mn0, Mn2))
dM = M0 - R0
small = np.abs(s2) <= 1 / 6 + 1e-9
A = np.stack([s2[small] ** k for k in (2, 3, 4)], 1)
a, b, c = np.linalg.lstsq(A, dM[small], rcond=None)[0]
fit = lambda x: a * x**2 + b * x**3 + c * x**4                                    # noqa: E731
print(f"[fig_mass] fit of the |S2| <= 1/6 spectral references: M0 - R0 = {a:.3f} S2^2 {b:+.3f} S2^3 "
      f"{c:+.2f} S2^4, max residual {np.abs(dM[small] - A @ [a, b, c]).max():.1e};  "
      f"at -1/4: fit {fit(-0.25):.4f} against {dM[s2 == -0.25][0]:.4f}, "
      f"at -0.30: {fit(-0.30):.4f} against {dM[s2 == -0.30][0]:.4f}")
xa, xb = -0.1667 - 1e-9, 0.1
x1, x2 = np.linspace(-1 / 6, 0.1, 200), np.linspace(-0.32, -1 / 6, 100)

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})
fig, ax = plt.subplots(1, 2, figsize=(3.4, 2.35))
ax[0].plot(x1, fit(x1), color=INK, lw=1.0, label=r"fit, $|S_2|\leq 1/6$")
ax[0].plot(x2, fit(x2), "--", color=INK, lw=1.0)
ax[0].plot(x1, a * x1**2, ":", color=MUTED, lw=1.0, label=rf"${a:.2f}\,S_2^2$")
ax[0].errorbar(s2, dM, yerr=e0, fmt="s", color=REFC, ms=4.5, mfc="none", lw=0.8, capsize=1.5,
               label="spectral reference")
ax[0].plot(sn, Mn0 - R0, "o", color=NETC, ms=3.5, label="network")
ax[0].set_ylabel(r"$M_{\rm ADM} - M_0$")
ax[1].plot(np.linspace(-0.32, 0.1, 50), -2 * np.linspace(-0.32, 0.1, 50), ":", color=MUTED, lw=1.0,
           label=r"$-2\,S_2$")
ax[1].errorbar(s2, M2, yerr=e2, fmt="s", color=REFC, ms=4.5, mfc="none", lw=0.8, capsize=1.5)
ax[1].plot(sn, Mn2, "o", color=NETC, ms=3.5)
ax[1].set_ylabel(r"$M_2$")
for p, tag in zip(ax, "ab"):
    p.set_xlabel(r"$S_2$")
    p.axhline(0, color=GRID, lw=0.6); p.axvline(0, color=GRID, lw=0.6)
    p.grid(True, which="major", color=GRID, lw=0.6)
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.text(0.0, 1.02, f"({tag})", transform=p.transAxes, fontsize=8, color=INK, va="bottom")
    p.set_xticks([-0.3, -0.15, 0])
hs, ls = ax[0].get_legend_handles_labels()
fig.legend(hs, ls, loc="upper center", ncol=2, frameon=False, handlelength=1.6, columnspacing=1.2)
ax[1].legend(frameon=False, loc="lower left", handlelength=1.6)
fig.tight_layout(pad=0.3, w_pad=0.8, rect=[0, 0, 1, 0.82])
fig.savefig(out, dpi=300)
fig.savefig(out.rsplit(".", 1)[0] + ".pdf")
print("[fig_mass] wrote", out)
