"""The scalar-field reference (reference_1d/sph_reference_scalar.py) against the vacuum one. (2026-11-10)

    python tools/plot_scalar_ref.py OUT.png SCALAR.npz VACUUM.npz [COARSE.npz COARSER.npz]

Panels: horizon mass M_AH(t) and coordinate radius r_AH(t) of both; Misner-Sharp mass against r and r phi against
r at t = 0, 2, 4, 6, 8; with the two coarser runs (dr x 2, x 4), the differences |U_2h - U_h| and |U_4h - U_2h| / 16
of alpha, chi and phi at the last time (4th-order convergence = the curves coincide).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference_1d"))
from sph_reference_scalar import diagnostics                       # noqa: E402

out, fs, fv = sys.argv[1:4]
S, V = np.load(fs), np.load(fv)
r, t, U = S["r"], S["t"], S["U"]
dS = [diagnostics(r, u) for u in U]
dV = [diagnostics(V["r"], u) for u in V["U"]]
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
A = ax[0, 0]
A.plot(t, [d[1] for d in dS], c="#d1495b", label=f"scalar shell A = {float(S['amp'])}, M_ADM = {float(S['M_adm']):.4f}")
A.plot(V["t"], [d[1] for d in dV], c="#222", ls="--", label="vacuum, M = 0.5")
A.set_title("horizon mass M_AH = R_areal(AH) / 2"); A.set_xlabel("t"); A.legend()
A = ax[0, 1]
A.plot(t, [d[0] for d in dS], c="#d1495b", label="scalar")
A.plot(V["t"], [d[0] for d in dV], c="#222", ls="--", label="vacuum")
A.plot(t, np.interp(t, V["t"], [d[0] for d in dV]) * np.array([d[1] for d in dS]) / 0.5, c="#2a6fdb", ls=":",
       label="vacuum r_AH(t) x M_AH(t) / 0.5")
A.set_title("coordinate radius of the horizon r_AH(t)"); A.set_xlabel("t"); A.legend()
cols = plt.cm.viridis(np.linspace(0, 0.9, 5))
for c, T in zip(cols, (0, 2, 4, 6, 8)):
    j = int(np.argmin(abs(t - T)))
    m = r > 0.15
    ax[0, 2].plot(r[m], dS[j][2][m], c=c, label=f"t = {T}")
    ax[1, 0].plot(r, r * U[j, 10], c=c, label=f"t = {T}")
    ax[1, 0].axvline(dS[j][0], c=c, lw=0.6, ls=":")
ax[0, 2].set_xscale("log"); ax[0, 2].set_ylim(0.49, 0.61); ax[0, 2].set_title("Misner-Sharp mass against r")
ax[0, 2].set_xlabel("r"); ax[0, 2].legend()
ax[1, 0].set_xlim(0, 12); ax[1, 0].set_title("r phi against r (dotted: the horizon)"); ax[1, 0].set_xlabel("r"); ax[1, 0].legend()
if len(sys.argv) >= 6:
    C1, C2 = np.load(sys.argv[4]), np.load(sys.argv[5])
    for A, T in zip(ax[1, 1:], (4, 8)):
        j, j1, j2 = (int(np.argmin(abs(Z["t"] - T))) for Z in (S, C1, C2))
        for i, nm, c in ((8, "alpha", "#222"), (4, "chi", "#2a6fdb"), (10, "phi", "#d1495b")):
            e1 = np.abs(np.interp(r, C1["r"], C1["U"][j1, i]) - U[j, i])
            e2 = np.abs(np.interp(r, C2["r"], C2["U"][j2, i]) - np.interp(r, C1["r"], C1["U"][j1, i])) / 16
            A.loglog(r, e1 + 1e-18, c=c, label=f"{nm}: |U(2h) - U(h)|")
            A.loglog(r, e2 + 1e-18, c=c, ls="--", label=f"{nm}: |U(4h) - U(2h)| / 16")
        A.set_title(f"convergence at t = {T} (h = {float(S['dr'])})"); A.set_xlabel("r"); A.set_ylim(1e-12, 1e-2)
        A.legend(fontsize=7); A.set_xlim(0.02, 30)
for A in ax.ravel():
    A.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(out, dpi=110)
print("wrote", out)
for T in (0, 2, 3, 4, 5, 6, 7, 8):
    j, jv = int(np.argmin(abs(t - T))), int(np.argmin(abs(V["t"] - T)))
    print(f"t = {T}: scalar r_AH {dS[j][0]:.4f} M_AH {dS[j][1]:.5f} | vacuum r_AH {dV[jv][0]:.4f} M_AH {dV[jv][1]:.5f}")
