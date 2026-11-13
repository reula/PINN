"""Test of the lapse laws with a forced Khat drift in the finite-difference code, against the networks. (2026-11-13)

    python tools/plot_lapse_law_test.py OUT.png LABEL=RUN_DIR [LABEL=RUN_DIR ...]

Files reference_1d/lapse_test/{plain_f0,plain_fm,pos_fm}_dr0.01.npz, made by
    sph_reference.py --dr 0.01 --tfinal 16 --lapse_K {plain,pos} [--k_force=-3.5e-5] --out ...
(--k_force: constant added to d_t Khat for r > 5, the size of the networks' Khat-row bias). Curves = difference from
the unforced 1+log run. Left: alpha error at r = 12 against t; right: alpha error against r at t = 8. Networks
(report.npz, network - reference) as markers / thin lines.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = sys.argv[1]
runs = [s.split("=", 1) for s in sys.argv[2:]]
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference_1d", "lapse_test")
ld = lambda n: np.load(os.path.join(D, n + "_dr0.01.npz"))
Z0 = ld("plain_f0")
r, t = Z0["r"], Z0["t"]
FD = (("plain_fm", "finite differences, 1+log, Khat forced by -3.5e-5 (r > 5)", "k", "-"),
      ("pos_fm", "finite differences, max(Khat, 0) in the lapse, same forcing", "k", "--"))
COLS = ["#2a6fdb", "#d1495b", "#1b9e77", "#e69f00"]
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
k12 = int(np.argmin(abs(r - 12)))
j8 = int(np.argmin(abs(t - 8)))
for n, lab, c, ls in FD:
    Z = ld(n)
    ax[0].plot(t[t <= 8], (Z["U"][:, 8, k12] - Z0["U"][:, 8, k12])[t <= 8], c=c, ls=ls, lw=2, label=lab)
    ax[1].plot(r, Z["U"][j8, 8] - Z0["U"][j8, 8], c=c, ls=ls, lw=2, label=lab)
for (lab, d), c in zip(runs, COLS):
    R = np.load(os.path.join(d, "report.npz"))
    k = int(np.argmin(abs(R["r"] - 12)))
    ax[0].plot(R["t"], R["err"][:, k, 8], "o", ms=3.5, c=c, label="network: " + lab)
    ax[1].plot(R["r"], R["err"][int(np.argmin(abs(R["t"] - 8))), :, 8], c=c, lw=1.2, label="network: " + lab)
ax[0].set_title("error of alpha at r = 12"); ax[0].set_xlabel("t")
ax[1].set_title("error of alpha at t = 8"); ax[1].set_xlabel("r"); ax[1].set_xscale("log"); ax[1].set_xlim(0.3, 24)
ax[1].set_ylim(-5e-4, 3e-3)
for A in ax:
    A.grid(alpha=0.3); A.axhline(0, c="0.6", lw=0.6)
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(out, dpi=110)
print("wrote", out)
