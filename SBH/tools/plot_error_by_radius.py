"""Field errors by radius at window ends: sing_bl_hard vs the baseline, with the template seam marked. (2026-10-30)

    python tools/plot_error_by_radius.py DATA_DIR [OUT.png]      (default plots/2026-10-30/error_by_radius.png)

DATA_DIR holds long_con10_w8.npz (baseline, tools/sphere_errors.py) and sing_bl_hard_sph.npz
(sing_pinn.py --eval_spheres). Rows: alpha, Khat, chi; columns: t = 0.25, 1, 2; rms over each sphere of
48 directions of the field minus the 1D reference. Shaded: r_c = 0.2 (template cutoff W = exp(-(r/r_c)^4),
rho = 1/2) to R_TPL = 3 r_c (W = e^-81, template gone); the network must bridge the two there.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots",
                                                         "2026-10-30", "error_by_radius.png")
RUNS = [("w8_qn_con10 (baseline)", "long_con10_w8.npz", "#222222"), ("sing_bl_hard", "sing_bl_hard_sph.npz", "#d1495b")]
Z = {n: np.load(os.path.join(D, f)) for n, f, _ in RUNS}
R_C = 0.2
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.25})
fig, axs = plt.subplots(3, 3, figsize=(14, 10), sharex=True, constrained_layout=True)
for i, (q, lab) in enumerate((("alpha_rms", "α"), ("Khat_rms", "K̂"), ("chi_rms", "χ"))):
    for j, t in enumerate((0.25, 1.0, 2.0)):
        ax = axs[i, j]
        for n, _, c in RUNS:
            z = Z[n]
            k = int(np.argmin(np.abs(z["t"] - t)))
            ax.loglog(z["r"], z[q][k], color=c, lw=1.6, label=n)
        ax.axvspan(R_C, 3 * R_C, color="#e07b00", alpha=0.12, lw=0)
        ax.set_title(f"rms of {lab} error on spheres, t = {t:g}")
        if i == 2:
            ax.set_xlabel("r (distance to the centre)")
axs[0, 0].legend(fontsize=8, loc="lower left")
fig.suptitle("Error against the 1D reference by radius at window ends. Shaded: the template seam "
             "r_c ≤ r ≤ 3 r_c (W from e⁻¹ to e⁻⁸¹, ρ from 1/2 to 0.9)", fontsize=10.5)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
