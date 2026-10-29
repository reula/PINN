"""Error of the initial data at t = 0, by radius, for several runs. (2026-10-29)

    python tools/plot_ic_error.py OUT.png NAME=RUN.npz [NAME=RUN.npz ...]

RUN.npz from `sing_pinn.py --eval_spheres` (also `--eval_pre`: the state after the --pre_iters IC phase) or
from tools/sphere_errors.py (main-script runs). Panels, all at t = 0, about the origin, on spheres of 48
directions: <m/M - 1> (signed), and the rms over each sphere of alpha, chi, Khat, Theta - the 1D reference
(every npz has these), then gt, At, Ghat, beta - the exact initial data (sing_pinn.py npz only: `ic_rms`).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out, runs = sys.argv[1], [s.rsplit("=", 1) for s in sys.argv[2:]]
Z = [(n, np.load(f)) for n, f in runs]
COLS = ["#222222", "#2a6fdb", "#e07b00", "#1b9e77", "#d1495b", "#7a5195"]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.25})
PANELS = [("mass_mean", "⟨m/M − 1⟩ (signed)"), ("alpha_rms", "α"), ("chi_rms", "χ"), ("Khat_rms", "K̂"),
          ("Theta_rms", "Θ"), ("ic:gt", "γ̃_ij (6 comps)"), ("ic:At", "Ã_ij (6 comps)"), ("ic:Ghat", "Γ̂^i"),
          ("ic:beta", "β^i")]
fig, axs = plt.subplots(3, 3, figsize=(14, 10.5), constrained_layout=True)
for ax, (key, title) in zip(axs.ravel(), PANELS):
    for (n, z), c in zip(Z, COLS):
        if key.startswith("ic:"):
            if "ic_rms" not in z.files:
                continue
            y = z["ic_rms"][list(z["ic_groups"]).index(key[3:])]
        else:
            y = z[key][0]
        ax.plot(z["r"], y, color=c, lw=1.6, label=n)
    ax.set_xscale("log")
    if key == "mass_mean":
        ax.set_yscale("symlog", linthresh=1e-5)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title("mass error " + title)
    else:
        ax.set_yscale("log")
        ax.set_title(f"rms of {title} error on spheres")
    ax.set_xlabel("r")
axs[0, 0].legend(fontsize=8)
fig.suptitle("Initial data (t = 0): error by radius. α, χ, K̂, Θ against the 1D reference at t = 0; "
             "γ̃, Ã, Γ̂, β against the exact Brill-Lindquist data (exact: γ̃ = δ, the others 0)", fontsize=10.5)
fig.savefig(out, dpi=130)
print("wrote", out)
