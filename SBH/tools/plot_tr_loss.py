"""The PINN residual of several 1D networks on the (t, r) plane of their windows. (2026-11-05)

    python tools/plot_tr_loss.py OUT.png "W1,W2,..." LABEL=PREFIX [LABEL=PREFIX ...]

PREFIX_wKKK.npz from `sing_1d.py --eval_tr PKL --eval_tr_out NPZ` (t (n_t,), r (n_r,), einstein, gauge, adm
(n_t, n_r): per point the Einstein and gauge parts as the loss weights them and the r^p constraint density).
One row per window W, one column per network: log10(einstein + gauge) in t (vertical) against log r, on a shared
colour scale. Last column: that quantity averaged over t, as a profile in r for every network; the shaded band is
r < 0.05, where the inner cloud puts 1/8 of the points. Missing files leave the panel blank.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out, wins = sys.argv[1], [int(w) for w in sys.argv[2].split(",")]
nets = [s.split("=", 1) for s in sys.argv[3:]]
COLS = ["#222222", "#d1495b", "#2a6fdb", "#1b9e77"]
LS = ["-", "--", "-.", ":"]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5})
Z = {}
for lab, pre in nets:
    for w in wins:
        f = f"{pre}_w{w:03d}.npz"
        if os.path.exists(f):
            z = np.load(f)
            Z[lab, w] = (z["t"], z["r"], z["einstein"] + z["gauge"])
vals = np.concatenate([np.log10(np.maximum(v[2], 1e-30)).ravel() for v in Z.values()])
lo, hi = np.percentile(vals, 1), np.percentile(vals, 99.9)
fig, axs = plt.subplots(len(wins), len(nets) + 1, figsize=(3.3 * (len(nets) + 1), 2.7 * len(wins)),
                        constrained_layout=True, squeeze=False)
for i, w in enumerate(wins):
    for j, (lab, _) in enumerate(nets):
        ax = axs[i, j]
        if (lab, w) not in Z:
            ax.set_axis_off()
            continue
        t, r, e = Z[lab, w]
        m = ax.pcolormesh(r, t, np.log10(np.maximum(e, 1e-30)), cmap="Blues", vmin=lo, vmax=hi, shading="auto")
        ax.set_xscale("log")
        ax.set_title(f"{lab}, window {w}: mean {e.mean():.1e}")
        ax.set_xlabel("r")
        ax.set_ylabel("t")
    ax = axs[i, -1]
    for j, (lab, _) in enumerate(nets):
        if (lab, w) in Z:
            t, r, e = Z[lab, w]
            ax.loglog(r, e.mean(axis=0), LS[j], color=COLS[j], lw=1.5, label=lab)
    ax.axvspan(r[0], 0.05, color="#999999", alpha=0.15, lw=0)
    ax.set_title(f"window {w}: averaged over t")
    ax.set_xlabel("r")
    ax.grid(True, alpha=0.25)
    if i == 0:
        ax.legend(frameon=False)
fig.colorbar(m, ax=axs[:, :-1], label="log10 (Einstein + gauge residual$^2$, as weighted in the loss)", shrink=0.8)
fig.suptitle("PINN residual on (t, r): per point, the Einstein + gauge part of the loss")
fig.savefig(out, dpi=130)
print("wrote", out)
