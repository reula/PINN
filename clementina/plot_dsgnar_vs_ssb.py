#!/usr/bin/env python
"""Paper configuration, one A30: DSGNAR vs SSBroyden-Wolfe -- loss and accuracy vs iteration.

Curves:
  * DSGNAR (exact residual, float64) from `mendieta_a30_paper_exact_history.json`
    (job 1181874, 200 steps from the random initialisation);
  * SSBroyden-Wolfe from the same random initialisation, 20000 iterations, from the
    `--ssb-only` run's history.json [iteration, loss, rel L2].

Both are the SAME objective on the SAME point sets (paper configuration: 60-60-60-60,
2^15/2^14 points, pde weight 1e-2), graded against the same reference field.

    python plot_dsgnar_vs_ssb.py --dsgnar-history .../history.json --ssb-history .../history.json \
                                 --out optimizers_loss_accuracy.png
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplc")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dsgnar-history", required=True)
ap.add_argument("--ssb-history", required=True)
ap.add_argument("--out", default="optimizers_loss_accuracy.png")
ap.add_argument("--ssb-max-iter", type=int, default=None,
                help="cut the SSBroyden-Wolfe curves at this iteration (DSGNAR unchanged)")
ap.add_argument("--title", default="paper configuration on one A30 (60-60-60-60, "
                                    "2$^{15}$/2$^{14}$ points): DSGNAR vs SSBroyden-Wolfe")
ARGS = ap.parse_args()

# ── DSGNAR: driver history rows [iter, loss, relL2, radius, rho, accepted, bits] ──
ds = json.loads(pathlib.Path(ARGS.dsgnar_history).read_text())["dsgnar"]
ds = np.asarray([(r[0], r[1], r[2]) for r in ds], dtype=float)

# ── SSBroyden-Wolfe: rows [iter, loss, relL2] ────────────────────────────────────
ssb_raw = json.loads(pathlib.Path(ARGS.ssb_history).read_text())
rows = ssb_raw.get("ssb_from_x0", ssb_raw)
ssb = np.asarray([(r[0], r[1], r[2]) for r in rows], dtype=float)
_pol = ssb_raw.get("ssb_polish") if isinstance(ssb_raw, dict) else None
pol = np.asarray([(r[0], r[1], r[2]) for r in _pol], dtype=float) if _pol else None
if ARGS.ssb_max_iter:
    ssb = ssb[ssb[:, 0] <= ARGS.ssb_max_iter]
    if pol is not None:
        pol = pol[pol[:, 0] <= ARGS.ssb_max_iter]

fig, axs = plt.subplots(1, 2, figsize=(13.0, 4.6))

axs[0].semilogy(ds[:, 0], ds[:, 1], "-", lw=2.0, color="C0",
                label=f"DSGNAR (exact residual, {int(ds[-1, 0])} it)")
axs[0].semilogy(ssb[:, 0], ssb[:, 1], "-", lw=2.0, color="C3",
                label=f"SSBroyden-Wolfe ({int(ssb[-1, 0])} it)")
axs[0].set(xlabel="iteration", ylabel=r"loss  $\sum_i w_i\,\mathrm{mean}(r_i^2)$",
           title="loss")
axs[0].grid(alpha=0.3, which="both")

axs[1].loglog(ds[:, 0], ds[:, 2], "-", lw=2.0, color="C0",
              label=f"DSGNAR: {ds[-1, 2]:.4f} at it {int(ds[-1, 0])}")
axs[1].loglog(ssb[:, 0], ssb[:, 2], "-", lw=2.0, color="C3",
              label=f"SSBroyden-Wolfe: {ssb[-1, 2]:.4f} at it {int(ssb[-1, 0])}")
if pol is not None:
    axs[0].semilogy(pol[:, 0], pol[:, 1], "--", lw=1.8, color="C2",
                    label=f"SSBroyden-Wolfe polish from DSGNAR ({int(pol[-1, 0])} it)")
    axs[1].loglog(pol[:, 0], pol[:, 2], "--", lw=1.8, color="C2",
                  label=f"SSB polish from DSGNAR: {pol[-1, 2]:.4f} at it {int(pol[-1, 0])}")
axs[1].set(xlabel="iteration", ylabel=r"rel. $\ell_2$ error vs reference (mean)",
           title="accuracy")
axs[1].grid(alpha=0.3, which="both")

for a in axs:
    a.legend(fontsize=9)
fig.suptitle(ARGS.title, fontsize=10, y=1.01)
plt.tight_layout()
out = pathlib.Path(ARGS.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"[fig] {out}")
print(f"[fig] DSGNAR final: loss {ds[-1,1]:.3e}  relL2 {ds[-1,2]:.4f}")
print(f"[fig] SSB    final: loss {ssb[-1,1]:.3e}  relL2 {ssb[-1,2]:.4f}")
