#!/usr/bin/env python
"""The paper's centre-of-mass figure: the same run without and with the centre-of-mass condition,
graded against the spectral reference.
    cd Stationary && .venv/bin/python Paper/fig_com.py [Paper/paper_metrics.json] [Paper/figs/com_steps.png]  (also writes the .pdf)
Input: scripts/report_metrics.py's JSON for fn_nocom_w2 and ab_none_w2 (the final formulation;
qp_nocom_w2 / qp_com_w2, with the extra outer conditions, until 2026-10-28).  Three panels: the angular
rms of lambda_net - lambda_ref, of max_ij |h_net - h_ref|, and the |l = 1| coefficient of
lambda_net - lambda_ref, against rho; dashed, the perturbation (reference minus spherical).
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "Paper/paper_metrics.json"
out = sys.argv[2] if len(sys.argv) > 2 else "Paper/figs/com_steps.png"
M = json.load(open(src))
RUNS = [("fn_nocom_w2", "without centre of mass", "#eb6834"),
        ("ab_none_w2", "with centre of mass", "#2a78d6")]
INK, MUTED, GRID = "#1f1f1e", "#898781", "#e1e0d9"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.35))
for name, lab, col in RUNS:
    p, o = M[name]["profile"], M[name]["odd"]
    ax[0].loglog(p["rho"], p["dlam_rms"], color=col, lw=1.4, label=lab)
    ax[1].loglog(p["rho"], p["dh_rms"], color=col, lw=1.4, label=lab)
    ax[2].loglog(o["rho"], np.maximum(np.abs(o["l1"]), 1e-16), color=col, lw=1.4, label=lab)
ref = M[RUNS[0][0]]["profile"]
for a, key in ((ax[0], "pert_lam_rms"), (ax[1], "pert_h_rms")):
    a.loglog(ref["rho"], ref[key], "--", color=INK, lw=1.0, label="perturbation")
ax[0].set_ylabel(r"rms $|\lambda_{\rm net}-\lambda_{\rm ref}|$")
ax[1].set_ylabel(r"rms $\max_{ij}|h_{\rm net}-h_{\rm ref}|$")
ax[2].set_ylabel(r"$|l=1$ coefficient of $\lambda_{\rm net}-\lambda_{\rm ref}|$")
for a, tag in zip(ax, "abc"):
    a.set_xlabel(r"$\rho$")
    a.grid(True, which="major", color=GRID, lw=0.6)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.text(0.93, 0.92, f"({tag})", transform=a.transAxes, fontsize=8, color=INK, ha="right")
hs, ls = ax[0].get_legend_handles_labels()
fig.legend(hs, ls, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(out, dpi=300)
fig.savefig(out.rsplit(".", 1)[0] + ".pdf")
print("wrote", out, "and the .pdf")
