#!/usr/bin/env python
"""One continuous run: DSGNAR, then SSBroyden-Wolfe continuing from the DSGNAR endpoint.

x-axis = global iteration: 1..N_ds are DSGNAR's iterations, N_ds+1..N_ds+5000 are the
SSBroyden-Wolfe iterations started from the parameters DSGNAR finished with (the `ssb_polish`
history, whose iteration 1 begins at DSGNAR's final loss/rel L2).  For reference the
independent SSBroyden-Wolfe run from the random initialisation is drawn faintly.

    python plot_dsgnar_then_ssb.py --history clementina/results/ldc_a30_exact_history.json \
        --ssb-iters 5000 --out results/ldc_a30_dsgnar_then_ssb.png
"""

from __future__ import annotations
import argparse, json, os, pathlib
os.environ.setdefault("MPLBACKEND", "Agg"); os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplc")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--history", required=True)
ap.add_argument("--ssb-iters", type=int, default=5000)
ap.add_argument("--out", default="dsgnar_then_ssb.png")
ap.add_argument("--no-reference", action="store_true",
                help="do not draw the independent SSBroyden run from the random init")
ap.add_argument("--title", default="A30 configuration on one A30: DSGNAR, then SSBroyden-Wolfe "
                                    "continued from the DSGNAR endpoint")
ARGS = ap.parse_args()

h = json.loads(pathlib.Path(ARGS.history).read_text())
ds = np.asarray([(r[0], r[1], r[2]) for r in h["dsgnar"]], float)
pol = np.asarray([(r[0], r[1], r[2]) for r in h["ssb_polish"]], float)
x0 = np.asarray([(r[0], r[1], r[2]) for r in h["ssb_from_x0"]], float)
pol = pol[pol[:, 0] <= ARGS.ssb_iters]
x0 = x0[x0[:, 0] <= ARGS.ssb_iters]
n_ds = int(ds[-1, 0])

# continuous sequence: DSGNAR iterations, then the continuation offset by n_ds
cont_x = n_ds + pol[:, 0]
cont_loss = np.concatenate([ds[:, 1], pol[:, 1]])
cont_acc = np.concatenate([ds[:, 2], pol[:, 2]])
allx = np.concatenate([ds[:, 0], cont_x])

fig, axs = plt.subplots(1, 2, figsize=(13.0, 4.7))
axs[0].semilogy(ds[:, 0], ds[:, 1], "-", lw=2.4, color="C0", label=f"DSGNAR ({n_ds} it)")
axs[0].semilogy(cont_x, pol[:, 1], "-", lw=2.4, color="C3",
                label=f"SSBroyden-Wolfe, continued from DSGNAR ({len(pol)} it)")
if not ARGS.no_reference:
    axs[0].semilogy(x0[:, 0], x0[:, 1], "-", lw=2.0, color="C1",
                    label=f"SSBroyden-Wolfe from the initial guess ({len(x0)} it)")
axs[0].axvline(n_ds, color="0.4", ls="--", lw=1)
axs[0].text(n_ds, 2e-2, " DSGNAR | Broyden continuation", fontsize=7, color="0.35")
axs[0].set(xlabel="iteration (DSGNAR, then Broyden continuation)", ylabel=r"loss  $\sum_i w_i\,\mathrm{mean}(r_i^2)$", title="loss")
axs[1].semilogy(ds[:, 0], ds[:, 2], "-", lw=2.4, color="C0", label=f"DSGNAR: {ds[-1,2]:.4f} at it {n_ds}")
axs[1].semilogy(cont_x, pol[:, 2], "-", lw=2.4, color="C3",
                label=f"continued: {pol[-1,2]:.4f} at it {int(cont_x[-1])}")
if not ARGS.no_reference:
    axs[1].semilogy(x0[:, 0], x0[:, 2], "-", lw=2.0, color="C1",
                    label=f"from the initial guess: {x0[-1,2]:.4f} at {len(x0)} it")
axs[1].axvline(n_ds, color="0.4", ls="--", lw=1)
k = int(np.argmin(pol[:, 2]))
axs[1].axhline(pol[k, 2], color="0.6", ls=":", lw=1,
               label=f"best along continuation: {pol[k,2]:.5f} at +{int(pol[k,0])} it")
axs[1].set(xlabel="iteration (DSGNAR, then Broyden continuation)", ylabel=r"rel. $\ell_2$ error vs reference (mean)", title="accuracy")
for a in axs:
    a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
fig.suptitle(ARGS.title, fontsize=10, y=1.02)
plt.tight_layout()
out = pathlib.Path(ARGS.out); out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"[fig] {out}")
print(f"[fig] DSGNAR end: loss {ds[-1,1]:.3e} relL2 {ds[-1,2]:.4f} | "
      f"continuation {len(pol)} it -> loss {pol[-1,1]:.3e} relL2 {pol[-1,2]:.4f}")
print(f"[fig] min relL2 along the continuation: {pol[:,2].min():.5f} at continuation iter {int(pol[np.argmin(pol[:,2]),0])}")
