"""Every run's error against ONE nonlinear reference, as a function of rho, in one figure.

    ./.venv/bin/python -m stationary.linref_summary --ref runs/<r>/linref_nonlinear_rhoW.npz \
        runs/<run> [runs/<run> ...] [--out plots/linref_summary.png]

The runs must share the reference's problem (`linref.load_reference` checks the inner lambda
data).  Per run: the angular rms of |lambda_net - lambda_ref| and of max_ij |h_net - h_ref| on
spheres, next to the size of the perturbation itself (reference - spherical) -- the scale an
error is read on -- and the P_2 coefficients of lambda and h_rr against the reference's.  Prints a
table of the same numbers per radial region, with each run's final training loss.
"""
from __future__ import annotations

import argparse
import json
import os

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

# the best few runs are coloured (validated categorical palette, dataviz skill), the rest grey
TOP = ["#2a78d6", "#eb6834", "#1baf7a"]
GREY = "#b4b2ac"


def run_errors(run_dir, ref_path, n_rho=40):
    from .evaluate import load_run
    from .linref import compare, load_reference, profiles
    from .model import point_fields
    from .train import exact_asset
    cfg, model, state = load_run(run_dir, "params.pkl")[:3]
    ex = exact_asset(cfg)
    ref, why = load_reference(run_dir, cfg, ex, path=ref_path)
    if ref is None:
        raise SystemExit(f"{run_dir}: {why}")
    pf = point_fields(model, state["net"])
    loss = float("nan")
    rp = os.path.join(run_dir, "report.json")
    if os.path.exists(rp):
        loss = float(json.load(open(rp)).get("final_loss", float("nan")))
    return {"name": os.path.basename(os.path.normpath(run_dir)), "loss": loss,
            "prof": profiles(pf, ref, ex, cfg, n_rho=n_rho),
            "bins": compare(pf, ref, ex, cfg, modes=False)["bins"]}


def plot(runs, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = sorted(runs, key=lambda r: r["loss"])
    col = {r["name"]: (TOP[i] if i < len(TOP) else GREY) for i, r in enumerate(runs)}
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
    ref = runs[0]["prof"]
    rho = ref["rho"]
    panels = [(ax[0, 0], "dlam_rms", "pert_lam_rms", r"$\lambda$: angular rms of network $-$ reference"),
              (ax[0, 1], "dh_rms", "pert_h_rms", r"$h$: angular rms of $\max_{ij}|$network $-$ reference$|$")]
    for a, key, pkey, t in panels:
        for r in runs[::-1]:
            c = col[r["name"]]
            top = c != GREY
            a.loglog(rho, r["prof"][key], color=c, lw=2.0 if top else 1.0, zorder=3 if top else 1,
                     label=f"{r['name']} (loss {r['loss']:.1e})" if top else None)
        a.loglog(rho, ref[pkey], "k--", lw=1.3, label="the perturbation itself (reference - spherical)")
        a.set_title(t, fontsize=11); a.set_xlabel(r"$\rho$")
    ax[0, 0].plot([], [], color=GREY, lw=1, label=f"the other {max(len(runs) - len(TOP), 0)} runs")
    ax[0, 0].legend(fontsize=7.5, loc="lower left")
    for a, key, t in ((ax[1, 0], "lam2", r"$|\lambda$ $P_2$ coefficient$|$"),
                      (ax[1, 1], "hrr2", r"$|h_{rr}$ $P_2$ coefficient$|$")):
        for r in runs[::-1]:
            c = col[r["name"]]
            a.loglog(rho, np.abs(r["prof"][key + "_net"]), color=c,
                     lw=2.0 if c != GREY else 1.0, zorder=3 if c != GREY else 1)
        a.loglog(rho, np.abs(ref[key + "_ref"]), "k--", lw=1.6, label="reference")
        a.loglog(rho, np.abs(ref[key + "_ref"][0]) * rho ** -3.0, ":", color="#898781", lw=1,
                 label=r"$\rho^{-3}$")
        a.set_title(t + ": networks (colours as above) and the reference", fontsize=11)
        a.set_xlabel(r"$\rho$"); a.legend(fontsize=8)
    for a in ax.ravel():
        a.grid(alpha=.3, which="both")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", required=True, help="the reference .npz (linref output)")
    ap.add_argument("--out", default="plots/linref_summary.png")
    a = ap.parse_args(argv)
    runs = []
    for r in a.runs:
        print(f"[summary] {r}", flush=True)
        runs.append(run_errors(r, a.ref))
    runs.sort(key=lambda r: r["loss"])
    ref_name = os.path.join(os.path.basename(os.path.dirname(os.path.abspath(a.ref))),
                            os.path.basename(a.ref))
    plot(runs, a.out, f"every run's error against the nonlinear reference {ref_name}")
    print(f"[summary] figure: {a.out}\n")
    edges = [(b["rho_lo"], b["rho_hi"]) for b in runs[0]["bins"]]
    head = "| run | final loss | " + " | ".join(f"rms dlam {lo:g}-{hi:g}" for lo, hi in edges) \
        + " | " + " | ".join(f"rms dh {lo:g}-{hi:g}" for lo, hi in edges) + " |"
    print(head)
    print("|" + "---|" * (2 + 2 * len(edges)))
    pert = runs[0]["bins"]
    print("| *perturbation itself* | -- | " + " | ".join(f"*{b['perturbation_rms_lam']:.1e}*" for b in pert)
          + " | " + " | ".join(f"*{b['perturbation_rms_h']:.1e}*" for b in pert) + " |")
    for r in runs:
        print(f"| `{r['name']}` | {r['loss']:.1e} | "
              + " | ".join(f"{b['rms_dlam']:.1e}" for b in r["bins"]) + " | "
              + " | ".join(f"{b['rms_dh']:.1e}" for b in r["bins"]) + " |")


if __name__ == "__main__":
    main()
