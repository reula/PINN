#!/usr/bin/env python
"""Training curves and the network's fields for the single-run phases I, J, L (off-axis quadrupole) and M (rotated Weyl).

    .venv/bin/python Paper/fig_training.py [--remote runs/_remote] [--recompute] [--only loss,error,surface]

Reads the local copies of the runs (history.json, params_itNNNNNN.pkl snapshots) and writes to Paper/figs:

    training_loss.png    loss vs iteration.  Left: the off-axis datum -- I (plain), J (inner data in H^1 +
                         rotation condition) and L (I's final network continued with the rotation condition,
                         drawn from I's last iteration on).  Right: the rotated Weyl test -- M (wr_rot45_w4
                         continued with the rotation condition; wr_rot45_w4 was staged, no history).
    training_error.png   rms error vs loss at the snapshots, h (left) and lambda (right).  Off-axis runs:
                         against the rotated spectral reference specref_nonlinear_fine (as grade_rotated);
                         Weyl runs: against the exact solution.  The off-axis points of I are its end of Adam
                         and its final network (I kept no snapshots).
    surface_L.png        surface plots of the L network's lambda and log10 R_ab R^ab (3-d Ricci of h, from
                         its autodiff Gamma derivatives, vtk.fields_at) on the plane y = 0, which contains
                         the datum axis (1,0,1)/sqrt 2; polar grid rho_in <= rho <= 4.

The snapshot errors are cached in Paper/aux/training_errors.json (--recompute to redo).
2026-11-10, Paper/NOTES.md "The rotation gauge condition".
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402

REF = os.path.join(ROOT, "runs/ab_none_w2/specref_nonlinear_fine.npz")
OFF = {"I": "rp_w60_n65k_one", "J": "rp_w60_n65k_tan_one", "L": "rp_w60_n65k_rot"}
WEYL = {"M": "wr_rot45_rot", "W4": "wr_rot45_w4"}
LEGEND = {
    "I": "I: width 60, 65536 pts, one run (Adam + SSBroyden)",
    "J": "J: as I + inner data in $H^1$ + rotation condition",
    "L": "L: I's network + rotation condition (SSBroyden)",
    "M": "M: wr_rot45_w4 + rotation condition (SSBroyden)",
    "W4": "wr_rot45_w4: width 40, staged, no rotation condition",
}
COL = {"I": "C0", "J": "C1", "L": "C2", "M": "C3", "W4": "C7"}


def history(run):
    h = json.load(open(os.path.join(run, "history.json")))
    return np.array([r["step"] for r in h]), np.array([r["loss"] for r in h])


def loss_at(run, step):
    s, l = history(run)
    return float(l[np.argmin(np.abs(s - step))])


def snapshots(run):
    out = []
    for p in sorted(glob.glob(os.path.join(run, "params_it*.pkl"))):
        out.append((int(re.search(r"params_it(\d+)\.pkl", p).group(1)), os.path.basename(p)))
    return out


def err_offaxis(run, params):
    from error_vs_loss import one
    r = one(run, params, REF)
    return {k: r[k] for k in ("rms_dh", "rms_dlam", "max_dh", "max_dlam", "abs_c")}


def err_exact(run, params):
    from stationary.evaluate import load_run
    from stationary.model import point_fields
    from stationary.specref import compare
    from stationary.train import exact_asset
    cfg, model, state = load_run(run, params)[:3]
    ex = exact_asset(cfg)
    cm = compare(point_fields(model, state["net"]), ex, ex, cfg, modes=False)
    return {k: cm[k] for k in ("rms_dh", "rms_dlam", "max_dh", "max_dlam")}


def errors(remote, recompute):
    cache = os.path.join(HERE, "aux", "training_errors.json")
    if os.path.exists(cache) and not recompute:
        return json.load(open(cache))
    res = {}
    rI = os.path.join(remote, OFF["I"])
    sI, _ = history(rI)
    aI = json.load(open(os.path.join(rI, "config.json")))
    res["I"] = [dict(step=int(aI.get("steps", 5000)), loss=None, **err_offaxis(rI, "params_adam.pkl")),
                dict(step=int(sI[-1]), loss=loss_at(rI, sI[-1]), **err_offaxis(rI, "params.pkl"))]
    # I's Adam stopped on its plateau rule at 1600; the history row closest to it is the Adam loss
    res["I"][0]["step"] = 1600
    res["I"][0]["loss"] = loss_at(rI, 1600)
    for k in ("J", "L"):
        run = os.path.join(remote, OFF[k])
        res[k] = []
        for step, p in snapshots(run):
            row = dict(step=step, loss=loss_at(run, step), **err_offaxis(run, p))
            print(f"[fig] {k} {step:6d} loss {row['loss']:.2e} rms dh {row['rms_dh']:.2e} "
                  f"rms dlam {row['rms_dlam']:.2e} |c| {row['abs_c']:.1e}", flush=True)
            res[k].append(row)
    run = os.path.join(remote, WEYL["M"])
    res["M"] = []
    for step, p in snapshots(run) + [(int(history(run)[0][-1]), "params.pkl")]:
        row = dict(step=step, loss=loss_at(run, step), **err_exact(run, p))
        print(f"[fig] M {step:6d} loss {row['loss']:.2e} rms dh {row['rms_dh']:.2e} rms dlam {row['rms_dlam']:.2e}",
              flush=True)
        res["M"].append(row)
    w4 = os.path.join(remote, WEYL["W4"])
    rep = json.load(open(os.path.join(w4, "report.json")))
    res["W4"] = [dict(step=0, loss=rep["final_loss"], **err_exact(w4, "params.pkl"))]
    json.dump(res, open(cache, "w"), indent=1)
    return res


def fig_loss(remote, out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    sI, lI = history(os.path.join(remote, OFF["I"]))
    ax[0].semilogy(sI, lI, color=COL["I"], label=LEGEND["I"])
    sJ, lJ = history(os.path.join(remote, OFF["J"]))
    ax[0].semilogy(sJ, lJ, color=COL["J"], label=LEGEND["J"])
    sL, lL = history(os.path.join(remote, OFF["L"]))
    ax[0].semilogy(sI[-1] + sL, lL, color=COL["L"], label=LEGEND["L"])
    ax[0].axvline(5000, color="0.6", lw=0.8, ls=":")
    k = int(np.argmax(lJ[1:] / lJ[:-1])) + 1                 # J's jump: the rotation term switched on
    ax[0].annotate("J resumed with the\nrotation condition", (sJ[k], lJ[k]), xytext=(-110, 10),
                   textcoords="offset points", fontsize=8, color=COL["J"],
                   arrowprops=dict(arrowstyle="->", color=COL["J"]))
    ax[0].text(5200, 1e5, "Adam stops on its plateau rule (I, J: 1600);\nSSBroyden counted from 5000",
               fontsize=7, color="0.4")
    ax[0].set(xlabel="iteration (Adam, then SSBroyden; L drawn from the end of I)", ylabel="loss",
              title="off-axis quadrupole datum, $S_2=-1/12$, axis $(1,0,1)$")
    ax[0].legend(fontsize=8)
    sM, lM = history(os.path.join(remote, WEYL["M"]))
    ax[1].semilogy(sM, lM, color=COL["M"], label=LEGEND["M"])
    w4 = json.load(open(os.path.join(remote, WEYL["W4"], "report.json")))
    ax[1].axhline(w4["final_loss"], color=COL["W4"], ls="--", label=LEGEND["W4"] + " (final loss)")
    ax[1].set(xlabel="SSBroyden iteration", ylabel="loss", title="Weyl (3.5, 1, 0.1, 0.5) rotated 45 deg")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[fig] wrote {out}")


def fig_error(res, out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for j, (key, lab) in enumerate((("rms_dh", r"rms $\delta h$"), ("rms_dlam", r"rms $\delta\lambda$"))):
        a = ax[j]
        for k in ("I", "J", "L", "M", "W4"):
            rows = [r for r in res[k] if r["loss"] is not None]
            x = [r["loss"] for r in rows]
            y = [r[key] for r in rows]
            a.loglog(x, y, "o-" if k != "W4" else "s", color=COL[k], ms=4, label=LEGEND[k])
            if k in ("J", "L", "M"):
                for r in (rows[0], rows[-1]):
                    a.annotate(f"{r['step']}", (r["loss"], r[key]), fontsize=7, color=COL[k],
                               xytext=(3, 3), textcoords="offset points")
            if k == "I":
                for r, t in zip(rows, ("end of Adam", "final")):
                    a.annotate(t, (r["loss"], r[key]), fontsize=7, color=COL[k], xytext=(3, 3),
                               textcoords="offset points")
        a.set(xlabel="loss", ylabel=lab,
              title=lab + ": off-axis vs spectral reference, Weyl vs exact")
        a.invert_xaxis()
        a.grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[fig] wrote {out}")


def fig_surface(remote, out, n=121, L=4.0, key="L"):
    from stationary.evaluate import load_run
    from stationary.model import point_fields
    from stationary.vtk import fields_at
    run = os.path.join(remote, OFF[key])
    cfg, model, state = load_run(run, "params.pkl")[:3]
    pf = point_fields(model, state["net"])
    # polar grid on the plane, so that the inner sphere is a clean edge
    rr = cfg.rho_in * (L / cfg.rho_in) ** np.linspace(0, 1, n // 2)
    ps = np.linspace(0, 2 * np.pi, 2 * n + 1)
    RR, PS = np.meshgrid(rr, ps, indexing="ij")
    X, Z = RR * np.cos(PS), RR * np.sin(PS)
    xs = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], -1)
    keep = np.ones(X.size, bool)
    f = fields_at(pf, jnp.asarray(xs[keep]))
    lam = np.full(X.size, np.nan)
    R2 = np.full(X.size, np.nan)
    lam[keep] = np.asarray(f["lambda"])
    R2[keep] = np.log10(np.abs(np.asarray(f["ricci_sq"])))
    fig = plt.figure(figsize=(13, 5.6))
    for i, (v, t, cm) in enumerate(((lam, r"$\lambda$", "viridis"),
                                    (R2, r"$\log_{10} R_{ab}R^{ab}$ of $h$", "magma"))):
        a = fig.add_subplot(1, 2, i + 1, projection="3d")
        V = v.reshape(X.shape)
        a.plot_surface(X, Z, np.ma.masked_invalid(V), cmap=cm, rstride=1, cstride=1, lw=0,
                       antialiased=False, vmin=np.nanmin(V), vmax=np.nanmax(V))
        s0 = cfg.rho_in / np.sqrt(2)
        for sg in (1, -1):                                   # the datum axis x = z, outside rho_in
            a.plot([sg * s0, sg * L], [sg * s0, sg * L], [np.nanmin(V)] * 2, "k--", lw=1)
        a.set(xlabel="x", ylabel="z", title=t + f"  (network {key}, plane y = 0)")
        a.view_init(elev=32, azim=-60)
    fig.suptitle(f"{key} network on the plane containing the datum axis (1,0,1) (dashed); "
                 r"inner sphere $\rho=1$ at the bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[fig] wrote {out}  lambda in [{np.nanmin(lam):.4f}, {np.nanmax(lam):.4f}], "
          f"log10 RabRab in [{np.nanmin(R2):.2f}, {np.nanmax(R2):.2f}]")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--remote", default=os.path.join(ROOT, "runs/_remote"))
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--only", default="loss,error,surface")
    a = ap.parse_args(argv)
    figs = os.path.join(HERE, "figs")
    want = a.only.split(",")
    if "loss" in want:
        fig_loss(a.remote, os.path.join(figs, "training_loss.png"))
    if "error" in want:
        fig_error(errors(a.remote, a.recompute), os.path.join(figs, "training_error.png"))
    if "surface" in want:
        fig_surface(a.remote, os.path.join(figs, "surface_L.png"))


if __name__ == "__main__":
    main()
