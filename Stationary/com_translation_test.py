"""Is the tail dipole an asymptotic TRANSLATION?  Shape fit and cross-sector prediction.

A translation by `a` acts on a spherically symmetric background (and to linear order on a nearly
spherical solution) as `delta F = -a . grad F_0 = -(a . n) F_0'(rho)`.  So the l = 1 radial profile
of any field F must satisfy

    F_{l=1}(rho) = -|a| F_0'(rho)          i.e.        F_{l=1}(rho) / F_0'(rho) = const

with the SAME constant for every field (`F = lambda, h_rr, h_tan`) if the whole configuration is a
translated copy of an untranslated one.  A genuine l = 1 source would instead show up as a decaying
harmonic `c rho^-2`, whose ratio `c/(rho^2 F_0')` is not constant.  This script measures that ratio.

    ./.venv/bin/python com_translation_test.py --run /tmp/com/hard2 [--label hard2] [--out plots/...]
"""
from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def profiles(run_dir, rhos, n_th=96, lmax=3):
    """l-th Legendre profiles of lambda, h_rr and h_tan on the given shells."""
    from stationary.evaluate import load_run

    cfg, model, state = load_run(run_dir)
    params = state["net"]
    mu, w = np.polynomial.legendre.leggauss(n_th)          # integrate over cos(theta)
    # Legendre basis sampled on the nodes; the projection is done on the host after the apply
    P = np.stack([np.polynomial.legendre.Legendre.basis(l)(mu) for l in range(lmax + 1)])
    mu_j = jnp.asarray(mu)
    sin_th = jnp.sqrt(1.0 - mu_j ** 2)

    def one(rho):
        x = jnp.stack([rho * sin_th, jnp.zeros_like(sin_th), rho * mu_j], axis=-1)
        f = model.apply(params, x)                         # Fields, batched
        h = f.h                                            # (N,3,3)
        lam = f.lam.reshape(-1)                            # (N,)
        n = x / rho
        h_rr = jnp.einsum("ni,nij,nj->n", n, h, n)
        h_tan = 0.5 * (jnp.einsum("nii->n", h) - h_rr)
        out = {}
        for name, vals in (("lam", lam), ("h_rr", h_rr), ("h_tan", h_tan)):
            a = np.asarray(vals)
            out[name] = np.array([(2 * l + 1) / 2 * np.sum(w * a * P[l]) for l in range(lmax + 1)])
        return out

    res = {k: [] for k in ("lam", "h_rr", "h_tan")}
    for rho in rhos:
        got = one(float(rho))
        for k, v in got.items():
            res[k].append(v)
    return {k: np.array(v) for k, v in res.items()}        # (n_rho, lmax+1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--n-rho", type=int, default=160)
    args = ap.parse_args()

    # a fine grid, logarithmic, away from both spheres
    rhos = np.exp(np.linspace(np.log(2.0), np.log(99.0), args.n_rho))
    prof = profiles(args.run, rhos)
    label = args.label or os.path.basename(args.run.rstrip("/"))
    d = {k: np.gradient(prof[k][:, 0], rhos) for k in prof}

    # Fit the translation coefficient from the field whose l = 1 part has the most signal, and
    # then PREDICT the others: a translated configuration has one and the same vector a.
    def fit(target, basis, mask):
        b, t = basis[mask], target[mask]
        c = float(np.dot(t, b) / np.dot(b, b))
        resid = float(np.linalg.norm(t - c * b) / np.linalg.norm(t))
        return c, resid

    # where the l = 1 part has signal (not in its noise-dominated tails)
    amp = np.abs(prof["lam"][:, 1])
    mask = amp > 0.2 * amp.max()
    a_z, lam_resid = fit(prof["lam"][:, 1], -d["lam"], mask)
    print(f"{label}:  l = 1 signal range rho = {rhos[mask][0]:.2f} .. {rhos[mask][-1]:.2f}")
    print(f"   lambda   : a_z = {a_z:+.6e} from the fit to -lambda_0';  misfit {lam_resid:.3f}")
    for cand, name in ((-d["lam"], "lambda_0'"), (-1.0 / rhos ** 2, "rho^-2"),
                       (-1.0 / rhos ** 3, "rho^-3"), (np.ones_like(rhos), "const"),
                       (-rhos, "rho")):
        c, r = fit(prof["lam"][:, 1], cand, mask)
        print(f"      candidate {name:10s}: coefficient {c:+.4e}   misfit {r:.3f}")
    for k, pretty in (("h_rr", "h_rr"), ("h_tan", "h_tan")):
        amp_k = np.abs(prof[k][:, 1])
        m = amp_k > 0.2 * amp_k.max()
        c, r = fit(prof[k][:, 1], -d[k], m)
        pred = -a_z * d[k]
        rp = float(np.linalg.norm(prof[k][:, 1][m] - pred[m]) / np.linalg.norm(prof[k][:, 1][m]))
        print(f"   {pretty:7s}: l=1 amplitude max {amp_k.max():.3e} (l=0 part "
              f"{np.abs(prof[k][:, 0]).mean():.3e}); a_z = {c:+.4e} from its own fit (misfit {r:.3f});"
              f"  PREDICTED by lambda's a_z: misfit {rp:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    ax = axes[0]
    ax.semilogx(rhos, prof["lam"][:, 1], lw=2.0, color="k", label=r"measured $\lambda_{l=1}$")
    ax.semilogx(rhos, a_z * (-d["lam"]), lw=1.6, color="tab:red", ls="--",
                label=rf"translation fit $-a_z\lambda_0'$, $a_z={a_z:+.2e}$")
    c2, _ = fit(prof["lam"][:, 1], -1.0 / rhos ** 2, mask)
    ax.semilogx(rhos, c2 * (-1.0 / rhos ** 2), lw=1.4, color="0.5", ls=":",
                label=rf"$c/\rho^2$ fit, $c={c2:+.2e}$")
    ax.axvspan(rhos[mask][0], rhos[mask][-1], color="tab:blue", alpha=0.06)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\lambda_{l=1}(\rho)$")
    ax.set_title("the dipole's radial profile against the two candidate shapes")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for k, col in (("h_rr", "tab:green"), ("h_tan", "tab:purple")):
        ax.semilogx(rhos, prof[k][:, 1], lw=1.8, color=col, label=rf"measured ${k}_{{l=1}}$")
        ax.semilogx(rhos, -a_z * d[k], lw=1.4, color=col, ls="--",
                    label=rf"$-a_z {k}_0'$ (predicted)")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"metric $l=1$ profile")
    ax.set_title(r"cross-sector check: does $\lambda$'s $a_z$ predict the metric's $l=1$?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(f"{label}: is the dipole a translation?  one $a_z$, three fields", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = args.out or f"plots/com_translation_{label}.png"
    fig.savefig(out, dpi=150)
    print(f"   wrote {out}")


if __name__ == "__main__":
    main()
