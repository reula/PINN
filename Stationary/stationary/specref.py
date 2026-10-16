"""A reference solution for the quadrupole runs, by a spectral solve of the SAME equations.

The pq / S runs impose `lambda(rho_in) = lambda0 + S2 P2(cos theta)` on the inner sphere, and that
problem has no closed-form solution: `--ref-solution` is the SPHERICAL member, so the report's
`rms_dh`, `rms_dlam` are distances to it, not errors (PERTURBATION.md, section 0).  This module
builds a reference that does solve the problem:

* the field is the spherical background (`exact_asset(cfg)`, i.e. S0 through `lam0`/`R0`) plus an
  axisymmetric, reflection-even perturbation, expanded in Legendre modes `l` and, radially, in
  Chebyshev polynomials of `log rho` on `[rho_in, rho_out]`:

      delta lambda = sum_l f0_l(rho) P_l
      delta h_ij   = sum_l [ f1_l P_l delta_ij + f2_l P_l n_i n_j
                             + f3_l P_l' (e_i n_j + n_i e_j)
                             + f4_l P_l'' (e_i e_j - (1 - mu^2)/2 (delta_ij - n_i n_j)) ]

  with `mu = z/rho`, `e = z_hat - mu n` (= rho grad mu, tangential).  For each `l` that is the
  complete even-parity set: h_rr, h_rA, the tangential trace and the tangential trace-free part
  (l = 0 has only the first two -- P_0' = P_0'' = 0);
* the residual is `losses.residual_vector` itself -- the run's PDE (Ricci, harmonic gauge, lambda
  equation), its inner datum, its outer Robin ladder and its pins -- so the reference solves the
  same boundary-value problem the network is trained on, with the same weights;
* LINEAR: modes `l = (2,)`, one least-squares solve `J c = -r(0)` at the background.  The l = 2
  response to first order in S2 (PERTURBATION.md: at linear order nothing else is excited);
* NONLINEAR: modes `l = (0, 2, 4, ...)`, Gauss-Newton on the same residual.  The full axisymmetric
  solution, to the truncation of the basis.

    ./.venv/bin/python -m stationary.linref runs/<run> [--ls 2] [--nonlinear --ls 0,2,4,6]

writes `<run>/linref.npz` and prints the run's network against it.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from .geometry import christoffel                     # noqa: E402
from .model import Fields                             # noqa: E402

ZHAT = jnp.array([0.0, 0.0, 1.0])
I3 = jnp.eye(3)


def legendre_with_derivs(l: int, mu):
    """P_l(mu), P_l'(mu), P_l''(mu) by the three-term recurrence (exact polynomials)."""
    def p(m):
        if l == 0:
            return jnp.ones_like(m)
        p0, p1 = jnp.ones_like(m), m
        for k in range(2, l + 1):
            p0, p1 = p1, ((2 * k - 1) * m * p1 - (k - 1) * p0) / k
        return p1
    d1 = jax.grad(p)
    d2 = jax.grad(d1)
    return p(mu), d1(mu), d2(mu)


def chebyshev(n: int, s):
    """T_0..T_{n-1}(s) by recurrence (smooth at s = +-1, unlike cos(k arccos s))."""
    out = [jnp.ones_like(s), s]
    for _ in range(2, n):
        out.append(2.0 * s * out[-1] - out[-2])
    return jnp.stack(out[:n])


class SpectralAxisym:
    """Background + spectral axisymmetric perturbation, with the model interface the losses use.

    `apply(c, X)` returns batched `Fields`, so `losses.residual_vector({"net": c}, ...)` evaluates
    this field exactly as it evaluates a network.  `derives_gamma = True`: Gamma is the metric's
    Christoffel symbol, as in the hybrid (metric-only) runs, so compatibility holds identically.
    """
    derives_gamma = True

    def __init__(self, cfg, exact_fields, ls=(2,), n_radial: int = 32):
        self.cfg = cfg
        self.exact = exact_fields
        self.ls = tuple(int(l) for l in ls)
        self.n = int(n_radial)
        self.nf = {l: (3 if l == 0 else 5) for l in self.ls}
        self.size = sum(self.nf[l] * self.n for l in self.ls)
        self.lo, self.hi = float(np.log(cfg.rho_in)), float(np.log(cfg.rho_out))

    def unpack(self, c):
        out, i = {}, 0
        for l in self.ls:
            m = self.nf[l] * self.n
            out[l] = c[i:i + m].reshape(self.nf[l], self.n)
            i += m
        return out

    def delta(self, c, x):
        """(delta h, delta lambda) at one point."""
        rho = jnp.linalg.norm(x)
        n = x / rho
        mu = n[2]
        s = 2.0 * (jnp.log(rho) - self.lo) / (self.hi - self.lo) - 1.0
        T = chebyshev(self.n, s)
        e = ZHAT - mu * n
        nn = jnp.outer(n, n)
        dh = jnp.zeros((3, 3))
        dl = 0.0
        for l, blk in self.unpack(c).items():
            f = blk @ T
            P, dP, ddP = legendre_with_derivs(l, mu)
            dl = dl + f[0] * P
            dh = dh + f[1] * P * I3 + f[2] * P * nn
            if l > 0:
                dh = dh + f[3] * dP * (jnp.outer(e, n) + jnp.outer(n, e))
                dh = dh + f[4] * ddP * (jnp.outer(e, e) - 0.5 * (1.0 - mu * mu) * (I3 - nn))
        return dh, dl

    def fields_of(self, c):
        def h_of(y):
            return self.exact(y).h + self.delta(c, y)[0]

        def fields(x):
            return Fields(h_of(x), christoffel(h_of, x), self.exact(x).lam + self.delta(c, x)[1])
        return fields

    def apply(self, c, X):
        return jax.vmap(self.fields_of(c))(X)


def solve(cfg, exact_fields, batch, ls=(2,), n_radial=32, nonlinear=False, iters=8,
          weights=None, verbose=True, c0=None):
    """Least-squares solve of the run's residual over the spectral field.  Returns (model, c, info)."""
    from .losses import residual_vector

    model = SpectralAxisym(cfg, exact_fields, ls=ls, n_radial=n_radial)
    lam_inf = jnp.asarray(cfg.lam_inf if cfg.lam_inf is not None else cfg.lam_inf_init)

    @jax.jit
    def res(c):
        return residual_vector({"net": c}, batch, cfg, model, exact_fields, weights=weights,
                               lam_inf=lam_inf)

    from .dsgnar import jvp_columns
    # J column by column in blocks of 32 tangents (a full jacfwd pushes every unknown through the
    # residual at once -- the pattern that ran DSGNAR out of memory on the hub's 13.5 GiB share)
    eye = jnp.eye(model.size)
    jac = jax.jit(lambda c: jvp_columns(res, c, eye, 32).T)
    c = jnp.zeros(model.size) if c0 is None else jnp.asarray(c0)
    r = res(c)
    info = {"r0_norm": float(jnp.linalg.norm(r)), "steps": [], "n_unknowns": model.size,
            "n_rows": int(r.shape[0])}
    if verbose:
        print(f"[linref] modes l = {model.ls}, {model.n} radial each: {model.size} unknowns, "
              f"{r.shape[0]} residual rows; |r(background)| = {info['r0_norm']:.3e}", flush=True)
    for it in range(iters if nonlinear else 1):
        t0 = time.time()
        J = jac(c)
        sv = np.linalg.svd(np.asarray(J), compute_uv=False)
        dc, *_ = np.linalg.lstsq(np.asarray(J), -np.asarray(r), rcond=1e-13)
        lin = float(np.linalg.norm(np.asarray(r) + np.asarray(J) @ dc))
        c = c + jnp.asarray(dc)
        r = res(c)
        step = {"iter": it, "linear_residual": lin, "residual": float(jnp.linalg.norm(r)),
                "loss": float(jnp.sum(r * r)), "sv_max": float(sv[0]), "sv_min": float(sv[-1]),
                "cond": float(sv[0] / max(sv[-1], 1e-300)), "wall": time.time() - t0}
        info["steps"].append(step)
        if verbose:
            print(f"[linref] {'Newton ' + str(it) if nonlinear else 'linear'}: "
                  f"|r0 + J dc| = {lin:.3e}, |r(c)| = {step['residual']:.3e} "
                  f"(loss {step['loss']:.3e}), cond(J) = {step['cond']:.2e}  "
                  f"({step['wall']:.1f}s)", flush=True)
        if nonlinear and it > 0 and step["residual"] > 0.999 * info["steps"][-2]["residual"]:
            break
    return model, c, info


def multipole(f, rho, l, n_gl=48):
    """(2l+1)/2 int_{-1}^{1} f(rho, mu) P_l(mu) dmu on Gauss-Legendre nodes (axisymmetric)."""
    mu, w = np.polynomial.legendre.leggauss(n_gl)
    xs = np.stack([rho * np.sqrt(1 - mu ** 2), np.zeros_like(mu), rho * mu], axis=-1)
    vals = np.asarray(f(jnp.asarray(xs)))
    P = np.asarray(jax.vmap(lambda m: legendre_with_derivs(l, m)[0])(jnp.asarray(mu)))
    return (2 * l + 1) / 2 * np.sum(vals * P * w)


def farfield(ref_fields, cfg, radii=(10.0, 20.0, 30.0, 50.0, 70.0, 100.0)):
    """Is the far field a decaying tail?  Per radius: the P_2 coefficient of the GH potential P,
    its log-slope (a tail of a2 rho^-3 gives -3), and the P_0 and P_2 parts of the outer Robin
    operator on P (the run's own kernel), which a decaying tail makes ~0 at EVERY radius, not
    only on the shells where it is imposed.  Radii are physical-chart radii scaled to the run's
    [rho_in, rho_out]."""
    from .losses import drop_power, gh_potential, robin_operator_exps
    orders = cfg.robin_orders or {"lam": cfg.robin_order}
    exps = [cfg.robin_exps["lam"] + i for i in range(orders["lam"])]
    if getattr(cfg, "robin_lam_drop_dipole", True):
        exps = drop_power(exps, 2.0)
    P = lambda y: gh_potential(ref_fields(y).lam)
    rob = jax.jit(jax.vmap(lambda x: robin_operator_exps(P, x, exps, 0.0)))
    pv = jax.jit(jax.vmap(P))
    scale = float(cfg.rho_out) / 100.0
    rows = []
    for r in radii:
        rho = r * scale
        a2 = multipole(pv, rho, 2)
        a2p = multipole(pv, rho * 1.02, 2)
        slope = float(np.log(abs(a2p) / abs(a2)) / np.log(1.02)) if a2 and a2p else float("nan")
        rows.append({"rho": rho, "P2": a2, "slope": slope,
                     "robin_l0": multipole(rob, rho, 0), "robin_l2": multipole(rob, rho, 2)})
    return rows


def _bins(xs, dh, dl, ph, pl, cfg):
    """The same rms numbers per radial region: the shell-wide rms is dominated by rho < 3, where
    the fields are largest, and says nothing about the far field."""
    rho = np.asarray(jnp.linalg.norm(xs, axis=-1))
    edges = [e for e in (cfg.rho_in, 3.0, 10.0, 30.0, cfg.rho_out)
             if cfg.rho_in <= e <= cfg.rho_out]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rho >= lo) & (rho <= hi)
        if not m.any():
            continue
        r = lambda v: float(np.sqrt(np.mean(np.asarray(v)[m] ** 2)))
        out.append({"rho_lo": float(lo), "rho_hi": float(hi), "n": int(m.sum()),
                    "rms_dh": r(dh), "rms_dlam": r(dl),
                    "perturbation_rms_h": r(ph), "perturbation_rms_lam": r(pl)})
    return out


def load_reference(run_dir, cfg, exact_fields, tag="nonlinear", path=None):
    """The spectral reference saved in `<run_dir>/linref_<tag>.npz`, as point fields.

    Returns `(fields, info)`, or `(None, reason)` when there is none or it does not belong to this
    run's problem.  The check is the one datum that tells problems apart: the reference must carry
    THIS run's inner lambda (lam0, S1, S2) to the accuracy of its own solve -- a reference copied
    from another run with different data misses it by ~|delta S2| ~ 1e-2.
    """
    from .problem import lam_inner_bc, sample_sphere
    path = path or os.path.join(run_dir, f"linref_{tag}.npz")
    if not os.path.exists(path):
        return None, f"no {os.path.basename(path)} (run `python -m stationary.linref {run_dir} --nonlinear`)"
    if exact_fields is None:
        return None, "the run has no spherical background to perturb"
    z = np.load(path)
    model = SpectralAxisym(cfg, exact_fields, ls=tuple(int(v) for v in z["ls"]),
                           n_radial=int(z["n_radial"]))
    fields = model.fields_of(jnp.asarray(z["c"]))
    xs = sample_sphere(jax.random.PRNGKey(3), 256, cfg.rho_in)
    miss = float(jnp.max(jnp.abs(jax.vmap(lambda x: fields(x).lam - lam_inner_bc(x, cfg))(xs))))
    if miss > 1e-3:
        return None, (f"{os.path.basename(path)} does not carry this run's inner lambda data "
                      f"(max miss {miss:.2e}): it was solved for a different problem")
    info = json.loads(str(z["info"])) if "info" in z.files else {}
    loss = info["steps"][-1]["loss"] if info.get("steps") else float("nan")
    return fields, {"path": path, "loss": loss, "inner_lam_miss": miss}


def compare(run_fields, ref_fields, exact_fields, cfg, n=4096, seed=999, modes=True):
    """The run's network against the reference: pointwise, per radial region, and mode by mode."""
    from .problem import sample_shell
    xs = sample_shell(jax.random.PRNGKey(seed), n, cfg)

    def one(x):
        a, b, e = run_fields(x), ref_fields(x), exact_fields(x)
        return (jnp.max(jnp.abs(a.h - b.h)), jnp.abs(a.lam - b.lam),
                jnp.max(jnp.abs(b.h - e.h)), jnp.abs(b.lam - e.lam))

    dh, dl, ph, pl = jax.vmap(one)(xs)
    out = {"rms_dh": float(jnp.sqrt(jnp.mean(dh ** 2))), "max_dh": float(jnp.max(dh)),
           "rms_dlam": float(jnp.sqrt(jnp.mean(dl ** 2))), "max_dlam": float(jnp.max(dl)),
           "perturbation_rms_h": float(jnp.sqrt(jnp.mean(ph ** 2))),
           "perturbation_rms_lam": float(jnp.sqrt(jnp.mean(pl ** 2))),
           "bins": _bins(xs, dh, dl, ph, pl, cfg)}
    if not modes:
        return out
    out["modes"] = {}
    radii = [r for r in (cfg.rho_in, 2.0, 5.0, 10.0, 25.0, 50.0, cfg.rho_out)
             if cfg.rho_in <= r <= cfg.rho_out]
    for name, get in (("lam", lambda F: F.lam), ("h_rr", None)):
        for l in (0, 1, 2, 4):
            rows = []
            for rho in radii:
                def f_of(fields):
                    def f(X):
                        def g(x):
                            F = fields(x)
                            if name == "lam":
                                return F.lam
                            nvec = x / jnp.linalg.norm(x)
                            return nvec @ F.h @ nvec
                        return jax.vmap(g)(X)
                    return f
                rows.append((rho, multipole(f_of(run_fields), rho, l),
                             multipole(f_of(ref_fields), rho, l)))
            out["modes"][f"{name}_l{l}"] = rows
    return out


def profiles(run_fields, ref_fields, exact_fields, cfg, n_rho=48, n_gl=32):
    """Radial profiles on log-spaced spheres: P_2 coefficients and angular rms differences.

    Returns arrays over rho: the P_2 coefficient of lambda and of h_rr for the network and the
    reference, and the angular rms of |lam_net - lam_ref| and max_ij |h_net - h_ref| next to the
    same for the perturbation itself (reference - spherical), which is the scale they are read on.
    """
    mu, w = np.polynomial.legendre.leggauss(n_gl)
    P2 = 0.5 * (3 * mu ** 2 - 1)
    rhos = np.geomspace(cfg.rho_in, cfg.rho_out, n_rho)

    def at(x):
        a, b, e = run_fields(x), ref_fields(x), exact_fields(x)
        n = x / jnp.linalg.norm(x)
        return (a.lam, b.lam, n @ a.h @ n, n @ b.h @ n,
                jnp.abs(a.lam - b.lam), jnp.max(jnp.abs(a.h - b.h)),
                jnp.abs(b.lam - e.lam), jnp.max(jnp.abs(b.h - e.h)))

    f = jax.jit(jax.vmap(at))
    out = {k: [] for k in ("lam2_net", "lam2_ref", "hrr2_net", "hrr2_ref",
                           "dlam_rms", "dh_rms", "pert_lam_rms", "pert_h_rms")}
    for rho in rhos:
        xs = jnp.asarray(np.stack([rho * np.sqrt(1 - mu ** 2), np.zeros_like(mu), rho * mu], -1))
        v = [np.asarray(t) for t in f(xs)]
        coef = lambda g: 2.5 * np.sum(g * P2 * w)
        ms = lambda g: np.sqrt(0.5 * np.sum(g ** 2 * w))      # angular rms (axisymmetric)
        out["lam2_net"].append(coef(v[0])); out["lam2_ref"].append(coef(v[1]))
        out["hrr2_net"].append(coef(v[2])); out["hrr2_ref"].append(coef(v[3]))
        out["dlam_rms"].append(ms(v[4])); out["dh_rms"].append(ms(v[5]))
        out["pert_lam_rms"].append(ms(v[6])); out["pert_h_rms"].append(ms(v[7]))
    out = {k: np.asarray(v) for k, v in out.items()}
    out["rho"] = rhos
    return out


def plot_profiles(prof, path, title):
    """Four panels against rho (log-log): the run against the reference, and the size of the miss.

    Network solid blue, reference / perturbation dashed orange (validated categorical pair; the
    line style is the secondary encoding).  One quantity per axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e4e3de"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    rho = prof["rho"]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    panels = [
        (axs[0, 0], "|lambda P2 coefficient|", [("network", np.abs(prof["lam2_net"]), BLUE, "-"),
                                                ("reference", np.abs(prof["lam2_ref"]), ORANGE, "--")]),
        (axs[0, 1], "|h_rr P2 coefficient|", [("network", np.abs(prof["hrr2_net"]), BLUE, "-"),
                                              ("reference", np.abs(prof["hrr2_ref"]), ORANGE, "--")]),
        (axs[1, 0], "lambda: angular rms", [("network - reference", prof["dlam_rms"], BLUE, "-"),
                                            ("perturbation (reference - spherical)",
                                             prof["pert_lam_rms"], ORANGE, "--")]),
        (axs[1, 1], "h: angular rms of max_ij |.|", [("network - reference", prof["dh_rms"], BLUE, "-"),
                                                     ("perturbation (reference - spherical)",
                                                      prof["pert_h_rms"], ORANGE, "--")]),
    ]
    for ax, ylabel, series in panels:
        for label, y, color, ls in series:
            ax.loglog(rho, np.maximum(y, 1e-18), ls, color=color, lw=2, label=label)
        ax.set_xlabel("rho")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.legend(frameon=False, fontsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax = axs[0, 0]
    r3 = rho ** -3.0 * np.abs(prof["lam2_ref"][0]) * rho[0] ** 3
    ax.loglog(rho, r3, ":", color=MUTED, lw=1, label="rho^-3 (linear theory)")
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    from .evaluate import load_run
    from .model import point_fields
    from .train import exact_asset, make_batch

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="a run directory (config.json, params.pkl)")
    ap.add_argument("--ls", default=None, help="Legendre modes, e.g. 2 or 0,2,4,6")
    ap.add_argument("--nonlinear", action="store_true", help="Gauss-Newton to the full solution")
    ap.add_argument("--n-radial", type=int, default=32)
    ap.add_argument("--n-coll", type=int, default=3000)
    ap.add_argument("--n-bnd", type=int, default=400)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--params", default="params.pkl")
    ap.add_argument("--scale-exps", default=None,
                    help="override the run's rho exponents compat,ricci,gauge,lam_eq, e.g. 3,4,3,4")
    ap.add_argument("--outer-shell-fracs", default=None,
                    help="override the Robin ladder (fractions of rho_out), e.g. 1,0.9,0.8,0.7,0.5,0.35")
    ap.add_argument("--tag", default="",
                    help="suffix of the output files, linref_nonlinear_<tag>.*, so variants do not "
                         "overwrite the reference")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse <run>/linref_<mode>.npz if present instead of solving again")
    a = ap.parse_args(argv)

    cfg, model, state, run_pf = None, None, None, None
    cfg, model, state = load_run(a.run, a.params)[:3]
    ls = tuple(int(v) for v in (a.ls or ("0,2,4,6" if a.nonlinear else "2")).split(","))
    exact_fields = exact_asset(cfg)
    if exact_fields is None:
        raise SystemExit("the run has no spherical background (--ref-solution): nothing to perturb")
    over = {"n_coll": a.n_coll, "n_bnd": a.n_bnd, "n_bnd_outer": a.n_bnd}
    if a.scale_exps:
        over["scale_exps"] = dict(zip(["compat", "ricci", "gauge", "lam_eq"],
                                      [float(v) for v in a.scale_exps.split(",")]))
    if a.outer_shell_fracs:
        over["outer_shell_fracs"] = tuple(float(v) for v in a.outer_shell_fracs.split(","))
    cfg_s = cfg.__class__(**{**cfg.__dict__, **over})
    if len(over) > 3:
        print(f"[linref] overrides: { {k: v for k, v in over.items() if k not in ('n_coll', 'n_bnd', 'n_bnd_outer')} }")
    batch = make_batch(jax.random.PRNGKey(cfg.seed + 4242), cfg_s)
    tag = ("nonlinear" if a.nonlinear else "linear") + (f"_{a.tag}" if a.tag else "")
    npz = os.path.join(a.run, f"linref_{tag}.npz")
    if a.reuse and os.path.exists(npz):
        z = np.load(npz)
        ref_model = SpectralAxisym(cfg_s, exact_fields, ls=tuple(int(v) for v in z["ls"]),
                                   n_radial=int(z["n_radial"]))
        c, info = jnp.asarray(z["c"]), json.loads(str(z["info"]))
        print(f"[linref] reusing {npz}")
    else:
        ref_model, c, info = solve(cfg_s, exact_fields, batch, ls=ls, n_radial=a.n_radial,
                                   nonlinear=a.nonlinear, iters=a.iters)
        np.savez(npz, c=np.asarray(c), ls=np.asarray(ls), n_radial=a.n_radial,
                 info=json.dumps(info))

    print("\n[linref] far field of the reference (GH potential P; a decaying tail has P2 slope -3 "
          "and Robin ~0 at every radius):")
    print("    rho        P2 coeff    slope    Robin(P) l=0   Robin(P) l=2")
    ff = farfield(ref_model.fields_of(c), cfg_s)
    for row in ff:
        print(f"    {row['rho']:6.1f}  {row['P2']: .3e}  {row['slope']: 6.2f}   {row['robin_l0']: .2e}"
              f"      {row['robin_l2']: .2e}")
    run_fields = point_fields(model, state["net"])
    cmp_ = compare(run_fields, ref_model.fields_of(c), exact_fields, cfg)
    print(f"\n[linref] {a.run} against the {tag} reference (S2 = {cfg.lam_bc_S2:g}):")
    print(f"  size of the perturbation itself (reference - spherical): "
          f"rms |dh| {cmp_['perturbation_rms_h']:.3e}, rms |dlam| {cmp_['perturbation_rms_lam']:.3e}")
    print(f"  network - reference: rms |dh| {cmp_['rms_dh']:.3e} (max {cmp_['max_dh']:.3e}), "
          f"rms |dlam| {cmp_['rms_dlam']:.3e} (max {cmp_['max_dlam']:.3e})")
    print("  modes (coefficient of P_l on the sphere):   rho      network      reference")
    for key, rows in cmp_["modes"].items():
        for rho, a_net, a_ref in rows:
            print(f"    {key:<9} {rho:8.2f}  {a_net:+.4e}  {a_ref:+.4e}")
    prof = profiles(run_fields, ref_model.fields_of(c), exact_fields, cfg)
    png = os.path.join(a.run, f"linref_{tag}.png")
    plot_profiles(prof, png, f"{os.path.basename(os.path.normpath(a.run))} against the {tag} "
                             f"reference (S2 = {cfg.lam_bc_S2:g}; loss of the reference "
                             f"{info['steps'][-1]['loss']:.1e})")
    print(f"[linref] figure: {png}")
    with open(os.path.join(a.run, f"linref_{tag}.json"), "w") as fh:
        json.dump({"info": info, "compare": cmp_, "farfield": ff,
                   "profiles": {k: np.asarray(v).tolist() for k, v in prof.items()}}, fh, indent=1)


if __name__ == "__main__":
    main()
