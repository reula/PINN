"""Fit test for BH (PINN_LESSONS 15.1 / 20.2), window 0 of the single puncture.

Question: how well can THIS network represent the answer if training were perfect?
It fits the PINN's own network by plain regression to the 1D reference on window 0
and prints the fit next to the PINN checkpoint of the same window, with the same
error measures on the same held-out points.

  - Same network: architecture, input scaling, flat-space output offsets and the
    moving chi background of the BH code (loaded from the code itself), initialised
    with model.init as window 0 of the PINN.
  - Target: the 1D reference (sph_reference.py: the code's own CCZ4 RHS and gauge),
    dense in time on [0, 0.25] (--snap_dt 0.0025), mapped to the 22 outputs:
    gt = a delta + b nn, At = c delta + d nn, Ghat = g n, beta = B n, chi, Khat,
    Theta, alpha. Bicubic spline in (t, r).
  - Loss: plain regression, mean over points and the 22 outputs of
    (out - target)^2. This is exactly the form of the PINN's soft-IC term
    (_ic_fit with unit weights). No PDE, BC or IC terms, no derivatives.
  - Same training: the code's point sampler (interior + core cloud, IC and
    terminal surfaces, boundary sphere; --min_puncture_dist rejection), the
    code's time schedule (truncated exponential for the first f_uniform of the
    restarts, then uniform), Crunch minimize with the production options
    (ssbroyden2, strong Wolfe, h0 scaling, carried-H rescale, scale-invariant),
    resample every N_change steps keeping H, EPOCHS_QN steps.
  - chi background: the trajectory of the PINN checkpoint (default, so fit and
    PINN share the same background) or the exact one at the origin
    (--traj origin).

Outputs in --out: fit_history.csv, fit_summary.txt, fit_alpha_puncture.csv,
fit_params.pkl, fit_window0.png.

Usage (MN5, 1 GPU; see fit_window.psubmit):
  python reference_1d/fit_window_bh.py RUN_DIR reference_1d/ref_single_m0.5_dr0.005_w0dense.npz --out fit_w0
RUN_DIR is a single-puncture PINN run with pinn_checkpoint_w0000.pkl.
CPU smoke test:
  JAX_PLATFORMS=cpu python3 reference_1d/fit_window_bh.py RUN_DIR REF --out /tmp/x --steps 500 \
      --Nint 2000 --NI 400 --Nb 400 --core_points 1000 --core_ic_points 200 --heldout 2000
"""
import argparse
import csv
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "tests"))

GROUPS = {"gt": slice(0, 6), "At": slice(6, 12), "chi": slice(12, 13), "Khat": slice(13, 14),
          "Theta": slice(14, 15), "Ghat": slice(15, 18), "alpha": slice(18, 19), "beta": slice(19, 22)}
BANDS = {"core": (0.0, 0.5), "mid": (0.5, 5.0), "far": (5.0, 1e9)}


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir")
    p.add_argument("ref")
    p.add_argument("--out", default="fit_w0")
    p.add_argument("--steps", type=int, default=24000, help="EPOCHS_QN of the PINN")
    p.add_argument("--N_change", type=int, default=250)
    p.add_argument("--Nint", type=int, default=80000)
    p.add_argument("--NI", type=int, default=8000)
    p.add_argument("--Nb", type=int, default=16000)
    p.add_argument("--core_points", type=int, default=48000)
    p.add_argument("--core_ic_points", type=int, default=4800)
    p.add_argument("--core_radius", type=float, default=0.5)
    p.add_argument("--time_dist", default="code", choices=["code", "uniform"],
                   help="code: the PINN's exponential-then-uniform schedule; uniform: uniform throughout")
    p.add_argument("--traj", default="pinn", choices=["pinn", "origin"])
    p.add_argument("--h0_scaling", type=int, default=1)
    p.add_argument("--h_rescale", type=int, default=1)
    p.add_argument("--scale_invariant", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--heldout", type=int, default=40000, help="held-out points (same distribution, uniform t)")
    p.add_argument("--eval_every", type=int, default=8, help="held-out evaluation every this many restarts")
    return p.parse_args()


def load_code(cfg, a):
    """The BH code's definitions with this run's configuration and the fit's point counts."""
    os.environ.setdefault("BH_SCRIPT", os.path.join(ROOT, "EEccz4_BH_puncture.py"))
    from load_module import load
    dom = cfg["domain"]
    argv = ["--N_LAYERS", str(cfg["N_LAYERS"]), "--HIDDEN", str(cfg["HIDDEN"]),
            "--ic_mode", cfg["ic_mode"], "--N_window", str(cfg["N_window"]),
            "--tfinal", repr(float(dom["tfinal"])), "--R_domain", repr(float(dom["R"])),
            "--n_stars", "1", "--BL_masses", repr(float(np.ravel(cfg["BL_masses"])[0])),
            "--BL_momenta", "0", "--BL_momenta_y", "0",
            "--chi_bg_mode", cfg.get("chi_bg_mode", "moving"),
            "--bh_center_mode", cfg.get("bh_center_mode", "shift_driven"),
            "--min_puncture_dist", repr(float(cfg.get("min_puncture_dist", 0.02))),
            "--Nint", str(a.Nint), "--NI", str(a.NI), "--Nb", str(a.Nb),
            "--core_points", str(a.core_points), "--core_ic_points", str(a.core_ic_points),
            "--core_radius", repr(a.core_radius), "--N_change", str(a.N_change),
            "--EPOCHS_QN", str(a.steps), "--SEED", str(a.seed),
            "--h0_scaling", str(a.h0_scaling), "--h_rescale", str(a.h_rescale),
            "--qn_scale_invariant", str(a.scale_invariant),
            "--outdir", os.path.abspath(a.out)]
    if cfg["ic_mode"] != "soft":
        raise SystemExit("fit test written for --ic_mode soft")
    return load(argv)


class Reference:
    """1D reference as a function of (t, r): bicubic splines of the 10 radial fields."""

    def __init__(self, path, t_max):
        from scipy.interpolate import RectBivariateSpline
        d = np.load(path)
        self.mass = float(d["mass"])
        t, r, U = d["t"], d["r"], d["U"]
        if t[-1] < t_max - 1e-12 or len(t) < 8:
            raise SystemExit(f"{path}: needs dense snapshots up to t = {t_max} "
                             "(sph_reference.py --tfinal 0.25 --snap_dt 0.0025)")
        self.t, self.r, self.U = t, r, U
        self.spl = [RectBivariateSpline(t, r, U[:, i, :], kx=3, ky=3) for i in range(10)]

    def radial(self, t, r):
        return np.stack([s.ev(t, r) for s in self.spl])            # (10, N)

    def two_grid_check(self):
        """Time-interpolation error: spline through every other snapshot vs the skipped ones."""
        from scipy.interpolate import RectBivariateSpline
        t2, U2 = self.t[::2], self.U[::2]
        rm = (self.r > 0.02) & (self.r < 24)
        worst = []
        for i in range(10):
            s = RectBivariateSpline(t2, self.r, U2[:, i, :], kx=3, ky=3)
            odd = self.t[1::2]
            T, R = np.meshgrid(odd, self.r[rm], indexing="ij")
            err = np.abs(s.ev(T.ravel(), R.ravel()).reshape(T.shape) - self.U[1::2, i][:, rm])
            worst.append(err.max())
        return worst


def targets22(ref, T, X):
    """(N,22) targets at points T (N,), X (N,3)."""
    r = np.linalg.norm(X, axis=1)
    n = X / r[:, None]
    a, b, c, d, chi, K, Th, g, al, B = ref.radial(T, r)
    nn = n[:, :, None] * n[:, None, :]
    I = np.eye(3)[None]
    gt = a[:, None, None] * I + b[:, None, None] * nn
    At = c[:, None, None] * I + d[:, None, None] * nn
    from_sym = lambda m: np.stack([m[:, 0, 0], m[:, 0, 1], m[:, 0, 2], m[:, 1, 1], m[:, 1, 2], m[:, 2, 2]], 1)
    return np.concatenate([from_sym(gt), from_sym(At), chi[:, None], K[:, None], Th[:, None],
                           g[:, None] * n, al[:, None], B[:, None] * n], axis=1)


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.run_dir, "pinn_checkpoint.pkl"), "rb") as f:
        man = pickle.load(f)
    cfg = man["model_config"]
    if cfg.get("n_stars", 1) != 1 or np.any(np.abs(np.ravel(cfg.get("BL_momenta", [0]))) > 0):
        raise SystemExit("fit test needs a single non-boosted puncture run")
    with open(os.path.join(a.run_dir, "pinn_checkpoint_w0000.pkl"), "rb") as f:
        w0 = pickle.load(f)
    t0, t1 = float(w0["t_start"]), float(w0["t_end"])

    ns = load_code(cfg, a)
    import jax
    import jax.numpy as jnp
    from jax import flatten_util
    jax.config.update("jax_enable_x64", True)
    minimize = ns["minimize"]

    # check the code's output ordering matches the one assumed in targets22
    comps = [tuple(c) for c in ns["CCZ4_METRIC3_COMPONENTS"]]
    assert comps == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)], comps
    assert (ns["CHI_IDX"], ns["ALPHA_IDX"]) == (12, 18), (ns["CHI_IDX"], ns["ALPHA_IDX"])

    ref = Reference(a.ref, t1)
    if not np.isclose(ref.mass, float(np.ravel(cfg["BL_masses"])[0])):
        raise SystemExit("mass mismatch between run and reference")
    tg = ref.two_grid_check()
    print("reference time-interpolation check (dt 0.005 spline vs dt 0.0025 data), max |err| per field:")
    print("   " + "  ".join(f"{f}={e:.1e}" for f, e in zip("a b c d chi Khat Theta g alpha B".split(), tg)))

    # chi background trajectory
    traj_tt = jnp.asarray(w0["traj_tt"])
    traj_xx = jnp.asarray(w0["traj_xx"])
    print(f"PINN w0 trajectory: max |x_p| = {float(jnp.abs(traj_xx).max()):.2e}")
    if a.traj == "origin":
        traj_xx = jnp.zeros_like(traj_xx)
    raw_apply = ns["apply_fn"]
    phys_apply = ns["_wrap_with_moving_chi_bg"](raw_apply) if cfg.get("chi_bg_mode") == "moving" else raw_apply
    bundle = (lambda p: ns["_bundle_params"](p, traj_tt, traj_xx)) if cfg.get("chi_bg_mode") == "moving" else (lambda p: p)
    ws, we = jnp.array(t0), jnp.array(t1)

    def out22(net_params, T, X):
        return phys_apply(bundle(net_params), T, X[:, 0:1], X[:, 1:2], X[:, 2:3], ws, we)

    # same initialisation as window 0 of the PINN
    model = ns["model"]
    one = jnp.ones((4, 1), dtype=jnp.float64)
    params0 = model.init(jax.random.PRNGKey(a.seed), one, one, one, one, ws, we)
    params0 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), params0)
    w_init, unflatten = flatten_util.ravel_pytree(params0)
    print(f"network {cfg['features']}, {w_init.size} parameters")

    @jax.jit
    def loss_fn(w, T, X, Y):
        return jnp.mean((out22(unflatten(w), T, X) - Y) ** 2)

    val_grad = jax.jit(jax.value_and_grad(loss_fn))

    # ---- points: the code's sampler -------------------------------------------------
    centers = [(0.0, 0.0, 0.0)]
    num_outer = max(1, a.steps // a.N_change)
    f_uniform = float(ns.get("f_uniform", 0.7))
    sigma_time_0 = float(ns.get("sigma_time_0", 0.001))
    k_uniform = f_uniform * num_outer

    def sigma_at(k):
        if a.time_dist == "uniform" or k_uniform <= 0 or k >= k_uniform:
            return None
        return (sigma_time_0 + (20.0 - sigma_time_0) * k / k_uniform) * (t1 - t0)

    def draw(key, sigma, nint=None, scale=1.0):
        Xc, key = ns["generate_interior"](nint or a.Nint, key, t0, t1, centers=centers, sigma_time=sigma)
        Xf0, key = ns["generate_ic_surface"](max(1, int(a.NI * scale)), key, t0, centers=centers)
        Xf1, key = ns["generate_ic_surface"](max(1, int(a.NI * scale)), key, t1, centers=centers)
        Xb, key = ns["generate_boundary"](max(1, int(a.Nb * scale)), key, t0, t1)
        P = np.asarray(jnp.vstack((Xc, Xf0, Xf1, Xb)))
        Y = targets22(ref, P[:, 0], P[:, 1:4])
        return jnp.asarray(P[:, :1]), jnp.asarray(P[:, 1:4]), jnp.asarray(Y), key

    # held-out: separate stream, uniform in t, interior+core only, scaled
    if a.heldout > 0:
        sc = a.heldout / (a.Nint + a.core_points)
        saved = ns["core_points"]
        ns["core_points"] = int(round(a.core_points * sc))
        Th, Xh, Yh, _ = draw(jax.random.PRNGKey(a.seed + 777), None, nint=int(round(a.Nint * sc)), scale=0)
        ns["core_points"] = saved
    rh = np.linalg.norm(np.asarray(Xh), axis=1)

    w_pinn, _ = flatten_util.ravel_pytree(jax.tree_util.tree_map(jnp.asarray, w0["params"]))
    if w_pinn.size != w_init.size:
        raise SystemExit("PINN checkpoint and fit network differ in size")
    if a.traj == "origin":
        print("note: --traj origin; the PINN is still evaluated with its own trajectory")

    def pinn_out(T, X):
        b = ns["_bundle_params"](unflatten(w_pinn), jnp.asarray(w0["traj_tt"]), jnp.asarray(w0["traj_xx"])) \
            if cfg.get("chi_bg_mode") == "moving" else unflatten(w_pinn)
        return phys_apply(b, T, X[:, 0:1], X[:, 1:2], X[:, 2:3], ws, we)

    def measures(O):
        """Held-out measures for (N,22) outputs O."""
        E = np.asarray(O) - np.asarray(Yh)
        m = {"loss": float(np.mean(E ** 2))}
        for bn, (lo, hi) in BANDS.items():
            sel = (rh >= lo) & (rh < hi)
            m[f"loss_{bn}"] = float(np.mean(E[sel] ** 2))
            for g, sl in GROUPS.items():
                m[f"{g}_{bn}"] = float(np.sqrt(np.mean(E[sel][:, sl] ** 2)))
        return m

    # alpha at r = 0.03 (and 0.1) along +z vs t inside the window
    t_line = np.linspace(t0, t1, 11)
    def alpha_line(fn, r):
        X = jnp.asarray(np.tile([[0.0, 0.0, r]], (len(t_line), 1)))
        return np.asarray(fn(jnp.asarray(t_line[:, None]), X))[:, 18]

    m_pinn = measures(pinn_out(Th, Xh))
    print(f"PINN w0 on held-out: regression loss {m_pinn['loss']:.3e}  (core {m_pinn['loss_core']:.2e}, "
          f"mid {m_pinn['loss_mid']:.2e}, far {m_pinn['loss_far']:.2e})")

    # ---- training: same loop shape as train_window --------------------------------
    opts = {"maxiter": a.N_change, "gtol": 1e-9, "update_method": "ssbroyden2",
            "initial_scale": bool(a.h0_scaling), "rescale_carried": bool(a.h_rescale),
            "scale_invariant": bool(a.scale_invariant), "alphamin": 1e-15, "max_iter": 100,
            "line_search": "wolfe", "c2": 0.9}
    if a.scale_invariant:
        opts["gtol"] = 0.0   # gtol is absolute (PINN_LESSONS 20.2): never let it stop a small-loss fit
    w = w_init
    H = jnp.eye(w.size, dtype=jnp.float64)
    key = jax.random.PRNGKey(a.seed + 1)
    hist, steps = [], 0
    hist_path = os.path.join(a.out, "fit_history.csv")
    tstart = time.time()
    for k in range(num_outer):
        T, X, Y, key = draw(key, sigma_at(k))
        res = minimize(fun_val=lambda v: loss_fn(v, T, X, Y),
                       fun_val_grad=lambda v: val_grad(v, T, X, Y),
                       x0=w, options=dict(opts, initial_H=H))
        if not (bool(jnp.isfinite(res.fun)) and bool(jnp.all(jnp.isfinite(res.x)))):
            print(f"restart {k}: non-finite, H reset and resample")
            H = jnp.eye(w.size, dtype=jnp.float64)
            continue
        w = res.x
        H = (res.hess_inv + res.hess_inv.T) / 2
        steps += int(res.nit)
        st = getattr(res, "stats", None) or {}
        row = {"restart": k, "steps": steps, "loss": float(res.fun), "nit": int(res.nit),
               "nskip": int(getattr(res, "nskip", st.get("n_skip", -1)) or 0),
               "nreset": int(getattr(res, "nreset", st.get("n_reset", -1)) or 0),
               "nfull": int(getattr(res, "nfull", -1) or 0),
               "wall_s": round(time.time() - tstart, 1)}
        if (k + 1) % a.eval_every == 0 or k == num_outer - 1:
            row.update({f"ho_{kk}": v for kk, v in measures(out22(unflatten(w), Th, Xh)).items()
                        if kk.startswith("loss") or kk.startswith("alpha") or kk.startswith("chi")})
            print(f"restart {k+1}/{num_outer} step {steps}: train {row['loss']:.3e}  held-out {row['ho_loss']:.3e} "
                  f"(core {row['ho_loss_core']:.2e})  skips {row['nskip']} resets {row['nreset']}  "
                  f"full steps {row['nfull']}  {row['wall_s']:.0f}s", flush=True)
        hist.append(row)
        with open(hist_path, "w", newline="") as f:
            keys = sorted({kk for r_ in hist for kk in r_}, key=lambda s: (s.startswith("ho_"), s))
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(hist)

    with open(os.path.join(a.out, "fit_params.pkl"), "wb") as f:
        pickle.dump({"params": jax.tree_util.tree_map(np.asarray, unflatten(w)), "args": vars(a),
                     "traj_tt": np.asarray(traj_tt), "traj_xx": np.asarray(traj_xx)}, f)

    # ---- side by side ---------------------------------------------------------------
    m_fit = measures(out22(unflatten(w), Th, Xh))
    fit_fn = lambda T, X: out22(unflatten(w), T, X)
    lines = [f"Fit test, window 0 [{t0}, {t1}], run {os.path.abspath(a.run_dir)}",
             f"reference {a.ref}; {steps} steps, {num_outer} restarts of {a.N_change}; points "
             f"Nint {a.Nint} + core {a.core_points}, NI {a.NI} + core {a.core_ic_points} (x2 surfaces), Nb {a.Nb}; "
             f"time_dist {a.time_dist}; traj {a.traj}; h0 {a.h0_scaling} rescale {a.h_rescale} SI {a.scale_invariant}",
             f"held-out: {len(rh)} points (uniform t; r<0.5: {(rh<0.5).sum()}, 0.5-5: {((rh>=0.5)&(rh<5)).sum()}, >5: {(rh>=5).sum()})",
             "",
             f"{'measure':<16}{'PINN':>12}{'fit':>12}{'PINN/fit':>10}"]
    for kk in m_pinn:
        lines.append(f"{kk:<16}{m_pinn[kk]:>12.3e}{m_fit[kk]:>12.3e}{m_pinn[kk] / max(m_fit[kk], 1e-300):>10.1f}")
    lines += ["", "alpha along +z vs t (reference | PINN | fit)"]
    rows_a = []
    for r0 in (0.03, 0.1):
        ra = ref.radial(t_line, np.full_like(t_line, r0))[8]
        pa, fa = alpha_line(pinn_out, r0), alpha_line(fit_fn, r0)
        lines.append(f"  r = {r0}: " + "  ".join(f"t={tt:.3f}: {x:.4f}|{y:.4f}|{z:.4f}" for tt, x, y, z in
                                                  list(zip(t_line, ra, pa, fa))[::5]))
        lines.append(f"    max |err| over t: PINN {np.abs(pa - ra).max():.2e}   fit {np.abs(fa - ra).max():.2e}")
        rows_a += [dict(r=r0, t=tt, ref=x, pinn=y, fit=z) for tt, x, y, z in zip(t_line, ra, pa, fa)]
    txt = "\n".join(lines)
    print("\n" + txt)
    open(os.path.join(a.out, "fit_summary.txt"), "w").write(txt + "\n")
    with open(os.path.join(a.out, "fit_alpha_puncture.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["r", "t", "ref", "pinn", "fit"])
        wr.writeheader()
        wr.writerows(rows_a)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        st = [r_["steps"] for r_ in hist]
        ax[0].semilogy(st, [r_["loss"] for r_ in hist], label="fit, training batch")
        ev = [r_ for r_ in hist if "ho_loss" in r_]
        for band, ls in (("", "-"), ("_core", "--"), ("_far", ":")):
            ax[0].semilogy([r_["steps"] for r_ in ev], [r_["ho_loss" + band] for r_ in ev], "k" + ls,
                           label="fit, held-out" + (band.replace("_", " ") or " all"))
            ax[0].axhline(m_pinn["loss" + band], color="C3", ls=ls, lw=1,
                          label="PINN w0" + (band.replace("_", " ") or " all"))
        ax[0].set_xlabel("step"); ax[0].set_ylabel("regression loss (mean over 22 outputs)")
        ax[0].legend(fontsize=7); ax[0].set_title("fit vs PINN, window 0")
        for r0, c in ((0.03, "C0"), (0.1, "C1")):
            ra = ref.radial(t_line, np.full_like(t_line, r0))[8]
            ax[1].semilogy(t_line, np.abs(alpha_line(pinn_out, r0) - ra) + 1e-16, c + "-", label=f"PINN r={r0}")
            ax[1].semilogy(t_line, np.abs(alpha_line(fit_fn, r0) - ra) + 1e-16, c + "--", label=f"fit r={r0}")
        ax[1].set_xlabel("t"); ax[1].set_ylabel("|alpha - ref| on +z"); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "fit_window0.png"), dpi=120)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
