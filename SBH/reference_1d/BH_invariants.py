#!/usr/bin/env python3
"""Gauge-invariant diagnostics of a BH PINN run vs time (2026-10-18; cf. Boson BS_invariants_eval.py,
PINN_LESSONS 20.4-20.5, BH/BH_INVARIANTS.md).

For every window, at --nt times from its start to its end (so growth INSIDE windows and jumps at
the hand-overs are visible), and for every puncture:
  1. Hawking mass m_H of coordinate spheres of radius r around the puncture's tracked position
     (the window's own track, interpolated in time):
         m_H = sqrt(A/16pi) [1 + (1/16pi) \\oint (P^2 - H^2) dA],
         H = D_i s^i (mean curvature of S in the slice), P = K - K_ij s^i s^j,
     s = outward unit normal (metric gamma_ij = gt_ij/chi). First derivatives of the network only
     (jax.jacfwd). In spherical symmetry m_H = Misner-Sharp mass; for a single puncture it equals M
     at EVERY radius and time (Schwarzschild in any slicing), so m_H - M is pure physical error.
     For a binary: small spheres measure each hole's mass (erasure -> m_H falls); large spheres
     around the origin approach the total (ADM) energy.
  2. Lapse at the puncture (r = 0.03 above it) and the tracked positions (translation mode:
     single puncture |x_p|; equal-mass binary: mirror sums x0+x1, y0+y1).
Writes <out>.csv and <out>.png.

Usage (CPU):  JAX_PLATFORMS=cpu python3 BH_invariants.py RUN_DIR --out RUN_DIR/analysis/BH_invariants
              [--radii 0.3,1,4,8] [--big_radii 15] [--nt 5] [--n_dir 200] [--every 1]
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def fib_dirs(n):
    i = np.arange(n) + 0.5
    ph = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--radii", default="0.3,1,4,8", help="sphere radii around each puncture")
    ap.add_argument("--big_radii", default="15", help="sphere radii around the origin (total mass)")
    ap.add_argument("--nt", type=int, default=5, help="times per window, start..end inclusive")
    ap.add_argument("--n_dir", type=int, default=200)
    ap.add_argument("--every", type=int, default=1, help="use every k-th window")
    ap.add_argument("--mass", type=float, default=None, help="expected mass per hole (default: BL masses)")
    ap.add_argument("--plot_only", action="store_true", help="re-plot <out>.csv without recomputing")
    a = ap.parse_args()
    out = a.out or os.path.join(a.run_dir, "analysis", "BH_invariants")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    radii = [float(s) for s in a.radii.split(",") if s]
    big = [float(s) for s in a.big_radii.split(",") if s]
    if a.plot_only:
        import csv, pickle
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(out + ".csv"))]
        cfg = pickle.load(open(os.path.join(a.run_dir, "pinn_checkpoint.pkl"), "rb"))["model_config"]
        n_p = int(cfg.get("n_stars", 1))
        m_ = np.ravel(cfg["BL_masses"]).astype(float)[:n_p]
        P_ = np.hypot(np.ravel(cfg.get("BL_momenta", [0.0] * n_p))[:n_p], np.ravel(cfg.get("BL_momenta_y", [0.0] * n_p))[:n_p])
        plot(rows, n_p, radii, big, m_ if a.mass is None else np.full(n_p, a.mass), out, a.run_dir, P_)
        return

    import compare_pinn as cp
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    man, wins = cp.load_run(a.run_dir)
    cfg = man["model_config"]
    ns = module_for(cfg)
    full_output, unpack = ns["full_output"], ns["unpack_sym3"]
    masses = np.ravel(cfg["BL_masses"]).astype(float)
    n_p = int(cfg.get("n_stars", len(masses)))
    masses = masses[:n_p]
    momenta = np.hypot(np.ravel(cfg.get("BL_momenta", [0.0] * n_p)).astype(float)[:n_p],
                       np.ravel(cfg.get("BL_momenta_y", [0.0] * n_p)).astype(float)[:n_p])

    dirs = fib_dirs(a.n_dir)

    def make_point_fns(w, t):
        fn, p = ns["_wrap_moving_chi_bg_if_needed"](ns["apply_fn"], w["params"], w.get("traj_tt"), w.get("traj_xx"))
        ws, we = jnp.array(w["t_start"]), jnp.array(w["t_end"])
        T = jnp.full((1, 1), float(t))

        def out1(x):            # (3,) -> (22,)
            return full_output(fn, p, T, x[0].reshape(1, 1), x[1].reshape(1, 1), x[2].reshape(1, 1), ws, we)[0]

        def geom(x, c):
            o = out1(x)
            gt = unpack(o[0:6]); At = unpack(o[6:12]); chi = o[12]; K = o[13] + 2.0 * o[14]
            gam = gt / chi
            gami = jnp.linalg.inv(gam)
            sqg = jnp.sqrt(jnp.linalg.det(gam))
            dF = (x - c) / jnp.linalg.norm(x - c)            # gradient of F = |x - c| - r (flat unit)
            nrm = jnp.sqrt(dF @ gami @ dF)
            s_up = gami @ dF / nrm
            Kij = (At + gt * K / 3.0) / chi
            P = K - s_up @ Kij @ s_up
            return sqg * s_up, sqg, nrm, P

        def per_point(x, c):
            v_fun = lambda xx: geom(xx, c)[0]
            J = jax.jacfwd(v_fun)(x)                         # d_k (sqrt(g) s^i), shape (3,3)
            v, sqg, nrm, P = geom(x, c)
            H = jnp.trace(J) / sqg
            dA = sqg * nrm                                   # area element per unit flat area
            return H, P, dA

        batched = jax.jit(jax.vmap(per_point, in_axes=(0, None)))

        def alpha_at(xyz):
            X = jnp.asarray(xyz)
            o = full_output(fn, p, jnp.full((X.shape[0], 1), float(t)), X[:, :1], X[:, 1:2], X[:, 2:3], ws, we)
            return np.asarray(o[:, 18])
        return batched, alpha_at

    def hawking(batched, c, r):
        c = jnp.asarray(c, dtype=jnp.float64)
        X = c[None, :] + r * jnp.asarray(dirs)
        H, P, dA = (np.asarray(v) for v in batched(X, c))
        w_flat = 4 * np.pi * r * r / len(dirs)
        A = np.sum(dA) * w_flat
        I = np.sum((P ** 2 - H ** 2) * dA) * w_flat
        return np.sqrt(A / (16 * np.pi)) * (1 + I / (16 * np.pi))

    def centres(w, t):
        tt, xx = w.get("traj_tt"), w.get("traj_xx")
        if tt is None:
            return np.array([np.array(cfg.get("star_centers", [[0, 0, 0]])[k])[:3] for k in range(n_p)], float)
        tt = np.asarray(tt); xx = np.asarray(xx)
        return np.array([[np.interp(t, tt, xx[k, 0]), np.interp(t, tt, xx[k, 1]), 0.0] for k in range(n_p)])

    rows = []
    for iw, w in enumerate(wins):
        if iw % a.every and iw != len(wins) - 1:
            continue
        for f in np.linspace(0.0, 1.0, a.nt):
            t = float(w["t_start"] + f * (w["t_end"] - w["t_start"]))
            batched, alpha_at = make_point_fns(w, t)
            C = centres(w, t)
            row = {"window": iw, "t": t, "frac": float(f)}
            for k in range(n_p):
                row[f"x{k}"], row[f"y{k}"] = C[k, 0], C[k, 1]
                row[f"alpha{k}"] = float(alpha_at(C[k][None, :] + np.array([[0, 0, 0.03]]))[0])
                for r in radii:
                    row[f"mH{k}_r{r:g}"] = float(hawking(batched, C[k], r))
            for R in big:
                row[f"mH_origin_R{R:g}"] = float(hawking(batched, np.zeros(3), R))
            rows.append(row)
        print(f"window {iw}: t={rows[-1]['t']:.2f} " + " ".join(
            f"{k}={v:.4f}" for k, v in rows[-1].items() if k.startswith("mH")), flush=True)

    keys = list(rows[0].keys())
    with open(out + ".csv", "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r_ in rows:
            fh.write(",".join(f"{r_[k]:.8g}" for k in keys) + "\n")
    plot(rows, n_p, radii, big, masses if a.mass is None else np.full(n_p, a.mass), out, a.run_dir, momenta)


def plot(rows, n_p, radii, big, masses, out, run_dir, momenta=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([r["t"] for r in rows]); win = np.array([r["window"] for r in rows])
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    def by_window(a_, y, **kw):
        lab = kw.pop("label", None)
        for iw in np.unique(win):
            m = win == iw
            a_.plot(t[m], y[m], label=lab if iw == win[0] else None, **kw)

    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # (a) hole mass from the innermost sphere (relative error), per puncture
    a0 = ax[0, 0]
    for k in range(n_p):
        y = np.array([r[f"mH{k}_r{radii[0]:g}"] for r in rows]) / masses[k] - 1
        by_window(a0, y, color=cols[k], lw=1.5, label=f"hole {k}, r = {radii[0]:g}")
    a0.axhline(0, color="0.5", lw=0.8)
    if momenta is not None and np.any(momenta > 0):
        e_kin = np.sqrt(masses[0] ** 2 + momenta[0] ** 2) / masses[0] - 1
        a0.axhline(e_kin, color="0.5", lw=0.8, ls=":")
        a0.text(t.min(), e_kin, f" sqrt(M^2+P^2)/M - 1 = {e_kin:.3f} (boosted hole)", fontsize=7, va="bottom", color="0.4")
        a0.set_title("Hole mass: Hawking mass of a sphere / M - 1\n(coordinate sphere: includes kinetic energy, slicing dependent)")
    else:
        a0.set_title("Hole mass: Hawking mass / M - 1 (exact 0 for a single puncture)")
    a0.set_xlabel("t (M)"); a0.legend(fontsize=8)
    # (b) Hawking mass at all radii, hole 0 (constraint violation by radius) + big spheres
    a1 = ax[0, 1]
    for j, r_ in enumerate(radii):
        y = np.array([r[f"mH0_r{r_:g}"] for r in rows]) / masses[0] - 1
        by_window(a1, y, color=cols[j % len(cols)], lw=1.2, label=f"hole 0, r = {r_:g}")
    for j, R in enumerate(big):
        y = np.array([r[f"mH_origin_R{R:g}"] for r in rows])
        y0 = y[0]
        by_window(a1, y / y0 - 1, color="k", lw=1.2, ls=["--", ":", "-."][j % 3], label=f"origin, R = {R:g} (rel. to t = 0)")
    a1.axhline(0, color="0.5", lw=0.8)
    a1.set_title("Hawking mass by radius (relative error)")
    a1.set_xlabel("t (M)"); a1.legend(fontsize=8)
    # (c) lapse at the punctures
    a2 = ax[1, 0]
    for k in range(n_p):
        by_window(a2, np.array([r[f"alpha{k}"] for r in rows]), color=cols[k], lw=1.5, label=f"hole {k}")
    ref = os.path.join(HERE, "ref_single_m0.5_dr0.005.npz")
    if n_p == 1 and os.path.exists(ref) and abs(masses[0] - 0.5) < 1e-12:
        d = np.load(ref); F = list(d["fields"])
        a2.plot(d["t"], [np.interp(0.03, d["r"], d["U"][k, F.index("alpha")]) for k in range(len(d["t"]))],
                "k--", lw=1, label="1D reference")
    a2.axhline(0, color="0.5", lw=0.8)
    a2.set_title("Lapse at the punctures (r = 0.03)"); a2.set_xlabel("t (M)"); a2.legend(fontsize=8)
    # (d) translation mode
    a3 = ax[1, 1]
    if n_p == 1:
        by_window(a3, np.hypot([r["x0"] for r in rows], [r["y0"] for r in rows]), color=cols[0], lw=1.5,
                  label="|tracked position| (exact 0)")
        a3.set_title("Translation mode: drift of the puncture")
    else:
        sx = np.array([r["x0"] + r["x1"] for r in rows]); sy = np.array([r["y0"] + r["y1"] for r in rows])
        by_window(a3, sx, color=cols[0], lw=1.5, label="x0 + x1 (exact 0, equal masses)")
        by_window(a3, sy, color=cols[1], lw=1.5, label="y0 + y1")
        a3.set_title("Translation mode: mirror symmetry of the tracks")
    a3.axhline(0, color="0.5", lw=0.8); a3.set_xlabel("t (M)"); a3.legend(fontsize=8)
    fig.suptitle(f"Gauge-invariant diagnostics: {os.path.basename(os.path.normpath(run_dir))}")
    fig.savefig(out + ".png", dpi=130)
    print("wrote", out + ".csv", out + ".png")


def module_for(cfg):
    """Load the BH script's definitions with the run's network/domain/puncture settings
    (as compare_pinn.module_for, but for any number of punctures and momenta)."""
    from load_module import load
    dom = cfg.get("domain", {})
    argv = ["--N_LAYERS", str(cfg["N_LAYERS"]), "--HIDDEN", str(cfg["HIDDEN"]),
            "--ic_mode", cfg["ic_mode"], "--N_window", str(cfg["N_window"]),
            "--tfinal", repr(float(dom["tfinal"]))]
    if "R" in dom:
        argv += ["--R_domain", repr(float(dom["R"]))]
    def lst(key):
        v = np.ravel(cfg.get(key, [0.0])).astype(float)
        return ",".join(repr(float(x)) for x in v)
    # "--opt=value" so that negative lists such as -0.16,0.16 are not taken for options
    argv += ["--n_stars", str(int(cfg.get("n_stars", 1))), "--BL_masses=" + lst("BL_masses"),
             "--BL_momenta=" + lst("BL_momenta"), "--BL_momenta_y=" + lst("BL_momenta_y")]
    for key in ("star_separation", "chi_bg_mode", "bh_center_mode"):
        if key in cfg:
            argv += ["--" + key, str(cfg[key])]
    return load(argv)


if __name__ == "__main__":
    main()
