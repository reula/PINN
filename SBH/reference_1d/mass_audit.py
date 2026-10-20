"""Finite-sphere mass audit (2026-10-16): how constant is the mass measured on
a sphere r = R in the moving-puncture gauge, in the 1D reference and in a
PINN run?  Two estimators, per R and t:

  M_psi = -2 R^2 d_r psi,  psi = chi^(-1/4)                (conformally flat)
  M_ADM = (R^2/2) (-A' + B/R),  gamma_ij = A delta_ij + B n_i n_j,
          A = a/chi, B = b/chi                             (flat-space ADM
          surface integral (1/16 pi) \oint (d_j g_ij - d_i g_jj) dS^i for a
          spherically symmetric metric)

Usage: JAX_PLATFORMS=cpu python3 mass_audit.py ref.npz [run_dir] --out prefix
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FIELDS = ("a", "b", "c", "d", "chi", "Khat", "Theta", "g", "alpha", "B")
RADII = (5.0, 10.0, 15.0, 20.0)


def masses_from_radial(r, a, b, chi, R):
    """Both estimators at radius R from radial profiles (FD derivative)."""
    psi = chi ** -0.25
    A = a / chi
    B = b / chi
    dpsi = np.gradient(psi, r)
    dA = np.gradient(A, r)
    m_psi = -2 * R**2 * np.interp(R, r, dpsi)
    m_adm = 0.5 * R**2 * (-np.interp(R, r, dA) + np.interp(R, r, B) / R)
    return m_psi, m_adm


def ref_audit(path):
    d = np.load(path)
    r, t, U = d["r"], d["t"], d["U"]
    ia, ib, ic = FIELDS.index("a"), FIELDS.index("b"), FIELDS.index("chi")
    rows = []
    for k, tk in enumerate(t):
        for R in RADII:
            mp, ma = masses_from_radial(r, U[k, ia], U[k, ib], U[k, ic], R)
            rows.append((float(tk), R, mp, ma))
    return rows, float(d["mass"])


def fib_dirs(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def pinn_audit(run_dir, times, n_dir=200, h=0.05):
    import compare_pinn as cp
    import jax.numpy as jnp
    man, wins = cp.load_run(run_dir)
    cfg = man["model_config"]
    ns = cp.module_for(cfg)
    full_output, unpack = ns["full_output"], ns["unpack_sym3"]

    def eval_window(w, t, xyz):
        if w.get("ic_mode", cfg["ic_mode"]) == "hard":
            fn = ns["make_hard_ic_apply_fn"](ns["model"].apply, w["t_start"], w["t_end"],
                                             ic_history=w.get("ic_history"))
        else:
            fn = ns["apply_fn"]
        fn, p = ns["_wrap_moving_chi_bg_if_needed"](fn, w["params"], w.get("traj_tt"), w.get("traj_xx"))
        X = jnp.asarray(xyz)
        T = jnp.full((X.shape[0], 1), float(t))
        return np.asarray(full_output(fn, p, T, X[:, :1], X[:, 1:2], X[:, 2:3],
                                      jnp.array(w["t_start"]), jnp.array(w["t_end"])))

    dirs = fib_dirs(n_dir)
    rows = []
    for t in times:
        # window whose interval ends at t (t = 0: window 0 at its start)
        w = wins[0] if t == 0 else min(wins, key=lambda w_: abs(w_["t_end"] - t))
        for R in RADII:
            per = {}
            for s in (-1, 0, 1):
                rr = R + s * h
                out = eval_window(w, t, rr * dirs)
                gt = np.asarray(unpack(jnp.asarray(out[:, 0:6])))
                chi = out[:, 12]
                # a = e.gt.e (tangential), b = n.gt.n - a, averaged over directions
                e = np.cross(dirs, np.where(np.abs(dirs[:, 2:3]) < 0.9, [[0, 0, 1.0]], [[1.0, 0, 0]]))
                e /= np.linalg.norm(e, axis=1, keepdims=True)
                a = np.einsum("ni,nij,nj->n", e, gt, e)
                b = np.einsum("ni,nij,nj->n", dirs, gt, dirs) - a
                per[s] = (chi ** -0.25, a / chi, b / chi)
            dpsi = (per[1][0] - per[-1][0]) / (2 * h)
            dA = (per[1][1] - per[-1][1]) / (2 * h)
            m_psi = -2 * R**2 * dpsi
            m_adm = 0.5 * R**2 * (-dA + per[0][2] / R)
            rows.append((t, R, float(m_psi.mean()), float(m_adm.mean()),
                         float(m_psi.std()), float(m_adm.std())))
            print(f"PINN t={t:5.2f} R={R:4.1f}  M_psi {m_psi.mean():.5f} (+-{m_psi.std():.1e})  "
                  f"M_ADM {m_adm.mean():.5f} (+-{m_adm.std():.1e})", flush=True)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ref")
    p.add_argument("run_dir", nargs="?")
    p.add_argument("--out", default="mass_audit")
    a = p.parse_args()
    rows, m = ref_audit(a.ref)
    with open(a.out + "_ref.csv", "w") as f:
        f.write("t,R,M_psi,M_ADM\n")
        for r_ in rows:
            f.write("%g,%g,%.6f,%.6f\n" % r_)
    print(f"reference, bare mass {m}")
    for R in RADII:
        sel = [r_ for r_ in rows if r_[1] == R]
        mp = np.array([r_[2] for r_ in sel]); ma = np.array([r_[3] for r_ in sel])
        print(f"  R={R:4.1f}: M_psi t=0 {mp[0]:.5f}, range [{mp.min():.5f}, {mp.max():.5f}]; "
              f"M_ADM t=0 {ma[0]:.5f}, range [{ma.min():.5f}, {ma.max():.5f}]")
    if a.run_dir:
        pr = pinn_audit(a.run_dir, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        with open(a.out + "_pinn.csv", "w") as f:
            f.write("t,R,M_psi,M_ADM,M_psi_dirstd,M_ADM_dirstd\n")
            for r_ in pr:
                f.write("%g,%g,%.6f,%.6f,%.2e,%.2e\n" % r_)
