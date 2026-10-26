"""Checks of adm_constraints.py (2026-10-26).

    JAX_ENABLE_X64=1 python tools/test_adm_constraints.py [RUN_DIR]

1. Exact Schwarzschild, isotropic (K_ij = 0): H = M_i = 0 (tests the Ricci scalar of gt/chi).
2. Exact Schwarzschild, Kerr-Schild (K_ij != 0): gamma_ij = delta_ij + 2 (m/r) n_i n_j,
   K_ij = (2 m alpha / r^2) [delta_ij - (2 + m/r) n_i n_j], alpha = (1 + 2m/r)^(-1/2), written in the
   CCZ4 variables (chi = det(gamma)^(-1/3), gt = chi gamma, At = chi (K_ij - gamma_ij K/3), Khat = K):
   H = M_i = 0 (tests the extrinsic-curvature terms and D_j K^j_i).
3. With RUN_DIR (a single-puncture run): on a trained window, the closed-form Taylor stack against autodiff,
   and H against an independent implementation (z4_bianchi.geometry on the metric function, as in
   tools/hamiltonian_shells.py).
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from load_defs import load

PHYS = ["--n_stars", "1", "--BL_masses", "0.5", "--BL_momenta", "0", "--BL_momenta_y", "0"]
ns = load(PHYS + ["--chi_bg_mode", "physical", "--bh_center_mode", "fixed"], outdir=tempfile.mkdtemp())
jax, jnp, np = ns["jax"], ns["jnp"], ns["np"]
import adm_constraints as adm
import z4_bianchi as zb

m = 0.5
pack = lambda S: ns["pack_sym3"](S)                                  # (N,3,3) -> (N,6)


def fields(gt, chi, At, K):
    out = jnp.zeros((gt.shape[0], 22))
    out = out.at[:, ns["GT_SLICE"]].set(pack(gt)).at[:, ns["AT_SLICE"]].set(pack(At))
    out = out.at[:, ns["CHI_IDX"]].set(chi).at[:, ns["KHAT_IDX"]].set(K)
    return out.at[:, ns["ALPHA_IDX"]].set(1.0)


def isotropic(params, t, x, y, z, ws, we):
    r = jnp.sqrt(x ** 2 + y ** 2 + z ** 2)[:, 0]
    n = x.shape[0]
    return fields(jnp.broadcast_to(jnp.eye(3), (n, 3, 3)), (1 + m / (2 * r)) ** -4, jnp.zeros((n, 3, 3)), jnp.zeros(n))


def kerr_schild(params, t, x, y, z, ws, we):
    X = jnp.concatenate([x, y, z], axis=1)
    r = jnp.linalg.norm(X, axis=1)
    nv = X / r[:, None]
    nn = nv[:, :, None] * nv[:, None, :]
    I = jnp.eye(3)[None]
    Hk = m / r
    gam = I + 2 * Hk[:, None, None] * nn
    alpha = (1 + 2 * Hk) ** -0.5
    Kij = (2 * m * alpha / r ** 2)[:, None, None] * (I - (2 + Hk)[:, None, None] * nn)
    gi = jnp.linalg.inv(gam)
    K = jnp.einsum("nij,nij->n", gi, Kij)
    chi = jnp.linalg.det(gam) ** (-1.0 / 3.0)
    return fields(chi[:, None, None] * gam, chi, chi[:, None, None] * (Kij - gam * K[:, None, None] / 3), K)


key = jax.random.PRNGKey(0)
for name, fn in (("isotropic", isotropic), ("Kerr-Schild", kerr_schild)):
    for rq in (0.4, 1.0, 3.0, 8.0):
        d = jax.random.normal(key, (16, 3))
        X = rq * d / jnp.linalg.norm(d, axis=1, keepdims=True)
        T = jnp.zeros((16, 1))
        H, M = adm.adm_constraints(ns, fn, None, T, X[:, :1], X[:, 1:2], X[:, 2:3], 0.0, 0.25)
        print(f"{name:12s} r = {rq:4.1f}: max|H| = {float(jnp.abs(H).max()):.1e}, max|M| = {float(jnp.abs(M).max()):.1e}"
              f"   (curvature scale m/r^3 = {m / rq ** 3:.1e})")

if len(sys.argv) > 1:
    import pickle
    run = sys.argv[1]
    ns2 = load(PHYS + ["--tfinal", "10", "--N_window", "40", "--deriv_mode", "taylor"], outdir=tempfile.mkdtemp())
    man = pickle.load(open(os.path.join(run, "pinn_checkpoint.pkl"), "rb"))
    st = pickle.load(open(os.path.join(run, man["window_files"][20]), "rb"))
    st["params"] = jax.tree_util.tree_map(jnp.asarray, st["params"])
    fn, p, ws, we = ns2["_network_apply_for_state"](st, jnp.asarray(st["traj_tt"]), jnp.asarray(st["traj_xx"]))
    d = jax.random.normal(key, (24, 3))
    X = d / jnp.linalg.norm(d, axis=1, keepdims=True) * jnp.linspace(0.3, 12, 24)[:, None]
    T = jnp.full((24, 1), float(st["t_start"]) + 0.1)
    args = (T, X[:, :1], X[:, 1:2], X[:, 2:3], ws, we)
    print("taylor-capable:", fn in ns2["_TAYLOR_KIND"])
    Ht, Mt = adm.adm_constraints(ns2, fn, p, *args)
    ns2["deriv_mode"] = "autodiff"
    Ha, Ma = adm.adm_constraints(ns2, fn, p, *args)
    print(f"trained window: Taylor vs autodiff: max|dH| = {float(jnp.abs(Ht - Ha).max()):.1e} (|H| up to "
          f"{float(jnp.abs(Ha).max()):.1e}), max|dM| = {float(jnp.abs(Mt - Ma).max()):.1e} (|M| up to {float(jnp.abs(Ma).max()):.1e})")
    GT, AT, CHI, KH, TH = ns2["GT_SLICE"], ns2["AT_SLICE"], ns2["CHI_IDX"], ns2["KHAT_IDX"], ns2["THETA_IDX"]
    unpack = ns2["unpack_sym3"]

    def H_indep(x3, t):
        out = lambda y: ns2["full_output"](fn, p, jnp.full((1, 1), t), y[0:1].reshape(1, 1), y[1:2].reshape(1, 1),
                                           y[2:3].reshape(1, 1), ws, we)[0]
        g, gi, _, _, Ric = zb.geometry(lambda y: unpack(out(y)[GT]) / out(y)[CHI], x3)
        o = out(x3)
        Kij = (unpack(o[AT]) + unpack(o[GT]) * (o[KH] + 2 * o[TH]) / 3) / o[CHI]
        trK = jnp.einsum("ij,ij->", gi, Kij)
        return jnp.einsum("ij,ij->", gi, Ric) + trK ** 2 - jnp.einsum("ij,ij->", Kij, gi @ Kij @ gi)
    Hi = jax.vmap(H_indep, in_axes=(0, None))(X, float(T[0, 0]))
    print(f"trained window: H vs independent geometry: max|dH| = {float(jnp.abs(Ht - Hi).max()):.1e}")
