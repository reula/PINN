"""Identity test for z4_bianchi.py: for ANY metric and covector,
    nabla^m (E_mn - 1/2 g_mn E) == box Z_n + R_nm Z^m.
Random smooth analytic fields (a Lorentzian perturbation of a lapse/shift/spatial-metric form and
a random covector), several points.   JAX_ENABLE_X64=1 python tools/test_z4_bianchi.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import z4_bianchi as zb

key = jax.random.PRNGKey(0)
ka, kb, kc, kd = jax.random.split(key, 4)
A = jax.random.normal(ka, (4, 4, 4)) * 0.3       # metric perturbation coefficients
P = jax.random.normal(kb, (4, 4, 4))             # phases
C = jax.random.normal(kc, (4, 4))                # covector coefficients
K = jax.random.normal(kd, (4, 4))


def g_fn(q):
    h = jnp.einsum("abk,abk->ab", A, jnp.sin(P @ q[:, None] * 0 + jnp.einsum("abk,k->abk", P, q)))
    h = 0.5 * (h + h.T)
    alpha = 1.2 + 0.2 * jnp.sin(q @ jnp.array([0.3, -0.7, 0.4, 0.5]))
    beta = 0.2 * jnp.cos(q[1:] + q[0])
    gam = jnp.eye(3) + 0.2 * h[1:, 1:]
    bl = gam @ beta
    top = jnp.concatenate([jnp.array([-alpha ** 2 + beta @ bl]), bl])
    return jnp.vstack([top[None, :], jnp.hstack([bl[:, None], gam])])


def z_fn(q):
    return jnp.einsum("ab,b->a", C, jnp.sin(K @ q)) + 0.1 * q


worst = 0.0
for i in range(5):
    q = jax.random.uniform(jax.random.PRNGKey(10 + i), (4,), minval=-1.0, maxval=1.0)
    lhs = zb.divergence_trace_reversed(g_fn, z_fn, q)
    rhs = zb.wave_z(g_fn, z_fn, q)
    rel = float(jnp.max(jnp.abs(lhs - rhs)) / jnp.max(jnp.abs(rhs)))
    worst = max(worst, rel)
    print(f"point {i}: |B| = {float(jnp.max(jnp.abs(rhs))):.3e}   max |div(E - gE/2) - B| / |B| = {rel:.1e}")
print("identity:", "PASS" if worst < 1e-10 else "FAIL", f"(worst {worst:.1e})")
# sanity: flat metric, Z = 0 -> 0;  flat metric -> B = box Z (ordinary wave operator, signature -+++)
eta = lambda q: jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
zq = lambda q: jnp.array([jnp.sin(q[1]) * jnp.cos(q[0]), 0.0, 0.0, 0.0])
q0 = jnp.array([0.3, 0.2, 0.0, 0.0])
B = zb.wave_z(eta, zq, q0)
exact = (-(-jnp.sin(q0[1]) * jnp.cos(q0[0])) + (-jnp.sin(q0[1]) * jnp.cos(q0[0])))   # (-d_t^2 + d_x^2)(sin x cos t) = 0
print(f"flat check: box(sin x cos t) = {float(B[0]):.2e} (exact {float(exact):.2e})")

# ---------------------------------------------------------------- 2. the CCZ4 -> (g, Z) mapping
# On a trained network the 4D Z4 tensor E_mn must be residual-sized with the RIGHT mapping and
# larger with a wrong one (Theta sign flipped, Z^i used as Z_i, Z ignored).
#   JAX_ENABLE_X64=1 python tools/test_z4_bianchi.py RUN_DIR WINDOW
if len(sys.argv) >= 3:
    import pickle, tempfile
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from load_defs import load
    run, wi = sys.argv[1], int(sys.argv[2])
    txt = open(os.path.join(run, "code_md5.txt")).read() if os.path.exists(os.path.join(run, "code_md5.txt")) else ""
    chi_mode = "physical" if "--chi_bg_mode physical" in txt else "moving"
    ns = load(["--n_stars", "1", "--BL_masses", "0.5", "--BL_momenta", "0", "--BL_momenta_y", "0",
               "--tfinal", "10", "--N_window", "40", "--chi_bg_mode", chi_mode]
              + (["--bh_center_mode", "fixed"] if chi_mode == "physical" else []), outdir=tempfile.mkdtemp())
    man = pickle.load(open(os.path.join(run, "pinn_checkpoint.pkl"), "rb"))
    st = pickle.load(open(os.path.join(run, man["window_files"][wi]), "rb"))
    st["params"] = jax.tree_util.tree_map(jnp.asarray, st["params"])
    tt = jnp.asarray(st["traj_tt"]) if "traj_tt" in st else None
    xx = jnp.asarray(st["traj_xx"]) if "traj_xx" in st else None
    fn, p, ws, we = ns["_network_apply_for_state"](st, tt, xx)
    kq = jax.random.PRNGKey(5)
    n_pts = 60
    rr = jnp.exp(jax.random.uniform(kq, (n_pts,), minval=jnp.log(0.3), maxval=jnp.log(6.0)))
    d = jax.random.normal(jax.random.PRNGKey(6), (n_pts, 3)); d = d / jnp.linalg.norm(d, axis=1, keepdims=True)
    tq = ws + (we - ws) * jax.random.uniform(jax.random.PRNGKey(7), (n_pts,))
    Q = jnp.concatenate([tq[:, None], rr[:, None] * d], axis=1)
    rc = ns["r_ccz4_evolution"](fn, p, Q[:, :1], Q[:, 1:2], Q[:, 2:3], Q[:, 3:4], ws, we)
    print(f"\n2. mapping test on {os.path.basename(run.rstrip('/'))} window {wi} (t in [{float(ws):g}, {float(we):g}]), "
          f"{n_pts} points at r in [0.3, 6]: CCZ4 residual rms {float(jnp.sqrt(jnp.mean(rc ** 2))):.2e}")
    for label, kw in (("Alic mapping (used)", {}), ("Theta sign flipped", {"theta_sign": -1.0}),
                      ("Z^i used as Z_i", {"z_mode": "contravariant_as_covariant"}), ("Z ignored", {"z_mode": "zero"})):
        def E_at(q):
            g_fn = lambda q_: ns["z4_spacetime_fields"](fn, p, q_, ws, we, **kw)[0]
            z_fn = lambda q_: ns["z4_spacetime_fields"](fn, p, q_, ws, we, **kw)[1]
            return zb.z4_tensor(g_fn, z_fn, q)
        E = jax.jit(jax.vmap(E_at))(Q)
        _, Zs = jax.vmap(lambda q: ns["z4_spacetime_fields"](fn, p, q, ws, we, **kw))(Q)
        eb = []
        for lo, hi in ((0.3, 0.7), (0.7, 1.5), (1.5, 3.0), (3.0, 6.0)):
            m = (rr >= lo) & (rr < hi)
            eb.append(f"[{lo:g},{hi:g}) {float(jnp.sqrt(jnp.mean(E[m] ** 2))):.2e}")
        print(f"   {label:22s} E_mn rms {float(jnp.sqrt(jnp.mean(E ** 2))):.3e}  by r: " + "  ".join(eb)
              + f"   (|Z_m| rms {float(jnp.sqrt(jnp.mean(Zs ** 2))):.2e})")
    B = jax.jit(lambda: ns["bianchi_residual"](fn, p, Q[:, :1], Q[:, 1:2], Q[:, 2:3], Q[:, 3:4], ws, we))()
    print(f"   Bianchi residual B_n (box Z + Ric Z) rms {float(jnp.sqrt(jnp.mean(B ** 2))):.3e}")

# ---------------------------------------------------------------- 3. exact Schwarzschild through the mapping
# Static isotropic Schwarzschild written in the CCZ4 variables and fed through z4_spacetime_fields
# as a fake apply_fn: E_mn and B_n must vanish (tests the 4-metric assembly from alpha, beta, chi, gt).
if "--schwarzschild" in sys.argv:
    import tempfile
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from load_defs import load
    ns = load(["--n_stars", "1", "--BL_masses", "0.5", "--chi_bg_mode", "physical", "--bh_center_mode", "fixed",
               "--BL_momenta", "0", "--BL_momenta_y", "0"], outdir=tempfile.mkdtemp())
    Mbh = 0.5

    def exact_apply(params, t, x, y, z, ws, we):
        r = jnp.sqrt(x ** 2 + y ** 2 + z ** 2)
        psi = 1.0 + Mbh / (2.0 * r)
        out = jnp.zeros((x.shape[0], 22), dtype=x.dtype)
        for k in (ns["_GT_XX_IDX"], ns["_GT_YY_IDX"], ns["_GT_ZZ_IDX"]):
            out = out.at[:, k].set(1.0)
        out = out.at[:, ns["CHI_IDX"]].set((psi ** -4)[:, 0])
        out = out.at[:, ns["ALPHA_IDX"]].set(((1 - Mbh / (2 * r)) / (1 + Mbh / (2 * r)))[:, 0])
        return out

    for rq in (0.4, 0.8, 2.0, 5.0):
        q = jnp.array([0.1, rq * 0.6, rq * 0.8, 0.0])
        g_fn = lambda q_: ns["z4_spacetime_fields"](exact_apply, None, q_, 0.0, 0.25)[0]
        z_fn = lambda q_: ns["z4_spacetime_fields"](exact_apply, None, q_, 0.0, 0.25)[1]
        E = zb.z4_tensor(g_fn, z_fn, q); B = zb.wave_z(g_fn, z_fn, q)
        _, _, _, _, Ric = zb.geometry(g_fn, q)
        print(f"3. exact Schwarzschild r = {rq}: max|E_mn| = {float(jnp.max(jnp.abs(E))):.1e}, "
              f"max|B_n| = {float(jnp.max(jnp.abs(B))):.1e}  (max|Riemann-scale g| ~ M/r^3 = {Mbh / rq ** 3:.1e})")
