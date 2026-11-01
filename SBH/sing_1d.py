"""1D (spherical) twin of sing_pinn.py: the same template + correction ansatz with a RADIAL network,
the 3D CCZ4 residual evaluated on the x-axis (cartoon reduction, as reference_1d/sph_reference.py). (2026-11-01)

    JAX_ENABLE_X64=1 python sing_1d.py --out OUT_DIR [options] -- <physics flags>

Per window [t0, t1], with T(t, r) the 1D template (10 radial functions a, b, c, d, chi, Khat, Theta, g, alpha,
B; sing_template.py), W = exp(-(r/r_c)^w_power), rho = r^p / (r^p + r_c^p) and N(t, r) an MLP with inputs
(2 (t - t0)/(t1 - t0) - 1, r / x_scale) -> 10:
    additive fields:  f = flat + W (T - flat) + rho N        chi, alpha:  f = T^W exp(rho N)
    --hard_ic:        f = h(r) + [above], s = t/t1 in window 0 (h = U0 - B(0), ln for chi, alpha; as sing_pinn.py)
The Cartesian fields at (t, r, 0, 0) are lift(f) (sing_template.lift), their derivative stacks come from autodiff
(jvp in t, jacfwd in x, y, z: spherical symmetry gives the y, z derivatives), and the residual is the main
script's _ccz4_residual_one. Loss = Einstein + gauge means on interior points, lam_tapa (end) at t1, Sommerfeld
on the radial functions at r = R, lamg0 IC, w_constraint mean(Theta^2 + |Z|^2): the weights of sing_pinn.py.
Points: the RADII of sing_pinn.py's 3D clouds (inner/core/focus Gaussians + uniform ball), so the loss weights
each shell as the 3D run does. SSBroyden, fresh points every --n_change, warm start across windows.
Report per window end (REPORT.npz, appended): signed f - f_ref for the 10 radial functions on a log grid of r
against the fine 1D reference (reference_1d/ref_single_m0.5_dr0.005_t20.npz), and the Misner-Sharp mass m/M - 1.
The exact solution of this problem IS the 1D reference, so the error is the error.
"""
import argparse
import csv
import os
import pickle
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tools"))
from load_defs import load
import sing_template as stpl

REF = os.path.join(HERE, "reference_1d")
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--template_file", default=None)
ap.add_argument("--r_c", type=float, default=0.2)
ap.add_argument("--w_power", type=float, default=4.0, help="W = exp(-(r/r_c)^w_power)")
ap.add_argument("--rho_power", type=float, default=2.0)
ap.add_argument("--x_scale", type=float, default=24.0, help="the network sees r / x_scale (24 = sing_pinn.py)")
ap.add_argument("--hard_ic", action="store_true")
ap.add_argument("--no_template", action="store_true", help="W = 0, rho = 1: the radial network alone (needs --hard_ic)")
ap.add_argument("--hidden", type=int, default=40)
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--tfinal", type=float, default=0.5)
ap.add_argument("--N_window", type=int, default=2)
ap.add_argument("--n_int", type=int, default=2000)
ap.add_argument("--n_end", type=int, default=250)
ap.add_argument("--n_ic", type=int, default=400)
ap.add_argument("--n_held", type=int, default=2000)
ap.add_argument("--inner_frac", type=float, default=0.125)
ap.add_argument("--inner_sigma", type=float, default=0.05)
ap.add_argument("--core_frac", type=float, default=0.375)
ap.add_argument("--core_sigma", type=float, default=0.5)
ap.add_argument("--focus_frac", type=float, default=0.25)
ap.add_argument("--focus_sigma", type=float, default=2.0)
ap.add_argument("--iters", type=int, default=4000)
ap.add_argument("--n_change", type=int, default=250)
ap.add_argument("--w_constraint", type=float, default=10.0)
ap.add_argument("--w_gauge", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--resume", action="store_true")
argv = sys.argv[1:]
rest = argv[argv.index("--") + 1:] if "--" in argv else []
a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
os.makedirs(a.out, exist_ok=True)

if a.no_template and not a.hard_ic:
    raise SystemExit("--no_template needs --hard_ic")
tfile = a.template_file or os.path.join(REF, "sing_bl_m0.5_dense.npz")
R_TPL = 3.0 * a.r_c
tab = stpl.load_table(tfile, "bl", R_TPL)
MASS = tab["mass"]
ns = load(["--n_stars", "1", "--BL_masses", repr(MASS), "--BL_momenta", "0", "--BL_momenta_y", "0",
           "--chi_bg_mode", "physical", "--bh_center_mode", "fixed"] + rest, outdir=tempfile.mkdtemp(prefix="sing1d_"))
jax, jnp, np = ns["jax"], ns["jnp"], ns["np"]
GT, CHI, TH, GH, AL = ns["GT_SLICE"], ns["CHI_IDX"], ns["THETA_IDX"], ns["GHAT_SLICE"], ns["ALPHA_IDX"]
DD = np.asarray(ns["_DD_OUTPUT_IDXS"])
R = float(ns["R_domain"])
FLAT = ns["_FLAT_CCZ4_VEC"]
pack, unpack = ns["pack_sym3"], ns["unpack_sym3"]
radial = stpl.make_radial(jnp, jax, tab)
FLAT_RAD = jnp.asarray(stpl.FLAT)
I_CHI, I_AL = stpl.I_CHI, stpl.I_ALPHA
LOG_EPS = 1e-10
REF_D = np.load(os.path.join(REF, "ref_single_m0.5_dr0.005_t20.npz"))
REF_R, REF_T, REF_U = REF_D["r"], REF_D["t"], REF_D["U"]              # U (n_t, 10, n_r)


def _pos(v):
    return 0.5 * (v + jnp.sqrt(v * v + 4.0 * LOG_EPS ** 2))


# ---- ansatz ----------------------------------------------------------------------------------------
def mlp_value(net, t, r, ws, we):
    mp = net["params"]["MLP"]
    h = jnp.stack([2.0 * (t - ws) / (we - ws) - 1.0, r / a.x_scale])
    for i in range(len(mp) - 1):
        h = jnp.tanh(h @ mp[f"Dense_{i}"]["kernel"] + mp[f"Dense_{i}"]["bias"])
    L = mp[f"Dense_{len(mp) - 1}"]
    return h @ L["kernel"] + L["bias"]


def init_net(key):
    sizes = [2] + [a.hidden] * a.layers + [10]
    mlp = {}
    for i in range(len(sizes) - 1):
        key, k = jax.random.split(key)
        kern = jax.nn.initializers.glorot_normal()(k, (sizes[i], sizes[i + 1]), jnp.float64)
        mlp[f"Dense_{i}"] = {"kernel": jnp.zeros_like(kern) if i == len(sizes) - 2 else kern,
                             "bias": jnp.zeros((sizes[i + 1],), jnp.float64)}
    return {"params": {"MLP": mlp}}


def U0_rad(r):
    psi = 1.0 + MASS / (2.0 * r)
    return FLAT_RAD.at[I_CHI].set(psi ** -4).at[I_AL].set(1.0 / psi)


def bg_rad(t, r):
    """Template part B (10,) and rho at (t, r): W-blended template, multiplicative chi and alpha."""
    T = radial(t, jnp.minimum(r, R_TPL))
    w = jnp.exp(-(r / a.r_c) ** a.w_power) * (0.0 if a.no_template else 1.0)
    B = FLAT_RAD + w * (T - FLAT_RAD)
    B = B.at[I_CHI].set(jnp.exp(w * jnp.log(_pos(T[I_CHI])))).at[I_AL].set(jnp.exp(w * jnp.log(_pos(T[I_AL]))))
    rho = 1.0 if a.no_template else r ** a.rho_power / (r ** a.rho_power + a.r_c ** a.rho_power)
    return B, rho


def hard_h(r):
    U0, B0 = U0_rad(r), bg_rad(0.0, r)[0]
    h = U0 - B0
    return h.at[I_CHI].set(jnp.log(_pos(U0[I_CHI])) - jnp.log(B0[I_CHI])).at[I_AL].set(jnp.log(_pos(U0[I_AL])) - jnp.log(B0[I_AL]))


def rad_fn(P, win, t, r):
    """The 10 radial functions of u at (t, r)."""
    t0, t1 = win
    B, rho = bg_rad(t, r)
    N = mlp_value(P["net"], t, r, t0, t1)
    if a.hard_ic:
        N = N * jnp.where(t0 == 0.0, (t - t0) / (t1 - t0), 1.0)
    phi = rho * N
    f = B + phi
    f = f.at[I_CHI].set(B[I_CHI] * jnp.exp(phi[I_CHI])).at[I_AL].set(B[I_AL] * jnp.exp(phi[I_AL]))
    if a.hard_ic:
        h = hard_h(r)
        f = (f + h).at[I_CHI].set(f[I_CHI] * jnp.exp(h[I_CHI])).at[I_AL].set(f[I_AL] * jnp.exp(h[I_AL]))
    return f


def u_point(P, win, q):
    """The 22 Cartesian fields at q = (t, x, y, z)."""
    xyz = q[1:4]
    return stpl.lift(jnp, pack, rad_fn(P, win, q[0], jnp.sqrt(jnp.sum(xyz ** 2))), xyz)


def stacks(P, win, Q):
    """(val (n,22), d_t, d_xyz (n,22,3), d_xyz d_xyz of the DD fields (n,11,3,3)) at Q (n,4) by autodiff."""
    def one(q):
        f = lambda qq: u_point(P, win, qq)
        v, vt = jax.jvp(f, (q,), (jnp.array([1.0, 0.0, 0.0, 0.0]),))
        g = lambda X: f(jnp.concatenate([q[:1], X]))
        return v, vt, jax.jacfwd(g)(q[1:]), jax.jacfwd(jax.jacfwd(g))(q[1:])[DD]
    return jax.vmap(one)(Q)


def residual(D):
    return jax.vmap(ns["_ccz4_residual_one"])(*D)


def zeta(D):
    val, _, d, _ = D
    gt = unpack(val[:, GT])
    gt_d = jnp.transpose(unpack(jnp.moveaxis(d[:, GT, :], 1, 2)), (0, 2, 3, 1))
    gu = jax.vmap(lambda g, dg: ns["_christoffel3"](g, dg)[2])(gt, gt_d)
    zv = val[:, CHI][:, None] * (val[:, GH] - gu) / 2.0
    return jnp.mean(val[:, TH] ** 2 + jnp.sum(zv ** 2, axis=1))


def ein_gauge(res):
    return (ns["lam_einstein"] * jnp.mean(res[:, :18] ** 2), ns["lam_gauge"] * a.w_gauge * jnp.mean(res[:, 18:22] ** 2))


def parts(P, win, batch):
    Qi, Qe, tb, ric, Uic = batch
    e_i, g_i = ein_gauge(residual(stacks(P, win, Qi)))
    e_e, g_e = ein_gauge(residual(stacks(P, win, Qe)))
    f = lambda t, r: rad_fn(P, win, t, r)
    fb, ft = jax.vmap(lambda t: jax.jvp(lambda tt: f(tt, R), (t,), (1.0,)))(tb)
    fr = jax.vmap(lambda t: jax.jvp(lambda rr: f(t, rr), (R,), (1.0,))[1])(tb)
    som = ft + ns["c"] * fr + ns["c"] * (fb - FLAT_RAD[None]) / R
    ic = jnp.mean((jax.vmap(lambda r: f(win[0], r))(ric) - Uic) ** 2)
    con = zeta(stacks(P, win, Qi)) if a.w_constraint > 0 else jnp.zeros(())
    return dict(einstein=e_i, gauge=g_i, end=ns["lam_tapa"] * (e_e + g_e), bc=ns["lam_bc"] * jnp.mean(som ** 2),
                ic=ns["lamg0"] * ic, constraint=a.w_constraint * con)


def total(p):
    return sum(p.values())


# ---- points: the radii of sing_pinn.py's clouds --------------------------------------------------
def radii(key, n):
    sizes = [2 * int(f * n / 2) for f in (a.inner_frac, a.core_frac, a.focus_frac)]
    keys = jax.random.split(key, 4)
    rs = [s * jnp.linalg.norm(jax.random.normal(k, (m, 3)), axis=1)
          for k, m, s in zip(keys[:3], sizes, (a.inner_sigma, a.core_sigma, a.focus_sigma))]
    rs.append(R * jax.random.uniform(keys[3], (n - sum(sizes),)) ** (1 / 3))
    return jnp.minimum(jnp.concatenate(rs), R)


def axis(t, r):
    return jnp.stack([t, r, jnp.zeros_like(r), jnp.zeros_like(r)], axis=1)


def draw(key, win, n_int, n_end, n_ic, Uic_fn):
    t0, t1 = float(win[0]), float(win[1])
    k = jax.random.split(key, 6)
    Qi = axis(t0 + (t1 - t0) * jax.random.uniform(k[0], (n_int,)), radii(k[1], n_int))
    Qe = axis(jnp.full((n_end,), t1), radii(k[2], n_end))
    tb = t0 + (t1 - t0) * jax.random.uniform(k[3], (max(n_end // 2, 8),))
    ric = radii(k[4], n_ic)
    return (Qi, Qe, tb, ric, Uic_fn(ric))


# ---- report: error against the 1D reference and the mass -------------------------------------------
REP_R = np.geomspace(0.03, 20.0, 240)


def mass_rad(P, win, t, r):
    """Misner-Sharp m/M - 1 at (t, r) from the radial functions: areal R = r sqrt(a/chi), g_rr = (a + b)/chi,
    K_thth / g_thth = (c + a K/3) / a with K = Khat + 2 Theta."""
    f = rad_fn(P, win, t, r)
    Rar, dR = jax.jvp(lambda rr: rr * jnp.sqrt(rad_fn(P, win, t, rr)[0] / rad_fn(P, win, t, rr)[4]), (r,), (1.0,))
    g_rr = (f[0] + f[1]) / f[4]
    K = f[5] + 2.0 * f[6]
    ratio = (f[2] + f[0] * K / 3.0) / f[0]
    return 0.5 * Rar * (1.0 - dR ** 2 / g_rr + Rar ** 2 * ratio ** 2) / MASS - 1.0


def report(P, win, t):
    j = int(np.argmin(np.abs(REF_T - t)))
    assert abs(REF_T[j] - t) < 1e-6, f"no reference snapshot at t = {t}"
    f = np.asarray(jax.vmap(lambda r: rad_fn(P, win, t, r))(jnp.asarray(REP_R)))          # (n_r, 10)
    ref = np.stack([np.interp(REP_R, REF_R, REF_U[j, i]) for i in range(10)], axis=1)
    m = np.asarray(jax.vmap(lambda r: mass_rad(P, win, t, r))(jnp.asarray(REP_R)))
    return f - ref, m


# ---- training ----------------------------------------------------------------------------------------
edges = np.linspace(0.0, a.tfinal, a.N_window + 1)
mb = ns["_minimize_mod"]
opts = {"maxiter": a.n_change, "gtol": 0.0, "update_method": "ssbroyden2", "initial_scale": True,
        "line_search": "wolfe", "c2": 0.9, "scale_invariant": True, "max_iter": 100, "alphamin": 1e-15}
key = jax.random.PRNGKey(a.seed)
P = {"net": init_net(jax.random.PRNGKey(a.seed + 1))}
prev, k0 = None, 0
REP = os.path.join(a.out, "report.npz")
rep = {"t": [], "err": [], "mass": []}
if a.resume:
    done = sorted(f for f in os.listdir(a.out) if f.startswith("window_") and f.endswith(".pkl"))
    if done:
        S = pickle.load(open(os.path.join(a.out, done[-1]), "rb"))
        P = jax.tree_util.tree_map(jnp.asarray, S["params"])
        prev, k0, key = (P, tuple(jnp.asarray(v) for v in S["win"])), S["window"] + 1, jnp.asarray(S["key"])
        z = np.load(REP)
        rep = {k: list(z[k]) for k in rep}
        print(f"resumed after window {S['window']}")
loss_fn = jax.jit(lambda w, unr, win, batch: total(parts(unr(w), win, batch)), static_argnums=1)
vg_fn = jax.jit(jax.value_and_grad(lambda w, unr, win, batch: total(parts(unr(w), win, batch))), static_argnums=1)
parts_jit = jax.jit(parts)
hist = os.path.join(a.out, "history.csv")
PART_NAMES = ("einstein", "gauge", "end", "bc", "ic", "constraint")
for k in range(k0, a.N_window):
    t0, t1 = float(edges[k]), float(edges[k + 1])
    win = (jnp.asarray(t0), jnp.asarray(t1))
    if prev is None:
        Uic_fn = jax.jit(jax.vmap(U0_rad))
    else:
        Pp, wp = prev
        Uic_fn = jax.jit(jax.vmap(lambda r, Pp=Pp, wp=wp: rad_fn(Pp, wp, wp[1], r)))
    w, unr = ns["flatten_util"].ravel_pytree(P)
    key, kh = jax.random.split(key)
    held = draw(kh, win, a.n_held, a.n_held // 8, a.n_held // 4, Uic_fn)
    if k == 0:
        e0 = np.asarray(jax.vmap(lambda r: rad_fn(P, win, 0.0, r) - U0_rad(r))(jnp.asarray(REP_R)))
        print(f"window 0: max |u(0) - U0| over the radial functions {np.abs(e0).max():.2e}", flush=True)
    print(f"window {k}: t in [{t0:g}, {t1:g}], {w.size} parameters", flush=True)
    t_start, it = time.time(), 0
    while it < a.iters:
        key, kd = jax.random.split(key)
        batch = draw(kd, win, a.n_int, a.n_end, a.n_ic, Uic_fn)
        w = mb.minimize(lambda v: loss_fn(v, unr, win, batch), lambda v: vg_fn(v, unr, win, batch), w, opts).x
        it += a.n_change
        hp = {n: float(v) for n, v in parts_jit(unr(w), win, held).items()}
        row = dict(window=k, iteration=it, seconds=round(time.time() - t_start, 1),
                   train=float(loss_fn(w, unr, win, batch)), held=sum(hp.values()), **{f"held_{n}": hp[n] for n in PART_NAMES})
        new = not os.path.exists(hist)
        with open(hist, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
        print(f"  it {it:6d} {row['seconds']:7.0f} s  train {row['train']:.3e}  held {row['held']:.3e} ("
              + " ".join(f"{n} {hp[n]:.1e}" for n in PART_NAMES) + ")", flush=True)
    P = unr(w)
    err, m = report(P, win, t1)
    rep["t"].append(t1)
    rep["err"].append(err)
    rep["mass"].append(m)
    np.savez(REP, r=REP_R, fields=np.array(stpl.FIELDS), t=np.array(rep["t"]), err=np.array(rep["err"]),
             mass=np.array(rep["mass"]), config=np.array(str(vars(a))))
    band = (REP_R >= 0.2) & (REP_R < 1.0)
    print(f"  t = {t1:g}: rms vs reference, alpha {np.sqrt((err[:, 8] ** 2).mean()):.2e} (0.2-1: "
          f"{np.sqrt((err[band, 8] ** 2).mean()):.2e}), chi {np.sqrt((err[:, 4] ** 2).mean()):.2e}, "
          f"Khat {np.sqrt((err[:, 5] ** 2).mean()):.2e} (0.2-1: {np.sqrt((err[band, 5] ** 2).mean()):.2e}); "
          f"m/M - 1 at r = 1: {np.interp(1.0, REP_R, m):+.2e}, max |alpha err| {np.abs(err[:, 8]).max():.2e} "
          f"at r = {REP_R[np.argmax(np.abs(err[:, 8]))]:.2f}", flush=True)
    S = {"params": jax.tree_util.tree_map(np.asarray, P), "win": [np.asarray(v) for v in win], "window": k,
         "key": np.asarray(key), "config": vars(a)}
    pickle.dump(S, open(os.path.join(a.out, f"window_{k:03d}.pkl"), "wb"))
    prev = (P, win)
print("done:", a.out)
