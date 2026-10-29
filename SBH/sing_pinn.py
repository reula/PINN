"""CCZ4 PINN with the puncture's singular part built in: u = template(x - a(t)) + correction. (2026-10-28)

    JAX_ENABLE_X64=1 python sing_pinn.py --out OUT_DIR [--template bl|trumpet] [options] -- <physics flags>

The main script's network has to reproduce the puncture itself: chi ~ r^4 (Brill-Lindquist) or ~ r^2
(trumpet), alpha ~ r (a kink), and tensor fields whose limit at the puncture depends on the direction.
Here those come from a template, the spherical single-puncture solution of reference_1d/sph_reference.py
(sing_template.py), centred at a trainable position a(t) and cut off inside the horizon, and the network
N only learns the departure from it. Per window [t0, t1] and puncture A, with xi_A = x - a_A(t), r_A = |xi_A|:

    W_A = exp(-(r_A / r_c)^4)                     cutoff: the template only acts for r_A <~ r_c
    rho = prod_A r_A^2 / (r_A^2 + r_c^2)          the correction vanishes like r^2 at each puncture, -> 1 far away
    chi, alpha:  u = exp(sum_A W_A ln T_A) * exp(rho N)          multiplicative: right vanishing order, positive
    beta:        u = sum_A W_A (T_A^beta - da_A/dt) + rho N       so beta(a_A) = -da_A/dt: the puncture moves with -beta
    the others:  u = flat + sum_A W_A (T_A - flat) + rho N

Template ('--template'): 'bl' is the time-dependent 1D evolution from Brill-Lindquist data (t = 0 is the BL
puncture, the same initial data as the main script); 'trumpet' is one late snapshot of it (t_s = 10 = 20 M),
frozen, and the run then starts from that snapshot (initial data = the full 1D slice at t_s, single puncture).

Centre: a_A(t) = a0 + v0 (t - t0) + s_a (p1 tau^2 + p2 tau^3), tau = (t - t0)/(t1 - t0), with a0, v0 the
previous window's end position and velocity (so a and da/dt are continuous) and p1, p2 trained with the
network (--fixed_center: not trained). No shift tracker.

Loss (the main script's weights, read from its flags): lam_einstein * Einstein rows + lam_gauge * w_gauge *
gauge rows on the interior points, lam_tapa x the same at t = t1, lam_bc * Sommerfeld at r = R_domain,
lamg0 * soft IC (exact initial data in window 0, the previous window's u after that), + w_constraint *
mean(Theta^2 + |Z^i|^2). SSBroyden on fresh points every --n_change iterations (as stage2.py).
--pre_iters N: each window (the first --pre_windows of them) starts with N iterations on the IC + Sommerfeld
terms alone, so the initial data are fitted before the interior residual is switched on; --pre_n_ic sets
the IC points per draw in that phase (same clouds, all at t = t0).
Derivatives: N from the main script's closed-form Taylor pass, the template part by autodiff (no network),
combined by the product/chain rule; --selftest checks this against autodiff of the whole u.

Outputs in OUT_DIR: window_KKK.pkl per window (params, window, centre), sing_history.csv (per draw),
sing_report.csv (per window end: rms field errors against the 1D reference, by radius band), config.json.
"""
import argparse
import csv
import json
import os
import pickle
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tools"))
from load_defs import load
from stage2_net import init_net as _init_net
import sing_template as stpl

REF = os.path.join(HERE, "reference_1d")
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--template", choices=["bl", "trumpet"], default="bl")
ap.add_argument("--template_file", default=None,
                help="default: reference_1d/sing_bl_m0.5_dense.npz (bl), reference_1d/ref_single_m0.5_dr0.005_t20.npz (trumpet)")
ap.add_argument("--t_snap", type=float, default=10.0, help="trumpet: time of the frozen snapshot")
ap.add_argument("--ref_file", default=None,
                help="1D reference for the per-window report (default reference_1d/ref_single_m0.5_dr0.005_t20.npz)")
ap.add_argument("--r_c", type=float, default=0.2, help="cutoff radius of the template and of rho")
ap.add_argument("--centers", default="0,0,0", help="initial puncture positions 'x,y,z;x,y,z'")
ap.add_argument("--fixed_center", action="store_true", help="do not train the puncture positions")
ap.add_argument("--a_scale", type=float, default=0.01, help="s_a: displacement per unit of p1, p2")
ap.add_argument("--tfinal", type=float, default=2.0)
ap.add_argument("--N_window", type=int, default=8)
ap.add_argument("--hidden", type=int, default=40)
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--n_int", type=int, default=128000,
                help="interior points per draw (the baseline w8_qn_con10: Nint 80000 + core_points 48000)")
ap.add_argument("--core_frac", type=float, default=0.375)
ap.add_argument("--focus_frac", type=float, default=0.25)
ap.add_argument("--core_sigma", type=float, default=0.5)
ap.add_argument("--focus_sigma", type=float, default=2.0)
ap.add_argument("--n_end", type=int, default=8000, help="points at t = t1 (the lam_tapa term; baseline NI)")
ap.add_argument("--n_ic", type=int, default=12800, help="IC points (baseline NI 8000 + core_ic_points 4800)")
ap.add_argument("--n_bc", type=int, default=16000, help="boundary points (baseline Nb)")
ap.add_argument("--n_held", type=int, default=8000)
ap.add_argument("--iters", type=int, default=24000, help="SSBroyden iterations per window (baseline --EPOCHS_QN)")
ap.add_argument("--n_change", type=int, default=250)
ap.add_argument("--pre_iters", type=int, default=0,
                help="SSBroyden iterations on the IC + boundary terms only at the start of a window, before the "
                     "--iters on the full loss (interior, end and constraint terms off)")
ap.add_argument("--pre_windows", type=int, default=-1, help="windows that get the --pre_iters phase (-1: all)")
ap.add_argument("--pre_n_ic", type=int, default=0,
                help="IC points per draw in the --pre_iters phase (0: --n_ic). The phase evaluates no derivatives "
                     "except on the boundary, so it can afford the whole point budget at t = t0 (e.g. 150000)")
ap.add_argument("--w_constraint", type=float, default=10.0)
ap.add_argument("--w_gauge", type=float, default=1.0)
ap.add_argument("--inner_frac", type=float, default=0.125,
                help="share of the interior points in the inner cloud: a narrow Gaussian (--inner_sigma) about each "
                     "puncture, so every batch has points right at it (no exclusion zone anywhere)")
ap.add_argument("--inner_sigma", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--resume", action="store_true", help="continue after the last saved window in OUT_DIR")
ap.add_argument("--selftest", action="store_true", help="check the template and the derivative stacks, then exit")
ap.add_argument("--eval_spheres", default=None, metavar="NPZ",
                help="evaluate the saved windows in --out on spheres (as tools/sphere_errors.py, same npz layout: "
                     "Misner-Sharp m/M - 1 about the trained centre, alpha/chi/Khat/Theta - 1D reference about the "
                     "origin; t = 0 and every window end; r in [0.05, 20], 100 radii x 48 directions), then exit")
argv = sys.argv[1:]
rest = argv[argv.index("--") + 1:] if "--" in argv else []
a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
os.makedirs(a.out, exist_ok=True)

tfile = a.template_file or os.path.join(REF, "sing_bl_m0.5_dense.npz" if a.template == "bl"
                                        else "ref_single_m0.5_dr0.005_t20.npz")
rfile = a.ref_file or os.path.join(REF, "ref_single_m0.5_dr0.005_t20.npz")   # t <= 20, the t <= 10 run extended
R_TPL = 3.0 * a.r_c                                     # W(3 r_c) = e^-81: the template is not needed beyond
tab = stpl.load_table(tfile, a.template, R_TPL, t_snap=a.t_snap)
MASS = tab["mass"]
ns = load(["--n_stars", "1", "--BL_masses", repr(MASS), "--BL_momenta", "0", "--BL_momenta_y", "0",
           "--chi_bg_mode", "physical", "--bh_center_mode", "fixed", "--deriv_mode", "taylor"] + rest,
          outdir=tempfile.mkdtemp(prefix="sing_"))
jax, jnp, np = ns["jax"], ns["jnp"], ns["np"]
for k, nk in (("eta", "eta_shift"), ("f_alpha", "f_alpha"), ("f_beta", "f_beta"), ("kappa_z", "kappa_z"),
              ("kappa_c", "kappa_c"), ("kappa_2", "kappa_2"), ("chi_floor", "chi_floor")):
    if k in tab and abs(float(ns[nk]) - tab[k]) > 1e-12:
        raise SystemExit(f"template {tfile} has {k} = {tab[k]}, the physics flags give {ns[nk]}")
GT, CHI, TH, GH, AL = ns["GT_SLICE"], ns["CHI_IDX"], ns["THETA_IDX"], ns["GHAT_SLICE"], ns["ALPHA_IDX"]
DD = np.asarray(ns["_DD_OUTPUT_IDXS"])
DCHI, DAL = int(ns["_DD_CHI_IDX"]), int(ns["_DD_ALPHA_IDX"])
R = float(ns["R_domain"])
FLAT = ns["_FLAT_CCZ4_VEC"]
pack, unpack = ns["pack_sym3"], ns["unpack_sym3"]
C0 = np.array([[float(v) for v in c.split(",")] for c in a.centers.split(";")])
NP = len(C0)
if a.template == "trumpet" and NP > 1:
    raise SystemExit("trumpet initial data are the 1D slice of ONE puncture")
T_OFF = tab["t_s"] if a.template == "trumpet" else 0.0     # run time t <-> 1D-evolution time t + T_OFF
if a.template == "bl" and a.tfinal > tab["t_end"] + 1e-9:
    raise SystemExit(f"the bl template ends at t = {tab['t_end']}")
radial = stpl.make_radial(jnp, jax, tab)
FLAT_RAD = jnp.asarray(stpl.FLAT)
SA = 0.0 if a.fixed_center else a.a_scale

# ---- the centre a_A(t) of one window ------------------------------------------------------------
# win = (t0, t1, a0 (NP,3), v0 (NP,3)), all arrays, so one compilation serves every window.
def center(acc, win, t):
    t0, t1, a0, v0 = win
    tau = (t - t0) / (t1 - t0)
    pos = a0 + v0 * (t - t0) + SA * (acc[:, 0] * tau ** 2 + acc[:, 1] * tau ** 3)
    vel = v0 + SA * (2 * acc[:, 0] * tau + 3 * acc[:, 1] * tau ** 2) / (t1 - t0)
    return pos, vel


# ---- template part at one point: B (22,) and rho, as one 23-vector ------------------------------
def tpl_fields(t_tpl, xi):
    """Template radial functions (10,) at offset xi, evaluated no farther out than R_TPL, and r."""
    r = jnp.sqrt(jnp.sum(xi ** 2))
    return radial(t_tpl, jnp.minimum(r, R_TPL)), r


def _pos(v):
    """Smooth positive floor for the log blend: the table's chi dips to -6e-7 at its innermost grid point
    (r = 0.0025: the chi floor and the dissipation of the 1D run); = v to O(LOG_EPS^2/v) wherever v >> LOG_EPS."""
    return 0.5 * (v + jnp.sqrt(v * v + 4.0 * LOG_EPS ** 2))


LOG_EPS = 1e-10


def bg_point(q, acc, win):
    t, X = q[0], q[1:4]
    pos, vel = center(acc, win, t)
    add = jnp.zeros(22)
    logs = jnp.zeros(2)                     # ln chi, ln alpha
    rho = 1.0
    for A in range(NP):
        xi = X - pos[A]
        rad, r = tpl_fields(t + T_OFF, xi)
        w = jnp.exp(-(r / a.r_c) ** 4)
        f = stpl.lift(jnp, pack, FLAT_RAD + w * (rad - FLAT_RAD), xi) - FLAT
        add = add + f
        add = add.at[ns["BETA_SLICE"]].add(-w * vel[A])
        logs = logs + w * jnp.log(_pos(jnp.stack([rad[stpl.I_CHI], rad[stpl.I_ALPHA]])))
        r2 = jnp.sum(xi ** 2)
        rho = rho * r2 / (r2 + a.r_c ** 2)
    B = FLAT + add
    B = B.at[CHI].set(jnp.exp(logs[0])).at[AL].set(jnp.exp(logs[1]))
    return jnp.concatenate([B, rho[None]])


def bg_derivs(acc, win, Q):
    """(val (n,23), d_t (n,23), d_xyz (n,23,3), d_xyz d_xyz (n,23,3,3)) of bg_point."""
    def one(q):
        f = lambda qq: bg_point(qq, acc, win)
        v, vt = jax.jvp(f, (q,), (jnp.array([1.0, 0.0, 0.0, 0.0]),))
        g = lambda X: f(jnp.concatenate([q[:1], X]))
        return v, vt, jax.jacfwd(g)(q[1:]), jax.jacfwd(jax.jacfwd(g))(q[1:])
    return jax.vmap(one)(Q)


# ---- the network N ------------------------------------------------------------------------------
_zero = _init_net(jax, jnp, jax.random.PRNGKey(0), 2, 1)
_shift = ns["_taylor_mlp"](_zero, *[jnp.zeros((1, 1))] * 4, 0.0, 1.0)[0][0]   # flat values the pass adds


def net_derivs(net, win, Q):
    v, vt, vd, vdd = ns["_taylor_mlp"](net, Q[:, 0:1], Q[:, 1:2], Q[:, 2:3], Q[:, 3:4], win[0], win[1])
    return v - _shift[None], vt, vd, vdd


def u_derivs(P, win, Q):
    """Derivative stack of u at points Q (n,4), in _ccz4_residual_one's layout."""
    bv, bt, bd, bdd = bg_derivs(P["acc"], win, Q)
    Bv, Bt, Bd, Bdd = bv[:, :22], bt[:, :22], bd[:, :22], bdd[:, DD]
    rv, rt, rd, rdd = bv[:, 22], bt[:, 22], bd[:, 22], bdd[:, 22]
    N, Nt, Nd, Ndd = net_derivs(P["net"], win, Q)
    ND, NdD = N[:, DD], Nd[:, DD]
    pv = rv[:, None] * N
    pt = rt[:, None] * N + rv[:, None] * Nt
    pd = rd[:, None, :] * N[:, :, None] + rv[:, None, None] * Nd
    pdd = (rdd[:, None] * ND[:, :, None, None] + rd[:, None, :, None] * NdD[:, :, None, :]
           + rd[:, None, None, :] * NdD[:, :, :, None] + rv[:, None, None, None] * Ndd)
    val, dt, d, dd = Bv + pv, Bt + pt, Bd + pd, Bdd + pdd
    for f, fD in ((CHI, DCHI), (AL, DAL)):          # multiplicative: u = B exp(phi)
        E = jnp.exp(pv[:, f])
        b, b1, b2 = Bv[:, f], Bd[:, f], Bdd[:, fD]
        p1, p2 = pd[:, f], pdd[:, fD]
        val = val.at[:, f].set(b * E)
        dt = dt.at[:, f].set((Bt[:, f] + b * pt[:, f]) * E)
        d = d.at[:, f].set((b1 + b[:, None] * p1) * E[:, None])
        dd = dd.at[:, fD].set(E[:, None, None] * (b2 + b1[:, :, None] * p1[:, None, :] + p1[:, :, None] * b1[:, None, :]
                                                  + b[:, None, None] * (p2 + p1[:, :, None] * p1[:, None, :])))
    return val, dt, d, dd


def mlp_value(net, t, Q3, ws, we):
    """N's value (n,22) without the Taylor pass (as _taylor_mlp's first output, minus its flat values)."""
    mp = net["params"]["MLP"]
    Z = jnp.concatenate([2.0 * (t - ws) / (we - ws) - 1.0, Q3 / R], axis=1)
    h = Z
    for i in range(len(mp) - 1):
        h = jnp.tanh(h @ mp[f"Dense_{i}"]["kernel"] + mp[f"Dense_{i}"]["bias"])
    L = mp[f"Dense_{len(mp) - 1}"]
    return h @ L["kernel"] + L["bias"]


def u_value(P, win, Q):
    bv = jax.vmap(lambda q: bg_point(q, P["acc"], win))(Q)
    B, rho = bv[:, :22], bv[:, 22:]
    phi = rho * mlp_value(P["net"], Q[:, 0:1], Q[:, 1:4], win[0], win[1])
    u = B + phi
    return u.at[:, CHI].set(B[:, CHI] * jnp.exp(phi[:, CHI])).at[:, AL].set(B[:, AL] * jnp.exp(phi[:, AL]))


u_value_jit = jax.jit(u_value)


# ---- initial data (window 0) --------------------------------------------------------------------
if a.template == "trumpet":
    _full = stpl.load_table(tfile, "trumpet", R + 1.0, t_snap=a.t_snap)
    _radial_full = stpl.make_radial(jnp, jax, _full)


def initial_data(X):
    """Exact initial data (n,22) at points X (n,3): BL punctures at the initial centres, or the 1D slice."""
    if a.template == "trumpet":
        def one(x):
            xi = x - jnp.asarray(C0[0])
            return stpl.lift(jnp, pack, _radial_full(0.0, jnp.sqrt(jnp.sum(xi ** 2))), xi)
        return jax.vmap(one)(X)
    psi = 1.0 + sum(MASS / (2.0 * jnp.linalg.norm(X - jnp.asarray(c), axis=1)) for c in C0)
    U = jnp.tile(FLAT[None], (X.shape[0], 1))
    return U.at[:, CHI].set(psi ** -4).at[:, AL].set(1.0 / psi)


# ---- loss ---------------------------------------------------------------------------------------
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
    return (ns["lam_einstein"] * jnp.mean(res[:, :18] ** 2),
            ns["lam_gauge"] * a.w_gauge * jnp.mean(res[:, 18:22] ** 2))


def parts(P, win, batch):
    Qi, Qe, Qb, Xic, Uic = batch
    Di = u_derivs(P, win, Qi)
    e_i, g_i = ein_gauge(residual(Di))
    e_e, g_e = ein_gauge(residual(u_derivs(P, win, Qe)))
    val, dt, d, _ = u_derivs(P, win, Qb)
    Xb = Qb[:, 1:4]
    rb = jnp.linalg.norm(Xb, axis=1)[:, None]
    som = dt + ns["c"] * jnp.einsum("nk,nfk->nf", Xb, d) / rb + ns["c"] * (val - FLAT[None]) / rb
    t0 = jnp.full((Xic.shape[0], 1), win[0])
    ic = jnp.mean((u_value(P, win, jnp.concatenate([t0, Xic], axis=1)) - Uic) ** 2)
    con = zeta(Di) if a.w_constraint > 0 else jnp.zeros(())
    return dict(einstein=e_i, gauge=g_i, end=ns["lam_tapa"] * (e_e + g_e), bc=ns["lam_bc"] * jnp.mean(som ** 2),
                ic=ns["lamg0"] * ic, constraint=a.w_constraint * con)


def parts_pre(P, win, batch):
    """The IC and Sommerfeld terms of parts() alone (the --pre_iters phase)."""
    Qi, Qe, Qb, Xic, Uic = batch
    val, dt, d, _ = u_derivs(P, win, Qb)
    Xb = Qb[:, 1:4]
    rb = jnp.linalg.norm(Xb, axis=1)[:, None]
    som = dt + ns["c"] * jnp.einsum("nk,nfk->nf", Xb, d) / rb + ns["c"] * (val - FLAT[None]) / rb
    t0 = jnp.full((Xic.shape[0], 1), win[0])
    ic = jnp.mean((u_value(P, win, jnp.concatenate([t0, Xic], axis=1)) - Uic) ** 2)
    return dict(bc=ns["lam_bc"] * jnp.mean(som ** 2), ic=ns["lamg0"] * ic)


def total(p):
    return sum(p.values())


DIST_BANDS = ((0.0, 0.01), (0.01, 0.05), (0.05, 0.5), (0.5, 1e9))
DIST_NAMES = ("d<0.01", "d0.01-0.05", "d0.05-0.5", "d>0.5")


def dist_bands(P, win, Q):
    """Einstein + gauge loss of the interior points split by distance to the nearest puncture: each band's
    contribution to the mean over all points (they add up to einstein + gauge), and its point count."""
    res = residual(u_derivs(P, win, Q))
    per = (ns["lam_einstein"] * jnp.sum(res[:, :18] ** 2, axis=1) / 18
           + ns["lam_gauge"] * a.w_gauge * jnp.sum(res[:, 18:22] ** 2, axis=1) / 4)
    pos = jax.vmap(lambda t: center(P["acc"], win, t)[0])(Q[:, 0])               # (n, NP, 3)
    d = jnp.min(jnp.linalg.norm(Q[:, None, 1:4] - pos, axis=2), axis=1)
    m = jnp.stack([(d >= lo) & (d < hi) for lo, hi in DIST_BANDS])
    return jnp.sum(jnp.where(m, per, 0.0), axis=1) / per.shape[0], jnp.sum(m, axis=1)


# ---- points -------------------------------------------------------------------------------------
def ball(key, n, rad):
    k1, k2 = jax.random.split(key)
    u = jax.random.normal(k1, (n, 3))
    return u / jnp.linalg.norm(u, axis=1, keepdims=True) * rad * jax.random.uniform(k2, (n, 1)) ** (1 / 3)


def clouds(key, t, pos_of_t):
    """Points at times t (n,): inner, core and focused Gaussian clouds about each puncture (antithetic pairs
    xi, -xi, so every cloud is symmetric about its centre) and a uniform bulk in the ball r <= R. No exclusion
    zone: the inner cloud (sigma 0.05) puts points right at the puncture."""
    n = t.shape[0]
    sizes = [2 * int(f * n / 2 / NP) for f in (a.inner_frac, a.core_frac, a.focus_frac)]
    pos = pos_of_t(t)                                                      # (n, NP, 3)
    keys = jax.random.split(key, 3 * NP + 1)
    X, used = [], 0
    for A in range(NP):
        for j, m in enumerate(sizes):
            kA = keys[3 * A + j]
            g = (a.inner_sigma, a.core_sigma, a.focus_sigma)[j] * jax.random.normal(kA, (m // 2, 3))
            X.append(pos[used:used + m, A] + jnp.concatenate([g, -g]))
            used += m
    X.append(ball(keys[-1], n - used, R))
    X = jnp.concatenate(X)
    rr = jnp.linalg.norm(X, axis=1, keepdims=True)
    return jnp.where(rr > R, X / rr * R, X)


def draw(key, P, win, n_int, n_end, n_ic, n_bc, Uic_fn):
    t0, t1 = float(win[0]), float(win[1])
    pos_of_t = jax.jit(jax.vmap(lambda tt: center(P["acc"], win, tt)[0]))
    k = jax.random.split(key, 6)
    ti = t0 + (t1 - t0) * jax.random.uniform(k[0], (n_int,))
    Qi = jnp.concatenate([ti[:, None], clouds(k[1], ti, pos_of_t)], axis=1)
    te = jnp.full((n_end,), t1)
    Qe = jnp.concatenate([te[:, None], clouds(k[2], te, pos_of_t)], axis=1)
    tb = t0 + (t1 - t0) * jax.random.uniform(k[3], (n_bc,))
    u = jax.random.normal(k[4], (n_bc, 3))
    Qb = jnp.concatenate([tb[:, None], R * u / jnp.linalg.norm(u, axis=1, keepdims=True)], axis=1)
    Xic = clouds(k[5], jnp.full((n_ic,), t0), pos_of_t)
    return Qi, Qe, Qb, Xic, Uic_fn(Xic)


# ---- report against the 1D reference (single puncture at the origin only) -----------------------
REF_D = np.load(rfile)
REF_R, REF_T, REF_U = REF_D["r"], REF_D["t"], REF_D["U"]
RAYS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                 [1, 1, 1], [-1, -1, -1]], float)
RAYS /= np.linalg.norm(RAYS, axis=1, keepdims=True)
REP_R = np.geomspace(0.003, 20.0, 160)
REP_BANDS = (("inner", 0.0, 0.05), ("core", 0.05, 0.5), ("mid", 0.5, 5.0), ("far", 5.0, 20.0))
REP_GROUPS = (("alpha", [18]), ("chi", [12]), ("Khat", [13]), ("Theta", [14]), ("gt", list(range(6))),
              ("At", list(range(6, 12))), ("Ghat", [15, 16, 17]), ("beta", [19, 20, 21]))


def report(P, win, t):
    """rms field errors vs the 1D reference at time t (run time), on 8 rays from the ORIGIN."""
    j = int(np.argmin(np.abs(REF_T - (t + T_OFF))))
    if abs(REF_T[j] - (t + T_OFF)) > 1e-6:
        return None
    X = (RAYS[:, None, :] * REP_R[None, :, None]).reshape(-1, 3)
    rr = np.linalg.norm(X, axis=1)
    rad = np.stack([np.interp(rr, REF_R, REF_U[j, f]) for f in range(10)], axis=1)
    ref = np.asarray(jax.vmap(lambda rv, x: stpl.lift(jnp, pack, rv, x))(jnp.asarray(rad), jnp.asarray(X)))
    Q = jnp.concatenate([jnp.full((len(X), 1), t), jnp.asarray(X)], axis=1)
    u = np.asarray(u_value_jit(P, win, Q))
    row = {}
    for bn, lo, hi in REP_BANDS:
        m = (rr >= lo) & (rr < hi)
        for gn, idx in REP_GROUPS:
            row[f"{gn}_{bn}"] = float(np.sqrt(np.mean((u[m][:, idx] - ref[m][:, idx]) ** 2)))
    i03 = np.argmin(np.abs(REP_R - 0.03))
    row["alpha_r0.03"] = float(np.mean(u.reshape(len(RAYS), -1, 22)[:, i03, 18]))
    row["alpha_r0.03_ref"] = float(np.interp(REP_R[i03], REF_R, REF_U[j, 8]))
    return row


# ---- selftest ------------------------------------------------------------------------------------
def random_params(key):
    P = {"net": _init_net(jax, jnp, key, a.hidden, a.layers), "acc": 0.5 * jax.random.normal(key, (NP, 2, 3))}
    P["net"]["params"]["MLP"][f"Dense_{a.layers}"]["kernel"] = 0.3 * jax.random.normal(
        jax.random.PRNGKey(4), (a.hidden, 22), jnp.float64)
    return P


if a.selftest:
    # 1. The template at t = 0 against Brill-Lindquist (bl), and its own CCZ4 residual: the 1D solution
    #    lifted to 3D without cutoff, through the spline and autodiff, should nearly solve the equations.
    rr = np.geomspace(0.03, R_TPL, 40)
    tq = 0.0 if a.template == "bl" else 0.0
    rv = np.asarray(jax.vmap(lambda r: radial(tq + T_OFF, r))(jnp.asarray(rr)))
    if a.template == "bl":
        psi = 1 + MASS / (2 * rr)
        print(f"template t=0 vs BL: max |chi - psi^-4| {np.abs(rv[:, 4] - psi ** -4).max():.2e}, "
              f"max |alpha - 1/psi| {np.abs(rv[:, 8] - 1 / psi).max():.2e}")

    def pure_tpl(params, t, x, y, z, ws, we):
        X = jnp.concatenate([x, y, z], axis=1)
        return jax.vmap(lambda tt, xx: stpl.lift(jnp, pack, radial(tt + T_OFF, jnp.sqrt(jnp.sum(xx ** 2))), xx))(t[:, 0], X)

    key = jax.random.PRNGKey(1)
    for t_test in ([0.1, 1.0, 5.0] if a.template == "bl" else [0.0]):
        dirs = jax.random.normal(key, (200, 3))
        dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
        rads = jnp.geomspace(0.05, R_TPL * 0.9, 200)[:, None]
        X = dirs * rads
        tt = jnp.full((200, 1), t_test)
        res = np.asarray(ns["r_ccz4_evolution"](pure_tpl, None, tt, X[:, :1], X[:, 1:2], X[:, 2:3], 0.0, 1.0))
        rn = np.asarray(rads[:, 0])
        for lo, hi in ((0.05, 0.1), (0.1, 0.3), (0.3, R_TPL)):
            m = (rn >= lo) & (rn < hi)
            print(f"template residual t={t_test:4.1f}  r in [{lo:.2f},{hi:.2f}): rms Einstein "
                  f"{np.sqrt(np.mean(res[m, :18] ** 2)):.2e}, gauge {np.sqrt(np.mean(res[m, 18:] ** 2)):.2e}")
    # 2. The stacks against autodiff of u, random N, nonzero p1, p2 and v0, off-centre puncture.
    t0, t1 = 0.25, 0.5
    win = (jnp.asarray(t0), jnp.asarray(t1), jnp.asarray(C0) + 0.01, jnp.full((NP, 3), 0.02))
    P = random_params(jax.random.PRNGKey(3))

    def apply_u(params, t, x, y, z, ws, we):
        return u_value(params, win, jnp.concatenate([t, x, y, z], axis=1))

    Qi = draw(jax.random.PRNGKey(5), P, win, 64, 4, 4, 4, lambda X: jnp.zeros((X.shape[0], 22)))[0]
    tn = jnp.full((6,), 0.4)
    dn = jnp.array([[1e-4, 0, 0], [0, -1e-4, 0], [7e-4, 7e-4, 0], [0, 0, -1e-3], [6e-3, -8e-3, 0], [0, 1e-2, 0]])
    Qi = jnp.concatenate([Qi, jnp.concatenate([tn[:, None], center(P["acc"], win, 0.4)[0][0] + dn], axis=1)])
    pos_s = jax.vmap(lambda tt: center(P["acc"], win, tt)[0][0])(Qi[:, 0])
    print("selftest points: min distance to the puncture "
          f"{float(jnp.min(jnp.linalg.norm(Qi[:, 1:4] - pos_s, axis=1))):.1e}")
    t, x, y, z = Qi[:, 0:1], Qi[:, 1:2], Qi[:, 2:3], Qi[:, 3:4]
    ref = ns["r_ccz4_evolution"](apply_u, P, t, x, y, z, t0, t1)
    D = u_derivs(P, win, Qi)
    mine = residual(D)
    print(f"stacks vs autodiff: max |residual difference| {float(jnp.abs(mine - ref).max()):.2e} "
          f"(max |residual| {float(jnp.abs(ref).max()):.2e}); value difference "
          f"{float(jnp.abs(D[0] - apply_u(P, t, x, y, z, t0, t1)).max()):.2e}")
    pos, vel = center(P["acc"], win, jnp.asarray(0.4))
    b = u_value(P, win, jnp.concatenate([jnp.full((1, 1), 0.4), pos[:1] + 1e-4], axis=1))[0, ns["BETA_SLICE"]]
    print(f"shift at the puncture {np.asarray(b)} vs -da/dt {-np.asarray(vel[0])}")
    sys.exit(0)

if a.eval_spheres:
    NR, NDIR = 100, 48
    rs = np.geomspace(0.05, 20.0, NR)
    kk = np.arange(NDIR) + 0.5                                   # Fibonacci sphere, as tools/sphere_errors.py
    cz = 1.0 - 2.0 * kk / NDIR
    ph = np.pi * (1.0 + 5 ** 0.5) * kk
    nd = np.stack([np.sqrt(1 - cz ** 2) * np.cos(ph), np.sqrt(1 - cz ** 2) * np.sin(ph), cz], axis=1)
    ed = np.cross(nd, np.where(np.abs(nd[:, 2:3]) < 0.9, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]))
    ed /= np.linalg.norm(ed, axis=1, keepdims=True)
    Nd, Ed = jnp.asarray(np.repeat(nd, NR, axis=0)), jnp.asarray(np.repeat(ed, NR, axis=0))
    Sd = jnp.asarray(np.tile(rs, NDIR)[:, None])
    KH, AT = ns["KHAT_IDX"], ns["AT_SLICE"]
    REFQ = {"chi": (12, 4), "Khat": (13, 5), "Theta": (14, 6), "alpha": (18, 8)}   # (u column, reference row)

    def sphere_eval(P, win, t):
        pos = center(P["acc"], win, jnp.asarray(t))[0][0]

        def out(s, c):
            X = c[None, :] + s * Nd
            return u_value(P, win, jnp.concatenate([jnp.full_like(s, t), X], axis=1))

        def metric(o):
            return unpack(o[:, GT]) / o[:, CHI][:, None, None]

        def R_of(s):
            return s[:, 0] * jnp.sqrt(jnp.einsum("pi,pij,pj->p", Ed, metric(out(s, pos)), Ed))

        o = out(Sd, pos)
        R, dR = jax.jvp(R_of, (Sd,), (jnp.ones_like(Sd),))
        g = metric(o)
        K = o[:, KH] + 2.0 * o[:, TH]
        Kij = (unpack(o[:, AT]) + unpack(o[:, GT]) * K[:, None, None] / 3.0) / o[:, CHI][:, None, None]
        m = 0.5 * R * (1.0 - dR ** 2 / jnp.einsum("pi,pij,pj->p", Nd, g, Nd)
                       + R ** 2 * (jnp.einsum("pi,pij,pj->p", Ed, Kij, Ed) / jnp.einsum("pi,pij,pj->p", Ed, g, Ed)) ** 2)
        return m, out(Sd, jnp.zeros(3)), pos

    sphere_jit = jax.jit(sphere_eval)
    files = sorted(f for f in os.listdir(a.out) if f.startswith("window_") and f.endswith(".pkl"))
    acc_ = {"t": [], "center": []}
    for i, f in enumerate(files):
        St = pickle.load(open(os.path.join(a.out, f), "rb"))
        P = jax.tree_util.tree_map(jnp.asarray, St["params"])
        win = tuple(jnp.asarray(v) for v in St["win"])
        for t in ([float(win[0])] if i == 0 else []) + [float(win[1])]:
            m, o0, pos = (np.asarray(v) for v in sphere_jit(P, win, t))
            res = {"mass": m.reshape(NDIR, NR) / MASS - 1.0}
            j = int(np.argmin(np.abs(REF_T - (t + T_OFF))))
            assert abs(REF_T[j] - (t + T_OFF)) < 1e-6, f"no reference snapshot at t = {t + T_OFF}"
            for q, (col, row) in REFQ.items():
                res[q] = o0[:, col].reshape(NDIR, NR) - np.interp(rs, REF_R, REF_U[j, row])[None, :]
            acc_["t"].append(t)
            acc_["center"].append(pos)
            for q, v in res.items():
                acc_.setdefault(q + "_mean", []).append(v.mean(axis=0))
                acc_.setdefault(q + "_rms", []).append(np.sqrt((v ** 2).mean(axis=0)))
            i1 = int(np.argmin(np.abs(rs - 1.0)))
            print(f"t = {t:6.3f}  <m/M-1>(r=1) {res['mass'].mean(0)[i1]:+.2e}  min over r {res['mass'].mean(0).min():+.2e}"
                  f"  centre {pos.round(5).tolist()}", flush=True)
    np.savez(a.eval_spheres, r=rs, M=MASS, T_OFF=T_OFF, **{q: np.asarray(v) for q, v in acc_.items()})
    print("wrote", a.eval_spheres)
    sys.exit(0)

# ---- training -------------------------------------------------------------------------------------
json.dump(vars(a) | {"physics": rest, "template_file": tfile, "ref_file": rfile, "mass": MASS, "T_OFF": T_OFF},
          open(os.path.join(a.out, "config.json"), "w"), indent=1)
edges = np.linspace(0.0, a.tfinal, a.N_window + 1)
mb = ns["_minimize_mod"]
opts = {"maxiter": a.n_change, "gtol": 0.0, "update_method": "ssbroyden2", "initial_scale": True,
        "line_search": "wolfe", "c2": 0.9, "scale_invariant": True, "max_iter": 100, "alphamin": 1e-15}
key = jax.random.PRNGKey(a.seed)
P = {"net": _init_net(jax, jnp, jax.random.PRNGKey(a.seed + 1), a.hidden, a.layers), "acc": jnp.zeros((NP, 2, 3))}
a0, v0 = jnp.asarray(C0), jnp.zeros((NP, 3))
prev = None                                                    # (P, win) of the previous window
k0 = 0
if a.resume:
    done = sorted(f for f in os.listdir(a.out) if f.startswith("window_") and f.endswith(".pkl"))
    if done:
        S = pickle.load(open(os.path.join(a.out, done[-1]), "rb"))
        P = jax.tree_util.tree_map(jnp.asarray, S["params"])
        prev = (P, tuple(jnp.asarray(v) for v in S["win"]))
        a0, v0 = jnp.asarray(S["a_end"]), jnp.asarray(S["v_end"])
        k0, key = S["window"] + 1, jnp.asarray(S["key"])
        print(f"resumed after window {S['window']}")

parts_jit = jax.jit(parts)
bands_jit = jax.jit(dist_bands)
loss_fn = jax.jit(lambda w, unr, win, batch: total(parts(unr(w), win, batch)), static_argnums=1)
vg_fn = jax.jit(jax.value_and_grad(lambda w, unr, win, batch: total(parts(unr(w), win, batch))), static_argnums=1)
loss_pre = jax.jit(lambda w, unr, win, batch: total(parts_pre(unr(w), win, batch)), static_argnums=1)
vg_pre = jax.jit(jax.value_and_grad(lambda w, unr, win, batch: total(parts_pre(unr(w), win, batch))), static_argnums=1)
hist = os.path.join(a.out, "sing_history.csv")
rep = os.path.join(a.out, "sing_report.csv")
PART_NAMES = ("einstein", "gauge", "end", "bc", "ic", "constraint")


def mem_gb():
    """Peak device memory so far in GB (None where the backend does not report it, e.g. CPU)."""
    st = jax.devices()[0].memory_stats()
    return st["peak_bytes_in_use"] / 1e9 if st and "peak_bytes_in_use" in st else None
for k in range(k0, a.N_window):
    t0, t1 = float(edges[k]), float(edges[k + 1])
    win = (jnp.asarray(t0), jnp.asarray(t1), a0, v0)
    P = {"net": P["net"], "acc": jnp.zeros((NP, 2, 3))}          # warm network, fresh centre correction
    if prev is None:
        Uic_fn = jax.jit(initial_data)
    else:
        Pp, wp = prev
        Uic_fn = jax.jit(lambda X, Pp=Pp, wp=wp: u_value(Pp, wp, jnp.concatenate([jnp.full((X.shape[0], 1), wp[1]), X], axis=1)))
    w, unr = ns["flatten_util"].ravel_pytree(P)
    key, kh = jax.random.split(key)
    held = draw(kh, P, win, a.n_held, a.n_held // 8, a.n_held // 4, a.n_held // 8, Uic_fn)
    print(f"window {k}: t in [{t0:g}, {t1:g}], {w.size} parameters, centre {np.asarray(a0).round(5).tolist()}, "
          f"velocity {np.asarray(v0).round(5).tolist()}", flush=True)
    t_start, it = time.time(), 0
    _, cnt = bands_jit(unr(w), win, draw(kh, P, win, a.n_int, 1, 1, 1, Uic_fn)[0])
    print("  interior points per batch by distance to the puncture: "
          + ", ".join(f"{n} {int(v)}" for n, v in zip(DIST_NAMES, np.asarray(cnt))), flush=True)
    n_pre = a.pre_iters if a.pre_windows < 0 or k < a.pre_windows else 0
    while it < n_pre + a.iters:
        key, kd = jax.random.split(key)
        pre = it < n_pre
        batch = draw(kd, unr(w), win, 1 if pre else a.n_int, 1 if pre else a.n_end,
                     (a.pre_n_ic or a.n_ic) if pre else a.n_ic, a.n_bc, Uic_fn)
        lf, vg = (loss_pre, vg_pre) if pre else (loss_fn, vg_fn)
        r = mb.minimize(lambda v: lf(v, unr, win, batch), lambda v: vg(v, unr, win, batch), w, opts)
        w = r.x
        it += a.n_change
        hp = {n: float(v) for n, v in parts_jit(unr(w), win, held).items()}
        pos, vel = center(unr(w)["acc"], win, jnp.asarray(t1))
        hb, hn = (np.asarray(v) for v in bands_jit(unr(w), win, held[0]))
        row = dict(window=k, iteration=it, phase="pre" if pre else "full", seconds=round(time.time() - t_start, 1),
                   train=float(lf(w, unr, win, batch)), held=sum(hp.values()),
                   **{f"held_{n}": hp[n] for n in PART_NAMES},
                   **{f"held_{n}": float(v) for n, v in zip(DIST_NAMES, hb)},
                   **{f"n_{n}": int(v) for n, v in zip(DIST_NAMES, hn)},
                   **{f"a{A}_{c}": float(pos[A, i]) for A in range(NP) for i, c in enumerate("xyz")})
        new = not os.path.exists(hist)
        with open(hist, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
        print(f"  it {it:6d}{'p' if pre else ' '} {row['seconds']:7.0f} s  train {row['train']:.3e}  held {row['held']:.3e} ("
              + " ".join(f"{n} {hp[n]:.1e}" for n in PART_NAMES) + ") by distance "
              + " ".join(f"{n} {v:.1e}" for n, v in zip(DIST_NAMES, hb)) + f"  a(t1) {np.asarray(pos[0]).round(5).tolist()}"
              + (f"  peak mem {mem_gb():.2f} GB" if mem_gb() is not None else ""),
              flush=True)
    P = unr(w)
    pos, vel = center(P["acc"], win, jnp.asarray(t1))
    rr = report(P, win, t1)
    if rr is not None:
        rr = dict(window=k, t=t1, ax=float(pos[0, 0]), ay=float(pos[0, 1]), az=float(pos[0, 2]), **rr)
        new = not os.path.exists(rep)
        with open(rep, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rr))
            if new:
                wr.writeheader()
            wr.writerow(rr)
        print("  vs 1D reference at t = %g: alpha(0.03) %.4f (ref %.4f); core rms alpha %.2e chi %.2e Khat %.2e gt %.2e; "
              "mid alpha %.2e Khat %.2e" % (t1, rr["alpha_r0.03"], rr["alpha_r0.03_ref"], rr["alpha_core"],
                                             rr["chi_core"], rr["Khat_core"], rr["gt_core"], rr["alpha_mid"], rr["Khat_mid"]),
              flush=True)
    S = {"params": jax.tree_util.tree_map(np.asarray, P), "win": [np.asarray(v) for v in win], "window": k,
         "a_end": np.asarray(pos), "v_end": np.asarray(vel), "key": np.asarray(key), "config": vars(a)}
    pickle.dump(S, open(os.path.join(a.out, f"window_{k:03d}.pkl.tmp"), "wb"))
    os.replace(os.path.join(a.out, f"window_{k:03d}.pkl.tmp"), os.path.join(a.out, f"window_{k:03d}.pkl"))
    prev, a0, v0 = (P, win), pos, vel
print("done:", a.out)
