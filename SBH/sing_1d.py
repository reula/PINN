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
With --ic_d1 / --ic_d2 the IC (window hand-off) term also matches r f' and r^2 f'' to the previous window's end
(a Sobolev hand-off; r d/dr keeps the terms scale-free near the puncture; 2026-11-03).
With --w_adm the division-free ADM penalty r^p [(chi H)^2 + |chi M|^2] is added on the interior points; with
--w_adm_ic the same density on the hand-off slice t = t0 at the IC points (H, M involve only the fields and their
spatial derivatives on that slice, so this asks each window to start on a constraint-satisfying slice instead of
matching the previous window's slopes; 2026-11-04). --adm_r0/--adm_q multiply the r^p weight of both by
(1 + (r/r0)^q), and --res_r0/--res_q weight the Einstein/gauge residual by (1 + (r/r0)^q) (interior and t1
points): heavier penalties on the outer points (the user's weight from another study; 2026-11-05).
Points: the RADII of sing_pinn.py's 3D clouds (inner/core/focus Gaussians + uniform ball), so the loss weights
each shell as the 3D run does; --log_frac F draws a share F log-uniform in r on [--log_rmin, R] instead (equal
points per decade, down to the puncture; 2026-11-05). SSBroyden, fresh points every --n_change, warm start across windows.
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
import adm_constraints as admc

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
ap.add_argument("--log_frac", type=float, default=0.0,
                help="share of every point set (interior, t1, IC, held) drawn log-uniform in r on [--log_rmin, R], "
                     "the same number of points per decade; the cloud shares are scaled by (1 - log_frac) (2026-11-05)")
ap.add_argument("--log_rmin", type=float, default=1e-3)
ap.add_argument("--iters", type=int, default=4000)
ap.add_argument("--iters0", type=int, default=0,
                help="iterations of window 0 when > 0 (default: --iters); window 0 starts from scratch and can sit on a "
                     "plateau longer than the warm-started windows (2026-11-06)")
ap.add_argument("--n_change", type=int, default=250)
ap.add_argument("--ic_d1", type=float, default=0.0,
                help="weight of the first radial derivative in the IC / window hand-off term: lamg0 [mean|f - U|^2 "
                     "+ ic_d1 mean|r f' - r U'|^2 + ic_d2 mean|r^2 f'' - r^2 U''|^2] (r d/dr is scale-free; 2026-11-03)")
ap.add_argument("--ic_d2", type=float, default=0.0, help="weight of the second radial derivative (see --ic_d1)")
ap.add_argument("--w_constraint", type=float, default=10.0)
ap.add_argument("--w_adm", type=float, default=0.0,
                help="weight of the ADM Hamiltonian/momentum penalty mean(r^p [(chi H)^2 + |chi M_i|^2]) on the interior "
                     "points, chi H and chi M in the division-free form (adm_constraints.conformal_constraints_one: every "
                     "1/chi cancelled by hand, regular at the puncture); 0 = off (2026-11-03)")
ap.add_argument("--w_adm_ic", type=float, default=0.0,
                help="weight of the same r^p [(chi H)^2 + |chi M_i|^2] penalty on the window's hand-off slice t = t0, at the IC "
                     "points (the constraints of the starting slice; independent of --w_adm); 0 = off (2026-11-04)")
ap.add_argument("--adm_r_power", type=float, default=2.0, help="p in the --w_adm weight r^p (the colleagues: r^2)")
ap.add_argument("--adm_r0", type=float, default=0.0,
                help="with --adm_q: the --w_adm / --w_adm_ic weight becomes r^p (1 + (r/adm_r0)^adm_q); 0 = plain r^p (2026-11-05)")
ap.add_argument("--adm_q", type=float, default=2.0, help="q of the outer boost (see --adm_r0)")
ap.add_argument("--res_r0", type=float, default=0.0,
                help="with --res_q: the Einstein and gauge residual means on the interior and t1 points are weighted by "
                     "(1 + (r/res_r0)^res_q); 0 = plain means (2026-11-05)")
ap.add_argument("--res_q", type=float, default=2.0, help="q of the residual's outer boost (see --res_r0)")
ap.add_argument("--fit_ref", default=None, metavar="NPZ",
                help="supervised fit: train the network on window --fit_window to the 1D reference NPZ (sph_reference.py "
                     "output with several snapshots per window), loss mean|f - f_ref|^2 on the reference's own (t, r) "
                     "nodes; print the PINN loss parts of the fitted network as it trains; write fit_ref_wNNN.npz/.pkl "
                     "in --out and exit (2026-11-05)")
ap.add_argument("--fit_window", type=int, default=0, help="window index for --fit_ref (edges from --tfinal, --N_window)")
ap.add_argument("--fit_d1", type=float, default=0.0,
                help="--fit_ref: also fit r f' to r f_ref' with this weight (f_ref' by 4th-order differences on the grid)")
ap.add_argument("--fit_d2", type=float, default=0.0, help="--fit_ref: also fit r^2 f'' to r^2 f_ref'' with this weight")
ap.add_argument("--eval_tr", default=None, metavar="PKL",
                help="evaluate the residual of the network in PKL (a window_KKK.pkl or fit_ref_wKKK.pkl) on a (t, r) grid "
                     "over its window: per point the Einstein and gauge parts as the loss weights them (lam x mean of the "
                     "squared rows) and the r^p constraint density; write --eval_tr_out and exit (2026-11-05)")
ap.add_argument("--eval_tr_out", default=None, metavar="NPZ")
ap.add_argument("--eval_constraints", action="store_true",
                help="evaluate H, M_r, chi H, chi M_r and the --w_adm density on the report radii at every saved window "
                     "end in --out, write constraints.npz there, and exit")
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


def outer_boost(r, r0, q):
    """(1 + (r/r0)^q), or 1 when r0 = 0 (off)."""
    return 1.0 + (r / r0) ** q if r0 > 0 else jnp.ones_like(r)


def ein_gauge(res, r):
    wr = outer_boost(r, a.res_r0, a.res_q)[:, None]
    return (ns["lam_einstein"] * jnp.mean(wr * res[:, :18] ** 2),
            ns["lam_gauge"] * a.w_gauge * jnp.mean(wr * res[:, 18:22] ** 2))


IC_ORDER = 2 if a.ic_d2 > 0 else (1 if a.ic_d1 > 0 else 0)
IC_W = (1.0, a.ic_d1, a.ic_d2)[:IC_ORDER + 1]


def taylor_r(g, r):
    """(g, r g', r^2 g'') (up to IC_ORDER) of a radial function g(r) -> (10,) at one r: (IC_ORDER + 1, 10)."""
    if IC_ORDER == 0:
        return g(r)[None]
    v, d1 = jax.jvp(g, (r,), (1.0,))
    out = [v, r * d1]
    if IC_ORDER == 2:
        out.append(r * r * jax.jvp(lambda rr: jax.jvp(g, (rr,), (1.0,))[1], (r,), (1.0,))[1])
    return jnp.stack(out)


def parts(P, win, batch):
    Qi, Qe, tb, ric, Uic = batch                     # Uic (n, IC_ORDER + 1, 10): the target and its r-derivatives
    Di = stacks(P, win, Qi)
    e_i, g_i = ein_gauge(residual(Di), Qi[:, 1])
    e_e, g_e = ein_gauge(residual(stacks(P, win, Qe)), Qe[:, 1])
    f = lambda t, r: rad_fn(P, win, t, r)
    fb, ft = jax.vmap(lambda t: jax.jvp(lambda tt: f(tt, R), (t,), (1.0,)))(tb)
    fr = jax.vmap(lambda t: jax.jvp(lambda rr: f(t, rr), (R,), (1.0,))[1])(tb)
    som = ft + ns["c"] * fr + ns["c"] * (fb - FLAT_RAD[None]) / R
    d = jax.vmap(lambda r: taylor_r(lambda rr: f(win[0], rr), r))(ric) - Uic
    ic = sum(wk * jnp.mean(d[:, k] ** 2) for k, wk in enumerate(IC_W))
    con = zeta(Di) if a.w_constraint > 0 else jnp.zeros(())
    out = dict(einstein=e_i, gauge=g_i, end=ns["lam_tapa"] * (e_e + g_e), bc=ns["lam_bc"] * jnp.mean(som ** 2),
               ic=ns["lamg0"] * ic, constraint=a.w_constraint * con)
    if a.w_adm > 0:
        out["adm"] = a.w_adm * jnp.mean(adm_density(Di, Qi[:, 1]))
    if a.w_adm_ic > 0:
        out["adm_ic"] = a.w_adm_ic * jnp.mean(adm_density(stacks(P, win, axis(jnp.full_like(ric, win[0]), ric)), ric))
    return out


def adm_density(D, r):
    """r^p [(chi H)^2 + |chi M_i|^2] per point, division-free, from a derivative stack D at radii r (the --w_adm integrand)."""
    cH, cM = admc.conformal_constraints_from_derivs(ns, *D)
    return r ** a.adm_r_power * outer_boost(r, a.adm_r0, a.adm_q) * (cH ** 2 + jnp.sum(cM ** 2, axis=1))


def total(p):
    return sum(p.values())


# ---- points: the radii of sing_pinn.py's clouds --------------------------------------------------
def radii(key, n):
    n_log = int(a.log_frac * n)
    sizes = [2 * int(f * (n - n_log) / 2) for f in (a.inner_frac, a.core_frac, a.focus_frac)]
    keys = jax.random.split(key, 5)
    rs = [s * jnp.linalg.norm(jax.random.normal(k, (m, 3)), axis=1)
          for k, m, s in zip(keys[:3], sizes, (a.inner_sigma, a.core_sigma, a.focus_sigma))]
    rs.append(R * jax.random.uniform(keys[3], (n - n_log - sum(sizes),)) ** (1 / 3))
    if n_log > 0:
        rs.append(a.log_rmin * (R / a.log_rmin) ** jax.random.uniform(keys[4], (n_log,)))
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


if a.eval_constraints:
    out = {"t": [], "H": [], "Mr": [], "cH": [], "cMr": [], "dens": [], "chi": []}
    for f in sorted(x for x in os.listdir(a.out) if x.startswith("window_") and x.endswith(".pkl")):
        S = pickle.load(open(os.path.join(a.out, f), "rb"))
        Pk, wk = jax.tree_util.tree_map(jnp.asarray, S["params"]), tuple(jnp.asarray(v) for v in S["win"])
        rr = jnp.asarray(REP_R)
        D = stacks(Pk, wk, jnp.stack([jnp.full_like(rr, wk[1]), rr, jnp.zeros_like(rr), jnp.zeros_like(rr)], axis=1))
        H, Mv = admc.constraints_from_derivs(ns, *D)
        cH, cM = admc.conformal_constraints_from_derivs(ns, *D)
        out["t"].append(float(wk[1]))
        out["H"].append(np.asarray(H))
        out["Mr"].append(np.asarray(Mv[:, 0]))                  # on the x-axis M_i = M_r n_i
        out["cH"].append(np.asarray(cH))
        out["cMr"].append(np.asarray(cM[:, 0]))
        out["dens"].append(np.asarray(adm_density(D, rr)))
        out["chi"].append(np.asarray(D[0][:, CHI]))
    np.savez(os.path.join(a.out, "constraints.npz"), r=REP_R, **{k: np.array(v) for k, v in out.items()},
             adm_r_power=a.adm_r_power)
    print("wrote", os.path.join(a.out, "constraints.npz"), len(out["t"]), "windows")
    sys.exit(0)


if a.eval_tr:
    S = pickle.load(open(a.eval_tr, "rb"))
    Pk = jax.tree_util.tree_map(jnp.asarray, S["params"])
    wk = tuple(jnp.asarray(float(v)) for v in S["win"][:2])
    tg = np.linspace(float(wk[0]), float(wk[1]), 41)
    rg = np.geomspace(1e-3, R, 240)
    T_, R_ = np.meshgrid(tg, rg, indexing="ij")
    Q = jnp.asarray(np.stack([T_.ravel(), R_.ravel(), 0 * T_.ravel(), 0 * T_.ravel()], axis=1))
    out = {k: [] for k in ("einstein", "gauge", "adm")}
    st = jax.jit(lambda Q: stacks(Pk, wk, Q))
    for i in range(0, Q.shape[0], 2048):
        D = st(Q[i:i + 2048])
        res = residual(D)
        out["einstein"].append(np.asarray(ns["lam_einstein"] * jnp.mean(res[:, :18] ** 2, axis=1)))
        out["gauge"].append(np.asarray(ns["lam_gauge"] * a.w_gauge * jnp.mean(res[:, 18:22] ** 2, axis=1)))
        out["adm"].append(np.asarray(adm_density(D, Q[i:i + 2048, 1])))
    shape = T_.shape
    np.savez(a.eval_tr_out, t=tg, r=rg, **{k: np.concatenate(v).reshape(shape) for k, v in out.items()},
             src=np.array(a.eval_tr))
    print("wrote", a.eval_tr_out)
    sys.exit(0)


if a.fit_ref:
    # Supervised fit to the 1D reference over one window, then the PINN loss of the fitted network (2026-11-05).
    FR = np.load(a.fit_ref)
    FR_T, FR_R, FR_U = FR["t"], FR["r"], np.transpose(FR["U"], (0, 2, 1))     # FR_U (n_t, n_r, 10)
    edges = np.linspace(0.0, a.tfinal, a.N_window + 1)
    t0, t1 = float(edges[a.fit_window]), float(edges[a.fit_window + 1])
    win = (jnp.asarray(t0), jnp.asarray(t1))
    J = np.where((FR_T >= t0 - 1e-9) & (FR_T <= t1 + 1e-9))[0]
    assert len(J) >= 3, f"--fit_ref has only {len(J)} snapshots in [{t0}, {t1}]"
    dr_ref = float(FR_R[1] - FR_R[0])
    TJ, UJ = jnp.asarray(FR_T[J]), jnp.asarray(FR_U[J])
    j0 = int(np.argmin(np.abs(FR_T - t0)))

    def fd_r(U, h, order):
        """4th-order central d/dr (order 1) or d^2/dr^2 (order 2) along axis 1; 2nd-order one-sided at the ends."""
        D = np.gradient(U, h, axis=1, edge_order=2) if order == 1 else \
            np.gradient(np.gradient(U, h, axis=1, edge_order=2), h, axis=1, edge_order=2)
        c = U
        if order == 1:
            D[:, 2:-2] = (-c[:, 4:] + 8 * c[:, 3:-1] - 8 * c[:, 1:-3] + c[:, :-4]) / (12 * h)
        else:
            D[:, 2:-2] = (-c[:, 4:] + 16 * c[:, 3:-1] - 30 * c[:, 2:-2] + 16 * c[:, 1:-3] - c[:, :-4]) / (12 * h * h)
        return D

    SOB = a.fit_d1 > 0 or a.fit_d2 > 0
    D1J = jnp.asarray(fd_r(FR_U[J], float(FR_R[1] - FR_R[0]), 1))
    D2J = jnp.asarray(fd_r(FR_U[J], float(FR_R[1] - FR_R[0]), 2))

    def fit_batch(key, n):
        """n points on the reference's own (t, r) nodes: snapshot in the window, radius = the run's cloud snapped to the grid."""
        k1, k2 = jax.random.split(key)
        it = jax.random.randint(k1, (n,), 0, len(J))
        ir = jnp.clip(jnp.round((radii(k2, n) - FR_R[0]) / dr_ref).astype(int), 0, len(FR_R) - 1)
        return TJ[it], jnp.asarray(FR_R)[ir], UJ[it, ir], D1J[it, ir], D2J[it, ir]

    def taylor2(g, r):
        v, d1 = jax.jvp(g, (r,), (1.0,))
        d2 = jax.jvp(lambda rr: jax.jvp(g, (rr,), (1.0,))[1], (r,), (1.0,))[1]
        return v, d1, d2

    def fit_loss(P, b):
        t, r, U, D1, D2 = b
        if not SOB:
            f = jax.vmap(lambda tt, rr: rad_fn(P, win, tt, rr))(t, r)
            return jnp.mean((f - U) ** 2)
        f, f1, f2 = jax.vmap(lambda tt, rr: taylor2(lambda q: rad_fn(P, win, tt, q), rr))(t, r)
        return (jnp.mean((f - U) ** 2) + a.fit_d1 * jnp.mean((r[:, None] * (f1 - D1)) ** 2)
                + a.fit_d2 * jnp.mean((r[:, None] ** 2 * (f2 - D2)) ** 2))

    Uic_fn = jax.jit(jax.vmap(lambda r: jnp.stack([jnp.interp(r, jnp.asarray(FR_R), jnp.asarray(FR_U[j0, :, i]))
                                                    for i in range(10)])[None]))
    mb = ns["_minimize_mod"]
    opts = {"maxiter": a.n_change, "gtol": 0.0, "update_method": "ssbroyden2", "initial_scale": True,
            "line_search": "wolfe", "c2": 0.9, "scale_invariant": True, "max_iter": 100, "alphamin": 1e-15}
    key = jax.random.PRNGKey(a.seed)
    P = {"net": init_net(jax.random.PRNGKey(a.seed + 1))}
    w, unr = ns["flatten_util"].ravel_pytree(P)
    fl = jax.jit(lambda v, b: fit_loss(unr(v), b))
    fval = jax.jit(lambda v, b: jnp.mean((jax.vmap(lambda tt, rr: rad_fn(unr(v), win, tt, rr))(b[0], b[1]) - b[2]) ** 2))
    fvg = jax.jit(jax.value_and_grad(lambda v, b: fit_loss(unr(v), b)))
    parts_jit = jax.jit(parts)
    key, kh, kf = jax.random.split(key, 3)
    held = draw(kh, win, a.n_held, a.n_held // 8, a.n_held // 4, Uic_fn)
    fheld = fit_batch(kf, a.n_held)
    names = ("einstein", "gauge", "end", "bc", "ic", "constraint") + (("adm",) if a.w_adm > 0 else ()) \
        + (("adm_ic",) if a.w_adm_ic > 0 else ())
    rows, t_start, it = [], time.time(), 0
    print(f"fit to {a.fit_ref}, window {a.fit_window}: t in [{t0:g}, {t1:g}], {len(J)} snapshots, {w.size} parameters"
          + (f", Sobolev weights r f' {a.fit_d1:g}, r^2 f'' {a.fit_d2:g}" if SOB else ""), flush=True)
    while it < a.iters:
        key, kd = jax.random.split(key)
        b = fit_batch(kd, a.n_int)
        w = mb.minimize(lambda v: fl(v, b), lambda v: fvg(v, b), w, opts).x
        it += a.n_change
        hp = {n: float(v) for n, v in parts_jit(unr(w), win, held).items()}
        rows.append(dict(iteration=it, seconds=round(time.time() - t_start, 1), fit=float(fl(w, fheld)), fit_val=float(fval(w, fheld)),
                         pinn=sum(hp.values()), **hp))
        print(f"  it {it:6d} {rows[-1]['seconds']:7.0f} s  fit loss (held) {rows[-1]['fit']:.3e} (values {rows[-1]['fit_val']:.3e})  PINN loss (held) "
              f"{rows[-1]['pinn']:.3e} (" + " ".join(f"{n} {hp[n]:.1e}" for n in names) + ")", flush=True)
    P = unr(w)
    err, m = report(P, win, t1)
    np.savez(os.path.join(a.out, f"fit_ref_w{a.fit_window:03d}.npz"), r=REP_R, t1=t1, err=err, mass=m,
             history=np.array([[r_[k] for k in ["iteration", "seconds", "fit", "fit_val", "pinn"] + list(names)] for r_ in rows]),
             history_cols=np.array(["iteration", "seconds", "fit", "fit_val", "pinn"] + list(names)), config=np.array(str(vars(a))))
    pickle.dump({"params": jax.tree_util.tree_map(np.asarray, P), "win": [t0, t1], "config": vars(a)},
                open(os.path.join(a.out, f"fit_ref_w{a.fit_window:03d}.pkl"), "wb"))
    bands = ((0.05, 0.2), (0.2, 1.0), (1.0, 3.0), (3.0, 10.0))
    print(f"  t = {t1:g}: " + " | ".join(lab + " " + " ".join(f"{np.sqrt((err[(REP_R >= lo) & (REP_R < hi), i] ** 2).mean()):.0e}"
                                                             for lo, hi in bands)
                                         for i, lab in ((8, "alpha"), (4, "chi"), (5, "Khat")))
          + "; m/M - 1 at r = 0.3 / 1 / 3 / 10: " + " ".join(f"{np.interp(x, REP_R, m):+.1e}" for x in (0.3, 1, 3, 10)),
          flush=True)
    print("done:", a.out)
    sys.exit(0)


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
PART_NAMES = (("einstein", "gauge", "end", "bc", "ic", "constraint") + (("adm",) if a.w_adm > 0 else ())
              + (("adm_ic",) if a.w_adm_ic > 0 else ()))
for k in range(k0, a.N_window):
    t0, t1 = float(edges[k]), float(edges[k + 1])
    win = (jnp.asarray(t0), jnp.asarray(t1))
    if prev is None:
        Uic_fn = jax.jit(jax.vmap(lambda r: taylor_r(U0_rad, r)))
    else:
        Pp, wp = prev
        Uic_fn = jax.jit(jax.vmap(lambda r, Pp=Pp, wp=wp: taylor_r(lambda rr: rad_fn(Pp, wp, wp[1], rr), r)))
    w, unr = ns["flatten_util"].ravel_pytree(P)
    key, kh = jax.random.split(key)
    held = draw(kh, win, a.n_held, a.n_held // 8, a.n_held // 4, Uic_fn)
    if k == 0:
        e0 = np.asarray(jax.vmap(lambda r: rad_fn(P, win, 0.0, r) - U0_rad(r))(jnp.asarray(REP_R)))
        print(f"window 0: max |u(0) - U0| over the radial functions {np.abs(e0).max():.2e}", flush=True)
    print(f"window {k}: t in [{t0:g}, {t1:g}], {w.size} parameters", flush=True)
    t_start, it = time.time(), 0
    n_iters = a.iters0 if (k == 0 and a.iters0 > 0) else a.iters
    while it < n_iters:
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
        st = jax.devices()[0].memory_stats()                     # None on the CPU
        print(f"  it {it:6d} {row['seconds']:7.0f} s  train {row['train']:.3e}  held {row['held']:.3e} ("
              + " ".join(f"{n} {hp[n]:.1e}" for n in PART_NAMES) + ")"
              + (f"  peak mem {st['peak_bytes_in_use'] / 1e9:.2f} GB" if st and "peak_bytes_in_use" in st else ""),
              flush=True)
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
