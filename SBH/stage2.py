"""Stage 2: one correction network over the whole span of a finished windowed run. (2026-10-26)

    JAX_ENABLE_X64=1 python stage2.py RUN_DIR --out OUT_DIR [options] -- <physics flags of the run>

The finished run (its windows' networks, frozen) is u1. A single new network N covers t in [0, T] and
the solution becomes
    u = u1 + s(t, r) N(t, x, y, z),     s = eps * (t / T) * (1 - r^2 / R^2),
so u keeps u1's initial data exactly (s = 0 at t = 0) and u1's values on the outer boundary r = R
(s = 0 there). N is trained, with SSBroyden on fresh points every --n_change iterations, so that u
satisfies the CCZ4 evolution equations (same residual and weights as the main script) plus, with
--w_constraint W, W * mean(Theta^2 + |Z^i|^2), and with --w_adm W', W' * mean(chi^p (H^2 + |M_i|^2)) of the
ADM constraints (adm_constraints.py). N's last layer starts at zero, so training starts
exactly from u1. eps (default: the rms of u1's residual on the first sample) only sets N's scale.

The residual needs the derivatives of the sum, which add: u1's value / d_t / d_xyz / d_xyz d_xyz come
from each window's own network (closed-form Taylor mode, all windows in one vmapped call, points drawn
in equal numbers per window span), N's the same way, times s by the product rule. The hand-off jumps
of u1 between windows stay: N is smooth in time.

Points per draw: --n_per_window in each window span, by default evenly in the ball r <= R (the
correction is for what propagates out, which the run's puncture-centred clouds sample thinly; see
--sampling for the alternatives), none closer than --min_puncture_dist; times uniform in the span. A fixed held-out sample
of the same kind grades u against u1 at every draw.

Outputs in OUT_DIR: stage2_history.csv (per draw: iteration, loss of u and of u1 on the training and
held-out samples, and their parts), stage2_params.pkl (N's parameters and the configuration, the
input of tools/sphere_errors.py --stage2).
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
from stage2_net import init_net as _init_net, make_correction
import adm_constraints as adm

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--out", required=True)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--layers", type=int, default=4, help="hidden layers of N")
ap.add_argument("--n_per_window", type=int, default=3200)
ap.add_argument("--n_heldout_per_window", type=int, default=800)
ap.add_argument("--iters", type=int, default=24000)
ap.add_argument("--n_change", type=int, default=250)
ap.add_argument("--w_constraint", type=float, default=10.0)
ap.add_argument("--w_adm", type=float, default=0.0,
                help="weight of mean(chi^p (H^2 + |M_i|^2)), the ADM constraints of the sum (adm_constraints.py)")
ap.add_argument("--adm_chi_power", type=float, default=2.0)
ap.add_argument("--eps", type=float, default=0.0, help="0: rms of u1's residual on the first sample")
ap.add_argument("--windows", type=int, default=0, help="use only the first K windows (tests); 0 = all")
ap.add_argument("--min_puncture_dist", type=float, default=0.02)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--resume", action="store_true", help="continue from OUT_DIR/stage2_params.pkl")
ap.add_argument("--sampling", choices=["final", "shell", "uniform"], default="uniform",
                help="uniform: evenly in the ball r <= R (volume); shell: radius uniform in [--shell_rmin, R] about "
                     "the puncture (density ~ 1/r^2); final: the main script's final-settings proportions")
ap.add_argument("--shell_rmin", type=float, default=0.5)
ap.add_argument("--core_taper", type=float, default=0.0,
                help="rc > 0: multiply the correction by 1 - exp(-r^2/rc^2), leaving the hole to the run")
ap.add_argument("--selftest", action="store_true", help="check the summed residual against autodiff and exit")
argv = sys.argv[1:]
rest = argv[argv.index("--") + 1:] if "--" in argv else []
a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
os.makedirs(a.out, exist_ok=True)

ns = load(rest + ["--chi_bg_mode", "moving", "--deriv_mode", "taylor"], outdir=tempfile.mkdtemp(prefix="s2_"))
jax, jnp, np = ns["jax"], ns["jnp"], ns["np"]
GT, CHI, TH, GH = ns["GT_SLICE"], ns["CHI_IDX"], ns["THETA_IDX"], ns["GHAT_SLICE"]
DD = np.asarray(ns["_DD_OUTPUT_IDXS"])
R = float(ns["R_domain"])
unpack = ns["unpack_sym3"]

# ---- u1: the frozen windows, stacked along a leading window axis --------------------------------
man = pickle.load(open(os.path.join(a.run, "pinn_checkpoint.pkl"), "rb"))
files = man["window_files"][:a.windows] if a.windows else man["window_files"]
states = [pickle.load(open(os.path.join(a.run, f), "rb")) for f in files]
assert all(st.get("ic_mode") == "soft" for st in states), "hard-IC windows are not Taylor-capable"
W = len(states)
stack = lambda *v: jnp.stack([jnp.asarray(x) for x in v])
U1 = {"net": jax.tree_util.tree_map(stack, *[st["params"] for st in states]),
      "traj_tt": stack(*[st["traj_tt"] for st in states]),
      "traj_xx": stack(*[st["traj_xx"] for st in states])}
WS = jnp.asarray([float(st["t_start"]) for st in states])
WE = jnp.asarray([float(st["t_end"]) for st in states])
T0, T = float(WS[0]), float(WE[-1])
assert T0 == 0.0


def _moving_kind():
    pass


ns["_TAYLOR_KIND"][_moving_kind] = "moving"


def u1_window(net, tt, xx, ws, we, t, x, y, z):
    return ns["_taylor_derivs"](_moving_kind, {"net": net, "traj_tt": tt, "traj_xx": xx}, t, x, y, z, ws, we)


@jax.jit
def u1_derivs(P):          # P: (W, n, 4) points, window k's in row k -> stacks flattened to (W*n, ...)
    d = jax.vmap(u1_window)(U1["net"], U1["traj_tt"], U1["traj_xx"], WS, WE,
                            P[..., 0:1], P[..., 1:2], P[..., 2:3], P[..., 3:4])
    return tuple(v.reshape((-1,) + v.shape[2:]) for v in d)


# ---- N: the correction network (stage2_net.py) --------------------------------------------------
def init_net(key):
    return _init_net(jax, jnp, key, a.hidden, a.layers)


def correction_derivs(p2, eps, t, x, y, z):
    return make_correction(ns, T, R, eps, a.core_taper)[0](p2, t, x, y, z)


def correction_value(p2, eps, t, x, y, z):
    return make_correction(ns, T, R, eps, a.core_taper)[1](p2, t, x, y, z)


def parts(derivs):
    res = jax.vmap(ns["_ccz4_residual_one"])(*derivs)
    ein = ns["lam_einstein"] * jnp.mean(res[:, :18] ** 2)
    gau = ns["lam_gauge"] * jnp.mean(res[:, 18:22] ** 2)
    val, _, d, _ = derivs
    gt = unpack(val[:, GT])
    gt_d = jnp.transpose(unpack(jnp.moveaxis(d[:, GT, :], 1, 2)), (0, 2, 3, 1))   # [n,a,b,c] = d_c gt_ab
    gu = jax.vmap(lambda g, dg: ns["_christoffel3"](g, dg)[2])(gt, gt_d)
    zv = val[:, CHI][:, None] * (val[:, GH] - gu) / 2.0
    con = jnp.mean(val[:, TH] ** 2 + jnp.sum(zv ** 2, axis=1))
    H, M = adm.constraints_from_derivs(ns, *derivs)
    hm = adm.adm_penalty_from(H, M, val[:, CHI], a.adm_chi_power)
    return ein, gau, con, hm


BANDS = ((0.0, 1.0), (1.0, 5.0), (5.0, 15.0), (15.0, 1e9))
BAND_NAMES = ("r<1", "r1-5", "r5-15", "r>15")


def band_losses(derivs, rr):
    """Einstein + gauge residual (same weights) split by radius from the origin: each band's
    contribution to the mean over ALL points, so the bands add up to ein + gau."""
    res = jax.vmap(ns["_ccz4_residual_one"])(*derivs)
    per = (ns["lam_einstein"] * jnp.sum(res[:, :18] ** 2, axis=1) / 18
           + ns["lam_gauge"] * jnp.sum(res[:, 18:22] ** 2, axis=1) / 4)
    return jnp.stack([jnp.sum(jnp.where((rr >= lo) & (rr < hi), per, 0.0)) / per.shape[0] for lo, hi in BANDS])


def total(ein, gau, con, hm):
    return ein + gau + a.w_constraint * con + a.w_adm * hm


# ---- points ----------------------------------------------------------------------------------
def draw(key, n):
    """(W, n, 4) points (t, x, y, z): n per window span, final-settings proportions."""
    kt, kc, kf, kb, kr = jax.random.split(key, 5)
    t = WS[:, None] + (WE - WS)[:, None] * jax.random.uniform(kt, (W, n))
    cx = jax.vmap(lambda tt, xx, tq: jnp.interp(tq, tt, xx[0, 0]))(U1["traj_tt"], U1["traj_xx"], t)
    cy = jax.vmap(lambda tt, xx, tq: jnp.interp(tq, tt, xx[0, 1]))(U1["traj_tt"], U1["traj_xx"], t)
    c = jnp.stack([cx, cy, jnp.zeros_like(cx)], axis=-1)        # (W,n,3) tracked puncture
    if a.sampling == "uniform":
        u = jax.random.normal(kc, (W, n, 3))
        X = u / jnp.linalg.norm(u, axis=-1, keepdims=True) * R * jax.random.uniform(kr, (W, n, 1)) ** (1 / 3)
    elif a.sampling == "shell":
        u = jax.random.normal(kc, (W, n, 3))
        rad = a.shell_rmin + (R - a.shell_rmin) * jax.random.uniform(kr, (W, n, 1))
        X = c + u / jnp.linalg.norm(u, axis=-1, keepdims=True) * rad
    else:
        n_core, n_foc = (3 * n) // 8, (3 * n) // 8
        n_bulk = n - n_core - n_foc
        off = jnp.concatenate([0.5 * jax.random.normal(kc, (W, n_core, 3)),
                               2.0 * jax.random.normal(kf, (W, n_foc, 3))], axis=1)
        u = jax.random.normal(kb, (W, n_bulk, 3))
        u = u / jnp.linalg.norm(u, axis=-1, keepdims=True) * R * jax.random.uniform(kr, (W, n_bulk, 1)) ** (1 / 3)
        X = jnp.concatenate([c[:, :n_core + n_foc] + off, u], axis=1)
    rel = X - c
    dist = jnp.linalg.norm(rel, axis=-1, keepdims=True)
    X = jnp.where(dist < a.min_puncture_dist, c + rel / jnp.maximum(dist, 1e-12) * a.min_puncture_dist, X)
    rr = jnp.linalg.norm(X, axis=-1, keepdims=True)
    X = jnp.where(rr > R, X / rr * R, X)
    return jnp.concatenate([t[..., None], X], axis=-1)


def flat_pts(P):
    P = P.reshape(-1, 4)
    return P[:, 0:1], P[:, 1:2], P[:, 2:3], P[:, 3:4]


if a.selftest:
    # The residual of u1 + s N from the derivative stacks above against the main script's own autodiff
    # residual of the summed function, in a middle window, with a random (nonzero) N.
    k = W // 2
    fn1, p1, ws, we = ns["_network_apply_for_state"](states[k], jnp.asarray(states[k]["traj_tt"]),
                                                     jnp.asarray(states[k]["traj_xx"]))
    pr = init_net(jax.random.PRNGKey(3))
    pr["params"]["MLP"][f"Dense_{a.layers}"]["kernel"] = 0.3 * jax.random.normal(
        jax.random.PRNGKey(4), (a.hidden, 22), jnp.float64)
    eps_t = 1e-2

    def summed_apply(params, t, x, y, z, ws_, we_):
        return fn1(p1, t, x, y, z, ws_, we_) + correction_value(params, eps_t, t, x, y, z)

    P = draw(jax.random.PRNGKey(5), 16)[k]                      # (16, 4) in window k
    t, x, y, z = P[:, 0:1], P[:, 1:2], P[:, 2:3], P[:, 3:4]
    ref = ns["r_ccz4_evolution"](summed_apply, pr, t, x, y, z, ws, we)    # autodiff path
    D1 = u1_window(jax.tree_util.tree_map(jnp.asarray, states[k]["params"]), U1["traj_tt"][k], U1["traj_xx"][k], WS[k], WE[k], t, x, y, z)
    D2 = correction_derivs(pr, eps_t, t, x, y, z)
    mine = jax.vmap(ns["_ccz4_residual_one"])(*(u + v for u, v in zip(D1, D2)))
    alone = ns["r_ccz4_evolution"](fn1, p1, t, x, y, z, ws, we)
    print(f"selftest: max |residual(stacks) - residual(autodiff)| = {float(jnp.abs(mine - ref).max()):.2e}; "
          f"max |residual| = {float(jnp.abs(ref).max()):.2e}; change made by N: {float(jnp.abs(ref - alone).max()):.2e}")
    sys.exit(0)

key = jax.random.PRNGKey(a.seed)
key, kh, k0 = jax.random.split(key, 3)
P_held = draw(kh, a.n_heldout_per_window)
D_held = u1_derivs(P_held)
held_pts = flat_pts(P_held)
held_r = jnp.sqrt(held_pts[1][:, 0] ** 2 + held_pts[2][:, 0] ** 2 + held_pts[3][:, 0] ** 2)

ckpt = os.path.join(a.out, "stage2_params.pkl")
if a.resume and os.path.exists(ckpt):
    C = pickle.load(open(ckpt, "rb"))
    p2, eps, it0 = jax.tree_util.tree_map(jnp.asarray, C["params"]), float(C["eps"]), int(C["iteration"])
    key = jnp.asarray(C["key"])
    print(f"resumed at iteration {it0}, eps {eps:.3e}")
else:
    p2, it0 = init_net(k0), 0
    e1, g1, c1, _ = parts(D_held)
    eps = a.eps or float(jnp.sqrt((e1 * 18 + g1 * 4) / 22))
    print(f"eps = {eps:.3e}  (u1 on the held-out sample: Einstein {float(e1):.3e}, gauge {float(g1):.3e}, "
          f"constraint {float(c1):.3e})")
w0, unravel = ns["flatten_util"].ravel_pytree(p2)
print(f"u1: {W} windows to t = {T:g}; N: {a.layers}x{a.hidden}, {w0.size} parameters; "
      f"{W * a.n_per_window} points per draw ({a.n_per_window} per window span)", flush=True)


def summed(w, D1, pts):
    D2 = correction_derivs(unravel(w), eps, *pts)
    return tuple(u + v for u, v in zip(D1, D2))


@jax.jit
def loss_fn(w, D1, pts):
    return total(*parts(summed(w, D1, pts)))


vg_fn = jax.jit(jax.value_and_grad(loss_fn))
parts_fn = jax.jit(lambda w, D1, pts: parts(summed(w, D1, pts)))
parts_u1 = jax.jit(parts)
bands_fn = jax.jit(lambda w, D1, pts, rr: band_losses(summed(w, D1, pts), rr))
bands_u1 = [float(v) for v in jax.jit(band_losses)(D_held, held_r)]
print("u1 held-out residual by radius: " + ", ".join(f"{n} {v:.2e} ({100 * v / sum(bands_u1):.0f} %)"
                                                      for n, v in zip(BAND_NAMES, bands_u1)), flush=True)

hist = os.path.join(a.out, "stage2_history.csv")
fields = ["iteration", "seconds", "train_u", "train_u1", "held_u", "held_u1",
          "held_einstein", "held_gauge", "held_constraint", "held_adm", "held_u1_einstein", "held_u1_gauge",
          "held_u1_constraint", "held_u1_adm"] \
    + [f"held_{n}" for n in BAND_NAMES] + [f"held_u1_{n}" for n in BAND_NAMES]
if not (a.resume and os.path.exists(hist)):
    with open(hist, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
mb = ns["_minimize_mod"]
opts = {"maxiter": a.n_change, "gtol": 0.0, "update_method": "ssbroyden2", "initial_scale": True,
        "line_search": "wolfe", "c2": 0.9, "scale_invariant": True, "max_iter": 100, "alphamin": 1e-15}
hu1 = [float(v) for v in parts_u1(D_held)]
w, it, t_start = w0, it0, time.time()
while it < a.iters:
    key, kd = jax.random.split(key)
    P = draw(kd, a.n_per_window)
    D1, pts = u1_derivs(P), flat_pts(P)
    tr_u1 = float(total(*parts_u1(D1)))
    r = mb.minimize(lambda v: loss_fn(v, D1, pts), lambda v: vg_fn(v, D1, pts), w, opts)
    w = r.x
    it += a.n_change
    hp = [float(v) for v in parts_fn(w, D_held, held_pts)]
    row = dict(iteration=it, seconds=round(time.time() - t_start, 1), train_u=float(loss_fn(w, D1, pts)),
               train_u1=tr_u1, held_u=total(*hp), held_u1=total(*hu1),
               held_einstein=hp[0], held_gauge=hp[1], held_constraint=hp[2], held_adm=hp[3],
               held_u1_einstein=hu1[0], held_u1_gauge=hu1[1], held_u1_constraint=hu1[2], held_u1_adm=hu1[3])
    hb = [float(v) for v in bands_fn(w, D_held, held_pts, held_r)]
    row.update({f"held_{n}": v for n, v in zip(BAND_NAMES, hb)})
    row.update({f"held_u1_{n}": v for n, v in zip(BAND_NAMES, bands_u1)})
    with open(hist, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(row)
    print(f"it {it:6d}  {row['seconds']:8.0f} s  train {row['train_u']:.3e} (u1 {tr_u1:.3e})  "
          f"held-out {row['held_u']:.3e} (u1 {row['held_u1']:.3e}, ratio {row['held_u'] / row['held_u1']:.3f}; "
          f"ADM {hp[3] / hu1[3]:.3f}); "
          "by radius u/u1: " + " ".join(f"{n} {v / u:.3f}" if u > 0 else f"{n} -" for n, v, u in zip(BAND_NAMES, hb, bands_u1)), flush=True)
    pickle.dump({"params": jax.tree_util.tree_map(np.asarray, unravel(w)), "eps": eps, "iteration": it,
                 "key": np.asarray(key), "T": T, "R": R, "hidden": a.hidden, "layers": a.layers,
                 "run": os.path.abspath(a.run), "windows": W, "w_constraint": a.w_constraint,
                 "core_taper": a.core_taper, "sampling": a.sampling, "w_adm": a.w_adm,
                 "adm_chi_power": a.adm_chi_power},
                open(ckpt + ".tmp", "wb"))
    os.replace(ckpt + ".tmp", ckpt)
print("done:", hist)
