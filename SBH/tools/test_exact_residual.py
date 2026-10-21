"""--dsgnar_residual_sketch none: (1) the Gram system equals the dense (J Omega S)^T (J Omega S),
(J Omega S)^T r; (2) from a point where the residual-sketched DSGNAR stalls (after SSBroyden),
compare 30 DSGNAR iterations with and without the residual sketch (paper step rule, no line scan).

    JAX_ENABLE_X64=1 python tools/test_exact_residual.py [--qn_its 1500]
"""
import argparse, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_defs import load
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--qn_its", type=int, default=1500); ap.add_argument("--its", type=int, default=30)
ap.add_argument("--production_net", action="store_true", help="6 x 40 network (n = 9302, s = 3100); skips the dense check")
ap.add_argument("--nint", type=int, default=1500)
a, _ = ap.parse_known_args()
NET = [] if a.production_net else ["--HIDDEN", "16", "--N_LAYERS", "3"]
ns = load(["--Nint", str(a.nint), "--NI", str(a.nint // 5), "--Nb", str(a.nint * 2 // 5)] + NET + ["--deriv_mode", "taylor",
           "--chi_bg_mode", "physical", "--bh_center_mode", "fixed", "--n_stars", "1", "--BL_masses", "0.5",
           "--BL_momenta", "0", "--BL_momenta_y", "0", "--dsgnar_jacobian", "rows"], outdir=tempfile.mkdtemp(prefix="te_"))
jax, jnp = ns["jax"], ns["jnp"]
import dsgnar
model = ns["model"]; key = jax.random.PRNGKey(4); ws, we = jnp.array(0.0), jnp.array(0.25)
params = jax.tree_util.tree_map(lambda v: v.astype(jnp.float64), model.init(key, *(jnp.ones((8, 1)),) * 4, ws, we))
td = ns["build_train_data"](ws, we, key, star_centers=[(0.0, 0.0, 0.0)]); td = td[0] if isinstance(td[0], tuple) else td
fn = ns["apply_fn"]
args = (ns["lam_bc"], ns["lamg0"], ns["w_pos"], None, None, ns["lam_einstein"], ns["lam_gauge"], ns["r_eps"], *td)
w0, unf = ns["flatten_util"].ravel_pytree(params)
obj = ns["_DsgnarObjective"](fn, unf, args, w0, row_batch=50)
n, M = int(w0.size), int(obj.residual(w0).shape[0]); s = n // 3

# (1) Gram vs dense
if a.production_net:
    print("1. dense check skipped (production network)")
srct = dsgnar.make_srct(jax.random.PRNGKey(9), n, s)
G, b = obj.gram_rows(w0, srct, s, jnp.ones((M,)))
J = None if a.production_net else jax.jacfwd(obj._residual)(w0)                          # (M, n) dense (small problem)
if J is not None:
    A = J @ dsgnar.srct_columns(srct, n, s, w0.dtype).T
    G0, b0 = A.T @ A, A.T @ obj.residual(w0)
    eg = float(jnp.max(jnp.abs(G - G0)) / jnp.max(jnp.abs(G0))); eb = float(jnp.max(jnp.abs(b - b0)) / jnp.max(jnp.abs(b0)))
    print(f"1. Gram vs dense: G rel {eg:.1e}, b rel {eb:.1e}  -> {'ok' if max(eg, eb) < 1e-11 else 'FAIL'}")

# (2) stall point
loss = jax.jit(lambda w: ns["compute_bfgs_loss"](fn, unf(w), *args))
vg = jax.jit(jax.value_and_grad(lambda w: ns["compute_bfgs_loss"](fn, unf(w), *args)))
r = ns["_minimize_mod"].minimize(loss, vg, w0, {"maxiter": max(a.qn_its, 1), "gtol": 0.0, "update_method": "ssbroyden2",
     "initial_scale": True, "line_search": "wolfe", "c2": 0.9, "scale_invariant": True, "max_iter": 100, "alphamin": 1e-15})
w1 = r.x if a.qn_its > 0 else w0
print(f"2. start: loss {float(loss(w1)):.4e} after {a.qn_its} SSBroyden its")
for mode in ("count", "none"):
    cfg = ns["_dsgnar_cfg"](seed=0, resample_every=0, dsgnar_sketch=s, dsgnar_steps=a.its, dsgnar_line_scan=False,
                            dsgnar_calibrate=False, dsgnar_grow_on_accept=False, dsgnar_radius_floor=0.0,
                            dsgnar_stage1_ratio=0.075, dsgnar_delta_min=1e-8, dsgnar_reweight_alpha=0.0,
                            dsgnar_residual_sketch=mode)
    flat, hist, info = dsgnar.dsgnar_phase(obj, w1, cfg, verbose=False)
    print(f"   residual sketch {mode:5s}: loss {hist[0]['loss']:.4e} -> {hist[-1]['loss']:.4e} in {info['iterations']} its, "
          f"{info['accepted']} accepted / {info['rejected']} rejected, stopped: {info['stopped']}, final radius {info['final_radius']:.1e}")
w2 = ns["_minimize_mod"].minimize(loss, vg, w1, {"maxiter": a.its * 10, "gtol": 0.0, "update_method": "ssbroyden2",
     "initial_scale": True, "line_search": "wolfe", "c2": 0.9, "scale_invariant": True, "max_iter": 100, "alphamin": 1e-15}).x
print(f"   for scale: {a.its * 10} more SSBroyden its -> {float(loss(w2)):.4e}")
