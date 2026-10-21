"""Is the loss DSGNAR evaluates the one it linearises? (2026-10-21)

The causal weights are recomputed from the residuals at every loss evaluation but are stop-gradiented
in the Jacobian. Along the descent direction d = -g (g = J^T r, the frozen-weight gradient), compare the
measured change of the loss, [L(w + h d) - L(w)] / h, with the predicted -|g|^2 (scaled), for decreasing h,
with --causal_weight on and off, at the high-loss start of a window (random init, core points).
    JAX_ENABLE_X64=1 python tools/test_causal_consistency.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_defs import load

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--production", action="store_true")
ap.add_argument("--floor_modes", action="store_true", help="compare --chi_floor_mode max vs smooth (causal on)")
ap.add_argument("--min_dist", action="store_true", help="compare --min_puncture_dist 0.02 vs 0.05 (max floor)")
a, _ = ap.parse_known_args()
PTS = ([] if a.production else ["--Nint", "2000", "--NI", "300", "--Nb", "600"]) + \
      ["--core_points", "48000" if a.production else "1500", "--core_ic_points", "4800" if a.production else "150"]
if a.min_dist:
    VARIANTS = [(True, "max", "0.02"), (True, "max", "0.05")]
else:
    VARIANTS = [(c, f, "0.02") for c, f in ([(True, "max"), (True, "smooth")] if a.floor_modes else [(True, "max"), (False, "max")])]
for causal, fmode, mind in VARIANTS:
    ns = load(PTS + ["--chi_floor_mode", fmode, "--min_puncture_dist", mind, 
               "--core_radius", "0.5", "--n_stars", "1", "--BL_masses", "0.5", "--BL_momenta", "0", "--BL_momenta_y", "0",
               "--deriv_mode", "taylor"] + ([] if causal else ["--no-causal_weight"]), outdir=tempfile.mkdtemp())
    jax, jnp = ns["jax"], ns["jnp"]; model = ns["model"]
    key = jax.random.PRNGKey(0); ws, we = jnp.array(0.0), jnp.array(0.25)
    params = jax.tree_util.tree_map(lambda v: v.astype(jnp.float64), model.init(key, *(jnp.ones((8, 1)),) * 4, ws, we))
    centers = ns["find_bh_centers_shift_driven"]()
    td = ns["build_train_data"](ws, we, key, star_centers=centers); td = td[0] if isinstance(td[0], tuple) else td
    fn = ns["_wrap_with_moving_chi_bg"](ns["apply_fn"])
    tt = jnp.linspace(0, .25, 9); xx = jnp.repeat(jnp.array([[c[0], c[1]] for c in centers])[:, :, None], 9, axis=2)
    args = (ns["lam_bc"], ns["lamg0"], ns["w_pos"], tt, xx, ns["lam_einstein"], ns["lam_gauge"], ns["r_eps"], *td)
    w0, unf = ns["flatten_util"].ravel_pytree(params)
    L = jax.jit(lambda w: ns["compute_bfgs_loss"](fn, unf(w), *args))
    g = jax.jit(jax.grad(lambda w: ns["compute_bfgs_loss"](fn, unf(w), *args)))(w0)   # frozen-weight gradient
    d = -g / jnp.linalg.norm(g)
    L0 = float(L(w0)); pred = float(g @ d)
    parts, _ = ns["apply_model"](fn, ns["_bundle_params"](params, tt, xx) if False else params,
                                 ns["lam_bc"], ns["lamg0"], ns["w_pos"], tt, xx, ns["lam_einstein"], ns["lam_gauge"],
                                 ns["r_eps"], *td)
    out0 = ns["full_output"](fn, ns["_bundle_params"](params, tt, xx), *td[0:4], ws, we)
    chi0 = out0[:, ns["CHI_IDX"]]
    print(f"   points with network chi < 1e-4: {int(jnp.sum(chi0 < 1e-4))}, < 2e-5: {int(jnp.sum(chi0 < 2e-5))}, "
          f"< 1e-5 (below floor): {int(jnp.sum(chi0 < 1e-5))}  (of {chi0.shape[0]})")
    print(f"causal_weight={causal} floor={fmode} min_puncture_dist={mind}: loss {L0:.4e}, predicted slope g.d = {pred:+.4e}; terms: "
          + ", ".join(f"{k} {float(v):.2e}" for k, v in parts.items() if k in ("loss_Einstein", "loss_gauge", "loss_BC", "loss_IC", "loss_pos")))
    for h in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        slope = (float(L(w0 + h * d)) - L0) / h
        print(f"   h = {h:.0e}: measured slope {slope:+.4e}   ratio measured/predicted {slope / pred:+.3f}")
