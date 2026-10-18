"""Tests for the DSGNAR port (run on CPU in a few minutes):

    JAX_ENABLE_X64=1 python tools/test_dsgnar_bh.py [--quick]

1. bfgs_residual_vector squares to compute_bfgs_loss (sum r^2 == loss) in every
   configuration that changes the loss's form: default (moving chi, Sommerfeld,
   causal weights), physical chi, Dirichlet BC, causal weights off, the log IC
   term, active positivity rows, --deriv_mode taylor;
2. its blocks cover the rows and are labelled in order;
3. a short DSGNAR phase on a small problem lowers the loss (and the history's
   loss is the true loss at the returned parameters).
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_defs import load

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="only the default configuration of test 1")
a = ap.parse_args()

SMALL = ["--Nint", "600", "--NI", "120", "--Nb", "240", "--HIDDEN", "16", "--N_LAYERS", "3"]
CONFIGS = {
    "default":     [],
    "physical":    ["--chi_bg_mode", "physical"],
    "dirichlet":   ["--metric_bc", "dirichlet"],
    "no_causal":   ["--no-causal_weight"],
    "ic_log":      ["--lam_ic_log", "0.3"],
    "taylor":      ["--deriv_mode", "taylor"],
}
if a.quick:
    CONFIGS = {"default": []}


def setup(extra, perturb=0.0):
    ns = load(SMALL + extra, outdir=tempfile.mkdtemp(prefix="tdb_"))
    jax, jnp = ns["jax"], ns["jnp"]
    model = ns["model"]
    key = jax.random.PRNGKey(3)
    ws, we = jnp.array(0.0), jnp.array(0.25)
    params = jax.tree_util.tree_map(lambda v: v.astype(jnp.float64),
                                    model.init(key, *(jnp.ones((8, 1)),) * 4, ws, we))
    if perturb:
        params = jax.tree_util.tree_map(lambda v: v + perturb * jax.random.normal(key, v.shape), params)
    centers = ns["find_bh_centers_shift_driven"]()
    td = ns["build_train_data"](ws, we, key, star_centers=centers)
    td = td[0] if isinstance(td[0], tuple) else td
    fn, tt, xx = ns["apply_fn"], None, None
    if ns["chi_bg_mode"] == "moving":
        fn = ns["_wrap_with_moving_chi_bg"](ns["apply_fn"])
        tt = jnp.linspace(0.0, 0.25, 9)
        xx = jnp.repeat(jnp.array([[c[0], c[1]] for c in centers])[:, :, None], 9, axis=2)
    args = (ns["lam_bc"], ns["lamg0"], ns["w_pos"], tt, xx,
            ns["lam_einstein"], ns["lam_gauge"], ns["r_eps"], *td)
    return ns, params, fn, args


fails = 0
print("1. sum(r^2) == compute_bfgs_loss")
for name, extra in CONFIGS.items():
    for perturb in (0.0, 0.8):            # 0.8: chi/alpha go negative somewhere -> positivity rows active
        ns, params, fn, args = setup(extra, perturb)
        jnp = ns["jnp"]
        loss = float(ns["compute_bfgs_loss"](fn, params, *args))
        blocks = []
        r = ns["bfgs_residual_vector"](fn, params, *args, _blocks=blocks)
        ss = float(jnp.sum(r ** 2))
        pos_rows = dict(blocks)["pos"]
        r_np = __import__("numpy").asarray(r)
        n_pos_active = int((r_np[-pos_rows:] != 0).sum())
        rel = abs(ss - loss) / abs(loss)
        ok = rel < 1e-12 and sum(b[1] for b in blocks) == r.shape[0]
        fails += not ok
        print(f"   {'ok ' if ok else 'FAIL'} {name:10s} perturb {perturb:.1f}  loss {loss:.15e}  "
              f"sum r^2 {ss:.15e}  rel {rel:.1e}  rows {r.shape[0]}  active pos rows {n_pos_active}")
print("   blocks:", blocks)

if not a.quick:
    print("3. a short DSGNAR phase lowers the loss")
    ns, params, fn, args = setup(["--deriv_mode", "taylor", "--dsgnar_steps", "6", "--dsgnar_sketch", "120"])
    jax, jnp = ns["jax"], ns["jnp"]
    flat0, unflatten = jax.flatten_util.ravel_pytree(params)
    obj = ns["_DsgnarObjective"](fn, unflatten, args, flat0)
    import dsgnar
    flat, hist, info = dsgnar.dsgnar_phase(obj, flat0, ns["_dsgnar_cfg"](seed=0, resample_every=0),
                                           verbose=False)
    l0, l1 = hist[0]["loss"], hist[-1]["loss"]
    true1 = float(ns["compute_bfgs_loss"](fn, unflatten(flat), *args))
    ok = l1 < l0 and abs(true1 - l1) / l1 < 1e-10
    fails += not ok
    print(f"   {'ok ' if ok else 'FAIL'} loss {l0:.6e} -> {l1:.6e} in {info['iterations']} iterations, "
          f"{info['accepted']} accepted; true loss at the result {true1:.6e}")

print("ALL PASSED" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
