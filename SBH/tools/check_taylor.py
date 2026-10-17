"""Check --deriv_mode taylor against autodiff on the FULL CCZ4 residual and its
parameter gradient, in both chi modes, and time both.

    JAX_ENABLE_X64=1 python tools/check_taylor.py [--n 2000] [--reps 5]
"""
import argparse, os, sys, time, statistics, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_defs import load

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--reps", type=int, default=5)
a, rest = ap.parse_known_args()

for chi_mode in ("moving", "physical"):
    ns = load(rest + ["--chi_bg_mode", chi_mode, "--Nint", str(a.n), "--NI", str(a.n // 10),
                      "--Nb", str(a.n // 5)], outdir=tempfile.mkdtemp(prefix="ct_"))
    jax, jnp = ns["jax"], ns["jnp"]
    model, fu = ns["model"], ns["flatten_util"]
    key = jax.random.PRNGKey(0); ws, we = jnp.array(0.0), jnp.array(0.25)
    params = jax.tree_util.tree_map(lambda v: v.astype(jnp.float64),
                                    model.init(key, *(jnp.ones((8, 1)),) * 4, ws, we))
    # perturb so every output (and its derivatives) is O(1), not the small-init ~1e-3
    params = jax.tree_util.tree_map(lambda v: v + 0.3 * jax.random.normal(key, v.shape), params)
    centers = ns["find_bh_centers_shift_driven"]()
    td = ns["build_train_data"](ws, we, key, star_centers=centers)
    td = td[0] if isinstance(td[0], tuple) else td
    t, x, y, z = td[0:4]
    w0, unf = fu.ravel_pytree(params)
    if chi_mode == "moving":
        fn = ns["_wrap_with_moving_chi_bg"](model.apply)
        tt = jnp.linspace(0.0, 0.25, 17)
        xx = jnp.repeat(jnp.array([[c[0], c[1]] for c in centers])[:, :, None], 17, axis=2)
        xx = xx + 0.05 * jnp.sin(tt)[None, None, :]       # a moving track
        P = lambda w: ns["_bundle_params"](unf(w), tt, xx)
    else:
        fn, P = model.apply, unf
    r_jit = ns["r_ccz4_evolution"].__wrapped__
    out = {}
    for mode in ("autodiff", "taylor"):
        ns["deriv_mode"] = mode
        res = jax.jit(lambda w: r_jit(fn, P(w), t, x, y, z, ws, we))
        vg = jax.jit(jax.value_and_grad(lambda w: jnp.mean(r_jit(fn, P(w), t, x, y, z, ws, we) ** 2)))
        R = jax.block_until_ready(res(w0)); V = jax.block_until_ready(vg(w0))
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter(); jax.block_until_ready(vg(w0)); ts.append(time.perf_counter() - t0)
        out[mode] = (R, V, statistics.median(ts))
    (Ra, Va, ta), (Rt, Vt, tt_) = out["autodiff"], out["taylor"]
    rel = lambda u, v: float(jnp.max(jnp.abs(u - v)) / jnp.max(jnp.abs(u)))
    print(f"[{chi_mode:8s}] residual rel.diff {rel(Ra, Rt):.1e}   loss rel.diff {abs(float(Va[0] - Vt[0]) / float(Va[0])):.1e}"
          f"   grad rel.diff {rel(Va[1], Vt[1]):.1e}   val+grad autodiff {1e3 * ta:.1f} ms, taylor {1e3 * tt_:.1f} ms"
          f"  ({ta / tt_:.2f}x)")
