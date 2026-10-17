"""Prototype: value, d_t, d_xyz and the 6 distinct second spatial derivatives
of the tanh MLP in ONE batched forward pass, checked against the script's
autodiff stack (jvp / jacfwd / jacfwd-of-jacfwd), then timed.

    JAX_ENABLE_X64=1 python tools/taylor_proto.py [--n 2000] [--reps 5] [--time]
"""
import argparse, os, sys, time, statistics, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_defs import load

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--time", action="store_true")
a, rest = ap.parse_known_args()
ns = load(rest, outdir=tempfile.mkdtemp(prefix="tp_"))
jax, jnp = ns["jax"], ns["jnp"]

PAIRS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
SYM = jnp.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])       # (k,l) -> pair index
DD_IDX = ns["_DD_OUTPUT_IDXS"]


def taylor_mlp(params, t, x, y, z, ws, we):
    """(N,1) inputs -> val (N,22), dt (N,22), d (N,22,3), dd (N,11,3,3)."""
    p = params["params"]["MLP"]
    layers = [p[f"Dense_{i}"] for i in range(len(p))]
    ww = jnp.maximum(we - ws, jnp.finfo(jnp.float64).eps)
    Z = jnp.concatenate([2.0 * (t - ws) / ww - 1.0,
                         (x - ns["x_center"]) / ns["x_half"],
                         (y - ns["y_center"]) / ns["y_half"],
                         (z - ns["z_center"]) / ns["z_half"]], axis=-1)
    scale = jnp.stack([2.0 / ww, 1.0 / ns["x_half"], 1.0 / ns["y_half"], 1.0 / ns["z_half"]])
    W0, b0 = layers[0]["kernel"], layers[0]["bias"]
    a = Z @ W0 + b0                                   # (N,H)
    a1 = scale[:, None] * W0                          # (4,H): same for every point
    h = jnp.tanh(a); s1 = 1.0 - h * h; s2 = -2.0 * h * s1
    h1 = s1[:, None, :] * a1[None]                    # (N,4,H)
    h2 = jnp.stack([s2 * a1[1 + k] * a1[1 + l] for k, l in PAIRS], axis=1)   # (N,6,H)
    for L in layers[1:-1]:
        W, b = L["kernel"], L["bias"]
        a, a1, a2 = h @ W + b, h1 @ W, h2 @ W
        h = jnp.tanh(a); s1 = 1.0 - h * h; s2 = -2.0 * h * s1
        h1n = s1[:, None, :] * a1
        sp = a1[:, 1:, :]                              # spatial first derivatives
        h2 = s1[:, None, :] * a2 + s2[:, None, :] * jnp.stack(
            [sp[:, k] * sp[:, l] for k, l in PAIRS], axis=1)
        h1 = h1n
    W, b = layers[-1]["kernel"], layers[-1]["bias"]
    val = h @ W + b
    val = val + ns["_FLAT_CCZ4_VEC"] * jnp.ones_like(val[:, :1]) \
        - (0.0 if ns["chi_bg_mode"] == "physical" else
           jnp.zeros(22).at[ns["CHI_IDX"]].set(1.0))
    d1 = h1 @ W                                         # (N,4,22)
    d2 = h2 @ W[:, DD_IDX]                              # (N,6,11)
    dt = d1[:, 0, :]
    d = jnp.moveaxis(d1[:, 1:, :], 1, 2)               # (N,22,3)
    dd = jnp.moveaxis(d2[:, SYM, :], 3, 1)             # (N,11,3,3)
    return val, dt, d, dd


def autodiff_derivs(apply_fn, params, t, x, y, z, ws, we):
    """The script's residual_one derivative stack, vmapped."""
    def one(q):
        f = lambda ql: ns["point_output_from_q"](apply_fn, params, ql, ws, we)
        qx = lambda p_: jnp.concatenate([q[0:1], p_])
        val = f(q)
        dt = jax.jvp(f, (q,), (jnp.array([1., 0., 0., 0.]),))[1]
        d = jax.jacfwd(lambda p_: f(qx(p_)))(q[1:4])
        dd = jax.jacfwd(jax.jacfwd(lambda p_: f(qx(p_))[DD_IDX]))(q[1:4])
        return val, dt, d, dd
    return jax.vmap(one)(jnp.concatenate([t, x, y, z], axis=1))


key = jax.random.PRNGKey(1)
model = ns["model"]
ws, we = jnp.array(0.0), jnp.array(0.25)
params = model.init(key, *(jnp.ones((8, 1)),) * 4, ws, we)
params = jax.tree_util.tree_map(lambda v: v.astype(jnp.float64), params)
# perturb the small-init last layer so second derivatives are not tiny
params = jax.tree_util.tree_map(lambda v: v + 0.3 * jax.random.normal(key, v.shape), params)
k1, k2 = jax.random.split(key)
X = jax.random.uniform(k1, (a.n, 4), minval=-1, maxval=1) * jnp.array([0.25, 6, 6, 6])
X = X.at[:, 0].set(jnp.abs(X[:, 0]))
t, x, y, z = (X[:, i:i + 1] for i in range(4))

ref = jax.jit(lambda p: autodiff_derivs(model.apply, p, t, x, y, z, ws, we))(params)
new = jax.jit(lambda p: taylor_mlp(p, t, x, y, z, ws, we))(params)
for name, r, n_ in zip(["val", "dt", "d", "dd"], ref, new):
    err = float(jnp.max(jnp.abs(r - n_))); sc = float(jnp.max(jnp.abs(r)))
    print(f"{name:3s} shape {tuple(r.shape)}  max|diff| {err:.2e}  (max|ref| {sc:.2e})")

if a.time:
    def loss_of(fn):
        return jax.jit(jax.value_and_grad(lambda p: sum(jnp.sum(o ** 2) for o in fn(p))))
    for name, fn in [("autodiff", lambda p: autodiff_derivs(model.apply, p, t, x, y, z, ws, we)),
                     ("taylor", lambda p: taylor_mlp(p, t, x, y, z, ws, we))]:
        for mode, f in [("fwd", jax.jit(fn)), ("val+grad", loss_of(fn))]:
            jax.block_until_ready(f(params)); ts = []
            for _ in range(a.reps):
                t0 = time.perf_counter(); jax.block_until_ready(f(params)); ts.append(time.perf_counter() - t0)
            print(f"{name:9s} {mode:9s} median {1e3 * statistics.median(ts):8.1f} ms")
