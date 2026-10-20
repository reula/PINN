"""Rows (the paper's DSGNAR code) vs columns (our port) for the sketched Jacobian.

    JAX_ENABLE_X64=1 python tools/bench_jacobian_sketch.py [--pts 64] [--s 1024] [--reps 3]

Both produce what DSGNAR needs for a block of interior points: the paper's code builds the DENSE
Jacobian of the residual rows (jax.jacrev, 22 rows per point) and then sketches it; our port
(Stationary/dsgnar.py, sketched_jvp_columns) pushes s forward tangents (JVPs) through the residual
and sketches as it goes. Times both on the same points with the production network, and checks
that the two give the same J @ Omega (with Omega an explicit random (n, s) matrix).
"""
import argparse, os, sys, tempfile, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_defs import load

ap = argparse.ArgumentParser()
ap.add_argument("--pts", type=int, default=64)
ap.add_argument("--s", type=int, default=1024)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--chunk", type=int, default=16)
a, rest = ap.parse_known_args()
ns = load(rest + ["--chi_bg_mode", "physical", "--bh_center_mode", "fixed", "--deriv_mode", "taylor"],
          outdir=tempfile.mkdtemp(prefix="bjs_"))
jax, jnp = ns["jax"], ns["jnp"]
model = ns["model"]
key = jax.random.PRNGKey(0)
ws, we = jnp.array(0.0), jnp.array(0.25)
params = jax.tree_util.tree_map(lambda v: v.astype(jnp.float64), model.init(key, *(jnp.ones((8, 1)),) * 4, ws, we))
w0, unf = ns["flatten_util"].ravel_pytree(params)
n = w0.size
X = jax.random.uniform(key, (a.pts, 4), minval=-1, maxval=1) * jnp.array([0.25, 6, 6, 6])
t, x, y, z = (X[:, i:i + 1] for i in range(4))
fn = ns["apply_fn"]
res = lambda w: ns["r_ccz4_evolution"](fn, unf(w), t, x, y, z, ws, we).reshape(-1)   # (22*pts,)
Om = jax.random.normal(key, (n, a.s)) / jnp.sqrt(a.s)

def res_pt(w, q):            # one point's 22 residual rows (the paper vmaps jacrev over sub-batches)
    q = q.reshape(1, 4)
    return ns["r_ccz4_evolution"](fn, unf(w), q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4], ws, we).reshape(-1)
Q = jnp.concatenate([t, x, y, z], axis=1)
rows = jax.jit(lambda w: jax.vmap(jax.jacrev(res_pt), in_axes=(None, 0))(w, Q).reshape(-1, n) @ Om)
rows_whole = jax.jit(lambda w: jax.jacrev(res)(w) @ Om)              # jacrev of the whole block (naive)
def cols_fn(w):
    _, jvp = jax.linearize(res, w)
    B = Om.T.reshape(a.s // a.chunk, a.chunk, n)
    return jax.lax.map(lambda b: jax.vmap(jvp)(b), B).reshape(a.s, -1).T
cols = jax.jit(cols_fn)

def tm(f):
    out = jax.block_until_ready(f(w0)); ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter(); jax.block_until_ready(f(w0)); ts.append(time.perf_counter() - t0)
    return out, statistics.median(ts)
Jr, tr = tm(rows); Jc, tc = tm(cols); _, tw = tm(rows_whole)
print(f"device {jax.devices()[0]}  n_params {n}  points {a.pts}  rows {22 * a.pts}  s {a.s}")
print(f"rows, jacrev of the whole block (naive): {1e3 * tw:8.1f} ms")
print(f"rows (paper: jacrev per point, vmapped):  {1e3 * tr:8.1f} ms")
print(f"cols (ours, {a.s} JVPs):                {1e3 * tc:8.1f} ms   ratio cols/rows = {tc / tr:.1f}")
print(f"agreement: max|diff|/max|J Om| = {float(jnp.max(jnp.abs(Jr - Jc)) / jnp.max(jnp.abs(Jr))):.1e}")
