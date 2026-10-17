"""Time the training loss and its pieces on the current device.

    JAX_ENABLE_X64=1 python tools/bench_loss.py [--n 4000] [--reps 5] [--script PATH] [-- <script flags>]

Builds a production-shaped problem (network from the script defaults unless
overridden after --, moving chi background, default BBH centres), then times
value_and_grad of: the full BFGS loss (bfgs_val_and_grad), the evolution
residual alone, the Sommerfeld boundary term, and the IC/positivity terms.
Prints compile time and median wall time per call.
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_defs import load, SCRIPT

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=4000, help="interior points (tapa/IC = n/10, boundary = n/5)")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--script", default=SCRIPT)
ap.add_argument("--only", default="", help="comma list of: full,evol,bc,ic")
a, rest = ap.parse_known_args()
if rest[:1] == ["--"]:
    rest = rest[1:]

import tempfile
ns = load(rest + ["--Nint", str(a.n), "--NI", str(a.n // 10), "--Nb", str(a.n // 5)],
          script=a.script, outdir=tempfile.mkdtemp(prefix="bench_"))
jax, jnp = ns["jax"], ns["jnp"]
model, flatten_util = ns["model"], ns["flatten_util"]

key = jax.random.PRNGKey(0)
ws, we = 0.0, float(ns["tfinal"]) / int(ns["N_window"])
dummy = (jnp.ones((8, 1)),) * 4
params = model.init(key, *dummy, jnp.array(ws), jnp.array(we))
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), params)
centers = ns["find_bh_centers_shift_driven"]() if "find_bh_centers_shift_driven" in ns else None
apply_fn = model.apply
traj_tt = traj_xx = None
if ns["chi_bg_mode"] == "moving":
    apply_fn = ns["_wrap_with_moving_chi_bg"](model.apply)
    n_grid = 17
    traj_tt = jnp.linspace(ws, we, n_grid)
    xy = jnp.array([[c[0], c[1]] for c in centers])            # (n_stars, 2)
    traj_xx = jnp.repeat(xy[:, :, None], n_grid, axis=2)
td = ns["build_train_data"](jnp.array(ws), jnp.array(we), key, star_centers=centers)
train_data = td[0] if isinstance(td[0], tuple) else td
w0, unflatten = flatten_util.ravel_pytree(params)
print(f"device {jax.devices()[0]}  params {w0.size}  interior {train_data[0].shape[0]}  "
      f"tapa {train_data[4].shape[0]}  boundary {train_data[8].shape[0]}  IC {train_data[13].shape[0]}")

args = (ns["lam_bc"], ns["lamg0"], ns["w_pos"], traj_tt, traj_xx,
        ns["lam_einstein"], ns["lam_gauge"], ns["r_eps"], *train_data)
P = (lambda p: p) if traj_tt is None else (lambda p: ns["_bundle_params"](p, traj_tt, traj_xx))
t, x, y, z = train_data[0:4]
tb, xb, yb, zb, gb = train_data[8:13]
WS, WE = train_data[-2], train_data[-1]


def vg(f):
    return jax.jit(jax.value_and_grad(lambda w: f(unflatten(w))))


cases = {
    "full": lambda: (lambda w: ns["bfgs_val_and_grad"](w, apply_fn, unflatten, *args)),
    "evol": lambda: vg(lambda p: jnp.mean(ns["r_ccz4_evolution"](apply_fn, P(p), t, x, y, z, WS, WE) ** 2)),
    "bc":   lambda: vg(lambda p: jnp.mean(ns["metric_sommerfeld_residual"](
        apply_fn, P(p), tb, xb, yb, zb, WS, WE, ns["c"], ns["lam_bc"], ns["r_eps"]) ** 2)),
    "fwd":  lambda: jax.jit(lambda w: jnp.mean(ns["r_ccz4_evolution"](apply_fn, P(unflatten(w)), t, x, y, z, WS, WE) ** 2)),
}
only = [s for s in a.only.split(",") if s] or ["full", "evol", "fwd", "bc"]
res = {}
for name in only:
    f = cases[name]()
    t0 = time.perf_counter(); out = f(w0); jax.block_until_ready(out); tc = time.perf_counter() - t0
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter(); out = f(w0); jax.block_until_ready(out); ts.append(time.perf_counter() - t0)
    val = out[0] if isinstance(out, tuple) else out
    res[name] = (statistics.median(ts), float(val))
    print(f"{name:5s} compile+first {tc:7.2f} s   median {1e3 * statistics.median(ts):9.1f} ms   value {float(val):.15e}")
