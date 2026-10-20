"""Which operations of the BH loss fail on the Intel oneAPI JAX backend? (Clementina, 2026-10-20)

    JAX_ENABLE_X64=1 python tools/oneapi_probe.py

The first bench there died in the jitted loss+gradient with oneDNN's "could not create a
primitive descriptor for the matmul primitive". This compiles each kind of product the code
uses separately, in float64 and float32, and prints PASS/FAIL per case, then tries the real
residual and its gradient in both --deriv_mode settings.
"""
import os
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
print("jax", jax.__version__, "| backend", jax.default_backend(), "|", jax.devices())
key = jax.random.PRNGKey(0)


def run(name, f, *a):
    try:
        out = jax.jit(f)(*a)
        jax.block_until_ready(out)
        print(f"PASS  {name}", flush=True)
        return True
    except Exception as e:
        msg = str(e).splitlines()[0][:140]
        print(f"FAIL  {name}: {type(e).__name__}: {msg}", flush=True)
        return False


for dt in (jnp.float64, jnp.float32):
    tag = "f64" if dt == jnp.float64 else "f32"
    r = lambda *s: jax.random.normal(key, s, dtype=dt)
    run(f"{tag} matmul (8000,40)@(40,40)", lambda A, B: A @ B, r(8000, 40), r(40, 40))
    run(f"{tag} matmul (8000,4)@(4,40)", lambda A, B: A @ B, r(8000, 4), r(4, 40))
    run(f"{tag} matmul (8000,40)@(40,22)", lambda A, B: A @ B, r(8000, 40), r(40, 22))
    run(f"{tag} batched (8000,4,40)@(40,40)", lambda A, B: A @ B, r(8000, 4, 40), r(40, 40))
    run(f"{tag} matvec (9302,9302)@(9302,)", lambda A, v: A @ v, r(9302, 9302), r(9302))
    run(f"{tag} outer (9302,)x(9302,)", lambda u, v: jnp.outer(u, v), r(9302), r(9302))
    run(f"{tag} bmm (8000,3,3)@(8000,3,3)", lambda A, B: jnp.einsum("nij,njk->nik", A, B), r(8000, 3, 3), r(8000, 3, 3))
    run(f"{tag} vmap einsum ij,ij->", jax.vmap(lambda A, B: jnp.einsum("ij,ij->", A, B)), r(8000, 3, 3), r(8000, 3, 3))
    run(f"{tag} vmap einsum ijk,k->ij", jax.vmap(lambda A, v: jnp.einsum("ijk,k->ij", A, v)), r(8000, 3, 3, 3), r(8000, 3))
    run(f"{tag} vmap einsum mn,ijmn->ij", jax.vmap(lambda A, B: jnp.einsum("mn,ijmn->ij", A, B)), r(8000, 3, 3), r(8000, 3, 3, 3, 3))
    run(f"{tag} vmap inv 3x3", jax.vmap(jnp.linalg.inv), r(8000, 3, 3) + 3 * jnp.eye(3, dtype=dt))
    run(f"{tag} vmap det 3x3", jax.vmap(jnp.linalg.det), r(8000, 3, 3))
    run(f"{tag} grad of sum((X@W)^2) wrt W", jax.grad(lambda W, X: jnp.sum((X @ W) ** 2)), r(40, 40), r(8000, 40))

# the real residual
from load_defs import load
for mode in ("autodiff", "taylor"):
    try:
        ns = load(["--Nint", "2000", "--NI", "200", "--Nb", "400", "--deriv_mode", mode],
                  outdir=tempfile.mkdtemp(prefix="oap_"))
        model = ns["model"]
        ws, we = jnp.array(0.0), jnp.array(0.25)
        params = jax.tree_util.tree_map(lambda v: v.astype(jnp.float64), model.init(key, *(jnp.ones((8, 1)),) * 4, ws, we))
        X = jax.random.uniform(key, (2000, 4), minval=-1, maxval=1) * jnp.array([0.25, 6, 6, 6])
        t, x, y, z = (X[:, i:i + 1] for i in range(4))
        fn = ns["apply_fn"]
        w0, unf = ns["flatten_util"].ravel_pytree(params)
        rr = lambda w: ns["r_ccz4_evolution"](fn, unf(w), t, x, y, z, ws, we)
        run(f"{mode}: network forward", lambda w: ns["full_output"](fn, unf(w), t, x, y, z, ws, we), w0)
        run(f"{mode}: residual forward", rr, w0)
        run(f"{mode}: residual value+grad", jax.value_and_grad(lambda w: jnp.mean(rr(w) ** 2)), w0)
    except Exception:
        print(f"FAIL  {mode}: setup\n" + traceback.format_exc()[-800:], flush=True)
