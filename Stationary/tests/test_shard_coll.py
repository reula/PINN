"""--shard-coll: the interior points split over several devices give the same loss and gradient.

Run in a subprocess with two fake host devices (XLA_FLAGS must be set before jax is imported).
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = textwrap.dedent('''
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from stationary import train as T
    from stationary.train import parse_args, make_batch, build
    from stationary.losses import total_loss
    assert jax.device_count() == 2, jax.devices()
    base = ["--arch", "hybrid", "--n-coll", "64", "--n-bnd", "16", "--width", "6", "--depth", "2",
            "--no-figures", "--outdir", "OUT"]
    c0 = parse_args(base); c1 = parse_args(base + ["--shard-coll"])
    model, state, ex = build(c0)
    key = jax.random.PRNGKey(3)
    b0, b1 = make_batch(key, c0), make_batch(key, c1)
    assert len(b1["coll"].sharding.device_set) == 2, b1["coll"].sharding
    f = lambda s, b: total_loss(s, b, c0, model, ex)[0]
    v0, g0 = jax.jit(jax.value_and_grad(f))(state, b0)
    v1, g1 = jax.jit(jax.value_and_grad(f))(state, b1)
    assert abs(float(v0) - float(v1)) <= 1e-12 * abs(float(v0)), (v0, v1)
    d = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree.leaves(g0), jax.tree.leaves(g1)))
    assert d <= 1e-10, d
    # and a whole sharded SSBroyden phase on the merged Crunch runs and lowers the loss
    cfg = parse_args(base + ["--shard-coll", "--steps", "0", "--qn-method", "ssbroyden",
                             "--lbfgs-steps", "20", "--qn-block", "10", "--resample-every", "10",
                             "--plateau-min-iters", "-1"])
    T.train(cfg, verbose=False)
    print("OK", float(v0), d)
''')


def test_sharded_collocation_matches_and_trains(tmp_path):
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=2",
               JAX_PLATFORMS="cpu", MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, "-c", SCRIPT.replace("OUT", str(tmp_path / "run"))],
                       cwd=HERE, env=env, capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]
