"""Size and time one training configuration on the device you are actually on.

The question this answers is the one a run cannot: will it FIT, and is it worth running
HERE?  A run that dies of device memory tells you nothing -- on an A30 handed out as a
12 GiB HAMI slice, `runs/production_quad_quarter_pinrad` was refused a 4.00 GiB allocation
at quasi-Newton iteration 0, after two hours of Adam setup, because JAX's default
preallocation had already taken 75% of the slice.  The numbers below say whether that was
avoidable before you spend the time.

    python -m stationary.bench --arch axisym_hybrid --steps 0 --n-coll 16384 \
        --R0 0.005773502691896258 --rho-in 0.01 --inner-radius 0.01 --rho-out 1.0 \
        --ref-solution --ref-asymptotic 1.0 --outer-bc robin --robin-exps 2,3,1 \
        --robin-orders h=3,lam=3 --no-robin-G --lam-inf 1.0 --decay-feature --radial log \
        --n-bnd 1024 --pin-far --w-pin 100 --w-lam-eq-radial 1 --lam-bc-S2 -0.08333333333333333

Accepts every flag `stationary.train` accepts (it goes through the same parser), plus:

    --reps N        timed repetitions (default 5; the first call is not counted)
    --no-time       sizes only, no execution -- use this when memory is the worry

and run it on each platform you are choosing between:

    JAX_PLATFORMS=cpu  python -m stationary.bench <flags>
    python -m stationary.bench <flags>                           # the GPU, on the hub

Read it like this.  It measures **float64** -- what the production runs use -- and says so in
its header; `--float32` asks for float32 explicitly, and only the float32 training runs want
it.  `temp` is what the compiled gradient needs in ONE piece, on top of the
parameters -- if it does not fit in what is free, the run dies wherever that graph is first
built, which for a quasi-Newton run is iteration 0 after the whole Adam phase.  The
per-value+gradient time times the iteration count is the floor on the wall time; a
quasi-Newton iteration also runs a line search, which costs 1-3 further evaluations.
"""
from __future__ import annotations

import argparse
import os
import time

import jax

# float64 by default, as in every post-processing module (evaluate, report, profile, vtk,
# plane, compare, invariants).  bench measures a TRAINING configuration, and the training runs
# are float64 too -- train.py leaves the flag alone and the runs are launched with
# JAX_ENABLE_X64=1.  A float32 measurement of a float64 run is not merely imprecise, it is
# plausible: an A30 measured that way reported 0.89 GiB and 0.026 s for a configuration that
# really needs 1.60 GiB and 0.048 s, and nothing in the output says the run would be float32.
# Doing this here rather than relying on the environment also means a shell that drops or
# overrides JAX_ENABLE_X64 cannot quietly change what is being measured.  Ask for float32
# explicitly with --float32.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from .losses import default_weights, total_loss
from .train import build, make_batch, parse_args


def _gib(n) -> str:
    return f"{n / 2**30:.2f} GiB"


def _device_memory():
    """(limit, in use) in bytes, or (None, None) where the backend does not report it."""
    try:
        st = jax.devices()[0].memory_stats()
    except Exception:
        return None, None
    if not st:
        return None, None
    return st.get("bytes_limit"), st.get("bytes_in_use")


def cfg_outdir(argv) -> str:
    """Honour a --outdir if the caller gave one, else a scratch dir (nothing is written)."""
    for i, a in enumerate(argv):
        if a == "--outdir" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--outdir="):
            return a.split("=", 1)[1]
    return os.path.join("runs", "_bench")


def main(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--no-time", action="store_true",
                   help="size the compiled gradient only; do not execute it")
    p.add_argument("--float32", action="store_true",
                   help="measure float32 instead of the default float64.  Only the float32 "
                        "training runs want this; the production runs are float64, and "
                        "measuring them in float32 understates the memory and the time")
    p.add_argument("-h", "--help", action="help")
    mine, rest = p.parse_known_args(argv)

    if mine.float32:
        # Before anything is built: build(), make_batch() and the compile below all read the
        # flag when they trace, not when this module was imported.
        jax.config.update("jax_enable_x64", False)

    cfg = parse_args(rest + ["--outdir", cfg_outdir(rest)])

    print("=" * 72)
    print("stationary.bench -- does this configuration fit HERE, and what does it cost?")
    print("=" * 72)
    print(f"  device      {jax.devices()}")
    for name, value in (("bytes_limit", _device_memory()[0]),
                        ("bytes_in_use", _device_memory()[1])):
        if value is not None:
            print(f"  {name:11s} {_gib(value)}   (reported by the backend, before we allocate)")
    x64 = bool(jax.config.read("jax_enable_x64"))
    print(f"  x64         {x64}   <- "
          + ("float64" if x64 else
             "float32, ASKED FOR with --float32: not what the production runs use"))
    print(f"  preallocate {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', '(default: 75% of the device!)')}")

    model, state, exact_fields = build(cfg)
    n_par = sum(int(v.size) for v in jax.tree.leaves(state))
    batch = make_batch(jax.random.PRNGKey(cfg.seed + 1), cfg)
    weights = default_weights(cfg)
    lam_inf = None if cfg.lam_inf is None else jnp.asarray(cfg.lam_inf)

    def loss_fn(st, b, sc, w):
        return total_loss(st, b, cfg, model, exact_fields, pde_scale=sc,
                          weights=w, lam_inf=lam_inf)

    print(f"  model       {cfg.arch}   {n_par} parameters")
    print(f"  sampling    n_coll {cfg.n_coll}   n_bnd {cfg.n_bnd}")
    print(f"  plan        {cfg.steps} Adam + {cfg.lbfgs_steps} quasi-Newton")

    t0 = time.time()
    compiled = jax.jit(jax.value_and_grad(loss_fn, has_aux=True)).lower(
        state, batch, 1.0, weights).compile()
    compile_s = time.time() - t0
    ma = compiled.memory_analysis()

    # The parameters are passed as arguments, so XLA reports them separately from the
    # temporaries.  Only the temporaries are what a device has to find room for on top of
    # everything else already resident.
    arg = getattr(ma, "argument_size_in_bytes", 0) if ma else 0
    out = getattr(ma, "output_size_in_bytes", 0) if ma else 0
    temp = getattr(ma, "temp_size_in_bytes", 0) if ma else 0
    alias = getattr(ma, "alias_size_in_bytes", 0) if ma else 0
    print()
    print(f"  compile     {compile_s:.1f} s")
    if ma is None:
        print("  memory      not reported by this backend")
    else:
        print(f"  arguments   {_gib(arg)}   (the parameters and the batch)")
        print(f"  outputs     {_gib(out)}")
        print(f"  TEMPORARIES {_gib(temp)}   <- what must fit in the device at once")
        print(f"  alias       {_gib(alias)}")
        print(f"  peak demand {_gib(arg + temp)}")
        limit = _device_memory()[0]
        if limit:
            print(f"  of the      {_gib(limit)} the backend reports")
            if arg + temp > 0.5 * limit:
                print("  WARNING: the peak is over half the device.  With JAX's default")
                print("           75% preallocation this will not fit: export")
                print("           XLA_PYTHON_CLIENT_PREALLOCATE=false before launching.")

    if mine.no_time:
        print("\n  (--no-time: nothing executed)")
        return 0

    r = compiled(state, batch, 1.0, weights)
    jax.block_until_ready(r)
    t0 = time.time()
    for _ in range(max(1, mine.reps)):
        r = compiled(state, batch, 1.0, weights)
    jax.block_until_ready(r)
    dt = (time.time() - t0) / max(1, mine.reps)

    print()
    print(f"  one value+gradient   {dt:.3f} s   (mean of {max(1, mine.reps)})")
    if cfg.steps:
        print(f"  Adam phase floor     {dt * cfg.steps / 60:.1f} min  ({cfg.steps} steps)")
    if cfg.lbfgs_steps:
        # A quasi-Newton iteration evaluates the objective at least once and then runs a
        # line search; 1-3 evaluations is the range the runs in this repo actually use.
        lo, hi = dt * cfg.lbfgs_steps, 3 * dt * cfg.lbfgs_steps
        print(f"  quasi-Newton floor   {lo / 60:.1f} - {hi / 60:.1f} min "
              f"({cfg.lbfgs_steps} iterations, 1-3 evaluations each)")
        print(f"                       (the plateau rule usually stops well short of the cap)")
    print("\n  Compare this line across platforms: the same command with JAX_PLATFORMS=cpu")
    print("  and with the GPU's x64 environment decides which one is worth the wall time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
