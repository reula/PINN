#!/usr/bin/env python
"""Recompute, on this machine, the paper's numbers that need no GPU: exact-solution checks, the
Robin exponents, the outer-pin floors, the parameter counts, and the Weyl runs re-evaluated from
their saved networks (copied to OUT so that the runs' own report_eval.json are not overwritten).
    cd Stationary && JAX_ENABLE_X64=1 .venv/bin/python Paper/recheck_local.py OUT_DIR
Prints one line per quantity and writes OUT_DIR/recheck_local.json.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.getcwd())

from stationary import exact                                         # noqa: E402
from stationary.losses import gh_potential                           # noqa: E402
from stationary.problem import Config                                # noqa: E402
from stationary.train import make_model, parse_args                  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "recheck"
os.makedirs(OUT, exist_ok=True)
res = {}


def put(key, val, note=""):
    res[key] = val
    print(f"[recheck] {key:42s} {val}   {note}", flush=True)


# 1. Robin exponents measured on the exact solution at rho = 20 (R0 = 1, k = 1)
f = exact.exact_fields(1.0, 1.0)
n = jnp.array([0.3, 0.4, math.sqrt(1 - 0.25)])
def mag(r):
    F = f(r * n)
    return (float(jnp.max(jnp.abs(F.h - jnp.eye(3)))), float(jnp.max(jnp.abs(F.G))),
            float(abs(F.lam - 1.0)))
a, b = mag(20.0 * (1 - 1e-4)), mag(20.0 * (1 + 1e-4))
ratios = [-(math.log(b[i]) - math.log(a[i])) / (math.log(1 + 1e-4) - math.log(1 - 1e-4)) for i in range(3)]
put("robin_decay_ratios_h_G_lam_at_20", [round(x, 3) for x in ratios], "paper: 2.01, 3.03, 0.955")

# 2. the order-1 outer pins on the exact spherical solution of the quadrupole problem (M = 1/sqrt 3)
M = 1 / math.sqrt(3)
fq = exact.exact_fields(M, 1.0)
def pin(fun, r=100.0):
    g = lambda rr: fun(fq(rr * jnp.array([0.0, 0.0, 1.0])).lam)
    v, d = jax.jvp(g, (r,), (1.0,))
    return float(r * d + (v - fun(jnp.asarray(1.0))))
p_lam = pin(lambda l: l)
p_P = pin(lambda l: gh_potential(l))
put("pin_lambda_residual_rho100", p_lam, "paper: -6.6e-5")
put("pin_lambda_cost_wpin100", 100 * p_lam ** 2, "paper: 4.3e-7")
put("pin_P_cost_wpin100", 100 * p_P ** 2, "paper: 1.5e-11")

# 3. parameter counts
for arch, extra in (("axisym_hybrid", []), ("axisym_hybrid", ["--decay-feature"]), ("hybrid", []),
                    ("mlp", [])):
    cfg = parse_args(["--arch", arch, "--width", "20", "--depth", "6", "--fourier", "0", "--outdir",
                      os.path.join(OUT, "x")] + extra)
    m = make_model(cfg)
    p = m.init(jax.random.PRNGKey(0), jnp.ones((4, 3)))
    put(f"params_{arch}{'_decay' if extra else ''}", int(sum(x.size for x in jax.tree.leaves(p))),
        "paper: 2265 / 2285 / 2347 / 2725")

# 4. the Weyl runs, re-evaluated from their saved networks
from stationary.evaluate import evaluate                             # noqa: E402
for run in ("weyl_prod", "weyl_unequal", "weyl_close4", "weyl_close35", "weyl_rot45",
            "weyl_rot45_long", "weyl_rot45_hub"):
    src = os.path.join("runs", run)
    dst = os.path.join(OUT, run)
    os.makedirs(dst, exist_ok=True)
    for fn in ("params.pkl", "config.json", "report.json", "history.json"):
        if os.path.exists(os.path.join(src, fn)):
            shutil.copy(os.path.join(src, fn), dst)
    rep = evaluate(dst, make_plots=False)
    own = json.load(open(os.path.join(src, "report.json")))
    put(f"{run}_final_loss", own.get("final_loss"))
    put(f"{run}_wall_min", round(own.get("wall_seconds", float("nan")) / 60, 1))
    for k in ("max_dh", "max_dG", "max_dlam", "rms_dh", "rms_dlam"):
        if k in rep:
            put(f"{run}_{k}", rep[k])

with open(os.path.join(OUT, "recheck_local.json"), "w") as fh:
    json.dump(res, fh, indent=1, default=float)
