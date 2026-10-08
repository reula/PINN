"""Measure the spectrum of the exact loss Hessian.

This is the measurement that decides whether Xu & Darve's premise applies to this
problem at all: they argue that BFGS/L-BFGS mislead PINN training because the true
Hessian is indefinite or positive *semi*-definite, while quasi-Newton updates
force it positive definite.  Here we just look.

    cd Evolution_try
    /Users/reula/jax_env/bin/python -m scripts.hessian_spectrum --n-coll 2201

Cost warning: forming the dense Hessian costs ``n_parameters`` forward-over-reverse
passes, so at 2201 parameters it is minutes, not seconds.  Use ``--n-coll`` to
shrink the residual batch (the Hessian cost is proportional to it) when you only
want the spectrum's shape.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from wave_pinn.config import Config
from wave_pinn.optim.trustregion import make_hessian
from wave_pinn.train import build, configure_jax


def main(argv=None) -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-coll", type=int, default=0, help="collocation points (0 = one per parameter)")
    p.add_argument("--init-from", default="", help="theta.npy to measure at (default: random init)")
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--out", default=os.path.join(here, "runs", "hessian_spectrum.json"))
    args = p.parse_args(argv)

    cfg = Config(T=args.T, n_coll=args.n_coll, init_from=args.init_from, tr_chunk=128)
    configure_jax(cfg)
    params, obj, cfg = build(cfg)
    flat, _ = jax.flatten_util.ravel_pytree(params)

    # Chunked forward-over-reverse.  A plain jax.hessian here pushes all
    # n_parameters tangents through at once and the process is killed; at
    # n = 2201 that is a multi-gigabyte intermediate.
    _loss, _grad, hess = make_hessian(obj, cfg)
    t0 = time.time()
    H = hess(flat)
    form_time = time.time() - t0
    w = np.linalg.eigvalsh(H)
    scale = float(np.max(np.abs(w)))
    thr = 1e-6 * scale
    n_pos = int(np.sum(w > thr))
    n_neg = int(np.sum(w < -thr))
    n_zero = int(w.size - n_pos - n_neg)

    result = {
        "T": cfg.T, "n_coll": cfg.n_coll, "n_parameters": int(flat.size),
        "init_from": args.init_from or None,
        "hessian_form_seconds": form_time,
        "lambda_min": float(w.min()), "lambda_max": float(w.max()),
        "abs_scale": scale, "zero_threshold": thr,
        "positive": n_pos, "negative": n_neg, "zero": n_zero,
        "fraction_negative": n_neg / w.size,
        "fraction_zero": n_zero / w.size,
    }
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    np.save(args.out.replace(".json", "_eigenvalues.npy"), w)
    print(f"\nwritten {args.out}")
    print(f"verdict: the exact Hessian is {'INDEFINITE' if n_neg else 'positive semi-definite'} "
          f"({n_neg}/{w.size} negative eigenvalues)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
