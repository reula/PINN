#!/usr/bin/env python
"""Is a trained run stable under resampling?  Its own loss on N fresh collocation samples.

    .venv/bin/python scripts/fresh_sample_loss.py runs/<run> [--n 5] [--params params.pkl]

Each sample is drawn exactly as training draws one (train.make_batch with the run's own config:
same sizes, same radial distribution, same shells), with seeds the run never used.  A solution that
fits only the sample it last saw shows a fresh-sample loss well above its training loss; a stable
one the same loss within the sampling spread.  Also prints the centre-of-mass flux d_z on four
spheres.  Writes <run>/fresh_samples.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stationary.evaluate import load_run                 # noqa: E402
from stationary.geometry import apply_config             # noqa: E402
from stationary.losses import com_flux, total_loss       # noqa: E402
from stationary.model import point_fields                # noqa: E402
from stationary.train import exact_asset, make_batch     # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--params", default="params.pkl")
    a = ap.parse_args(argv)
    cfg, model, state = load_run(a.run, a.params)[:3]
    apply_config(cfg, derives_gamma=getattr(model, "derives_gamma", None))
    ex = exact_asset(cfg)
    lam_inf = jnp.asarray(cfg.lam_inf if cfg.lam_inf is not None else cfg.lam_inf_init)
    f = jax.jit(lambda b: total_loss(state, b, cfg, model, ex, lam_inf=lam_inf)[0])
    losses = [float(f(make_batch(jax.random.PRNGKey(90001 + 17 * i), cfg))) for i in range(a.n)]
    rep = json.load(open(os.path.join(a.run, "report.json"))) if os.path.exists(
        os.path.join(a.run, "report.json")) else {}
    train = rep.get("final_loss")
    cc = cfg.__class__(**{**cfg.__dict__, "com_radii": (1.0, 2.0, 10.0, 100.0)})
    d = [float(v) for v in np.asarray(com_flux(point_fields(model, state["net"]), cc)).reshape(-1, 3)[:, 2]]
    out = {"training_loss": train, "fresh_losses": losses, "fresh_mean": float(np.mean(losses)),
           "fresh_min": float(np.min(losses)), "fresh_max": float(np.max(losses)),
           "ratio_mean_to_training": (float(np.mean(losses)) / train) if train else None, "d_z": d}
    print(f"[fresh] {a.run}: training loss {train:.3e}" if train else f"[fresh] {a.run}")
    print(f"[fresh] {a.n} fresh samples: " + " ".join(f"{v:.3e}" for v in losses))
    print(f"[fresh] mean {out['fresh_mean']:.3e}  spread [{out['fresh_min']:.3e}, {out['fresh_max']:.3e}]"
          + (f"  mean/training {out['ratio_mean_to_training']:.2f}" if train else ""))
    print("[fresh] d_z on spheres 1, 2, 10, 100: " + " ".join(f"{v:+.1e}" for v in d))
    with open(os.path.join(a.run, "fresh_samples.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
