"""How much of a run's loss is the SAMPLE, not the solution.

The training loss is a mean over a finite collocation sample, so it is an estimate.  Two batches
of the same parameters, drawn from the same law, give different losses; the scatter across those
draws is the floor below which "the loss went down" stops meaning "the solution improved" --
below it, a lower number is a smaller number on *that* sample.

This prints, for one run directory:

  * the loss on each of N fresh batches (same n_coll/n_bnd as the run, different seed);
  * mean, std, min, max of those losses;
  * the terms of the last batch, so a scatter can be attributed to a group;
  * the unweighted group means (the weight-independent numbers HUB.md says to judge on) per
    batch, with their scatter.

Usage (from Stationary/):

    ./.venv/bin/python loss_scatter.py --outdir runs/pq_c100_vacF_phihyb25
    ./.venv/bin/python loss_scatter.py --outdir runs/pq_c100_vacF_phihyb25 --batches 8 \
        --n-coll 65536            # what would more points buy?
    ./.venv/bin/python loss_scatter.py --outdir <run> --params params_adam.pkl
"""
from __future__ import annotations

import argparse

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from stationary.evaluate import load_run
from stationary.losses import group_terms, total_loss
from stationary.train import make_batch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--params", default="params.pkl")
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--n-coll", type=int, default=None, help="override the run's n_coll")
    ap.add_argument("--n-bnd", type=int, default=None, help="override the run's n_bnd")
    a = ap.parse_args()

    cfg, model, state = load_run(a.outdir, a.params)
    if a.n_coll is not None:
        cfg.n_coll = a.n_coll
    if a.n_bnd is not None:
        cfg.n_bnd = a.n_bnd
        cfg.n_bnd_outer = a.n_bnd
    # the exact asset the run used, for the boundary terms (may be None)
    from stationary.train import exact_asset
    exact = exact_asset(cfg)

    print(f"run {a.outdir}  params {a.params}")
    print(f"  n_params {jax.flatten_util.ravel_pytree(state['net'])[0].size}  "
          f"n_coll {cfg.n_coll}  n_bnd {cfg.n_bnd}/{cfg.n_bnd_outer}  "
          f"radial {cfg.radial}  lam_eq_form {cfg.lam_eq_form}  scale_ref {cfg.scale_ref}")

    losses, groups = [], []
    parts_last = None
    for i in range(a.batches):
        batch = make_batch(jax.random.PRNGKey(a.seed0 + i), cfg)
        loss, parts = total_loss(state, batch, cfg, model, exact)
        losses.append(float(loss))
        gp = group_terms(state, batch, cfg, model, exact)
        groups.append({k: float(v) for k, v in gp.items()})
        parts_last = {k: float(v) for k, v in parts.items()}
    L = np.array(losses)
    print(f"\n  loss over {a.batches} batches: mean {L.mean():.6e}   std {L.std():.3e}"
          f"   min {L.min():.6e}   max {L.max():.6e}   spread {(L.max()-L.min())/L.mean():.1%}")
    print("    " + "  ".join(f"{x:.3e}" for x in L))
    print("\n  unweighted group means (weight-independent; HUB.md: judge on these):")
    for k in groups[0]:
        v = np.array([g[k] for g in groups])
        print(f"    {k:16s} mean {v.mean():.6e}   std {v.std():.3e}   "
              f"({v.std()/max(v.mean(), 1e-300):.1%} across batches)")
    print("\n  terms of the last batch:")
    for k, v in sorted(parts_last.items(), key=lambda kv: -abs(kv[1]))[:8]:
        print(f"    {k:18s} {v:.6e}")
    print("\n  A loss below the batch scatter is a number on one sample, not a better solution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
