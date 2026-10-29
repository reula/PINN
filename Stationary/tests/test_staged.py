"""--stage-base: a second-stage correction network on a frozen first stage (model.StagedNet).

Three statements: with eps = 0 the staged model IS the base (fields and Christoffel symbols); the
trainable parameters are the correction's only, and the loss has a nonzero gradient in them; and a
staged run is rebuilt from its config (make_model reloads the base from its run directory).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import dataclasses
import json
import os
import pickle

import jax.numpy as jnp
import numpy as np

from stationary.losses import total_loss
from stationary.model import StagedNet, point_fields
from stationary.problem import sample_shell, sample_sphere
from stationary.train import build, make_model, parse_args

BASE = ["--arch", "hybrid", "--n-coll", "64", "--n-bnd", "24", "--n-bnd-outer", "24",
        "--width", "8", "--depth", "2", "--fourier", "0", "--no-figures",
        "--R0", "0.5773502691896258", "--rho-in", "1.0", "--rho-out", "100.0",
        "--inner-radius", "1.0", "--lam0", "0.33333333333333337", "--lam-inf", "1.0",
        "--lam-bc-S2", "-0.08333333333333333", "--no-robin-G"]


def _base_run(tmp_path):
    cfg = parse_args(BASE + ["--outdir", str(tmp_path / "base")])
    model, state, _ = build(cfg)
    # a base that is not near flat space, so that equality below is not trivial
    state["net"] = jax.tree_util.tree_map(lambda p: p + 0.3, state["net"])
    os.makedirs(cfg.outdir, exist_ok=True)
    with open(os.path.join(cfg.outdir, "config.json"), "w") as fh:
        json.dump(dataclasses.asdict(cfg), fh)
    with open(os.path.join(cfg.outdir, "params.pkl"), "wb") as fh:
        pickle.dump(state, fh)
    return cfg, model, state


def _staged(tmp_path, eps):
    return parse_args(BASE + ["--outdir", str(tmp_path / "stage"), "--stage-base",
                              str(tmp_path / "base"), "--stage-eps", str(eps)])


def test_zero_eps_is_the_base(tmp_path):
    _, bmodel, bstate = _base_run(tmp_path)
    cfg = _staged(tmp_path, 0.0)
    model, state, _ = build(cfg)
    assert isinstance(model, StagedNet)
    fb, fs = point_fields(bmodel, bstate["net"]), point_fields(model, state["net"])
    for x in sample_shell(jax.random.PRNGKey(3), 8, cfg):
        a, b = fb(x), fs(x)
        np.testing.assert_allclose(b.h, a.h, rtol=0, atol=1e-14)
        np.testing.assert_allclose(b.lam, a.lam, rtol=0, atol=1e-14)
        np.testing.assert_allclose(b.G, a.G, rtol=0, atol=1e-12)


def test_trains_the_correction_only(tmp_path):
    _, _, bstate = _base_run(tmp_path)
    cfg = _staged(tmp_path, 1e-2)
    model, state, exact = build(cfg)
    nb = sum(p.size for p in jax.tree_util.tree_leaves(bstate["net"]))
    ns = sum(p.size for p in jax.tree_util.tree_leaves(state["net"]))
    assert ns == (4 + 1) * 8 + 9 * 8 + 9 * 7          # the correction: 4 inputs, 2 x 8, 7 outputs
    assert nb == ns                                    # same architecture as this base
    key = jax.random.PRNGKey(4)
    batch = {"coll": sample_shell(key, 64, cfg), "inner": sample_sphere(key, 24, cfg.rho_in),
             "outer": sample_sphere(key, 24, cfg.rho_out)}
    g = jax.grad(lambda s: total_loss(s, batch, cfg, model, exact_fields=exact)[0])(state)
    assert float(sum(jnp.sum(v ** 2) for v in jax.tree_util.tree_leaves(g["net"]))) > 0


def test_rebuilt_from_config(tmp_path):
    _base_run(tmp_path)
    cfg = _staged(tmp_path, 1e-3)
    m = make_model(cfg)
    assert isinstance(m, StagedNet) and m.eps == 1e-3 and m.width == 8
