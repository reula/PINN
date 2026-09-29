"""The compatibility group, and the formulations that do not need it.

`compat[a,b,c] = d_a h_bc - G^d_ab h_dc - G^d_ac h_bd` is a real equation for the first-order
formulation, where the network outputs Gamma (18 of the 25 fields) and nothing ties it to h.
For the metric-only ("hybrid") formulations Gamma IS the Christoffel symbol of h, computed by
autodiff from the same h that enters the residual, so compatibility holds identically: the
group is a constraint on nothing, sitting at the round-off floor (measured 3.5e-36 in the
production runs).  Carrying it in the loss therefore costs a group, invites a reader of
report.json to count four equations where there are three, and -- the reason it is worth a test
rather than a comment -- leaves a term in the objective that no data can move.

These tests pin the boundary: metric-only models exclude it, first-order models keep it, and
the exclusion is keyed off the MODEL (its `derives_gamma`) rather than `cfg.arch`, so a caller
that builds one of these nets by hand gets the right loss whatever the config says.

The residual itself is still computed for every model and still lands in report.json and in
history.json as `pde_compat`.  That is deliberate: it is the structural check that Gamma really
is h's Christoffel symbol, and it is the first number that would move if that derivation broke.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from stationary import losses as L
from stationary import train as T
from stationary.losses import (FIRST_ORDER_PDE_KEYS, METRIC_ONLY_PDE_KEYS, default_weights,
                               equation_keys, total_loss)
from stationary.model import (AxisymHybridNet, FieldNet, HybridNet, SymFieldNet,
                              SymHybridNet)
from stationary.train import _pde_str, make_model

TINY = ["--n-coll", "32", "--n-bnd", "8", "--width", "4", "--depth", "2", "--no-figures",
        "--ref-solution", "--ref-asymptotic", "1.0"]

METRIC_ONLY = ["sym_hybrid", "hybrid", "axisym_hybrid"]
FIRST_ORDER = ["sym", "mlp"]


def build(arch):
    cfg = T.parse_args(["--arch", arch, "--outdir", "/tmp/eqk", "--steps", "1",
                        "--lbfgs-steps", "0"] + TINY)
    model, state, exact_fields = T.build(cfg)
    return cfg, model, state, exact_fields


@pytest.mark.parametrize("arch", METRIC_ONLY)
def test_metric_only_models_declare_that_they_derive_gamma(arch):
    cfg, model, _, _ = build(arch)
    assert model.derives_gamma is True
    assert equation_keys(model) == METRIC_ONLY_PDE_KEYS
    assert "compat" not in equation_keys(model)


@pytest.mark.parametrize("arch", FIRST_ORDER)
def test_first_order_models_keep_compatibility(arch):
    cfg, model, _, _ = build(arch)
    assert model.derives_gamma is False
    assert equation_keys(model) == FIRST_ORDER_PDE_KEYS


def test_every_architecture_is_classified():
    """A new net class must not silently inherit FieldNet's answer."""
    for cls, expected in ((FieldNet, False), (SymFieldNet, False),
                          (HybridNet, True), (SymHybridNet, True), (AxisymHybridNet, True)):
        assert cls.derives_gamma is expected, cls.__name__
    for arch in METRIC_ONLY + FIRST_ORDER:
        cfg = T.parse_args(["--arch", arch, "--outdir", "/tmp/eqk"] + TINY)
        assert isinstance(make_model(cfg).derives_gamma, bool)


@pytest.mark.parametrize("arch", METRIC_ONLY + FIRST_ORDER)
def test_forcing_compat_absurd_moves_the_loss_only_for_the_first_order_models(arch):
    """The decisive property: put 1e6 in the compat residual and see whether the loss notices.

    A mean square, so a first-order loss must move by exactly that amount (its weight is 1).
    A metric-only loss must not move at all -- if it does, the group is still being counted.
    """
    cfg, model, state, exact_fields = build(arch)
    batch = T.make_batch(jax.random.PRNGKey(0), cfg)
    base, parts_base = total_loss(state, batch, cfg, model, exact_fields)
    compat_old = float(parts_base["pde_compat"])

    real = L.pde_terms

    def sabotaged(pf, xs, cfg=None, gauge_src=None, _real=real):
        out = dict(_real(pf, xs, cfg, gauge_src))
        out["compat"] = jnp.asarray(1e6)
        return out

    L.pde_terms = sabotaged
    try:
        blown, parts = total_loss(state, batch, cfg, model, exact_fields)
    finally:
        L.pde_terms = real

    moved = abs(float(blown) - float(base))
    assert float(parts["pde_compat"]) == pytest.approx(1e6), "the sabotage did not take"
    if model.derives_gamma:
        assert moved == 0.0, (
            f"{arch}: compat is not supposed to be in the loss, but forcing it to 1e6 moved "
            f"the loss by {moved:.3e}")
    else:
        # Weight 1, and the group is replaced rather than added to, so the move is
        # 1e6 - compat_old.  (compat_old is ~0.012 at a random init, not 0: for a first-order
        # model the network's Gamma really is not yet h's Christoffel symbol.)
        assert moved == pytest.approx(1e6 - compat_old, rel=1e-9), (
            f"{arch}: compat IS a constraint for the first-order formulation, but forcing it "
            f"to 1e6 moved the loss by {moved:.3e} instead of {1e6 - compat_old:.3e}")


def test_the_structural_check_is_still_reported():
    """Dropping it from the loss must not drop it from the diagnostics."""
    cfg, model, state, exact_fields = build("axisym_hybrid")
    batch = T.make_batch(jax.random.PRNGKey(0), cfg)
    _, parts = total_loss(state, batch, cfg, model, exact_fields)
    assert "pde_compat" in parts, "the structural check that Gamma == Christoffel(h) is gone"
    assert float(parts["pde_compat"]) < 1e-12, "and it is not at the floor for a hybrid model"


def test_the_log_does_not_advertise_a_group_the_loss_ignores():
    """compat must be absent from a metric-only `pde(...)` fragment and present otherwise."""
    hybrid = {"pde_ricci": 1e-3, "pde_gauge": 1e-4, "pde_lam_eq": 1e-5, "pde_compat": 3e-36}
    line = _pde_str(hybrid, METRIC_ONLY_PDE_KEYS)
    assert "compat" not in line and "ricci" in line and "lam=1.00e-05" in line
    assert "compat=3.00e-36" in _pde_str(hybrid, FIRST_ORDER_PDE_KEYS)


def test_the_radial_term_still_prints_when_it_is_on():
    """`_pde_str` shares the Adam line and the quasi-Newton line; neither may lose lam_rad."""
    parts = {"pde_ricci": 1e-3, "pde_gauge": 1e-4, "pde_lam_eq": 1e-5,
             "pde_lam_eq_radial": 2e-7}
    assert "lam_rad=2.00e-07" in _pde_str(parts, METRIC_ONLY_PDE_KEYS)
