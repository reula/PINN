"""`--pde-radial`: the radial derivative of each equation group at the collocation points.

The design, as specified: only the radial derivative; at the collocation points; the SAME weight
as the group it comes from (so it rides on `weights[k]`, with no weight of its own); ONE EXTRA rho
for dimensions (`rho**(scale_exps[g]+1)`); and no `compat`.  Off by default, so every existing run
and test is unaffected.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from stationary.geometry import apply_config
from stationary.losses import pde_radial_groups_raw, residual_vector, total_loss
from stationary.problem import Config
from stationary.train import make_batch, make_model

jax.config.update("jax_enable_x64", True)


def _setup(**kw):
    c = Config(arch="axisym_hybrid", width=6, depth=2, fourier=0, rho_in=1.0, rho_out=10.0,
               inner_radius=1.0, lam0=1.0 / 3.0, lam_bc_S2=-1.0 / 12.0, R0=1.0 / 3.0 ** 0.5,
               ref_solution=True, n_coll=48, n_bnd=16, n_bnd_outer=16, outer_bc="robin",
               robin_order=3, robin_orders={"h": 3, "lam": 3}, robin_include_G=False,
               robin_exps={"h": 2.0, "G": 3.0, "lam": 1.0}, **kw)
    c.__post_init__()
    apply_config(c)
    model = make_model(c)
    state = {"net": model.init(jax.random.PRNGKey(0), jnp.ones((4, 3)))}
    return c, model, state, make_batch(jax.random.PRNGKey(1), c)


def test_off_by_default_and_nothing_is_differentiated_then():
    c, _, _, _ = _setup()
    assert c.pde_radial is False
    assert pde_radial_groups_raw(None, None, c) == {}      # returns before touching anything


def test_no_compat():
    c, m, s, b = _setup(pde_radial=True)
    from stationary.model import point_fields
    xs = b["coll"]
    assert set(pde_radial_groups_raw(point_fields(m, s["net"]), xs, c)) <= {"ricci", "gauge",
                                                                           "lam_eq"}


def test_it_carries_the_group_weight_and_not_one_of_its_own():
    """With every equation weight at zero, switching the term on must change nothing."""
    c0, m, s, b = _setup()
    c1, _, _, _ = _setup(pde_radial=True)
    zeros = {"compat": 0.0, "ricci": 0.0, "gauge": 0.0, "lam_eq": 0.0}
    c0.eq_weights, c1.eq_weights = dict(zeros), dict(zeros)
    l0, _ = total_loss(s, b, c0, m)
    l1, _ = total_loss(s, b, c1, m)
    assert float(l0) == pytest.approx(float(l1), rel=1e-12), (float(l0), float(l1))


def test_the_objective_grows_and_the_residual_vector_still_squares_to_it():
    off, m, s, b = _setup()
    on, _, _, _ = _setup(pde_radial=True)
    l_off, _ = total_loss(s, b, off, m)
    l_on, _ = total_loss(s, b, on, m)
    assert float(l_on) > float(l_off)                     # a positive term was added
    r = residual_vector(s, b, on, m)
    assert float(jnp.sum(r ** 2)) == pytest.approx(float(l_on), rel=1e-9)


def test_the_flag_reaches_the_config():
    from stationary.train import parse_args
    assert parse_args(["--outdir", "/tmp/x"]).pde_radial is False
    assert parse_args(["--outdir", "/tmp/x", "--pde-radial"]).pde_radial is True
