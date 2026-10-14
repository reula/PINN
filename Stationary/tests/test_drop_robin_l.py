"""`--drop-robin-l 1` removes the dipole from the outer condition's action, and nothing else.

The outer Robin residual is a function on the sphere at `rho_out`.  A run can ask for a multipole
to be exempt from it, which is a diagnostic: the sector is then unconstrained at the outer boundary
and whatever the interior pushes there is unopposed.  What this file pins is that the exemption is
exact for the requested `l` and leaves every other multipole alone -- it is a projection, so a
mistake in the coefficient would quietly de-weight neighbours.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from stationary.geometry import Fields, apply_config
from stationary.losses import _legendre, outer_bc_raw
from stationary.problem import Config, sample_sphere

jax.config.update("jax_enable_x64", True)
I3 = jnp.eye(3)


def _cfg(drop):
    # The default scalar condition (on the Geroch-Hansen potential, dipolar power dropped) is
    # what the runs impose, so it is what is tested here.
    c = Config(outer_bc="robin", robin_order=3, robin_orders={"h": 3, "G": 3, "lam": 3},
               robin_include_G=False, lam_inf=1.0, rho_out=10.0, rho_in=1.0,
               drop_robin_ls=tuple(drop))
    c.__post_init__()
    return c


def _content(xs, R, l):
    mu = xs[:, 2] / jnp.linalg.norm(xs, axis=-1)
    return float((2 * l + 1) / R.shape[0] * jnp.sum(R * _legendre(l, mu)))


def _residual(drop, lmax=3):
    cfg = _cfg(drop)
    apply_config(cfg)
    # A lambda with l = 1 and l = 2 content that the Robin condition does not satisfy, SMALL and
    # decaying: the scalar condition is imposed on the Geroch-Hansen potential, which is nonlinear
    # in lambda, so a polynomial that wanders far from 1 makes P meaningless (and P diverges where
    # lambda crosses zero).  With a small perturbation P = -delta_lambda/2 to leading order and the
    # multipole content is preserved.
    def fields(x):
        r = jnp.linalg.norm(x)
        mu = x[2] / r
        return Fields(I3, jnp.zeros((3, 3, 3)),
                      1.0 + 0.2 * mu / r + 0.05 * (1.5 * mu ** 2 - 0.5) / r ** 2)
    xs = sample_sphere(jax.random.PRNGKey(0), 4096, cfg.rho_out)
    R = outer_bc_raw(fields, xs, cfg)["lam"]
    return xs, R


def test_dropping_l1_removes_the_dipole_and_only_the_dipole():
    """The projection is a Monte-Carlo estimate, so the assertions are the physical ones: the
    dipole collapses, its dominant neighbour does not move, and no other multipole is grossly
    disturbed.  Measured: l = 1 falls 40x at 4096 points (146x at 16384, the estimator's noise
    floor), while the l = 2 content is unchanged to five digits."""
    xs, full = _residual(())
    _, kept = _residual((1,))
    before = [_content(xs, full, l) for l in range(4)]
    after = [_content(xs, kept, l) for l in range(4)]
    assert abs(after[1]) < abs(before[1]) / 20, (before, after)
    assert abs(after[2] - before[2]) < 0.01 * abs(before[2]), (before, after)
    # not a global de-weighting: the neighbours keep their size
    assert abs(after[2]) > 0.9 * abs(before[2])
    assert max(abs(after[0] - before[0]), abs(after[3] - before[3])) < 0.05 * abs(before[2])


def test_the_flag_reaches_the_config():
    from stationary.train import parse_args
    assert parse_args(["--outdir", "/tmp/x", "--drop-robin-l", "1"]).drop_robin_ls == (1,)
    assert parse_args(["--outdir", "/tmp/x", "--drop-robin-l", "1", "--drop-robin-l", "2"]
                      ).drop_robin_ls == (1, 2)
    assert parse_args(["--outdir", "/tmp/x"]).drop_robin_ls == ()


def cfg_factory(drop):
    c = Config(inner_bc="spherical", rho_in=1.0, rho_out=10.0, inner_radius=1.0,
               lam0=1.0 / 3.0, lam_bc_S2=-1.0 / 12.0, drop_inner_ls=tuple(drop))
    c.__post_init__()
    apply_config(c)
    return c


def test_dropping_l1_also_works_on_the_inner_datum():
    """The inner data here carry `P2` and no `P1`, so the inner condition is a dipole condition
    too: `--drop-inner-l 1` is what makes the sector free at the inner sphere."""
    from stationary.losses import inner_bc_raw
    cfg = cfg_factory((1,))
    # a field with a deliberate l = 1 part: the datum has none, so the residual is the field's l = 1
    def fields(x):
        r = jnp.linalg.norm(x)
        return Fields(I3 + 0.05 * jnp.outer(x / r, x / r), jnp.zeros((3, 3, 3)),
                      cfg.lam0 + 0.1 * (x[2] / r))
    xs = sample_sphere(jax.random.PRNGKey(1), 8192, cfg.rho_in)
    full = inner_bc_raw(fields, xs, cfg_factory(()))["lam"]
    kept = inner_bc_raw(fields, xs, cfg)["lam"]
    before, after = _content(xs, full, 1), _content(xs, kept, 1)
    assert abs(before) > 0.05, before                      # the field's dipole is really there
    assert abs(after) < 0.1 * abs(before), (before, after)  # and the datum no longer sees it
    assert abs(_content(xs, kept, 0) - _content(xs, full, 0)) < 0.05 * abs(before)
