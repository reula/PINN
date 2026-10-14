"""The outer Robin condition on a ladder of spheres, and why that is legitimate.

A homogeneous Robin condition of this family is satisfied IDENTICALLY by the exact solution at
every radius (it annihilates a window of powers, and the solution lives in that window), so
imposing it at rho_out/2, rho_out/4 ... is not an approximation -- it is the same condition, read
where a thin layer at rho_out cannot satisfy it on the solution's behalf.

The measurement that motivated it: on pq_r100_phihyb025_R0, B_3[h - I3] was 1.30, 0.52 and 0.059 at
rho = 50, 90 and 99 but 4.8e-05 at rho = 100.0, while the metric kept a constant l = 2 part
(h_rr = 1.0288 on the equator, 0.9418 on the axis, spherical mean 1.00015).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from stationary.geometry import apply_config
from stationary.losses import I3, robin_operator
from stationary.problem import Config, sample_sphere
from stationary.train import exact_asset, make_batch

jax.config.update("jax_enable_x64", True)


def _cfg(shells=1):
    # R0 is NOT optional: without it the Weyl solution is a different (degenerate) one, whose
    # Robin residual is O(10) while the run's reference sits at 1e-10 (cost me two wrong readings).
    c = Config(arch="sym_hybrid", width=6, depth=2, fourier=0, rho_in=1.0, rho_out=100.0,
               inner_radius=1.0, lam0=1.0 / 3.0, lam_bc_S2=-1.0 / 12.0, lam_inf=1.0,
               R0=1.0 / 3.0 ** 0.5, ref_asymptotic=1.0, decay_feature=True,
               n_coll=64, n_bnd=32, n_bnd_outer=32, outer_bc="robin", robin_order=3,
               robin_orders={"h": 3, "G": 3, "lam": 3}, robin_include_G=False,
               robin_exps={"h": 2.0, "G": 3.0, "lam": 1.0},
               ref_solution=True, outer_shells=shells)
    c.__post_init__()
    return c


def test_the_ladder_places_the_extra_spheres_where_it_says():
    cfg = _cfg(shells=3)
    b = make_batch(jax.random.PRNGKey(0), cfg)
    assert b["outer"].shape[0] == cfg.n_bnd_outer
    extra = b["outer_extra"]
    assert extra.shape[0] == 2 * cfg.n_bnd_outer
    r = np.asarray(jnp.linalg.norm(extra, axis=-1))
    assert np.allclose(np.unique(np.round(r, 6)), [25.0, 50.0]), np.unique(r)
    assert make_batch(jax.random.PRNGKey(0), _cfg(shells=1)).get("outer_extra") is None


def test_the_exact_solution_satisfies_the_condition_on_every_sphere():
    """If it did not, the ladder would be imposing physics the solution does not obey.

    Measured on the run's own reference (pq_r100_phihyb025_R0's configuration): 1.5e-10 at
    rho = 100, 4e-9 at 50, 2e-8 at 35, 1e-7 at 25, 5.7e-6 (lambda) at 12.5.  The growth inward is the
    quadrature and derivative accuracy of the reference itself, not a violation, so the tolerance
    has to allow it -- and the ladder is used at rho >= rho_out/4, where the reference is at 1e-7.
    """
    cfg = _cfg(shells=1)
    apply_config(cfg)
    ref = exact_asset(cfg)
    for rho in (100.0, 50.0, 25.0):
        xs = sample_sphere(jax.random.PRNGKey(1), 8, rho)
        for base, order, inf, field in ((cfg.robin_exps["h"], cfg.robin_orders["h"], I3,
                                         lambda y: ref(y).h),
                                        (cfg.robin_exps["lam"], cfg.robin_orders["lam"], 1.0,
                                         lambda y: ref(y).lam)):
            vals = jnp.array([robin_operator(field, x, base, order, inf) for x in xs])
            assert float(jnp.max(jnp.abs(vals))) < 1e-5, (rho, float(jnp.max(jnp.abs(vals))))


def test_the_ladder_is_configurable_and_defaults_off():
    from stationary.train import parse_args
    assert parse_args(["--outdir", "/tmp/x"]).outer_shells == 1
    a = parse_args(["--outdir", "/tmp/x", "--outer-shells", "5", "--outer-shell-ratio", "3"])
    assert (a.outer_shells, a.outer_shell_ratio) == (5, 3.0)
