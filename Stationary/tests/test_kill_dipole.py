"""`--kill-dipole h`: the outer condition run WITHOUT the dipolar power in its kernel.

The distinction this pins down, because the two are easy to confuse and mean opposite things:

  --drop-robin-l 1   EXEMPTS the dipole: the l = 1 residual is removed from the loss, so the dipole
                     is unconstrained at the outer boundary and whatever the interior pushes there
                     survives.
  --kill-dipole h    KILLS it: the same Robin condition is run with the factor (rho d_rho + 2) --
                     the one that annihilates rho^-2, the dipolar power -- taken out of the product,
                     so the l = 1 rho^-2 coefficient is now the one thing the condition forces to
                     zero.  The rest of the spectrum is untouched.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from stationary.geometry import Fields, apply_config
from stationary.losses import (I3, dipole_killing_exps, outer_bc_raw, robin_operator,
                               robin_operator_exps)
from stationary.problem import Config, sample_sphere

jax.config.update("jax_enable_x64", True)


def _cfg(**kw):
    c = Config(outer_bc="robin", robin_order=3, robin_orders={"h": 3, "G": 3, "lam": 3},
               robin_include_G=False, robin_exps={"h": 2.0, "G": 3.0, "lam": 1.0},
               lam_inf=1.0, rho_out=100.0, rho_in=1.0, lam0=1 / 3, R0=1 / 3 ** 0.5,
               ref_solution=True, **kw)
    c.__post_init__()
    return c


def _field(l, power, amp=1.0):
    """A field whose ONLY content is the l-th Legendre mode decaying as rho^-power."""
    def fields(y):
        r = jnp.linalg.norm(y)
        mu = y[2] / r
        P = {0: 1.0, 1: mu, 2: 1.5 * mu ** 2 - 0.5}[l]
        e = amp * P / r ** power
        return Fields(I3 + e * I3, jnp.zeros((3, 3, 3)), 1.0 + e)
    return fields


def test_the_exponents_are_the_window_with_two_removed():
    assert dipole_killing_exps([2.0, 3.0, 4.0]) == [3.0, 4.0, 5.0]     # h:  rho^-2 killed
    assert dipole_killing_exps([1.0, 2.0, 3.0]) == [1.0, 3.0, 4.0]     # lam: rho^-2 killed
    assert dipole_killing_exps([3.0, 4.0, 5.0]) == [3.0, 4.0, 5.0]     # 2 absent: nothing to do


def test_a_pure_dipole_is_invisible_to_the_standard_condition_and_visible_to_the_killing_one():
    """The whole point: rho^-2 is inside the standard kernel, so a dipole tail is accepted."""
    y = jnp.array([60.0, 0.0, 80.0])          # mu = 0.8: P_1 must not vanish
    dip = _field(1, 2.0)
    std = robin_operator(lambda z: dip(z).h, y, 2.0, 3, I3)
    mod = robin_operator_exps(lambda z: dip(z).h, y, dipole_killing_exps([2.0, 3.0, 4.0]), I3)
    assert float(jnp.max(jnp.abs(std))) < 1e-9, "the standard window accepts rho^-2"
    assert float(jnp.max(jnp.abs(mod))) > 1e6 * float(jnp.max(jnp.abs(std))), "the dipole must be SEEN"
    assert float(jnp.max(jnp.abs(mod))) > 1e-6


def test_the_rest_of_the_spectrum_is_untouched():
    """l = 0 and l = 2, at the powers they are supposed to have, stay invisible."""
    y = jnp.array([100.0, 0.0, 0.0])
    for l, power in ((0, 3.0), (2, 3.0), (2, 4.0)):
        f = _field(l, power)
        mod = robin_operator_exps(lambda z: f(z).h, y, dipole_killing_exps([2.0, 3.0, 4.0]), I3)
        assert float(jnp.max(jnp.abs(mod))) < 1e-9, (l, power, float(jnp.max(jnp.abs(mod))))


def test_the_l0_part_of_a_rho_minus_two_tail_is_still_accepted():
    """The modified kernel cannot tell the modes apart by itself -- it is RADIAL -- which is exactly
    why it is applied only to the l = 1 projection.  A pure l = 0, rho^-2 tail must therefore end up
    with a vanishing residual even with the dipole killed."""
    cfg = _cfg(kill_dipole=("h",))
    apply_config(cfg)
    xs = sample_sphere(jax.random.PRNGKey(3), 1024, cfg.rho_out)
    mono = _field(0, 2.0)
    out = outer_bc_raw(mono, xs, cfg)
    # The substitution is exact in the mode; what is left is the projection's own sampling noise,
    # so the honest criterion is relative: the residual must be far below the modified operator's
    # own action on this field (which is what would appear if the l = 0 part were being constrained).
    y = jnp.array([60.0, 0.0, 80.0])
    raw_mod = float(jnp.max(jnp.abs(robin_operator_exps(
        lambda z: mono(z).h, y, dipole_killing_exps([2.0, 3.0, 4.0]), I3))))
    assert raw_mod > 1e-4, raw_mod
    # with N = 1024 the unbiased projection's own noise is O(1/sqrt(N)) ~ 3%
    assert float(jnp.max(jnp.abs(out["h"]))) < 0.1 * raw_mod, (float(jnp.max(jnp.abs(out["h"]))), raw_mod)
    # and directly: the l = 0 projection of the modified operator still annihilates it
    y = jnp.array([60.0, 0.0, 80.0])
    mod = robin_operator_exps(lambda z: mono(z).h, y, dipole_killing_exps([2.0, 3.0, 4.0]), I3)
    kept = kill_dipole_part(jnp.zeros((1, 1)), jnp.zeros((1, 1)), y[None, :], l=1)
    assert float(jnp.max(jnp.abs(kept))) == 0.0


def test_outer_bc_raw_kills_the_dipole_only_in_the_requested_field():
    cfg = _cfg(kill_dipole=("h",))
    apply_config(cfg)
    xs = sample_sphere(jax.random.PRNGKey(0), 128, cfg.rho_out)
    dip = _field(1, 2.0)
    killed = outer_bc_raw(dip, xs, cfg)
    cfg.kill_dipole = ()
    plain = outer_bc_raw(dip, xs, cfg)
    assert float(jnp.max(jnp.abs(plain["h"]))) < 1e-9, "standard: the dipole is accepted"
    assert float(jnp.max(jnp.abs(killed["h"]))) > 1e-6, "killed: it must not be"
    assert float(jnp.max(jnp.abs(killed["h"]))) > 1e3 * float(jnp.max(jnp.abs(plain["h"])))
    assert float(jnp.max(jnp.abs(killed["lam"] - plain["lam"]))) == 0.0
