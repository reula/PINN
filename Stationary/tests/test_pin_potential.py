"""Config.pin_lam_potential: the averaged order-1 Robin pin on the GH potential P instead of lam.

The lam form <(rho d_rho + 1)(lam - 1)> is violated by the exact spherical solution at O(M^2/rho^2)
(lam = 1 - 2M/rho + 2M^2/rho^2 - ...); on P, whose monopole has no rho^-2 term, it holds to
O(M^3/rho^3).  Measured at rho_out = 100 on the pq problem: 6.6e-5 against 3.8e-7."""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from stationary.losses import outer_pin_raw  # noqa: E402
from stationary.problem import Config, sample_sphere  # noqa: E402
from stationary.train import exact_asset  # noqa: E402


def _cfg(on_P):
    return Config(R0=0.5773502691896258, rho_in=1.0, rho_out=100.0, inner_radius=1.0,
                  lam0=0.33333333333333337, lam_bc_S2=-0.08333333333333333, lam_inf=1.0,
                  ref_solution=True, ref_asymptotic=1.0, outer_bc="robin",
                  pin_lam_robin=True, pin_lam_potential=on_P)


def test_exact_solution_satisfies_the_P_pin_not_the_lam_pin():
    xs = sample_sphere(jax.random.PRNGKey(0), 64, 100.0)
    out = {}
    for on_P in (False, True):
        cfg = _cfg(on_P)
        ex = exact_asset(cfg)
        averaged, _ = outer_pin_raw(ex, xs, cfg, ex)
        out[on_P] = abs(float(jnp.mean(averaged["lam"])))
    assert 5e-5 < out[False] < 8e-5            # 2 M^2 / rho_out^2 with M = R0
    assert out[True] < 1e-6                    # O(M^3 / rho_out^3)
    assert out[True] < out[False] / 100
