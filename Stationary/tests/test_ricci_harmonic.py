"""ricci_form = 'harmonic' (DeTurck): R_ij - nabla_(i xi_j), xi^k = h^ab Gamma^k_ab.

At a metric in harmonic gauge it equals full Ricci; linearised about flat space it is
-1/2 Laplacian(h_ij) -- the elliptic principal part -- which full Ricci is not."""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from stationary import geometry  # noqa: E402
from stationary.geometry import christoffel, residuals_at  # noqa: E402
from stationary.model import Fields  # noqa: E402
from stationary.problem import Config  # noqa: E402
from stationary.train import exact_asset  # noqa: E402


def _fields_of(h_of, lam_of=lambda x: 1.0):
    return lambda x: Fields(h_of(x), christoffel(h_of, x), jnp.asarray(lam_of(x)))


def _ric(fields, x, form):
    geometry.set_ricci_form(form)
    try:
        return np.asarray(residuals_at(fields, x, want_compat=False)["ricci"])
    finally:
        geometry.set_ricci_form("full")


def test_equal_to_full_ricci_in_harmonic_gauge():
    cfg = Config(R0=0.5773502691896258, rho_in=1.0, rho_out=100.0, inner_radius=1.0,
                 lam0=0.33333333333333337, lam_inf=1.0, ref_solution=True, ref_asymptotic=1.0,
                 outer_bc="robin", lam_bc_S2=-0.08333333333333333)
    ex = exact_asset(cfg)
    for x in (jnp.array([0.3, 1.1, 0.7]), jnp.array([2.0, -3.0, 5.0])):
        np.testing.assert_allclose(_ric(ex, x, "harmonic"), _ric(ex, x, "full"), atol=1e-10)


def test_linearised_principal_part_is_minus_half_laplacian():
    def f(x):                                   # a smooth, non-harmonic-gauge perturbation
        s = jnp.sin(0.7 * x[0]) * jnp.cos(0.4 * x[1]) + 0.3 * x[2] ** 2 * x[0]
        return jnp.array([[s, x[1] * x[2], 0.2 * s],
                          [x[1] * x[2], jnp.exp(0.1 * x[0]), x[0] ** 2],
                          [0.2 * s, x[0] ** 2, s * x[1]]])
    eps = 1e-6
    fields = _fields_of(lambda x: jnp.eye(3) + eps * f(x))
    x = jnp.array([0.4, -0.3, 0.8])
    lap = sum(jax.hessian(lambda y: f(y))(x)[:, :, k, k] for k in range(3))
    np.testing.assert_allclose(_ric(fields, x, "harmonic") / eps, -0.5 * np.asarray(lap),
                               rtol=1e-4, atol=1e-5)
    assert np.max(np.abs(_ric(fields, x, "full") / eps + 0.5 * np.asarray(lap))) > 1e-2
