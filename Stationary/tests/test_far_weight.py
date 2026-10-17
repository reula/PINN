"""Config.far_weight: a loss weight 1 + (rho/rho_c)^q on the interior equations, on top of the
dimensionless exponents.  Off by default (the residuals are unchanged); on, every residual carries
exactly sqrt(1 + (rho/rho_c)^q)."""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from stationary.geometry import scaled_residuals_batch  # noqa: E402
from stationary.model import Fields  # noqa: E402


def _fields(x):
    # any smooth non-solution, so the residuals are nonzero
    r = jnp.linalg.norm(x)
    h = jnp.eye(3) * (1.0 + 0.1 / r) + 0.01 * jnp.outer(x, x) / r ** 3
    G = jnp.zeros((3, 3, 3)) + 0.001 / r ** 2
    return Fields(h, G, 1.0 - 0.3 / r + 0.02 * x[2] / r ** 2)


def test_far_weight_is_the_stated_factor():
    xs = jnp.array([[1.0, 0.0, 0.0], [0.0, 3.0, 4.0], [10.0, 0.0, 0.0], [0.0, 0.0, 80.0]])
    exps = dict(compat=1.0, ricci=2.0, gauge=1.0, lam_eq=2.0)
    off = scaled_residuals_batch(_fields, xs, exps)
    same = scaled_residuals_batch(_fields, xs, exps, far_weight=None)
    on = scaled_residuals_batch(_fields, xs, exps, far_weight=(10.0, 4.0))
    rho = np.linalg.norm(np.asarray(xs), axis=-1)
    fac = np.sqrt(1.0 + (rho / 10.0) ** 4)
    for k in off:
        np.testing.assert_array_equal(np.asarray(off[k]), np.asarray(same[k]))
        a, b = np.asarray(off[k]), np.asarray(on[k])
        shape = (-1,) + (1,) * (a.ndim - 1)
        np.testing.assert_allclose(b, a * fac.reshape(shape), rtol=1e-12, atol=1e-300)
