"""Config.lam_bc_axis: the S1/S2 inner data about an arbitrary axis e.  e = z reproduces the
default; a tilted e gives the default data rotated: lam_e(R x) = lam_z(x) for R z = e."""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from stationary.problem import Config, lam_inner_bc


def test_axis_z_is_the_default_and_a_tilted_axis_rotates_the_data():
    base = dict(lam0=0.3, lam_bc_S1=0.05, lam_bc_S2=-0.08, rho_in=1.0)
    xs = jnp.asarray(np.random.default_rng(0).normal(size=(20, 3)))
    np.testing.assert_allclose(lam_inner_bc(xs, Config(**base, lam_bc_axis=(0.0, 0.0, 2.0))),
                               lam_inner_bc(xs, Config(**base)), rtol=1e-14)
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    R = jnp.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])        # R z = (s, 0, c)
    e = tuple(float(v) for v in R @ jnp.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(lam_inner_bc(xs @ R.T, Config(**base, lam_bc_axis=e)),
                               lam_inner_bc(xs, Config(**base)), rtol=1e-12)
