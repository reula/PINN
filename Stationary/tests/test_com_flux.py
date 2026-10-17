"""Config.com_radii / losses.com_flux: the conserved dipole flux d = D/(4 pi) (COM_GAUGE.md 4b).

0 for the exact (centred) solution on every sphere; for the same solution translated by a -- still a
solution, translations are harmonic -- the same value on every sphere, M a; and the residual vector
still squares to the total loss with the term on."""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from stationary.losses import com_flux, residual_vector, total_loss  # noqa: E402
from stationary.problem import Config  # noqa: E402
from stationary.train import exact_asset, make_batch, make_model  # noqa: E402


def _cfg(**kw):
    base = dict(R0=0.5773502691896258, rho_in=1.0, rho_out=100.0, inner_radius=1.0,
                lam0=0.33333333333333337, lam_bc_S2=-0.08333333333333333, lam_inf=1.0,
                ref_solution=True, ref_asymptotic=1.0, outer_bc="robin",
                com_radii=(1.5, 3.0, 10.0, 40.0))
    base.update(kw)
    return Config(**base)


def test_flux_is_zero_centred_and_conserved_translated():
    cfg = _cfg()
    ex = exact_asset(cfg)
    d0 = np.asarray(com_flux(ex, cfg)).reshape(-1, 3)
    assert np.max(np.abs(d0)) < 1e-10
    a = 0.05
    shifted = lambda x: ex(x - jnp.array([0.0, 0.0, a]))
    d = np.asarray(com_flux(shifted, cfg)).reshape(-1, 3)
    assert np.max(np.abs(d[:, :2])) < 1e-10            # x, y components stay zero
    dz = d[:, 2]
    assert np.ptp(dz) < 1e-3 * abs(dz.mean())            # the same on every sphere
    M = -0.5 * 100.0 ** 2 * float(jax.grad(lambda r: ex(jnp.array([0., 0., r])).lam)(100.0)) * -1
    assert abs(dz.mean() - M * a) < 0.05 * M * a, (dz, M * a)


def test_residual_vector_still_squares_to_the_loss():
    cfg = _cfg(arch="axisym_hybrid", width=8, depth=2, fourier=2, n_coll=64, n_bnd=16,
               n_bnd_outer=16, com_radii=(1.0, 5.0))
    model = make_model(cfg)
    batch = make_batch(jax.random.PRNGKey(0), cfg)
    params = model.init(jax.random.PRNGKey(1), batch["coll"][:2])
    state = {"net": params}
    ex = exact_asset(cfg)
    r = residual_vector(state, batch, cfg, model, ex, lam_inf=jnp.asarray(1.0))
    L, parts = total_loss(state, batch, cfg, model, ex, lam_inf=jnp.asarray(1.0))
    assert "com" in parts
    np.testing.assert_allclose(float(jnp.sum(r ** 2)), float(L), rtol=1e-10)
