"""AxisymHybridNet(reflect=True): z -> -z symmetry by construction, h(Rx) = R h(x) R and
lambda(Rx) = lambda(x) with R = diag(1, 1, -1), for any parameters."""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from stationary.model import AxisymHybridNet  # noqa: E402


def test_reflected_net_is_exactly_symmetric_and_the_plain_one_is_not():
    R = jnp.diag(jnp.array([1.0, 1.0, -1.0]))
    xs = jax.random.normal(jax.random.PRNGKey(0), (16, 3)) * 5.0
    xs = xs * (1.5 / jnp.minimum(jnp.linalg.norm(xs, axis=-1, keepdims=True), 1.5))
    for reflect in (False, True):
        net = AxisymHybridNet(width=12, depth=3, fourier=3, out_std=0.3, reflect=reflect)
        params = net.init(jax.random.PRNGKey(1), xs)
        f = net.apply(params, xs)
        g = net.apply(params, xs @ R)
        dh = float(jnp.max(jnp.abs(jnp.einsum("ij,njk,kl->nil", R, f.h, R) - g.h)))
        dl = float(jnp.max(jnp.abs(f.lam - g.lam)))
        if reflect:
            assert dh < 1e-14 and dl < 1e-14, (dh, dl)
        else:
            assert dh > 1e-6 or dl > 1e-6          # a random plain net is not symmetric


def test_reflect_off_keeps_the_parameter_layout():
    xs = jnp.ones((2, 3))
    a = AxisymHybridNet(width=8, depth=2, fourier=2).init(jax.random.PRNGKey(0), xs)
    b = AxisymHybridNet(width=8, depth=2, fourier=2, reflect=True).init(jax.random.PRNGKey(0), xs)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
