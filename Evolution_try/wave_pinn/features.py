"""Input feature maps for the network, including the periodic (cos/sin) embedding.

Why a feature map at all
------------------------
The boundary condition is periodic, and the network part of the ansatz must obey
it *exactly*.  Feeding the network ``cos(pi x / L)`` and ``sin(pi x / L)`` does
that: both are 2L-periodic, and so is any function the network builds out of
them.  With ``features="plain"`` periodicity would only be approximate and would
have to be bought with a penalty term.

The available maps
------------------
``periodic``     ``[t/T, cos(pi x/L), sin(pi x/L)]`` -- the default.  A *plain*
                 network on the two features that carry periodicity, at unit
                 amplitude, with no frequency expansion of any kind.
``periodic_ic``  the above plus the hard-coded initial data ``u0(x)``, ``v0(x)``.
``fourier``      ``[t/T, cos(n pi x/L), sin(n pi x/L) for n = 1..K]``: a
                 band-limited Fourier series.  Kept because it is the obvious
                 alternative, but the sweep in the README shows it is *worse*
                 here at equal budget the more harmonics it is given.
``fourier_ic``   the above plus ``u0``, ``v0``.
``plain``        ``[t/T, x/L]`` -- not periodic; only for diagnostics.
``plain_ic``     the above plus ``u0``, ``v0``.

Why no frequency expansion is needed
------------------------------------
``x -> (cos(pi x/L), sin(pi x/L))`` maps ``[-L, L]`` one-to-one onto the unit
circle, so a *nonlinear* network fed those two numbers can represent **any**
2L-periodic function of ``x`` -- including the width-0.2 Gaussian pulse.  Adding
high harmonics does not add representational power; it adds frequencies the
optimiser must fit, and it makes the landscape worse.  A single unit-amplitude
sine/cosine pair is therefore both the minimal and the best-conditioned way to
make the network periodic.

The ``*_ic`` variants additionally hand the network ``u0(x)`` and ``v0(x)`` as
inputs.  That removes the need to reconstruct the narrow profile from scratch --
the exact remainder is ``u0(x)`` times a smooth function -- but it also feeds
the network a sharp input, and in the sweep it trained *worse*, not better.  It
is kept as an option, not used as the default.
"""

from __future__ import annotations

import jax.numpy as jnp

from .config import Config


def has_ic_features(cfg: Config) -> bool:
    return cfg.features.endswith("_ic")


def feature_dim(cfg: Config) -> int:
    """Number of columns produced by :func:`features`."""
    if cfg.features.startswith("periodic"):
        d = 3
    elif cfg.features.startswith("fourier"):
        d = 1 + 2 * cfg.n_modes
    elif cfg.features.startswith("plain"):
        d = 2
    else:
        raise ValueError(f"unknown feature map {cfg.features!r}")
    if has_ic_features(cfg):
        d += 2
    return d


def features(cfg: Config, t, x, u0=None, v0=None):
    """Build the feature matrix for the batches ``t``, ``x`` (any broadcast shape).

    ``u0``/``v0`` are only needed (and only used) by the ``*_ic`` maps; passing
    them in avoids differentiating the profile twice.
    """
    t_hat = t / cfg.T
    x_hat = x / cfg.L
    cols = [t_hat]
    if cfg.features.startswith("periodic"):
        k = jnp.pi / cfg.L
        cols.append(jnp.cos(k * x))
        cols.append(jnp.sin(k * x))
    elif cfg.features.startswith("fourier"):
        for n in range(1, cfg.n_modes + 1):
            k = n * jnp.pi / cfg.L
            cols.append(jnp.cos(k * x))
            cols.append(jnp.sin(k * x))
    elif cfg.features.startswith("plain"):
        cols.append(x_hat)
    else:
        raise ValueError(f"unknown feature map {cfg.features!r}")

    if has_ic_features(cfg):
        if u0 is None or v0 is None:
            raise ValueError(f"features={cfg.features!r} needs u0 and v0")
        cols.append(u0)
        cols.append(v0)

    return jnp.stack(cols, axis=-1)
