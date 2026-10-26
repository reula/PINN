"""The stage-2 correction network s(t, x) N(t, x, y, z), shared by stage2.py and the evaluation tools.

    s = eps * (t / T) * (1 - r^2 / R^2) * (1 - exp(-r^2 / rc^2))      (last factor only with rc > 0)

N is a tanh MLP with the main script's layout (inputs scaled as there, over the window [0, T]); its
value and derivatives come from the main script's closed-form Taylor pass, minus the flat-space values
that pass adds. s vanishes at t = 0 (keeps the run's initial data), on the outer boundary r = R
(keeps its boundary values) and, with rc > 0, at the origin (leaves the hole to the run).
"""
import numpy as np


def make_correction(ns, T, R, eps, rc=0.0):
    jax, jnp = ns["jax"], ns["jnp"]
    DD = np.asarray(ns["_DD_OUTPUT_IDXS"])
    shift = np.zeros(22)
    for i in (ns["_GT_XX_IDX"], ns["_GT_YY_IDX"], ns["_GT_ZZ_IDX"], ns["ALPHA_IDX"]):
        shift[int(i)] = 1.0
    shift = jnp.asarray(shift)

    def spatial(X):
        """g = (1 - r^2/R^2)(1 - exp(-r^2/rc^2)) with its gradient (n,3) and Hessian (n,3,3)."""
        r2 = jnp.sum(X ** 2, axis=1)
        I = jnp.eye(3)[None]
        w, w1, w2 = 1.0 - r2 / R ** 2, -2.0 * X / R ** 2, -2.0 / R ** 2 * I
        if rc > 0:
            e = jnp.exp(-r2 / rc ** 2)
            c, c1 = 1.0 - e, (2.0 / rc ** 2) * e[:, None] * X
            c2 = e[:, None, None] * (2.0 / rc ** 2 * I - 4.0 / rc ** 4 * X[:, :, None] * X[:, None, :])
        else:
            c, c1, c2 = jnp.ones_like(r2), jnp.zeros_like(X), jnp.zeros_like(w2) * r2[:, None, None]
        g = w * c
        g1 = w1 * c[:, None] + w[:, None] * c1
        g2 = (w2 * c[:, None, None] + w1[:, :, None] * c1[:, None, :] + c1[:, :, None] * w1[:, None, :]
              + w[:, None, None] * c2)
        return g, g1, g2

    def derivs(p2, t, x, y, z):
        """(value, d_t, d_xyz, d_xyz d_xyz of the DD fields) of s * N at (n,1) points."""
        v, nt, nd, ndd = ns["_taylor_mlp"](p2, t, x, y, z, 0.0, T)
        v = v - shift[None, :]
        g, g1, g2 = spatial(jnp.concatenate([x, y, z], axis=1))
        tau = t[:, 0] / T
        s, s_t = eps * tau * g, eps * g / T
        s_k, s_kl = eps * tau[:, None] * g1, eps * tau[:, None, None] * g2
        val = s[:, None] * v
        dt = s_t[:, None] * v + s[:, None] * nt
        d = s_k[:, None, :] * v[:, :, None] + s[:, None, None] * nd
        vD, nD = v[:, DD], nd[:, DD, :]
        dd = (s_kl[:, None] * vD[:, :, None, None] + s_k[:, None, :, None] * nD[:, :, None, :]
              + s_k[:, None, None, :] * nD[:, :, :, None] + s[:, None, None, None] * ndd)
        return val, dt, d, dd

    def value(p2, t, x, y, z):
        return derivs(p2, t, x, y, z)[0]

    return derivs, value


def init_net(jax, jnp, key, hidden, layers):
    """MLP params in the main script's layout; the last layer is zero, so N = 0 at the start."""
    sizes = [4] + [hidden] * layers + [22]
    mlp = {}
    for i in range(len(sizes) - 1):
        key, k = jax.random.split(key)
        kern = jax.nn.initializers.glorot_normal()(k, (sizes[i], sizes[i + 1]), jnp.float64)
        if i == len(sizes) - 2:
            kern = jnp.zeros_like(kern)
        mlp[f"Dense_{i}"] = {"kernel": kern, "bias": jnp.zeros((sizes[i + 1],), jnp.float64)}
    return {"params": {"MLP": mlp}}


def load_correction(ns, path):
    """(value_fn(t, x, y, z) of s * N, config) from a stage2_params.pkl."""
    import pickle
    jax, jnp = ns["jax"], ns["jnp"]
    C = pickle.load(open(path, "rb"))
    p2 = jax.tree_util.tree_map(jnp.asarray, C["params"])
    _, value = make_correction(ns, C["T"], C["R"], C["eps"], C.get("core_taper", 0.0))
    return (lambda t, x, y, z: value(p2, t, x, y, z)), C
