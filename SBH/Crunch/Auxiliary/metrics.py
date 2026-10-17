import jax.numpy as jnp
import numpy as np

def relative_l2(pred, ref):
    return float(jnp.linalg.norm(pred - ref) / (jnp.linalg.norm(ref) + 1e-12))

def mean_absolute_error(pred, ref):
    return float(jnp.mean(jnp.abs(pred - ref)))
