"""Check the spherical reduction used by sph_reference.py.

Generic smooth spherically symmetric fields (Z, Theta, beta, Khat != 0) are
evaluated two ways with the BH code's r_ccz4_evolution:
  (3D)   the exact Cartesian fields, at random points off the axis;
  (1D)   the radial Taylor model of sph_reference.py with EXACT radial
         derivatives, on the x-axis at the same radius,
and the 1D radial RHS, mapped back to Cartesian components at the 3D point,
must equal the 3D RHS. Expected: ~1e-14 (round-off). This tests the
cartoon reduction and the field layout, not the CCZ4 equations themselves
(tests/test_random.py does that).

    JAX_PLATFORMS=cpu python3 test_sph_reduction.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = [sys.argv[0], "--kappa_z", "0.3", "--kappa_c", "0.2", "--kappa_2", "0.1",
            "--f_alpha", "1.3", "--f_beta", "0.8", "--eta", "0.7", "--chi_floor", "1e-12"]
import sph_reference as S

args = S.parse()
ns, jax, jnp, apply_fn, r_ccz4 = S.build(args)
fields_from_radial = None
pack = ns["pack_sym3"]

# generic radial functions (smooth, nonzero derivatives; b, d ~ r^2 near 0)
def rad(r):
    return jnp.stack([
        1.0 + 0.1 * jnp.sin(1.3 * r),            # a
        0.05 * r**2 / (1 + r**2),                # b
        0.07 * jnp.cos(0.9 * r),                 # c
        0.04 * r**2 * jnp.exp(-0.3 * r),         # d
        0.6 + 0.2 * jnp.tanh(r - 1.0),           # chi
        0.15 * jnp.exp(-r),                      # Khat
        0.1 * jnp.sin(r),                        # Theta
        0.2 * r * jnp.exp(-0.5 * r),             # g (odd)
        0.7 + 0.2 * jnp.cos(0.7 * r),            # alpha
        0.12 * r / (1 + r**2),                   # B (odd)
    ])

def cart(rv, xyz):
    a, b, c, d, chi, Kh, Th, g, al, B = rv
    r = jnp.sqrt(jnp.sum(xyz**2)); n = xyz / r; nn = jnp.outer(n, n); I = jnp.eye(3)
    return jnp.concatenate([pack((a * I + b * nn)[None])[0], pack((c * I + d * nn)[None])[0],
                            jnp.stack([chi, Kh, Th]), g * n, al[None], B * n])

def apply_exact(params, t, x, y, z, ws, we):
    xyz = jnp.concatenate([x, y, z], axis=1)
    return jax.vmap(lambda p: cart(rad(jnp.sqrt(jnp.sum(p**2))), p))(xyz)

rng = np.random.default_rng(1)
P = rng.normal(size=(40, 3)); P *= (rng.uniform(0.3, 4.0, 40) / np.linalg.norm(P, axis=1))[:, None]
rr = np.linalg.norm(P, axis=1)
z = jnp.zeros((40, 1))
res3 = r_ccz4(apply_exact, None, z, jnp.asarray(P[:, :1]), jnp.asarray(P[:, 1:2]), jnp.asarray(P[:, 2:3]), 0.0, 1.0)

# 1D: Taylor model with exact derivatives, one point per radius
d1 = jax.vmap(jax.jacfwd(rad))(jnp.asarray(rr))
d2 = jax.vmap(jax.jacfwd(jax.jacfwd(rad)))(jnp.asarray(rr))
v0 = jax.vmap(rad)(jnp.asarray(rr))
R1 = []
for k in range(40):     # one radius at a time (the model looks up the nearest grid point)
    coef = jnp.stack([v0[k], d1[k], d2[k]])[None]          # (1,3,10)
    res = r_ccz4(apply_fn, (coef, float(rr[k]), 1.0), z[:1], jnp.asarray([[rr[k]]]), z[:1], z[:1], 0.0, 1.0)
    R = -res[0]
    R1.append(jnp.stack([R[3], R[0] - R[3], R[9], R[6] - R[9], R[12], R[13], R[14], R[15], R[18], R[19]]))
R1 = jnp.stack(R1)
pred = jax.vmap(cart)(R1, jnp.asarray(P))       # radial RHS mapped to Cartesian at the 3D point
err = np.max(np.abs(np.asarray(pred) - np.asarray(-res3)), axis=0)
scale = np.max(np.abs(np.asarray(res3)), axis=0)
names = ["gt"] * 6 + ["At"] * 6 + ["chi", "Khat", "Theta"] + ["Ghat"] * 3 + ["alpha"] + ["beta"] * 3
for nm in dict.fromkeys(names):
    idx = [i for i, n in enumerate(names) if n == nm]
    print(f"  {nm:6s} max|RHS|={scale[idx].max():.3e}   max|1D - 3D|={err[idx].max():.3e}")
worst = err.max()
print("PASS" if worst < 1e-12 else "FAIL", f"(worst {worst:.2e})")
