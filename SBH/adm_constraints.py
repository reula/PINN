"""The ADM Hamiltonian and momentum constraints of the network's 3+1 data (2026-10-26).

    H   = R + K^2 - K_ij K^ij
    M_i = D_j K^j_i - d_i K

with gamma_ij = gt_ij / chi, K_ij = (At_ij + gt_ij K / 3) / chi, K = Khat + 2 Theta (the CCZ4 variables)
and K in H, M taken as the trace gamma^ij K_ij. Both vanish for any vacuum solution. They are NOT the
script's `hamiltonian_constraint` / `gauge_source_vector`, which return the Z4 variables Theta and Z^i
(kept under those inherited names); with Theta = Z = 0 a network can still violate H and M (the
outgoing mass deficit, README).

Everything is computed from the derivative stack the CCZ4 residual uses -- value (22,), d_t (22,),
d_xyz (22,3), d_xyz d_xyz of the _DD_OUTPUT_IDXS fields (11,3,3) -- so it works with the closed-form
Taylor pass, with autodiff, and with summed stacks (stage2.py). `ns` is the script's namespace
(tools/load_defs.py, or globals() inside the script).
"""
import jax
import jax.numpy as jnp
import numpy as np

import z4_bianchi as _zb


def _unpack_last(ns, a):
    """(6, ...) packed symmetric components on axis 0 -> (3, 3, ...)."""
    return jnp.moveaxis(ns["unpack_sym3"](jnp.moveaxis(a, 0, -1)), (-2, -1), (0, 1))


def constraints_one(ns, val, d, dd):
    """(H, M_i) at one point from its derivative stack."""
    GT, AT = ns["GT_SLICE"], ns["AT_SLICE"]
    CHI, KH, TH = ns["CHI_IDX"], ns["KHAT_IDX"], ns["THETA_IDX"]
    dd_idx = list(np.asarray(ns["_DD_OUTPUT_IDXS"]))
    i_gt = [dd_idx.index(k) for k in range(GT.start, GT.stop)]
    i_chi = dd_idx.index(CHI)
    gt = ns["unpack_sym3"](val[GT])
    At = ns["unpack_sym3"](val[AT])
    chi = val[CHI]
    dgt = _unpack_last(ns, d[GT])                                   # [a,b,c] = d_c gt_ab
    ddgt = _unpack_last(ns, dd[jnp.asarray(i_gt)])                  # [a,b,c,e] = d_c d_e gt_ab
    dchi, ddchi = d[CHI], dd[i_chi]
    dAt = _unpack_last(ns, d[AT])
    # physical metric and its derivatives
    g = gt / chi
    dg = dgt / chi - gt[:, :, None] * dchi[None, None, :] / chi ** 2
    ddg = (ddgt / chi
           - (dgt[:, :, :, None] * dchi[None, None, None, :] + dgt[:, :, None, :] * dchi[None, None, :, None]) / chi ** 2
           - gt[:, :, None, None] * ddchi[None, None] / chi ** 2
           + 2.0 * gt[:, :, None, None] * dchi[None, None, :, None] * dchi[None, None, None, :] / chi ** 3)
    _, gi, Gam, _, Ric = _zb.geometry_from_derivs(g, dg, ddg)
    # extrinsic curvature and its derivatives
    Kc = val[KH] + 2.0 * val[TH]
    dKc = d[KH] + 2.0 * d[TH]
    Kij = (At + gt * Kc / 3.0) / chi
    dK = ((dAt + dgt * Kc / 3.0 + gt[:, :, None] * dKc[None, None, :] / 3.0) / chi
          - (At + gt * Kc / 3.0)[:, :, None] * dchi[None, None, :] / chi ** 2)     # [a,b,c] = d_c K_ab
    trK = jnp.einsum("ij,ij->", gi, Kij)
    Kup = gi @ Kij @ gi
    H = jnp.einsum("ij,ij->", gi, Ric) + trK ** 2 - jnp.einsum("ij,ij->", Kij, Kup)
    dgi = -jnp.einsum("ia,abc,bj->ijc", gi, dg, gi)
    dtrK = jnp.einsum("ijc,ij->c", dgi, Kij) + jnp.einsum("ij,ijc->c", gi, dK)
    # D_j K_ki = d_j K_ki - Gamma^l_jk K_li - Gamma^l_ji K_kl ; M_i = gamma^jk D_j K_ki - d_i trK
    DK = (jnp.einsum("kij->jki", dK) - jnp.einsum("ljk,li->jki", Gam, Kij) - jnp.einsum("lji,kl->jki", Gam, Kij))
    M = jnp.einsum("jk,jki->i", gi, DK) - dtrK
    return H, M


def constraints_from_derivs(ns, val, dt, d, dd):
    """(H (N,), M (N,3)) from a batched derivative stack."""
    return jax.vmap(lambda v, a, b: constraints_one(ns, v, a, b))(val, d, dd)


def derivative_stack(ns, apply_fn, params, t, x, y, z, window_start, window_end):
    """The residual's derivative stack at (N,1) points: closed-form Taylor when the script would use it,
    otherwise autodiff (value, d_t, d_xyz of all 22 outputs, d_xyz d_xyz of the _DD_OUTPUT_IDXS ones)."""
    if ns["deriv_mode"] == "taylor" and apply_fn in ns["_TAYLOR_KIND"]:
        return ns["_taylor_derivs"](apply_fn, params, t, x, y, z, window_start, window_end)
    DD = ns["_DD_OUTPUT_IDXS"]

    def one(q):
        f = lambda qq: ns["point_output_from_q"](apply_fn, params, qq, window_start, window_end)
        fx = lambda p: f(jnp.concatenate([q[0:1], p]))
        return (f(q), jax.jvp(f, (q,), (jnp.array([1.0, 0.0, 0.0, 0.0]),))[1], jax.jacfwd(fx)(q[1:]),
                jax.jacfwd(jax.jacfwd(lambda p: fx(p)[DD]))(q[1:]))
    return jax.vmap(one)(jnp.concatenate([t, x, y, z], axis=1))


def adm_constraints(ns, apply_fn, params, t, x, y, z, window_start, window_end, with_chi=False):
    """(H (N,), M (N,3)) of the network at (N,1) points; with_chi: also chi (N,), for weights."""
    stack = derivative_stack(ns, apply_fn, params, t, x, y, z, window_start, window_end)
    H, M = constraints_from_derivs(ns, *stack)
    return (H, M, stack[0][:, ns["CHI_IDX"]]) if with_chi else (H, M)


def adm_penalty_from(H, M, chi, chi_power):
    """mean(chi^p (H^2 + |M|^2)): chi^p fades out the puncture's singular region (chi -> 0 there)."""
    return jnp.mean(chi ** chi_power * (H ** 2 + jnp.sum(M ** 2, axis=1)))
