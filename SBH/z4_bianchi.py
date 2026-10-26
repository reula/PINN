"""Covariant 4D machinery for the Z4 constraint-propagation (contracted Bianchi) residual. (2026-10-21)

The Z4 field equations (vacuum, no damping) are
    E_mn = R_mn + nabla_m Z_n + nabla_n Z_m = 0.
For ANY metric g and covector Z, the contracted Bianchi identity nabla^m G_mn = 0 and the commutator
of covariant derivatives give the identity
    nabla^m (E_mn - 1/2 g_mn E) = box Z_n + R_nm Z^m          (E = g^mn E_mn)
so the constraint variables Z obey a linear (in Z) second-order wave equation, sourced only by the
DIVERGENCE of the field-equation residual. `wave_z` computes B_n = box Z_n + R_nm Z^m; as a loss term
it penalises exactly the part of the error that pumps the constraint-violating modes, and vanishes
on the true solution (Z = 0).

All functions are per point and generic: `g_fn(q) -> (4, 4)`, `z_fn(q) -> (4,)` with q = (t, x, y, z);
derivatives by nested forward-mode autodiff. Index convention for derivative arrays: the LAST axes
are the derivative directions (dg[a, b, c] = d_c g_ab).
"""
import jax
import jax.numpy as jnp


def geometry(g_fn, q):
    """(g, ginv, Gamma, dGamma, Ricci) at q; dGamma[l,m,n,r] = d_r Gamma^l_mn."""
    g = g_fn(q)
    dg = jax.jacfwd(g_fn)(q)                         # (4,4,4)
    ddg = jax.jacfwd(jax.jacfwd(g_fn))(q)            # (4,4,4,4): ddg[a,b,c,d] = d_d d_c g_ab
    return geometry_from_derivs(g, dg, ddg)


def geometry_from_derivs(g, dg, ddg):
    """geometry() from given metric derivatives (any dimension): dg[a,b,c] = d_c g_ab,
    ddg[a,b,c,d] = d_d d_c g_ab. (2026-10-26: also used by adm_constraints.py.)"""
    ginv = jnp.linalg.inv(g)
    # T[s,m,n] = d_m g_sn + d_n g_sm - d_s g_mn  and its derivative dT[s,m,n,r]
    T = jnp.einsum("snm->smn", dg) + jnp.einsum("smn->smn", dg) - jnp.einsum("mns->smn", dg)
    dT = (jnp.einsum("snmr->smnr", ddg) + jnp.einsum("smnr->smnr", ddg) - jnp.einsum("mnsr->smnr", ddg))
    Gam = 0.5 * jnp.einsum("ls,smn->lmn", ginv, T)
    dginv = -jnp.einsum("la,abr,bs->lsr", ginv, dg, ginv)                     # d_r g^ls
    dGam = 0.5 * (jnp.einsum("lsr,smn->lmnr", dginv, T) + jnp.einsum("ls,smnr->lmnr", ginv, dT))
    # R_mn = d_l Gamma^l_mn - d_n Gamma^l_ml + Gamma^l_ls Gamma^s_mn - Gamma^l_ns Gamma^s_ml
    Ric = (jnp.einsum("lmnl->mn", dGam) - jnp.einsum("lmln->mn", dGam)
           + jnp.einsum("lls,smn->mn", Gam, Gam) - jnp.einsum("lns,sml->mn", Gam, Gam))
    return g, ginv, Gam, dGam, Ric


def nabla_z(Gam, z, dz):
    """N[m,n] = nabla_m Z_n = d_m Z_n - Gamma^l_mn Z_l; dz[n,m] = d_m Z_n."""
    return dz.T - jnp.einsum("lmn,l->mn", Gam, z)


def wave_z(g_fn, z_fn, q):
    """B_n = box Z_n + R_nm Z^m at q."""
    g, ginv, Gam, dGam, Ric = geometry(g_fn, q)
    z = z_fn(q)
    dz = jax.jacfwd(z_fn)(q)                          # dz[n, r] = d_r Z_n
    ddz = jax.jacfwd(jax.jacfwd(z_fn))(q)             # ddz[n, m, r] = d_r d_m Z_n
    N = nabla_z(Gam, z, dz)                           # N[m, n] = nabla_m Z_n
    # d_r N[m,n] = d_r d_m Z_n - d_r Gamma^l_mn Z_l - Gamma^l_mn d_r Z_l
    dN = (jnp.einsum("nmr->mnr", ddz) - jnp.einsum("lmnr,l->mnr", dGam, z)
          - jnp.einsum("lmn,lr->mnr", Gam, dz))
    # nabla_r N_mn = d_r N_mn - Gamma^l_rm N_ln - Gamma^l_rn N_ml ;  box Z_n = g^rm nabla_r N_mn
    nnN = (jnp.einsum("mnr->rmn", dN) - jnp.einsum("lrm,ln->rmn", Gam, N) - jnp.einsum("lrn,ml->rmn", Gam, N))
    box = jnp.einsum("rm,rmn->n", ginv, nnN)
    return box + Ric @ (ginv @ z)


def z4_tensor(g_fn, z_fn, q):
    """E_mn = R_mn + nabla_m Z_n + nabla_n Z_m at q (the Z4 field equations, vacuum, kappa = 0)."""
    g, ginv, Gam, dGam, Ric = geometry(g_fn, q)
    z = z_fn(q)
    N = nabla_z(Gam, z, jax.jacfwd(z_fn)(q))
    return Ric + N + N.T


def divergence_trace_reversed(g_fn, z_fn, q):
    """nabla^m (E_mn - 1/2 g_mn E) at q, computed independently of wave_z (for the identity test)."""
    def S_fn(qq):
        g = g_fn(qq)
        E = z4_tensor(g_fn, z_fn, qq)
        return E - 0.5 * g * jnp.einsum("mn,mn->", jnp.linalg.inv(g), E)
    g, ginv, Gam, _, _ = geometry(g_fn, q)
    S = S_fn(q)
    dS = jax.jacfwd(S_fn)(q)                          # dS[m,n,r] = d_r S_mn
    # nabla_r S_mn = d_r S_mn - Gamma^l_rm S_ln - Gamma^l_rn S_ml ; divergence g^rm nabla_r S_mn
    nS = (jnp.einsum("mnr->rmn", dS) - jnp.einsum("lrm,ln->rmn", Gam, S) - jnp.einsum("lrn,ml->rmn", Gam, S))
    return jnp.einsum("rm,rmn->n", ginv, nS)
