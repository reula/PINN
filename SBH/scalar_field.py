"""Massless scalar field minimally coupled to the CCZ4 system of EEccz4_BH_puncture.py. (2026-11-10)

The user (2026-11-10): a black hole's mass grows when radiation falls in, so the template's mass and the size of
its excised core must be able to change; a scalar field is the way to test that in spherical symmetry.

Variables: phi and Pi = n^a d_a phi = (d_t phi - beta^i d_i phi) / alpha  (n^a the unit normal of the slices).

    T_ab  = d_a phi d_b phi - (1/2) g_ab (d phi)^2,          (d phi)^2 = -Pi^2 + gamma^ij d_i phi d_j phi
    rho   = n^a n^b T_ab         = (1/2) (Pi^2 + gamma^ij d_i phi d_j phi)
    S_i   = -gamma_i^a n^b T_ab  = -Pi d_i phi
    S_ij  = gamma gamma T        = d_i phi d_j phi + (1/2) gamma_ij (Pi^2 - gamma^kl d_k phi d_l phi)

    d_t phi = beta^i d_i phi + alpha Pi
    d_t Pi  = beta^i d_i Pi + alpha D^2 phi + gamma^ij d_i alpha d_j phi + alpha K Pi,        K = Khat + 2 Theta
    D^2 phi = chi gt^ij d_i d_j phi - chi Gt^k d_k phi - (1/2) gt^ij d_i chi d_j phi           (as lap_alpha there)

(rho, S_i, S_ij) are what `_ccz4_residual_one(..., matter=(tau, S_i, S_ij))` takes (tau = rho). The constraints
with matter are H - 16 pi rho = 0 and M_i - 8 pi S_i = 0 with (H, M_i) of adm_constraints.constraints_one.

Stacks: `val`, `d` are the 22-field value and gradient stacks of the BH script ((22,), (22, 3)); `sv` = (phi, Pi),
`sd` (2, 3) their gradients, `phi_dd` (3, 3) the Hessian of phi.
"""
import jax.numpy as jnp


def _metric(ns, val, d):
    gt = ns["unpack_sym3"](val[ns["GT_SLICE"]])
    gt_d = ns["_unpack_sym3_leading"](d[ns["GT_SLICE"]])
    _, _, Gt_upper, gt_inv = ns["_christoffel3"](gt, gt_d)
    return gt, gt_inv, Gt_upper, ns["_chi_safe"](val[ns["CHI_IDX"]])


def matter_one(ns, val, d, sv, sd):
    """(rho, S_i, S_ij) at one point."""
    gt, gt_inv, _, chi = _metric(ns, val, d)
    dphi, Pi = sd[0], sv[1]
    D2 = chi * jnp.einsum("ij,i,j->", gt_inv, dphi, dphi)
    rho = 0.5 * (Pi ** 2 + D2)
    return rho, -Pi * dphi, jnp.outer(dphi, dphi) + 0.5 * (gt / chi) * (Pi ** 2 - D2)


def kg_rhs_one(ns, val, d, sv, sd, phi_dd):
    """(d_t phi, d_t Pi) at one point."""
    _, gt_inv, Gt_upper, chi_s = _metric(ns, val, d)
    chi, chi_d = val[ns["CHI_IDX"]], d[ns["CHI_IDX"]]
    alpha, alpha_d = val[ns["ALPHA_IDX"]], d[ns["ALPHA_IDX"]]
    beta = val[ns["BETA_SLICE"]]
    K = val[ns["KHAT_IDX"]] + 2.0 * val[ns["THETA_IDX"]]
    dphi, dPi, Pi = sd[0], sd[1], sv[1]
    lap = (chi * jnp.einsum("ij,ij->", gt_inv, phi_dd) - chi * jnp.dot(Gt_upper, dphi)
           - 0.5 * jnp.einsum("ij,i,j->", gt_inv, chi_d, dphi))
    rhs_phi = jnp.dot(beta, dphi) + alpha * Pi
    rhs_Pi = (jnp.dot(beta, dPi) + alpha * lap + chi * jnp.einsum("ij,i,j->", gt_inv, alpha_d, dphi)
              + alpha * K * Pi)
    return rhs_phi, rhs_Pi
