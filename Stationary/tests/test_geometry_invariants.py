"""The geometry layer, validated against closed forms rather than against itself.

Three independent checks, in increasing order of what they prove:

 1. flat space             every curvature scalar vanishes, and the sphere machinery returns
                           the elementary values (A = 4 pi rho^2, k = 2/rho, m_H = 0);
 2. Schwarzschild          a metric whose invariants are textbook -- Kretschmann 48 M^2/r_a^6,
                           tidal eigenvalues (-2, 1, 1) M/r_a^3, type D, m_H = M;
 3. the repo's own assets  the spherical asset of exact.py IS Schwarzschild M = R0 in the
                           vacuum reading (this is the test that fixes the reading, see the
                           module docstring), the Weyl asset is Israel-Khan of mass 1.1, and
                           the residual system's own identities hold in the scalar reading.

Everything asserted here holds to the tolerances stated inline; the numbers were checked
interactively before being written down.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from stationary import exact, geometry_invariants as gi
from stationary.weyl import Rods, h_cart, lam_of

M1 = 1.0


# ------------------------------------------------------------------ 1. flat space
def test_flat_space_has_no_curvature():
    h = lambda x: jnp.eye(3)                                    # noqa: E731
    lam = lambda x: 1.0                                         # noqa: E731
    x = jnp.array([1.3, 0.4, -0.7])
    for reading in ("scalar", "vacuum"):
        c = gi.curvature_at(h, lam, x, reading)
        assert abs(float(c["R"])) < 1e-12
        assert float(jnp.linalg.norm(c["Ric"])) < 1e-12
        assert abs(float(c["K"])) < 1e-12
        assert abs(float(c["C2"])) < 1e-12
        assert abs(float(c["CdotC"])) < 1e-12
    s = gi.spatial_at(h, x)
    assert abs(float(s["R"])) < 1e-12
    assert abs(float(s["rank1_defect"])) < 1e-12
    assert abs(float(s["K3"])) < 1e-12
    assert abs(float(s["cotton2"])) < 1e-12


def test_flat_space_sphere_geometry():
    h = lambda x: jnp.eye(3)                                    # noqa: E731
    lam = lambda x: 1.0                                         # noqa: E731
    rho = 2.0
    sc = gi.sphere_geometry(h, lam, rho, "scalar", n_mu=16, n_phi=8)
    assert float(sc["area"]) == pytest.approx(4.0 * jnp.pi * rho**2, rel=2e-6)
    assert float(sc["r_areal"]) == pytest.approx(rho, rel=1e-6)
    assert float(sc["k_mean"]) == pytest.approx(2.0 / rho, rel=1e-5)
    assert abs(float(sc["hawking_mass"])) < 1e-6


# ------------------------------------------------------- 2. Schwarzschild, closed forms
def schwarzschild_asset(M=M1):
    """Schwarzschild in harmonic coordinates: the repo's own spherical asset with R0 = M.

    Verified independently (module docstring, and test_spherical_asset_is_...): in the vacuum
    reading this is Schwarzschild of mass M with areal radius rho + M, so every closed form
    below is textbook.  A hand-written 'harmonic' metric was tried first and was NOT
    Schwarzschild -- h = d rho^2 + (rho+M)^2 dOmega^2 has zero spatial Ricci, so it cannot
    solve the residual system at all; the asset's h = d rho^2 + (rho^2-M^2) dOmega^2 is the
    right one.
    """
    fields = exact.exact_fields(M, k=1.0)
    return (lambda x: fields(x).h), (lambda x: fields(x).lam)


def test_schwarzschild_invariants_are_textbook():
    h, lam = schwarzschild_asset()
    for rho in (1.5, 4.0, 12.0):
        x = jnp.array([rho, 0.0, 0.0])
        ra = rho + M1
        c = gi.curvature_at(h, lam, x, "vacuum")
        assert abs(float(c["R"])) < 1e-11
        assert float(jnp.linalg.norm(c["Ric"])) < 1e-11
        assert float(c["K"]) == pytest.approx(48.0 * M1**2 / ra**6, rel=1e-8)
        assert float(c["C2"]) == pytest.approx(48.0 * M1**2 / ra**6, rel=1e-8)
        assert abs(float(c["CdotC"])) < 1e-14          # static: no Pontryagin density
        assert abs(float(c["H_norm"])) < 1e-12         # purely electric
        assert abs(complex(c["S"]) - 1.0) < 1e-8       # type D
        assert c["petrov"] == "D"
        # tidal eigenvalues (-2, 1, 1) M/r_a^3
        eigs = sorted(float(v) for v in c["E_eigs"])
        assert eigs[0] == pytest.approx(-2.0 * M1 / ra**3, rel=1e-6)
        assert eigs[1] == pytest.approx(M1 / ra**3, rel=1e-6)
        assert eigs[2] == pytest.approx(M1 / ra**3, rel=1e-6)
        # exactly one nonzero NP scalar, |psi_2| = M/r_a^3
        mags = [abs(complex(v)) for v in c["psi"]]
        assert max(mags) == pytest.approx(M1 / ra**3, rel=1e-6)
        assert sorted(mags)[-2] < 1e-9


def test_schwarzschild_hawking_mass_is_M():
    h, lam = schwarzschild_asset()
    for rho in (3.0, 10.0):
        sc = gi.sphere_geometry(h, lam, rho, "vacuum", n_mu=24, n_phi=12)
        assert float(sc["r_areal"]) == pytest.approx(rho + M1, rel=1e-6)
        assert float(sc["hawking_mass"]) == pytest.approx(M1, rel=1e-4)
        assert float(sc["lam_mean"]) == pytest.approx(1.0 - 2.0 * M1 / (rho + M1), rel=1e-6)


# --------------------------------------------- 3. the repo's assets and identities
def test_spherical_asset_is_schwarzschild_of_mass_R0():
    """The finding the module docstring is built on: M = R0, areal radius rho + R0."""
    R0 = 1.0
    fields = exact.exact_fields(R0, k=1.0)
    h = lambda x: fields(x).h                                # noqa: E731
    lam = lambda x: fields(x).lam                            # noqa: E731
    for rho in (1.5, 3.0, 8.0):
        x = jnp.array([rho, 0.0, 0.0])
        ra = rho + R0
        assert float(lam(x)) == pytest.approx(1.0 - 2.0 * R0 / ra, rel=1e-12)
        c = gi.curvature_at(h, lam, x, "vacuum")
        assert float(jnp.linalg.norm(c["Ric"])) < 1e-12
        assert float(c["K"]) == pytest.approx(48.0 * R0**2 / ra**6, rel=1e-8)
        assert abs(float(c["CdotC"])) < 1e-14
        assert abs(complex(c["S"]) - 1.0) < 1e-8


def test_scalar_reading_satisfies_the_residual_identities():
    """In the code's own reading R_ij = (1/2) phi_i phi_j, so R_ij is rank one, R = |grad phi|^2/2,
    and the 3-d identities K3 = 4 R_ij R^ij - R^2 and K3 = 3 R^2 both hold."""
    R0 = 1.0
    fields = exact.exact_fields(R0, k=1.0)
    h = lambda x: fields(x).h                                # noqa: E731
    lam = lambda x: fields(x).lam                            # noqa: E731
    for x in (jnp.array([1.5, 0.0, 0.0]), jnp.array([1.5, 0.7, 0.3])):
        s = gi.spatial_at(h, x)
        # R_ij = (1/2) phi_i phi_j  =>  rank one  =>  R_ij R^ij = R^2, and the 3-d identity
        # R_abcd R^abcd = 4 R_ij R^ij - R^2 then reads K3 = 3 R^2.
        assert abs(float(s["rank1_defect"])) < 1e-10 * max(1.0, float(s["R"]) ** 2)
        assert float(s["RijRij"]) == pytest.approx(float(s["R"]) ** 2, rel=1e-9)
        assert float(s["K3"]) == pytest.approx(
            4.0 * float(s["RijRij"]) - float(s["R"]) ** 2, rel=1e-9)
        assert float(s["K3"]) == pytest.approx(3.0 * float(s["R"]) ** 2, rel=1e-8)


def test_cotton_vanishes_for_the_conformally_flat_spherical_slice():
    """A spherically symmetric 3-metric is conformally flat; the two-rod slice is not."""
    R0 = 1.0
    fields = exact.exact_fields(R0, k=1.0)
    s = gi.spatial_at(lambda x: fields(x).h, jnp.array([1.5, 0.4, 0.2]))
    assert abs(float(s["cotton2"])) < 1e-12
    rods = Rods.pair(1.0, 0.1, 0.5)
    s2 = gi.spatial_at(lambda x: h_cart(x, rods, 400), jnp.array([1.0, 0.3, 0.4]))
    assert float(s2["cotton2"]) > 1e-6


def test_weyl_two_rod_asset_is_vacuum_israel_khan():
    rods = Rods.pair(1.0, 0.1, 0.5)
    h = lambda x: h_cart(x, rods, 400)                       # noqa: E731
    lam = lambda x: lam_of(x, rods)                          # noqa: E731
    for x in (jnp.array([0.5, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.5])):
        c = gi.curvature_at(h, lam, x, "vacuum")
        assert float(jnp.linalg.norm(c["Ric"])) < 1e-12
        assert abs(float(c["R"])) < 1e-14
        assert abs(float(c["CdotC"])) < 1e-13
        assert abs(float(c["H_norm"])) < 1e-12
        assert abs(complex(c["S"]) - 1.0) > 1e-3             # type I, not D
        assert c["petrov"] == "I"
        # NB: psi_1 = psi_3 is NOT asserted to vanish here.  For a static vacuum solution the
        # static tetrad is purely ELECTRIC (H = 0, checked above), but its legs are not the
        # repeated principal null directions of a type I spacetime, so psi_1, psi_3 are
        # generically nonzero -- only for type D (Schwarzschild) does the static tetrad
        # coincide with the principal one and leave psi_2 alone.


def test_near_axis_is_type_D_and_frame_is_well_conditioned():
    """Near the axis an axisymmetric static vacuum solution is type D by symmetry.

    Two things are pinned here.  (1) A real bug: the tangent frame used a fixed seed vector
    (z, then y), which degenerates exactly on the axis, where the outward normal IS z -- e1
    came out None and the cross product raised.  The seed is now the coordinate direction least
    aligned with n.  (2) The physical statement: on the axis the tangential eigenvalues of the
    tidal tensor are equal by axisymmetry, so S = 1, while off the axis S departs from 1 (the
    previous test).  The approach is O(rho^2), which is why the points sit at rho = 0.02.

    EXACTLY on the axis the invariants come back NaN, and that is not this module's bug: the
    Cartesian components of the Weyl metric have direction-dependent second derivatives at
    rho_cyl = 0.  `weyl.curvature_invariant` documents the same limitation and solves it by
    evaluating in the cylindrical chart; the same route is a follow-up here (see the module
    docstring).
    """
    rods = Rods.pair(1.0, 0.1, 0.5)
    h = lambda x: h_cart(x, rods, 400)                       # noqa: E731
    lam = lambda x: lam_of(x, rods)                          # noqa: E731
    for rho, z in ((0.02, 0.0), (0.02, 3.2)):                # in the gap, and beyond both rods
        c = gi.curvature_at(h, lam, jnp.array([rho, 0.0, z]), "vacuum")
        assert float(jnp.linalg.norm(c["Ric"])) < 1e-8
        assert abs(complex(c["S"]) - 1.0) < 1e-5
        assert c["petrov"] == "D"


def test_weyl_two_rod_mass_is_1p1_far_away():
    """The far field of Israel-Khan is Schwarzschild-like with the total mass."""
    rods = Rods.pair(1.0, 0.1, 0.5)
    h = lambda x: h_cart(x, rods, 400)                       # noqa: E731
    lam = lambda x: lam_of(x, rods)                          # noqa: E731
    m = gi.sphere_geometry(h, lam, 40.0, "vacuum", n_mu=48, n_phi=24)["hawking_mass"]
    assert float(m) == pytest.approx(rods.total_mass, rel=2e-2)


# ------------------------------------------------- 4. the report summary function
def test_geometry_report_on_the_exact_asset():
    """With the asset as its own reference: no defects, type D, mass M, zero K-error."""
    M0 = 1.0
    fields = exact.exact_fields(M0, k=1.0)
    h, lam = (lambda x: fields(x).h), (lambda x: fields(x).lam)
    out = gi.geometry_report(h, lam, (h, lam), rho_lo=1.0, rho_hi=8.0, reading="vacuum",
                             n_dir=8, n_mu=12, n_phi=6)
    assert out["geom_n_used"] == out["geom_n_points"] == 24
    assert out["geom_vacuum_defect_median"] < 1e-12
    assert out["geom_pontryagin_max"] < 1e-15
    assert out["geom_rank1_defect_median"] < 1e-12
    assert out["geom_eigenvalue_defect_median"] < 1e-12
    assert out["geom_type_D_fraction"] == 1.0
    assert out["geom_kretschmann_error_median"] < 1e-14
    for row in out["geom_spheres"]:                 # rho, m_H, r_areal, m_lapse
        rho, m_h, r_a, m_lapse = row
        assert m_h == pytest.approx(M0, rel=1e-4)
        assert m_lapse == pytest.approx(M0, rel=1e-4)
        assert r_a == pytest.approx(rho + M0, rel=1e-5)
    # the reference profile is computed at the outer radius only, for cost
    assert out["geom_m_h_ref_outer"] == pytest.approx(M0, rel=1e-4)
    assert out["geom_r_areal_ref_outer"] == pytest.approx(
        out["geom_spheres"][-1][0] + M0, rel=1e-5)
    assert "    reading vacuum" in gi.format_geometry(out)[0]


def test_geometry_report_separates_the_two_identities():
    """The rank-one defect constrains h alone, the eigenvalue defect needs lambda too.

    Perturbing lambda (by a position-dependent factor, so it is not the lambda -> c lambda
    symmetry of the system) must move the eigenvalue and vacuum defects and must NOT move the
    rank-one one.  This is the reason both are reported.
    """
    fields = exact.exact_fields(1.0, k=1.0)
    h = lambda x: fields(x).h                                # noqa: E731
    lam = lambda x: fields(x).lam                            # noqa: E731
    lam_bad = lambda x: lam(x) * (1.0 + 0.01 * jnp.linalg.norm(x))   # noqa: E731
    good = gi.geometry_report(h, lam, None, 1.0, 8.0, n_dir=8, n_mu=12, n_phi=6)
    bad = gi.geometry_report(h, lam_bad, None, 1.0, 8.0, n_dir=8, n_mu=12, n_phi=6)
    assert good["geom_rank1_defect_median"] < 1e-12
    assert bad["geom_rank1_defect_median"] < 1e-12            # h untouched
    assert good["geom_eigenvalue_defect_median"] < 1e-12
    assert bad["geom_eigenvalue_defect_median"] > 1e-3        # lambda broken
    assert bad["geom_vacuum_defect_median"] > 1e-2
    assert "geom_kretschmann_error_median" not in bad         # no reference given


# --------------------------------------------------- 5. the Hawking mass profile
def test_hawking_profile_is_flat_for_schwarzschild():
    """m_H = M at every radius, and m_lapse = r_a(1-lambda)/2 agrees with it here.

    The Gauss-Legendre rule was introduced for this: with the midpoint rule in mu the mass at
    n_mu = 10 was 0.2% high, which was larger than the error the profile is meant to measure.
    """
    M0 = 1.0
    fields = exact.exact_fields(M0, k=1.0)
    h, lam = (lambda x: fields(x).h), (lambda x: fields(x).lam)
    for n_mu, n_phi in ((6, 4), (12, 8)):
        p = gi.hawking_profile(h, lam, [1.5, 3.0, 8.0], "vacuum", n_mu=n_mu, n_phi=n_phi)
        for i, rho in enumerate(p["radii"]):
            assert float(p["m_h"][i]) == pytest.approx(M0, rel=1e-11)
            assert float(p["m_lapse"][i]) == pytest.approx(M0, rel=1e-11)
            assert float(p["r_areal"][i]) == pytest.approx(float(rho) + M0, rel=1e-11)


def test_hawking_profile_of_flat_space_is_zero():
    p = gi.hawking_profile(lambda x: jnp.eye(3), lambda x: 1.0, [1.0, 2.0, 5.0], "scalar",
                           n_mu=8, n_phi=6)
    for i in range(3):
        assert abs(float(p["m_h"][i])) < 1e-12


def test_hawking_profile_approaches_the_total_mass():
    """Israel-Khan: m_H rises with radius and converges to the sum of the rod masses (1.1).

    It approaches from BELOW because a sphere of radius rho only encloses the rods once it is
    outside them (the upper rod ends at 2.5), and it is flat at 1.1 once it does -- which is
    the statement that all the mass is inside.
    """
    rods = Rods.pair(1.0, 0.1, 0.5)
    h = lambda x: h_cart(x, rods, 400)                       # noqa: E731
    lam = lambda x: lam_of(x, rods)                          # noqa: E731
    p = gi.hawking_profile(h, lam, [1.0, 2.0, 5.0, 20.0, 80.0], "vacuum", n_mu=16, n_phi=10)
    m = [float(v) for v in p["m_h"]]
    assert all(a < b for a, b in zip(m, m[1:]))              # monotone increasing
    assert m[-1] == pytest.approx(rods.total_mass, rel=1e-5)
    assert m[-2] == pytest.approx(rods.total_mass, rel=1e-3)
    assert all(v <= rods.total_mass * (1.0 + 1e-9) for v in m)


def test_hawking_mass_field_is_the_profile_at_the_node_radii():
    """The 3-d field written into the VTK is m_H(|x|), exact at the levels it is built on."""
    M0 = 1.0
    fields = exact.exact_fields(M0, k=1.0)
    h, lam = (lambda x: fields(x).h), (lambda x: fields(x).lam)
    levels = jnp.linspace(1.5, 8.0, 6)
    xs = jnp.stack([levels, jnp.zeros_like(levels), jnp.zeros_like(levels)], axis=-1)
    out = gi.hawking_mass_field(h, lam, xs, "vacuum", n_radii=6, n_mu=10, n_phi=8)
    vals = out["hawking_mass"]
    assert vals.shape == (6,)
    for v in vals:
        assert float(v) == pytest.approx(M0, rel=1e-10)


# ------------------------------------------------- 6. the figure machinery (no plotting)
def test_horizon_table_is_the_exact_schwarzschild_geometry():
    """A rod of mass m is a horizon of area 16 pi m^2 and surface gravity 1/(4m)."""
    from stationary.geometry_figures import horizon_table
    from stationary.problem import Config

    cfg = Config(weyl=True, weyl_half_length=1.0, weyl_half_length_b=0.1, weyl_half_gap=0.5)
    ht = horizon_table(cfg)
    assert len(ht) == 2
    for m, span, area, kappa in ht:
        assert area == pytest.approx(16.0 * jnp.pi * m**2, rel=1e-12)
        assert kappa == pytest.approx(1.0 / (4.0 * m), rel=1e-12)
        assert span[0] < span[1]
    # unequal masses: the two spans are NOT mirror images about z = 0
    assert abs(ht[0][1][0] + ht[1][1][1]) > 1e-6


def test_slice_grid_lies_in_the_shell_and_avoids_the_axis():
    from stationary.geometry_figures import slice_grid
    from stationary.problem import Config

    cfg = Config(weyl=True, weyl_half_length=0.028571428571,
                 weyl_half_length_b=0.002857142857, weyl_half_gap=0.014285714286,
                 rho_in=0.1, rho_out=1.0)
    R, Z, ys, mask = slice_grid(cfg, n_r=6, n_theta=8)
    assert R.shape == Z.shape == mask.shape == (6, 8)
    assert np.all(mask)                                 # shell-conforming: nothing to mask
    assert float(ys[..., 0].min()) >= 0.1 * cfg.rho_in  # floored off the axis
    r = np.linalg.norm(np.asarray(ys)[..., [0, 2]], axis=-1)
    levels = np.unique(np.round(r, 12))
    assert levels.size == 6                             # every node on one of the 6 levels
    assert np.all((r >= cfg.rho_in - 1e-12) & (r <= cfg.rho_out + 1e-12))
    # radial levels geometric, which is what makes the inner sphere resolved
    assert np.allclose(np.diff(np.log(levels)), np.diff(np.log(levels))[0], rtol=1e-9)


# ------------------------------------------- 7. the two evaluation routes and the frame
def test_frames_agree_off_axis():
    """The coordinate and orthonormal contractions are the same scalars, to round-off.

    This is the check that makes the orthonormal option trustworthy: it changes nothing
    analytically (frame invariance) and everything numerically (no 1/rho^2 cancellation).
    """
    fields = exact.exact_fields(1.0, k=1.0)
    h, lam = (lambda x: fields(x).h), (lambda x: fields(x).lam)
    for x in (jnp.array([1.5, 0.7, 0.3]), jnp.array([3.0, 0.2, -0.4])):
        a = gi.curvature_at(h, lam, x, "vacuum")
        b = gi.curvature_at(h, lam, x, "vacuum", frame="orthonormal")
        assert float(b["K"]) == pytest.approx(float(a["K"]), rel=1e-11)
        assert float(b["C2"]) == pytest.approx(float(a["C2"]), rel=1e-11)
        assert abs(float(b["CdotC"])) < 1e-13
        assert complex(b["S"]) == pytest.approx(complex(a["S"]), rel=1e-6)
        for p, q in zip(a["E_eigs"], b["E_eigs"]):
            assert float(p) == pytest.approx(float(q), rel=1e-10)


def test_cylindrical_route_agrees_with_cartesian_off_axis():
    """Same scalar, two charts: the cylindrical route is the one that survives near the axis."""
    fields = exact.exact_fields(1.0, k=1.0)
    h, lam = (lambda x: fields(x).h), (lambda x: fields(x).lam)
    axis = jnp.array([0.0, 0.0, 1.0])
    for rho, z in ((1.5, 0.3), (1.2, -0.7)):          # |x| > R0 = 1: a Riemannian view
        x = rho * jnp.array([1.0, 0.0, 0.0]) + z * axis
        cart = float(gi.curvature_at(h, lam, x, "vacuum")["K"])
        cyl = float(gi.curvature_at_cylindrical(h, lam, x, "vacuum", axis=axis)["K"])
        assert cyl == pytest.approx(cart, rel=1e-6)
    # and the exact chart metric is smooth where the Cartesian one is not: a REGULAR axis
    # point (beyond the rod ends) converges instead of blowing up
    from stationary.geometry_figures import exact_chart_fields
    from stationary.problem import Config
    cfg = Config(weyl=True, weyl_half_length=0.028571428571,
                 weyl_half_length_b=0.002857142857, weyl_half_gap=0.014285714286,
                 rho_in=0.1, rho_out=1.0)
    hc, lc = exact_chart_fields(cfg, n_quad=160)
    # the chart metric is evaluated at chart coordinates directly.  Two things to pin: the
    # orthonormal contraction agrees with the coordinate one wherever the latter is still
    # accurate (it is a frame change, not a different answer), and the values stay finite and
    # smooth as rho -> 0 instead of diverging.
    pts = [jnp.array([r, 0.0, 0.25]) for r in (3e-2, 1e-2, 3e-3)]
    ks = [float(gi.curvature_at(hc, lc, y, "vacuum", frame="orthonormal")["K"]) for y in pts]
    kc = [float(gi.curvature_at(hc, lc, y, "vacuum")["K"]) for y in pts]
    assert all(np.isfinite(ks))
    for a, b in zip(ks, kc):
        assert a == pytest.approx(b, rel=1e-8)
    assert max(ks) / min(ks) < 1.1                      # smooth in rho, no divergence
