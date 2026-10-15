# Perturbation analysis: a quadrupole datum on a spherical background

The runs impose an angular inner datum `lambda(rho_in) = lambda0 + S2 P2(cos theta)` with
`S2 = -1/12 = -0.0833`, on a background that is the spherically symmetric Weyl solution of mass
`M = R0 = 1/sqrt(3) = 0.5774` (`lambda_B = (r-M)/(r+M)`, `lambda0 = 1/3` at the inner sphere).
This note records what the leading response should be, and then what the runs actually show.

## 0. Expansion parameter, and what the expansion can be trusted for

Perturb the logarithm and the metric: `lambda = lambda_B exp(dphi)`, `h = h_B + dh`. At the inner
sphere the datum gives

    dphi(rho_in) = (S2/lambda0) P2 = 0.25 P2,

so the **relative** perturbation is 25 %, not 8 %.  Consequences, stated once and used throughout:

* the *structure* -- which powers of rho appear, in which angular mode -- is exact;
* the *amplitudes* are indicative only: the neglected second-order terms are `O(0.0625)`.

Because the background is spherically symmetric and the chart is harmonic (`x^i = rho n^i`), the
angular functions on a sphere of constant `rho` are exactly `P_l(cos theta)`, and the linearised
equations are diagonal in `l`.  So a pure `l = 2` datum produces, at linear order,

    only l = 2.   No l = 1.  No change in l = 0.

That is the sharpest statement here, and it is testable (section 3).

## 1. Leading terms

    quantity        l = 0                                l = 2                     l = 1
    phi = ln lambda lambda_B, i.e. 1 - 2M/rho + ...      A(rho) P2, A -> a2 rho^-3  0
    lambda          1 - 2M/rho + 2M^2/rho^2 - ...        lambda_B A, -> a2 rho^-3   0
    P (Geroch)      M/rho + M^3/(3 rho^3) + ...          -> a2' rho^-3              0
    h_rr - 1        O(rho^-3)                            -> b2 rho^-3               0
    g2 - 1          -M^2/rho^2 - M^4/rho^4 - ...         -> c2 rho^-3               0

Read off:

1. **The l = 2 response is linear in S2 and decays as rho^-3**, in the scalar *and* in the metric:
   the metric's source is the cross term `d_a lambda_B d_b (d lambda)`, itself `O(S2)`.  A
   quadrupole datum therefore *must* show up in the metric at the few-per-cent level.  What has to
   be checked is not its presence but its **radial power**.
2. **rho^-2 is occupied twice** in lambda (`2M^2`, from the exponential) and in g2 (`-M^2`), and
   also by the `l = 1` dipole.  A radial operator cannot separate angular modes from powers, which
   is why the scalar condition moved to `P` (whose `l = 0` part has no rho^-2 at all) and why the
   metric condition must be left alone.
3. **`l = 1` vanishes at linear order.**  Whatever `l = 1` content a run has is second order in
   `S2` (`S2^2 = 6.9e-3`), a coordinate offset (the origin not at the centre of mass), or
   numerical.
4. **The `l = 0` back-reaction of the quadrupole is `O(S2^2/rho^6)`**: a few 1e-3 at rho = 1, and
   1e-14 at rho = 100.  So the mass and the far-field level are untouched by the quadrupole, and
   the agreement of `lambda(rho_out)` with the spherical value is meaningful.
5. The scalar kernel `{1,3,4}` on `P` annihilates `rho^-1` and `rho^-3`, forces `rho^-2` (the
   dipole) to vanish, and its first unintended victim is `rho^-5`, whose coefficient is
   `O(M^5) = 0.06` -- i.e. `6e-12` at rho = 100.  Below anything measured here.

## 2. The radial equation in the scalar sector

Writing `phi = phi_B + delta` and keeping linear order, the equation `Delta_h phi = 0` becomes, on
the spherical background,

    Delta_{h_B} delta = 0,        delta = sum_l A_l(rho) P_l(cos theta).

The background is static and spherically symmetric, so the angular part is the ordinary sphere
Laplacian and each `l` decouples.  The chart is harmonic but *not* areal-spherical, so the radial
operators are not the flat ones; asymptotically, however, the two solutions of the `l`-th radial
equation behave as `rho^{-(l+1)}` and `rho^{l}` plus subleading powers.  The inner datum fixes the
mixture at `rho_in` and the outer decay condition selects the falling branch, so

    A_l(rho) = a_l rho^{-(l+1)} (1 + O(1/rho^2)),        l = 2: a2 rho^-3.

That `O(1/rho^2)` is the reason the *multipole coefficient* `S_l = rho^{l+1} A_l` is not
rho-independent at moderate rho: it approaches its asymptotic value only as the subleading powers
die away.  **A drift measured over rho = 1 ... 100 is therefore not evidence of anything; it must
be measured in the asymptotic zone.**

## 3. The check, measured

`S_l(rho) = rho^{l+1} * (2l+1)/2 * integral f P_l dmu` on Gauss-Legendre nodes, at rho = 25, 50, 100
so that the subleading powers are small.  A clean multipole has S_l rho-independent; the
*amplitude* is `a_l = S_l / rho^{l+1}`.

    run                                        l=2: S_2 at 25 / 50 / 100          amplitude a_2
    pq_r100_phihyb025_R0   h_rr   -4.59e+02  -3.69e+03  -2.87e+04   drift 98%   ->  -2.9e-02  CONSTANT
    pq_r100_phihyb025_S1   h_rr   -4.64e+02  -3.77e+03  -2.98e+04   drift 98%   ->  -3.0e-02  CONSTANT
    pq_r100_phihyb025_S2   h_rr   -2.49e+02  -1.74e+03  -8.60e+03   drift 97%   ->  -2.9e-02  CONSTANT
    pq_r100_phihyb025_S4   h_rr   -3.18e+01  -8.73e+01  -2.05e+02   drift 84%   ->  -2.0e-03 ... -2.1e-04

The verdict is the opposite of "not settled yet":

* **S1, S2 and R0 carry an `l = 2` metric part that does not decay at all**: `a_2` is `-0.029` at
  rho = 25, 50 *and* 100.  A constant is `rho^0`, which the Robin window `{rho^-2, rho^-3, rho^-4}`
  *forbids*: this is not the physical `rho^-3` response but the forbidden mode, and it is exactly
  the ~2-3 % offset seen on the `h_rho_rho` equator panel.
* **S4 (four shells) does decay**: `a_2 = -2.0e-3, -7.0e-4, -2.1e-4`, i.e. roughly `rho^-1.7` over
  this range.  Not yet `rho^-3`, but the forbidden constant is gone, which is the effect the
  four-shell ladder was supposed to have.  (S4 differs from S2 in two ways -- four shells versus
  three, and hybrid versus log sampling -- so this cannot be credited to the shell count alone.)
* In the scalar sector `P`'s `l = 2` decays as well (`S4`: 1.11, 1.93, 2.84, i.e. `a_2` falling
  roughly as `rho^-2.3`), and `P`'s `l = 1` is at the `1e-4 ... 1e-7` level, i.e. small.

## 4. The scaling test (the prediction this note exists for)

Halve the datum and the linear analysis predicts:

    l = 2 amplitudes:  half            (linear in S2)
    l = 1 amplitudes:  a quarter       (there is no linear l = 1; it can only be O(S2^2))
    l = 0 quantities:  unchanged to ~1e-12

    RHO_OUT=100 TAG=_Q2 SHELLS=1.0,0.9,0.8,0.7 NCOLL=16384 NBND=1024 RADIAL=log \
        EXTRA="--lam-bc-S2 -0.0416666667" bash run_s2.sh

If `l = 1` falls by 4, it is second order and harmless; if it falls by 2, something enters that the
analysis above has missed; if it does not fall at all, it is a coordinate offset and the answer is
the centre-of-mass condition, not another decay kernel.
