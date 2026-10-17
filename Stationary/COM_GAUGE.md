# The tail dipole as an asymptotic translation, and the center of mass that fixes it

*Working hypothesis*: the `l = 1` ("dipole") coefficient we keep finding in the tail of `lambda` is
not a physical multipole.  It is the coordinate image of the freedom to **translate the asymptotic
chart** — the same freedom that makes the `l = 1` coefficient of an asymptotically flat metric
non-invariant — and the invariant object that replaces it is the **center of mass**.  Pinning the
center of mass (or, equivalently, translating the chart so that it sits at the origin) is what
kills the dipole.  This note collects what the literature says, states why the mechanism fits our
runs, and gives the two-step plan: a diagnostic that discriminates gauge from physics, and the gauge
pin to implement afterwards.

## 1. The gauge statement, in the notation we use

For an asymptotically Euclidean initial data set `(M^3, g, K, mu, J)` one defines the ADM energy and
linear momentum as flux integrals at infinity,

    E = (1/16 pi) lim_{r->inf} ∮_{|x|=r} sum_ij (d_i g_ij - d_j g_ii) x^j/r  dmu_delta
    P^j = (1/8 pi) lim_{r->inf} ∮_{|x|=r} sum_i pi_ij x^i/r dmu_delta ,      pi = (tr K) g - K

and the mass `m = sqrt(E^2 - |P|^2)` — this is the setup of Cederbaum–Sakovich
([arXiv:1901.00028](https://ar5iv.labs.arxiv.org/html/1901.00028), Eqs. (3)–(5)).  The center of
mass is *not* of the same kind: the natural flux-integral definition, due to Beig and Ó Murchadha
(and in parts Regge and Teitelboim), is

    C^l_BOM = (1/16 pi E) lim_{r->inf} ∮_{|x|=r} [ x^l sum_ij (d_i g_ij - d_j g_ii) x^j/r
                                                  - sum_i ( g_il x^i/r - g_ii x^l/r ) ] dmu_delta

(Cederbaum–Sakovich Eq. (6)).  Two properties are the whole point here:

* **Under a translation the center of mass shifts with the chart**, while the `l = 1` coefficient of
  the metric does not stay put.  Cederbaum–Sakovich state the covariant version of this:
  their STCMC center of mass "transforms **equivariantly** under the asymptotic Poincaré group of
  the ambient spacetime and in particular evolves under the Einstein evolution equations like a
  point particle in Special Relativity".  Equivariance is exactly `C -> C + a` under `x -> x + a`:
  the center of mass is a position, the dipole coefficient is a chart-dependent number whose change
  is `m a` to leading order.  So *the dipole measures where the origin of the chart sits relative to
  the center of mass*, plus whatever physical `l = 1` content the solution genuinely has.
* **The BÓM integral does not always converge.**  Cederbaum–Sakovich are explicit that it "will in
  general *not* converge for initial data sets which are merely `C^{2}_{1/2+eps}`-asymptotically
  Euclidean ... it will however converge once one assumes that the initial data set satisfies
  certain asymptotic symmetry conditions in the given chart, as for example the Regge–Teitelboim
  conditions".  Those conditions are parity statements on the asymptotic fields — in the BMS/ADM
  Hamiltonian formulation they are written `h_ij(-n^k) = h_ij(n^k)`, `pi_ij(-n^k) = -pi_ij(n^k)`
  ([arXiv:1801.03718](https://ar5iv.labs.arxiv.org/html/1801.03718)) — i.e. *the metric's even part
  must have no `l = 1` and its odd part no `l = 0`*.  A dipole in `h` is precisely the violation of
  those conditions, which is why the center-of-mass integral and the dipole are two faces of the
  same gauge question.

Cederbaum–Nerz, *Explicit Riemannian manifolds with unexpectedly behaving center of mass*
([zbMATH 06457411](https://zbmath.org/pdf/06457411.pdf)), show that the earlier CMC-foliation route
to a center of mass misbehaves on explicit examples; this is what the STCMC construction of
Cederbaum–Sakovich repairs, and it is the reason to prefer a covariant definition over a naive
flux formula evaluated on whatever sphere is at hand.  Szabados's review
([Living Reviews](https://pmc.ncbi.nlm.nih.gov/articles/PMC5255888/pdf/41114_2016_Article_4.pdf))
is the general reference for how these quantities transform.

## 2. Why this is our situation

* Our runs are **stationary, axisymmetric and single-body**, and the inner datum carries **no
  dipole** (`S1 = 0`; the datum is `lam0 + S2 P2(n_z)`).  A physical `l = 1` moment is therefore not
  expected, while a chart offset is not forbidden by anything in the objective.
* The **inner boundary pins the shape, not the gauge**.  In the flat reduction
  (`FLAT_DIPOLE.md`) both boundary conditions are imposed on *fixed spheres* `|x| = rho_in`,
  `|x| = rho_out`, so a translation is not a symmetry of the problem and the origin is pinned — and
  the measured dipole there is zero.  In the full problem the metric's boundary weights are switched
  off (`inner_h = 0, outer_h = 0`), so the only control on the metric's angular content is the Ricci
  equation and the gauge condition; the `lambda` equation is `Delta_h lambda`, so an `l = 1`
  component of `h` — a translation-like mode of the chart — is an `l = 1` **source** for `lambda`,
  and the tail dipole follows.
* That reading also explains the two numbers we have: with the *soft* inner datum the dipole is
  `|S_10| = 1.07e-02`, and with the *hard* inner datum — the datum enforced exactly, leaving no
  freedom to absorb a chart offset in the inner region — it grows to `3.67e-01`
  (`pq_r100_phihyb025_floor2` vs `hard2`).  Harder inner data, larger tail dipole, is what a gauge
  mismatch looks like.

So the user's remark about the inner boundary is the right one, and it has a precise form: the
center-of-mass flux formula is meaningful only under the parity (Regge–Teitelboim) conditions, which
a dipole *violates*.  Applying it at `rho_in` to a field that already has a dipole is therefore
circular — the formula may not converge, and if it does, it is measuring the violation rather than
removing it.  Used at `rho_out`, where the asymptotics are controlled, or through the covariant
STCMC construction, it becomes a legitimate gauge-fixing condition.

## 3. Step one: the diagnostic that separates gauge from physics (cheap, do this first)

A translation by a vector `a` acts on a spherically symmetric background as `delta lam = -a . grad
lam_0(rho)`, i.e.

    lam_1(rho, theta) = -(a . n) lam_0'(rho)      =>      lam_1(rho) / lam_0'(rho) = const = -|a| cos(angle)

So the test is a **radial-profile test**, and it needs no new physics:

1. decompose the converged solution's `lambda` into `l = 0, 1, ...` radial profiles
   (`multipoles.py` already has the `S_lm` machinery; a small driver is enough);
2. form the ratio `lam_{l=1}(rho) / lam_{l=0}'(rho)` over `rho in [rho_in, rho_out]`.

* If the ratio is **constant in `rho`**, the dipole is a pure asymptotic translation: a gauge mode,
  removable by shifting the chart by `a`, with no physical content.
* If it is not constant, part of the dipole is a genuine `l = 1` source and a gauge pin would remove
  only the translational piece.  The same test applies to `h_rr - 1` and `h_tan - 1`, which should
  show the same proportionality with the same vector `a` if the whole configuration is translated.

The prediction to falsify: the ratio is constant, and the same `a` comes out of the `lambda` and the
metric sectors.  A second, independent check of the same statement: the `l = 1` radial profile
should be `propto lam_0'`, not a decaying harmonic `rho^{-2}` with an independent coefficient —
these two profiles are different functions of `rho` on a shell of ratio 100 and are easy to tell
apart.

## 3b. The measurement: it is NOT a translation (and not even a clean harmonic)

`com_translation_test.py` runs the Sec. 3 test on the two finished c100 runs (`floor2`,
`hard2`), with a positive control first.  The control is a field that *is* a rigid translation,
`F(x) = 1 - 0.1/|x - d z|`, put through the identical projection and fit:

| control | `l = 1` max | ratio median | spread | exact `-d_z` |
|---|---|---|---|---|
| `d_z = 0` | 6.2e-17 | — | — | 0 |
| `d_z = 0.05` | 1.25e-03 | **-0.0500** | 2.2e-04 | -0.05 |
| `d_z = 0.20` | 5.00e-03 | **-0.1999** | 8.7e-04 | -0.20 |

So the ratio test would have found a translation: it is constant to 0.4 % and returns `d_z` exactly.
On the runs it does not come out at all:

| run | `l = 1` of `lambda` | ratio over `rho in [2,99]` | far field `[60,99]` | `rho^-2` explains |
|---|---|---|---|---|
| `floor2` | max 1.43e-05 **at `rho ~ 3.5`** | -2.5e-04 +- 3.9e-03 | +2.7e-03 +- 2.7e-03 | 51 % of the `l = 1` norm in `[40,99]` |
| `hard2` | max 2.63e-05 **at `rho ~ 9.1`** | +6.7e-04 +- 2.7e-02 | +6.1e-02 +- 4.1e-02 | **3.4 %** |

Three things are wrong with the translation reading, in increasing order of importance:

1. the ratio is **not constant and changes sign** as the fit window moves (and the spread is larger
   than the median), where the control's spread was 0.4 %;
2. the `l = 1` amplitude peaks in the **near field** (`rho ~ 3.5` and `9.1`), not in the tail, and
   oscillates with several nodes in `log rho` — visible in
   `plots/com_translation_floor2.png` — which is the shape of the network's own radial basis
   (`sin/cos(pi i t)`, `t = log(rho/rho_in)/log(ratio)`), not of a smooth gauge mode;
3. the far-field `l = 1` content is **not a decaying harmonic**: a `c/rho^2` fit explains 51 % of its
   norm for `floor2` and only 3.4 % for `hard2`, the run with the largest reported `|S_10|`.

The cross-sector check fails the same way: the metric's `l = 1` amplitude is ~1e-04 and its own fit
gives `a_z ~ 0.8-1.3`, four orders of magnitude from the `a_z ~ 1e-04` that `lambda` implies, so `h`
and `lambda` are not two views of one translated configuration.

**What this means.**  The dipole is not a rigid translation of the configuration, and — more
usefully — the `l = 1` content of these solutions is *not resolved as an asymptotic harmonic at all*.
That is exactly what the reported multipole tables had been hinting at: the `|S_10|` lines carry a
"drift over the table" of 100-163 %, i.e. the extracted dipole changes with the extraction radius.
The numbers in those lines are therefore not converged multipole measurements, and building a
center-of-mass argument on top of them would be building on sand.

**Consequence for the plan.**  The gauge pin of Sec. 4 keeps its *principle* — the asymptotic chart
does have a translation freedom and a center of mass is the invariant that fixes it — but here it
would act on a field whose `l = 1` sector is dominated by ansatz-resolution structure.  The order of
operations is therefore reversed: pin the `l = 1` (or the center of mass) *first*, as a
**regularisation** that forces that sector to be a clean decaying harmonic, and only then read the
dipole as a gauge mode.  Route 1 of Sec. 4 does both at once, and the diagnostic to confirm it is
the same ratio test: after the pin, the ratio should be constant and the far-field `c/rho^2` fraction
should approach 1.

## 4. Step two: the gauge pin to implement afterwards

Three routes, in increasing order of fidelity:

1. **No-dipole pin (the `l = 1` analogue of the existing monopole pin).**  `losses.outer_pin_raw`
   already reduces the `lambda` field to its `l = 0` harmonic and pins the order-1 Robin combination
   of that; the generalisation is to pin the `l = 1` coefficient at `rho_out` to zero as well.  Cheap,
   in the existing idiom, and it fixes precisely the gauge mode the diagnostic identifies.  Risk: it
   pins one chart, and if a physical dipole were present it would be fighting it — which is what the
   diagnostic of Sec. 3 rules out first.
2. **Center-of-mass flux integral** (Beig–Ó Murchadha, the formula in Sec. 1) evaluated on the outer
   sphere: it is a *linear* functional of the asymptotic fields, so it can be added to the loss
   exactly like the pin, with the invariant meaning.  This is the "formula of Cederbaum's that would
   kill the dipole" in its most direct form, and it is where the parity caveat bites: to use it we
   must be in the RT window (metric's even part without `l = 1`), so it should be applied as a
   *constraint on the solution* rather than as a post-hoc measurement of a field that already
   violates it — or applied to the `lambda` sector alone, where we control the parity by
   construction (the datum is `P2`, i.e. even).
3. **Trainable chart offset.**  Give the ansatz one extra parameter pair (axisymmetry leaves a single
   `d_z`) — the network is evaluated at `x - d` — and pin `d` through the dipole/COM condition.
   This is the most explicit realisation of "the dipole is where the origin sits": the optimiser can
   then slide the chart instead of contorting the field.  Two extra scalars, no new physics, and it
   makes the gauge mode visible in the logs as a number `d_z` that can be monitored.

The covariant STCMC center of mass of Cederbaum–Sakovich (a foliation by constant-spacetime-mean-
curvature spheres) is the principled version of route 2, and the one to adopt if the naive flux
integral turns out not to converge on our solutions — which is exactly what Cederbaum–Nerz warn
about and what the parity discussion above predicts for a field with a dipole.

## 4b. A centre-of-mass flux that is conserved, so it can be imposed INSIDE (2026-10-18)

With g = -lambda dt^2 + lambda^-1 h the static vacuum equations are Ric(h) = (1/2) d ln lambda (x)
d ln lambda and Delta_h ln lambda = 0, so U = (1/2) ln lambda is h-harmonic; and in harmonic gauge
the coordinates x^i are h-harmonic (Delta_h x^i = -h^ab Gamma^i_ab = 0).  Green's identity for two
h-harmonic functions then makes

    D^i(S) = oint_S ( x^i dU/dn - U dx^i/dn ) dA_h
           = oint_{|x| = rho} sqrt(h) h^jk ( x^i d_k U - U delta^i_k ) n_j rho^2 dOmega

the SAME on every closed surface S enclosing the inner sphere.  Far out, a mass M displaced by a
(U ~ -M/|x - a|) gives D = 4 pi M a: D is the centre of mass, and it is conserved through the shell.
No quasi-local centre-of-mass condition of this kind turned up in a literature search (the standard
ones are fluxes at infinity -- ADM, Beig-O Murchadha, Regge-Teitelboim -- or quasi-local
constructions like Wang-Yau's); it is the elementary Green-identity statement for this formulation.

**Measured** (D_z on spheres rho = 1 ... 100, 32 x 8 quadrature):

| | rho = 1 | 2 | 10 | 100 |
|---|---|---|---|---|
| fine nonlinear reference (even basis) | 2e-15 | 0 | -6e-15 | -7e-14 |
| `m5_ssb_norsmp` | -0.32 | -0.32 | -0.32 | -0.32 |
| `m5_dsg_cyc` | -0.43 | -0.43 | -0.43 | -0.43 |
| `m5_w40` | -0.11 | -0.11 | -0.11 | -0.11 |
| `ib16k_ssbroyden` | -0.083 | -0.083 | -0.083 | -0.022 |

The networks' D is constant in rho -- they are consistent solutions -- but NOT zero: their centre of
mass is displaced along z by a = D/(4 pi M) ~ -0.04 (best run), although the data are symmetric.

**Why the data allow it: a residual harmonic gauge mode.**  A coordinate change x -> x + xi with
Delta_h xi = 0, xi = 0 on the inner sphere and xi -> a at infinity keeps the chart harmonic and the
inner data (the points of the inner sphere do not move: same induced metric, same lambda datum),
and translates only the asymptotic chart, changing D by 4 pi M a.  Its effect on lambda at fixed
coordinates, xi . grad lambda, vanishes at rho_in, peaks near rho ~ 2 and decays like
grad lambda ~ rho^-2 -- the shape of the l = 1 difference network - reference measured in DSGNAR.md
9.2 (3.3e-3 at rho = 2, 4e-4 at rho = 10).  So that "error" is, at least in large part, this gauge
mode: the same geometry in a translated asymptotic chart (to be confirmed by a gauge-invariant
comparison).  The outer Robin condition on P without rho^-2 penalises it only at the level of the
loss floor: for a ~ 0.04 the P dipole at rho_out is ~2.5e-6, whose Robin residual (factor 2)
costs ~1e-9 at w_outer = 100 -- the size of the whole loss.

**What it gives.**  D = 0 fixes the asymptotic origin at the centre of mass.  It is a gauge choice,
available for ANY data (the mode xi exists in general), it involves only l = 1, and because D is
conserved it can be imposed on any sphere -- the inner one included, where the network is best
resolved -- instead of through a tail coefficient at rho_out.  As a loss term: three averaged
numbers per sphere, (D^x, D^y, D^z)^2, on a few spheres (pin-like; cheap).

Implemented (2026-10-18): `--com-pin RADII [--w-com W]` (Config.com_radii, w_com = 100): the loss
gets w_com * sum over the spheres of |d|^2, d = D/(4 pi), by fixed Gauss-Legendre x uniform-phi
quadrature (24 x 8 per sphere), in `total_loss` and `residual_vector` alike (`losses.com_flux`).
tests/test_com_flux.py: d = 0 for the exact solution on every sphere; for it translated by
a = 0.05 the same value on every sphere, = M a; the residual vector still squares to the loss.
First runs: Mendieta round 7 (`m7_ssb_com`, `m7_dsg_com`, DSGNAR.md 9.2).

**Confirmed far out (2026-10-18).**  For `m5_ssb_norsmp`, `com_flux` gives d_z = -0.02567 on the
spheres rho = 2, 10, 50 (constant to 2e-5 relative), M = 0.5771 (= R0), so a = d/M = -0.0445.  The
l = 1 coefficient of lambda_net(x) - lambda_ref(x), against lambda_net(x) - lambda_ref(x - a z):

| rho | plain | reference translated by a |
|---|---|---|
| 10 | 4.1e-4 | 5.4e-5 |
| 30 | 5.3e-5 | 2.5e-6 |
| 100 | 5.4e-6 | 2.4e-7 |

Far out the network IS the reference in a chart translated by a (8-23x less l = 1 difference);
near the inner sphere translating makes it worse (-6.6e-3 at rho = 1.5), as it must: there the gauge
vector xi -> 0 and is not a translation.  So the odd "error" is the residual harmonic gauge mode of
section 4b, not wrong geometry.

## 5. What this would and would not explain

It would explain, in one stroke: the size of the tail dipole in the production runs, its growth when
the inner datum is enforced exactly, its absence in the flat reduction (fixed spheres), and its
indifference to the multipole content of the objective.  It would *not* by itself explain the
`|S_10| = 3.67e-01` being an order of magnitude larger than the soft-BC run's `1.07e-02` — that
still needs the numbers from the diagnostic in Sec. 3, which is the next thing to run.

## References

* C. Cederbaum, A. Sakovich, *On center of mass and foliations by constant spacetime mean curvature
  surfaces for isolated systems in General Relativity*,
  [arXiv:1901.00028](https://ar5iv.labs.arxiv.org/html/1901.00028) — the STCMC center of mass, its
  Poincaré equivariance, and the quotation of the Beig–Ó Murchadha flux integral used above.
* C. Cederbaum, C. Nerz, *Explicit Riemannian manifolds with unexpectedly behaving center of mass*,
  [zbMATH 06457411](https://zbmath.org/pdf/06457411.pdf); see also
  [arXiv:1501.05559](https://ar5iv.labs.arxiv.org/html/1501.05559) on the center of mass of
  asymptotically hyperbolic data.
* R. Beig, N. Ó Murchadha, *The Poincaré group as the symmetry group of canonical general
  relativity*, Ann. Phys. 174 (1987) 463 — the flux-integral center of mass.
* T. Regge, C. Teitelboim, *Role of surface integrals in the Hamiltonian formulation of general
  relativity*, Ann. Phys. 88 (1974) 286 — the parity conditions that make the flux integrals
  converge.
* L. B. Szabados, *Quasi-local energy-momentum and angular momentum in general relativity*,
  [Living Reviews](https://pmc.ncbi.nlm.nih.gov/articles/PMC5255888/pdf/41114_2016_Article_4.pdf) —
  how these quantities transform under chart changes.
* BMS group at spatial infinity, the Hamiltonian (ADM) approach,
  [arXiv:1801.03718](https://ar5iv.labs.arxiv.org/html/1801.03718) — the parity conditions written
  as `h_ij(-n) = h_ij(n)`, `pi_ij(-n) = -pi_ij(n)`.
* Our own reduction: [FLAT_DIPOLE.md](FLAT_DIPOLE.md) — the flat problem with fixed spheres has no
  dipole, which is the consistency check for the whole picture.
