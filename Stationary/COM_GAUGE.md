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
