# Plan: from static to genuinely stationary (rotating) vacuum

Status: **plan for review, nothing implemented.**  Scope chosen with the user: rotation
(`g_tphi` / twist / Ernst), a first-class geometric-indicator layer, and acceptance as three
successive gates (no-reference reproduction -> convergence -> physical observables).

Companion documents: `README.md` (system, results), `Laplace_Robin.md` (the Robin operator),
`HUB.md` (the hub workflow and its reference numbers).

---

## 0. What exists, and precisely what is missing

`stationary/geometry.py` states the system the code solves:

    Ricci(h)_ab                = (1/(2 lambda^2)) grad_a lambda grad_b lambda        (6)
    h^{ab} grad_a grad_b lambda = (1/lambda) h^{ab} grad_a lambda grad_b lambda      (1)
    Gamma^a_{bc} h^{bc}        = 0                                                   (3)

with `h` a 3-metric on the spatial slices and `lambda` the lapse.  That is the **static**
reduction of the 4-D vacuum equations: the space-time is

    ds^2 = -lambda^2 dt^2 + h_ij dx^i dx^j ,          (no cross term)

so `g_{t phi} = 0` identically and there is no momentum/rotation sector to solve for.  Every
run in `runs/` is of this kind, validated against the static Weyl two-rod (Israel-Khan)
solution, in a harmonic chart (or with the inhomogeneous cylindrical de Donder source), with
inner-sphere data, order-3 Robin decay at `rho_out`, and far-field value pins.

The extension is therefore exactly: **let the shift exist.**

    ds^2 = -lambda^2 dt^2 + h_ij (dx^i + beta^i dt)(dx^j + beta^j dt) ,

study the full `R_ab(g) = 0` (10 components, all `t`-independent), and recover the two
Killing vectors, the frame dragging, the horizon, the ergosphere and the multipole moments.
Nothing in the current pipeline becomes wrong; the static path must remain bit-identical
(regression gate P3).

Missing pieces, in dependency order:

1. an exact rotating reference with a *known* chart and gauge (Kerr first, then NUT and the
   rotating two-object families), plus its 3+1 split `(lambda, h_ij, beta_i)`;
2. a 4-D metric / Christoffel / Ricci / gauge-source layer built from `(h, lambda, beta)`;
3. rotational boundary data: the inner induced 4-metric (rigid rotation `Omega_in` or a
   horizon), the correct far-field falloffs of `beta`, the higher-order Robin operator for the
   rotational sector, and **amplitude pins for `J`** -- the same "kernel vs amplitude" branch
   problem documented for `lambda` in `HUB.md` section 6d, one derivative order over;
4. the **geometric indicator layer** (curvature invariants, Petrov type, horizon, ergosphere,
   Geroch-Hansen multipoles, no-hair identities) -- gauge invariant, so it is also what scores
   a solve that was never told the answer;
5. no-reference solving: an asymptotic **tail model** built only from the measurable moments
   (`M`, `J`, then `M_l`, `J_l`), iterated self-consistently, which is what replaces
   `--ref-solution` in the loss.

---

## 1. Formulation: two routes, one spine

### Route A -- Papapetrou / Ernst, 2-D (`(rho, z)`)

    ds^2 = -f (dt - omega dphi)^2 + f^{-1} [ e^{2 gamma} (drho^2 + dz^2) + rho^2 dphi^2 ]

with the Ernst potential `E = f + i psi` (`psi` the twist) obeying

    (E + Ebar) [ d_rho^2 + d_z^2 + (1/rho) d_rho ] E = 2 [ (d_rho E)^2 + (d_z E)^2 ]

and `gamma` by the two quadratures (the (rho,z) gradient of gamma; the exact sign and factor
conventions are to be **pinned by a test on Kerr**, as `test_robin_operator_expansions_...`
pins the Robin expansions):

    gamma_,rho = (rho/(4 f^2)) [ f_,rho^2 - f_,z^2 + psi_,rho^2 - psi_,z^2 ]
    gamma_,z   = (rho/(2 f^2)) [ f_,rho f_,z + psi_,rho psi_,z ]

* unknowns: 2 real functions of 2 variables -> the "small net + dense quasi-Newton" recipe of
  `README.md` section 6b applies directly;
* the `1/rho` is removed by solving the regularised form
  `rho (E+Ebar)(E_,rho,rho + E_,zz) + (E+Ebar) E_,rho - 2 rho (E_,rho^2 + E_,z^2) = 0` with
  `f`, `psi` even in `rho` -- no axis singularity, and axis regularity becomes a boundary
  condition at `rho = 0`;
* the horizon is the rod `rho = 0`, `|z| <= sigma`, which is a **coordinate-regular** surface in
  prolate spheroidal coordinates `x = (r_+ + r_-)/(2 sigma)`, `y = (r_+ - r_-)/(2 sigma)`,
  `r_+-^2 = rho^2 + (z -+ sigma)^2`, `sigma = sqrt(M^2 - a^2)`.  So `A`, `kappa`, `Omega_H`
  and the horizon 2-metric are measurable, not just predicted;
* exact assets are cheap and closed-form (Kerr, NUT, double-Kerr/Kramer-Neugebauer, the
  Neugebauer-Meinel rotating disk if matter is wanted later), and the Geroch-Hansen moments
  come straight out of `E` by the Fodor-Hoenselaers-Perjes expansion
  `E = (1 - xi)/(1 + xi)`, `xi = sum_l (M_l + i J_l)/rho^{l+1}`.

### Route B -- keep the 3-D harmonic-gauge solver, add the shift

Keep `Fields`-style first-order variables and the existing model/sampling/reweighting/
quasi-Newton/diagnostics/VTK machinery; make the geometry 4-D and add `beta`:

    g_ab = [[ -lambda^2 + beta_k beta^k , beta_j ], [ beta_i , h_ij ]]

and impose `R_ab(g) = 0` with an inhomogeneous de Donder condition
`Gamma^a_{bc} g^{bc} = U^a`, where -- exactly as `weyl.gauge_source_from_metric` does today for
the static cylindrical chart -- **`U^a` is derived from the candidate's own metric** for the
rotating (Papapetrou) chart.  That trick is what lets the exact reference solve the system
without a coordinate transformation, and it is already proven in this repo for the static case.

* unknowns: `h` (5 or 6) + `lambda` (1) + `beta` (1 for axisymmetry about the rotation axis,
  3 in general) -> for the axisymmetric case the existing `AxisymHybridNet` gains **one scalar
  output** (`w(rho, mu)` with `beta_i = w * phi_hat_i`);
* scales to non-axisymmetric rotation and to matter later, which Route A does not;
* costs the horizon: with a shell domain `rho_in >` horizon the metric functions are singular
  at the rod, so horizon-local quantities are reached either by the Route A companion or by a
  later horizon-adapted domain (P8).

### Recommendation

**B is the spine, A is the companion tool.**  Reasons: B is a strict generalisation of the
validated pipeline (one extra output field, the same losses/optimiser/report), it keeps the
road to non-axisymmetric rotation and matter open, and the "source from the candidate's own
metric" device already solves B's main technical risk (gauge/reference mismatch).  A is small,
and it is needed anyway for (i) the exact references and their `gamma`, (ii) the asymptotic
tail model and the multipole extraction that gate 3 depends on, (iii) an independent
cross-check of the same configuration by a different formulation, and (iv) horizon-local
observables.  **Decision D1** at the end asks you to confirm this.

### Boundary data (both routes)

* **inner sphere** `rho = rho_in`: the induced 4-metric.  Static runs impose the round metric
  of areal radius `inner_radius` plus `lambda_0` (derived from `k = 1`) and leave `h_rr` free.
  Rotating: the same for `h` and `lambda`, plus the rotational content -- either a prescribed
  rigid rotation `Omega_in` (a material surface) or the no-rotation/co-rotating condition of a
  horizon.  Which one is a physics choice: **Decision D5**.
* **outer sphere** `rho = rho_out`: higher-order Robin on `h`, `Gamma` and `lambda` as today,
  plus a Robin condition for `beta` whose base exponent **is measured from the exact asset**
  (the covariant `beta_phi ~ 2J/rho` and the contravariant `beta^phi ~ 2J/rho^3` suggests the
  exponent is not the same for the two index positions -- this must be pinned, not guessed);
  plus **pins on the amplitudes** `lambda_inf`, `h_tan`, `h_rr` **and now `J`**, since the
  rotational Robin operator annihilates the leading `1/rho^n` modes and is therefore blind to
  the amplitude that *is* the angular momentum.
* **tail model** (no-reference runs): replace the exact Robin source by
  `B[field_net] = B[field_tail]`, with `field_tail` the asymptotic expansion in `(M_l, J_l)`;
  iterate solve -> extract moments from the far field -> rebuild the tail -> re-solve.

---

## 2. Conventions to be pinned by named tests (repo discipline)

Every one of these gets a test that fails loudly if the convention moves, in the spirit of
`tests/test_pipeline.py::test_robin_operator_expansions_match_the_documented_forms`:

| # | convention | pinned by |
|---|---|---|
| 1 | signature and `(f, omega) <-> (lambda, h, beta)` identification: `g_ab` rebuilt from `(lambda,h,beta)` equals the Papapetrou metric | `test_rot_conventions.py` |
| 2 | twist definition and the two `gamma` quadratures (signs and factors) | same, on Kerr |
| 3 | Kerr closed form in Weyl-Papapetrou coordinates (prolate `x,y`), including `a -> 0` and `a -> M` | same |
| 4 | the inhomogeneous de Donder source `U^a` of the rotating chart | same, exact-solution residual < 1e-14 |
| 5 | far-field exponents of `h`, `Gamma`, `lambda`, `beta` (covariant and contravariant) | same, measured |
| 6 | axis regularity orders (`g_phi,phi/rho^2 -> 1`, `g_t,phi = O(rho^2)`), no conical singularity | same |
| 7 | `Omega_H`, `kappa`, `A` sign and normalisation (Smarr must close) | same |
| 8 | multipole normalisation `M_l + i J_l = M (i a)^l` | `test_invariants_rot.py` |

---

## 3. The geometric indicator layer (headline deliverable)

A single module that, given any `(lambda, h, beta)` field (network or exact), produces a
**gauge-invariant characterisation** of the geometry.  It is what makes a no-reference solve
scoreable, and it is the "more geometrical description" the user asked for.

**Algebraic / curvature**
* `R`, `R_ab R^ab` (= 0 is a solver quality check), `R_abcd R^abcd`, `C_abcd C^abcd`
* Pontryagin density `C_abcd *C^abcd` -- non-zero only with rotation, a clean "is it spinning"
  indicator that survives any gauge
* Weyl scalars in a null tetrad `psi_0..psi_4`; `I = psi_0 psi_4 - 4 psi_1 psi_3 + 3 psi_2^2`,
  `J = det`; **speciality index `S = 27 J^2 / I^3`** (= 1 for Kerr/Schwarzschild, type D)
* `psi_2` asymptotics vs `M/rho^3` (a curvature-level mass measurement)

**Stationary structure**
* lapse squared `-xi.xi`, frame dragging `omega = -g_t,phi/g_phi,phi`, ZAMO angular velocity
* the twist 1-form and the twist potential `psi`
* **ergosphere**: `g_tt = 0`, its shape, its area, and `r_ergo(theta)`
* **horizon**: the Killing horizon of `xi + Omega_H eta`; `A`, `kappa`, `Omega_H`,
  `T_H = kappa/2pi`, `S = A/4`, irreducible mass
* horizon intrinsic geometry: its 2-metric, its **horizon multipoles** (`mu_l`, the
  Ashtekar-Krishnan style), and the Gauss-Bonnet Euler characteristic (must be 2 -- a
  topological check independent of everything else)

**Asymptotic / conserved**
* ADM/Komar `M`, `J`, centre of mass; quasi-local (Hawking, Komar-on-spheres) masses
* **Geroch-Hansen multipoles** `M_l`, `J_l` (FHPS from the Ernst potential)
* no-hair / consistency identities, each as a test: `M_l + i J_l = M(ia)^l`,
  `chi = J/M^2 <= 1`, Penrose `A <= 8 pi M^2`,
  **Smarr** `M = kappa A/(4 pi) + 2 Omega_H J`,
  **Christodoulou** `M^2 = M_irr^2 + J^2/(4 M_irr^2)`,
  spin-induced quadrupole `Q = -chi^2 M^3` (Kerr: coefficient 1)

**Observable description (post-processing, from the reconstructed metric)**
* ISCO radius, photon ring, light deflection, epicyclic frequencies, periapsis precession --
  compared with the closed-form Kerr expressions (Bardeen) where they exist

**Outputs**
* `geometry.json` + `geometry.txt` per run (every quantity above, its reference value, its
  tolerance, and the verdict), a figure sheet, and new `stationary/plane.py` /
  `stationary/vtk.py` panels: ergosphere + horizon surfaces, frame-dragging vector field,
  curvature-scalar maps, horizon embedding diagrams.

---

## 4. Phases and gates

Gate G0 (asset), G1 (regression), G2 (manufactured), G3 (no reference in the loss),
G4 (convergence), G5 (physical observables).  **P0-P2 are route-independent** and can start
immediately, so the plan is not blocked on D1.

### P0 -- conventions and theory note (no solver)
`Stationary/Stationary_Rotating.md`: the equations, the two routes, the gauge sources, the
falloffs, the invariant identities, each convention labelled with the test that pins it.  This
is the `Laplace_Robin.md` of the rotating extension and it is written *before* the code.
Gate: the note exists and every numbered convention in section 2 above points at a test name.

### P1 -- exact rotating assets
* `stationary/rotating.py`: Kerr in Weyl-Papapetrou (prolate `x,y`: `f`, `omega`, `gamma`),
  its 3+1 split `(lambda, h, beta)`, NUT, and (later) double-Kerr / Kramer-Neugebauer,
  Neugebauer-Meinel disk.
* `verify_kerr.py`, standalone, in the style of `verify_weyl.py`.
* Tests: `f, omega, gamma` satisfy the Ernst equation and the quadratures to machine precision;
  `a -> 0` gives the static Weyl/Schwarzschild limit already in `exact.py`; `a -> M` extremal;
  invariants equal the closed forms (`A = 8 pi M (M + sigma)`,
  `Omega_H = a/(2 M (M + sigma))`, `kappa = sigma/(2 M (M + sigma))`, `S = 1`,
  `M_l = M(ia)^l`); `R_ab = 0` in 4-D (after P3) and the de Donder source identity.

**G0**: an exact rotating asset that is *verified*, not assumed -- because everything
downstream is scored against it.

### P2 -- geometric indicator layer
`stationary/geometry_invariants.py` (new) + extensions to `invariants.py`, `multipoles.py`,
`diagnostics.py`, `report.py`, `plane.py`, `vtk.py`; `stationary/geometry_report.py` for
`geometry.json`/`geometry.txt` + figures.
Tests: every identity of section 3 on the exact Kerr asset; the Euler characteristic is 2; the
speciality index is 1; Smarr closes to 1e-14.

**Status: the static half is DONE** (commit pending), the rotating half waits for P3/P4.

* `stationary/geometry_invariants.py` exists and is validated by
  `tests/test_geometry_invariants.py` (10 tests) against **closed forms**, not against itself:
  flat space, Schwarzschild (Kretschmann, tidal eigenvalues, `|psi_2|`, type D, Hawking mass),
  the repo's spherical asset and the Weyl two-rod asset.
* The experiment that had to come first was *which 4-metric the code's `(h, lambda)`
  represents*.  Two completions are exact for the same data — `g = -lambda^a dt^2 + lambda^-a h`
  with `a = 0` (the scalar reading of README section 1) and `a = 1` (the vacuum one) — and they
  are different space-times.  The `a = 1` one is the physical one (see README section 11.1):
  the spherical asset is Schwarzschild of mass `M = R0` with areal radius `rho + R0`, and the
  Weyl asset is Israel-Khan of mass 1.1.  Both readings are supported by the `reading` argument.
* Already available for the static runs: `R`, `R_ab R^ab`, `R_abcd R^abcd`, `C^2`, the
  **Pontryagin density** (identically zero while static: the staticity certificate *and* the
  regression baseline for the rotating work), the electric/magnetic parts of the Weyl tensor
  with their tidal eigenvalues, `psi_0..psi_4`, `I`, `J`, `S = 27J^2/I^3`, coarse Petrov type,
  the spatial `R_ij` with the **rank-one defect** `R_ij R^ij - R^2` (zero on any solution, and
  it needs no reference solution), the 3-d identity `K3 = 4 R_ij R^ij - R^2`, the Cotton norm,
  and the area / areal radius / mean curvature / **Hawking mass** of a coordinate sphere.
* Wired into the run report: `geometry_report` + `format_geometry` reduce all of it to robust
  numbers (medians/maxima over a Fibonacci sample of directions and radii), and
  `stationary.report` prints the block, so every `report.txt` that `postprocess.sh` writes now
  carries the geometry.  Two additions there are worth naming: the **eigenvalue defect**
  `R - (1/2)|grad phi|^2`, which is the identity a wrong `lambda` moves while the rank-one
  defect does not, and the **Kretschmann error against the exact reference**, the only
  gauge-invariant error in the report.  Measured: the converged `weyl_rot45_hub` gives
  6.6e-02 / 4.3e-02 / 7.6e-04 for those three, the barely-trained `_smoke` 5.5 / 0.45 / 9.2e-02.
* In the VTK export: `stationary.vtk` writes `kretschmann`, `weyl_c2`, `pontryagin`,
  `ricci_abs`, `speciality_dev`, `petrov_D` beside `lambda`/`lambda_err`, with the nodes on the
  run's symmetry axis evaluated at `1e-2 rho_in` from it (side limit).  That clamp is needed
  because the Cartesian components of an axisymmetric metric are not smooth across the axis:
  measured `K = 2.9e+08` on the axis against `3.7e+04` at `rho_axis = 0.05`, for a true value
  that is finite.  The cylindrical-chart route (`curvature_at_axisym`) is exact off the axis but
  *worse* on it for a network, which is why it is kept for chart-native metrics.
* The Hawking mass is now a profile rather than two numbers: `hawking_profile` returns
  `m_H(r)` (Gauss-Legendre in mu, exact to round-off at 6 x 4 nodes) and `hawking_mass_field`
  puts it on a grid; the report prints three radii plus the mass the configuration must have
  (the sum of the rod masses for a Weyl run), and the VTK export carries `hawking_mass`.
  Measured on `weyl_rot45_hub`: m_H +0.061% of 0.03142857 against the exact solution's
  -0.005%, with the profile flat to 5e-5 across the shell.
* Figures: `stationary.geometry_figures --outdir runs/<name>` writes the geometry slice
  (Kretschmann network vs exact, the gauge-invariant error, the vacuum defect, the speciality
  and the Pontryagin density, with the two horizon rods marked and their exact
  `A = 16 pi m^2`, `kappa = 1/(4m)` in the title) and the Hawking profile.  Following `plane.py`:
  the grid is the shell-conforming POLAR one (geometric radial, uniform theta), both panels are
  chart fields about the run's OWN axis contracted in an orthonormal frame -- the exact from its
  analytic chart metric `diag(e^{2k}, rho^2, e^{2k})`, the network through `cylindrical_metric`
  (a genuine change of coordinates, azimuth included) -- and `rho` is floored at `0.1 rho_in`
  because the chart's Christoffel symbols lose precision below ~1e-3 of the rod scale before any
  frame change can help.  `curvature_at(..., frame="orthonormal")` is the option that makes the
  contraction stable; it is the same scalar as the coordinate contraction to 1e-8.
* Still to do in P2: Geroch-Hansen multipoles, the geodesic observables (ISCO, photon ring), the
  Euler-characteristic and horizon-multipole checks, and `geometry.json` plus the `plane.py`
  curvature panel.
* Still to do for rotation (P4): `beta` in the tetrad and the electric/magnetic split, the
  Kerr/NUT assets as the closed-form checks, and the Smarr / Christodoulou / no-hair
  identities.

**Value of doing P2 before any new solver work**: the existing static runs immediately gain the
invariant report, and the same code scores the rotating ones.

### P3 -- 4-D residual layer and the static-limit regression
* `geometry.py`: dimension-generic packing (`sym_d`, `gamma_d`) and `Fields4`
  `(h, lam, beta)` with `Gamma` derived; `residuals4_at`, `scaled_residuals4_batch`.
* `stationary/rotating.py`: `gauge_source_rotating_from_metric(h, beta, axis)` -- the
  candidate-only de Donder source of the rotating chart.
* `problem.py`: `Config` gains `rotate`, `spin`/`j`, `inner_omega`, `beta_robin_exp`,
  `pin_j`, `tail_model`.
* Tests: **with `beta = 0` the 4-D residual groups reduce exactly to the documented 3-D groups**
  of the `geometry.py` docstring (this is the regression gate); the de Donder source makes the
  exact Kerr asset solve the system to machine precision; existing static runs unchanged.

**G1**: the extension is a strict superset -- the static results stay reproducible.

### P4 -- rotating ansatz, losses, manufactured Kerr solve
* `model.py`: `AxisymRotHybridNet` (one extra output, `beta = w(rho, mu) * phi_hat`) and a
  general 3-D rotating variant.
* `losses.py`: the rotational residual group; rotational Robin with per-field exponents and
  orders; `outer_pin_terms` extended to `J`; `equation_keys` updated (and `test_equation_keys`
  extended -- the repo already centralises this).
* `train.py`: `exact_asset` for the rotating assets, banner, weights.
* `bench.py`: unchanged (same parser) -- **but bench the new config before any hub launch**;
  4-D second derivatives and an extra field raise both time and the XLA temporary, which is
  what the `HUB.md` section 5 memory table is for.
* Tests: `test_pins_rot.py` (the `J` pin fixes the branch on a deliberately wrong-`J` start),
  `test_robin_terms` extended, axis regularity.

**G2**: solve Kerr with exact inner data and an exact Robin source; errors at or below the
static manufactured levels (`max|dh|` ~1e-3, and better with the pins).

### P5 -- no-reference solve (the real target)
* `stationary/asymptotic.py`: the `(M_l, J_l)` tail model for `h`, `lambda`, `beta` and its
  Robin contraction; the self-consistent iteration loop.
* Tests: the tail model reproduces Kerr's far field to its truncation order; the loop converges
  (with the measured number of iterations written into the docs).

**G3**: the exact solution is used **only to score** -- never in the inner data, the Robin
source or the pins -- and the gauge-invariant errors land at or below the manufactured levels.

### P6 -- convergence and self-consistency study
A ladder script (`rot_ladder.sh`, in the style of `run_ladder.sh`): error vs `n_coll`, net
width/depth, Robin order, domain ratio, tail order, and float64-vs-float32; plus an a
posteriori, residual-based error estimate and Richardson extrapolation in `n_coll`.

**G4**: a measured convergence order and a self-consistent error estimate with no reference
solution at all.

### P7 -- physical observables
`geometry_report.py` numbers for a family of spins: `M`, `J`, `chi`, `S`, `M_l`, `J_l`,
ergosphere, horizon quantities where the domain reaches them, ISCO/photon ring; comparison
tables via `compare.py`; the Kerr relation as the calibration, and the *deviation* from it as
the physics for configurations that are not Kerr.

**G5**: quantitative agreement with the analytic values on Kerr, then the first genuinely new
configuration.

### P8 -- horizon-adapted domain and the first new configuration
Either prolate-coordinate horizon domain in the Route A companion, or excision /
horizon-penetrating coordinates in the spine; then a rotating two-object configuration (no
closed-form balanced solution) as the first result that the exact-reference pipeline could
never have produced.

### P9 -- integration and documentation
`README.md` new section, `HUB.md` new section with the bench numbers and the hub commands,
`Stationary_Rotating.md` updated from "plan" to "as built", postprocess/VTK/report extended,
`bench` numbers recorded.

---

## 5. Test plan (summary)

| file | what it pins |
|---|---|
| `tests/test_rot_conventions.py` | the eight conventions of section 2 |
| `tests/test_kerr_asset.py` | Ernst + quadratures + closed-form invariants + limits |
| `tests/test_geometry4.py` | 4-D residuals; **exact static reduction at `beta = 0`** |
| `tests/test_invariants_rot.py` | Smarr, Christodoulou, Penrose, `S = 1`, Euler = 2, `M_l = M(ia)^l` |
| `tests/test_pins_rot.py` | the `J` branch pin |
| `tests/test_asymptotic.py` | tail model vs Kerr far field; loop convergence |
| `tests/test_equation_keys.py` (extend) | the new groups in loss/reweight/log |

---

## 6. Risks

1. **Gauge/reference mismatch (the main technical risk of the spine).**  Mitigated by the
   candidate-only de Donder source, which the repo already uses for the static cylindrical
   chart, plus the machine-precision asset test.  Fallback: transform Kerr to harmonic
   coordinates numerically and verify the gauge residual directly.
2. **The rotational branch problem.**  The Robin operator is blind to `J`'s amplitude, exactly
   as it was blind to `lambda`'s level.  Mitigated by reusing the pin machinery (P4) and by
   knowing that the fix is a *value* condition, not a weight.
3. **Horizon not in the domain.**  A shell-only spine cannot measure `A`, `kappa`, `Omega_H`.
   Mitigated by the Route A companion (prolate coordinates make the rod regular) and by P8.
4. **Cost.**  4-D second derivatives plus `beta` raise time and memory; bench before launching,
   and remember the 12 GiB-slice lesson (`HUB.md` sections 5, 6d) -- it is the bench number
   that decides, not the loss.
5. **Regularity at the axis for `beta`.**  Getting the parity/falloff wrong produces a silent
   conical or frame-dragging singularity.  Mitigated by convention 6's test and by the
   invariants (the Pontryagin density and the horizon Euler characteristic would notice).
6. **No exact reference for the object the current pipeline models.**  The current inner sphere
   is a shell; there is no closed-form rotating shell.  Mitigated by validating the *manufactured*
   step on Kerr (exterior region on a shell around the horizon) and then removing the reference
   for the target configuration -- which is the point of gate G3.
7. **Wall-clock discipline.**  Every heavy solve is a hub run sized with `bench` first; a
   *training* launch is the one path that does not set the JAX compilation cache directory, so
   post-processing and `--check` must keep `JAX_COMPILATION_CACHE_DIR` off `$HOME`
   (`HUB.md` section 13).  Do not launch from a dirty tree (`HUB.md` section 5).

---

## 7. Decisions needed

* **D1 (spine).**  Confirm B as the spine + A as the companion tool, or choose A only (fastest
  to horizon physics, abandons the 3-D machinery for now) or B only (no horizon-local numbers
  until P8).
* **D2 (first self-contained target).**  Kerr's exterior on a shell (the manufactured test with
  the reference removed) or straight to a rotating two-object configuration?
* **D3 (gate G5 scope).**  Are `A`, `kappa`, `Omega_H` from the solution *required*, or is the
  multipole/no-hair route from a shell-only domain enough?  This decides whether P8 is on the
  critical path.
* **D4 (gauge).**  Inhomogeneous de Donder source built from the candidate (recommended), or an
  explicit Kerr-in-harmonic-coordinates asset?
* **D5 (inner object).**  Prescribed rigid rotation `Omega_in` on a material inner sphere, or a
  horizon (no-rotation/co-rotating condition)?  Matter (a rotating fluid/disk) later is a
  separate extension (option C of the original choice).

---

## 8. Suggested order of work and effort

| phase | depends on | rough effort | gate |
|---|---|---|---|
| P0 theory note + conventions | -- | 2-3 days | conventions listed |
| P1 Kerr/NUT assets | P0 | 3-5 days | G0 |
| P2 invariant layer | -- (parallel with P1) | 4-6 days | identities on Kerr |
| P3 4-D residuals + static regression | P1 | 4-6 days | G1 |
| P4 ansatz + losses + manufactured Kerr | P3, P2 | 1-2 weeks | G2 |
| P5 no-reference + tail model | P4 | 1-2 weeks | **G3** |
| P6 convergence study | P5 | 1 week + hub time | **G4** |
| P7 observables report | P6 | 1 week | **G5** |
| P8 horizon domain / new configuration | P7 | 2-4 weeks | new physics |
| P9 docs, hub integration | continuous | 2-3 days | -- |

Effort is focused-work estimates for one experienced developer working with the assistant; the
hub runs (and their queueing) dominate the calendar from P4 on.  Each phase ends with a review
before the next one starts -- the standing instruction from `prompts.txt`.
