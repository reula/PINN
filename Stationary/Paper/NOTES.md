# Paper draft: sources, open items and conflicts

`paper.tex` (REVTeX 4.2, PRD), `refs.bib`, `figs/`. Build: `latexmk -pdf paper.tex`. For Overleaf,
upload `paper.tex`, `refs.bib` and `figs/`. For CQG, switch to the `iopart` class.

The draft follows the outline in `prompts.md`. Every number in it comes from the project notes;
nothing was recomputed for the paper.

## Where each section's numbers come from

| section | sources |
|---|---|
| II equations, gauge, BCs, COM flux | README.md 1-4, 8.4-8.8; COM_GAUGE.md 4b; PERTURBATION.md; stationary/losses.py, geometry.py, problem.py |
| III networks, loss, optimisers, specref | README.md 6.1, 8.6; REPORT_OPTIMISERS.md; DSGNAR.md 1-9; HUB.md (timings); stationary/specref.py |
| IV tests | README.md 8.1-8.12 (m2R3_symhybrid, m2R4_realrobin, weyl_prod, weyl_unequal, weyl_close4/35, weyl_rot45*) |
| V quadrupole | REPORT_PIN_COM.md; PERTURBATION.md; COM_GAUGE.md; DSGNAR.md rounds 3-27; MULTIPOLES_GH_THORNE.md |
| figures | figs/optimisers.png = plots/report_optimisers.png; figs/com_steps.png = plots/report_pin_com.png; figs/weyl_plane.png = runs/weyl_prod/plane/plane.png |

## Open items (marked TODO in the text)

1. **Authors, affiliations, funding.**
2. **Citations to choose:**
   - earlier numerical work on the stationary exterior problem;
   - minimal-energy data sets;
   - higher-order Newtonian-limit expansions;
   - the Crunch implementation (Palma group: names, and how they want to be cited).
3. **References to verify.** Every `refs.bib` entry marked VERIFY was written from memory or from
   the project notes and must be checked:
   - Reula 1989;
   - Gürsel 1983;
   - FHP 1989;
   - Cederbaum–Sakovich;
   - Urbán–Stefanou–Pons (SSBroyden);
   - DSGNAR. Its authors and title are as given in DSGNAR.md (Webb, Jerad and Cartis, arXiv:2607.02194).
4. **COM attribution (done 2026-10-23).** The flux is Cederbaum's pseudo-Newtonian centre of mass
   (thesis arXiv:1201.5433, §4.3: 4πG m z^k = ∮(∂U/∂ν x^k − U ∂x^k/∂ν) dσ in γ-harmonic coordinates,
   by Green's formula; quasi-local; with an addition theorem; equal to the ADM/CMC centres; with a
   Newtonian limit). The paper cites the thesis and Cederbaum–Nerz 2015 (Ann. Henri Poincaré 16,
   1609). It also cites the earlier works they build on, taken from their bibliographies:
   - centre-of-mass definitions: Regge–Teitelboim, Beig–Ó Murchadha, Huisken–Yau, Metzger, Huang
     (2009, 2010), Corvino–Wu, Eichmair–Metzger, and the Corvino–Pollack review;
   - static asymptotics: Beig 1980, Beig–Simon 1980, Kennefick–Ó Murchadha 1995;
   - multipoles: Beig 2000.

   Entries still marked VERIFY: Huang's years (the two bibliographies disagree), the Corvino–Wu
   article number, and the Eichmair–Metzger volume.
5. **Multipole moments (done 2026-10-23).** Section V.D now has Geroch–Hansen moments extracted
   from the axis (stationary/gh_moments.py, scripts/gh_moments_runs.py, tests/test_gh_moments.py)
   and the tail-decay checks (scripts/tail_decay.py). See MULTIPOLES_GH_THORNE.md §6.2b. Still
   open: the window spread of M4 (7e-4) is the limiting error bar, and the FHPS route through U
   differs from the Ernst route by 1.7% in M2 for the quadrupolar solution (truncation of the ln
   series, as on exact Schwarzschild). The paper quotes the Ernst route.
6. **The 0.062 ratio.** It is measured (Table IV) but not derived. A short linearised calculation
   of the translation mode's effect on λ near the inner sphere would explain it.
7. **The weakly constrained mode of the 3-D ansatz** is not identified (end of V.E). An angular
   decomposition of network − reference would settle it.
8. **Figures.** All three carry internal titles and run names. Regenerate them for the paper, in
   journal style and without titles. In the optimisers figure, the third panel's "SSBroyden stalls"
   refers to the legacy line search; the caption says so.
9. **Rotated Weyl with the metric-only 3-D network.** The rotated Weyl runs used the first-order
   network. For consistency with the paper's recommendation, a run with `--arch hybrid` would be
   worth adding.
10. **Round 28** (merged-Crunch Wolfe on the rotated quadrupole) is still running and not in the draft.

## Conflicts between sources, and the choice made

- **`weyl_prod` max |δh|.** report.txt gives 4.83e-7 and report_eval.json gives 6.70e-7. The paper
  uses 6.7e-7, from report_eval.json, as the README's ratio table does. Wall time: the README says
  7424 s, report.txt says 124.4 min. The paper quotes no wall time for it.
- **Rotated Weyl errors.** The paper uses the README table (report_eval.json). report.txt's
  "over the shell" lines differ: 5.6e-2, 2.3e-1 and 7.9e-4.
- **Closer Weyl shells, max |δh|.** The README gives 1.57e-5 and 7.72e-5; report_eval.json gives
  1.540e-5 and 7.295e-5. The paper rounds to the README's 1.6e-5 and 7.7e-5. Check which
  evaluation grid is meant.
- **4-d line element.** ROTATING_PLAN.md writes -λ² dt² + h. The code and README use
  g = -λ dt² + λ⁻¹ h, and the paper follows the code.
- **F₂ in exact.py.** The docstring disagrees with the code. The paper uses the code's (and the
  README's) F₂ = 2R₀ + ρ ln((ρ−R₀)/(ρ+R₀)).
- **Axisymmetric ansatz.** The model.py docstring omits the ½ on the c term. The paper uses the
  code's ½.
- **Parameter counts.** 2265 (Weyl runs, no decay feature) against 2285 (quadrupole runs, with
  decay feature). Both are stated.
- **Regression-fit file.** `runs/rot_ssbm3/fit_reference.json` on disk may still hold the 512-point
  smoke fit. The paper quotes the full fit from the job logs (rms δh 7.5e-6, PINN loss 3.4e-2).

## Add in Conclusions: 
Other boundary conditions, like Barnik's proposal would be investigated in the future. 

## Bib:

Quote other people working on the topic, check who quotes my 1988 paper. In particular Michel Andersson did also some work without citing my paper. There is in arxiv a recent paper on this, I am one of the authors which have citations on other papers, notably on the use of the SSBroyden crunch. 

## Are the Robin ladder and the averaged pins still needed? (2026-10-27)

All three were introduced while the inner boundary still imposed h_rA = 0 (removed in 0773b6a,
10-15), when no decaying solution existed: pin_lam_robin 09-29, pin_h_robin 09-30, and
outer_shell_fracs (the ladder) 10-14.  Their motivating symptoms fit that over-determination.
The ladder was added against a constant l = 2 metric part of ~6% in the bulk, the same
non-decaying l = 2 mode the spectral solver stalled on under h_rA = 0.  The metric pins were added
against h drifting to a multiple of the identity.  The P pin (10-17) and the centre-of-mass
condition (10-18) came after the fix.  The ablations at the end of Paper/hub_qp.sh (ab_noladder,
ab_nopins, ab_none: qp_com from scratch without the ladder, without the pins, without both) test
whether the method can drop them.


## The author's \claude{} notes of 2026-10-29, and what was done

- Weyl two-rod subsubsection: moved from II.C to the tests section (V.C, label sec:weylsol kept);
  its rotation paragraph merged into "A rotated Weyl configuration". II.C is now "The spherical
  solutions" (sec:spherical).
- "spectral solution" in II.D: now "a spectral approximation" with a forward reference; the
  spectral method is introduced in III ("Spectral approximations", eq:spectral: the basis of
  stationary/specref.py, Gauss-Newton on the loss's residual vector) with its two uses (grading;
  how small the loss can be: a floor reached by both methods is the problem's). The 1.8e-5 / 3e-9
  pair is the l <= 6, 24-mode approximation (VERIFICATION.md, PERTURBATION.md:287,304).
- lambda_0 "chosen so that lambda -> 1": replaced. lambda_0, S_1, S_2, r_in are free data; the
  symmetry is lambda -> k lambda (scaling, not adding a constant), fixed by lambda_infty = 1; for
  spherical data M = r_in (1 - lambda_0) / (2 sqrt(lambda_0)) (1/sqrt 3 for the quadrupole
  problem, checked).
- Parameter counts: explained in III from eq:mlp, (n_in+1) n_out per layer: axisym 3 in / 5 out
  -> 2285, 3-D 4 in (n, t; no decay feature in HybridNet) / 7 out -> 2347 (stationary/model.py).
- Loss: what a loss and a residual are, each term named, the bar defined.
- Spectral paragraph reworded; "l = 2 tail decays as rho^-3" now says it is the l = 2 part; the
  reference's specifics (L = 10, J = 36, 1008 unknowns, 2500 points, 37 min, DSGNAR.md:674-679)
  moved to IV.A, with its mass and the remark that the best networks now have a lower loss
  (1.3e-11 vs 2.8e-11), so the quoted errors compare two approximations of similar quality.
- Mass: M0 = 0.583847 vs R0 = 0.577350, back-reaction 6.5e-3 ~ S2^2 = 6.9e-3. Scaling with S2 NOT
  checked (would need a spectral solve at S2/2).
- Pipeline test: explained, with the closed-form Schwarzschild fields as the example.
- "standard shell" (Weyl): rho_in = 7.5 (3 x 2.5, the distance to the rods' ends), rho_out = 75;
  table caption "standard units" -> the units of masses 1 and 0.1. (HUB.md calls [3.5, 35] the
  standard shell of the rotated runs: a different usage, not in the paper.)
- Ablation subsection and tab:ablation removed; its numbers are two sentences in II.D. The table
  rows remain in VERIFICATION.md.
- Also edited in the author's new text: typos (choosen, Schwarszchild, Schwarszshild, ofcourse,
  firts), "the activation function is a tanh multilayer perceptron", "root seeking algorithms"
  (Adam and SSBroyden minimise), "asserts the stability of the approximation parameters"
  (-> overfitting check).

Conflict (resolved 2026-10-31): PERTURBATION.md item 4 said the mass is untouched by the
quadrupole; the moments give +1.1%, and the S2 scan (hub_mass.sh) confirms a 0.83 S2^2 shift.
PERTURBATION.md carries a dated correction; the paper has tab:mass.
