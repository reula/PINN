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
4. **Novelty of the COM condition.** COM_GAUGE.md says a literature search found no quasi-local
   centre-of-mass condition of this form. That claim needs a proper check before it is stated.
5. **Multipole moments.** No Geroch–Hansen or Thorne moment of a computed solution exists yet. The
   paper reports only harmonic tail coefficients of P and of λ, and says so. Computing M0 and M2 of
   the quadrupole solution is the main missing result for that section (MULTIPOLES_GH_THORNE.md §6.2).
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
