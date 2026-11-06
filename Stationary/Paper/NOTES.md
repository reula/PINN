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

## Working on paper.tex (2026-11-01)

- Editor: VS Code with the LaTeX Workshop extension (settings in the repository's .vscode/settings.json:
  latexmk on save, PDF in a tab, word wrap in .tex).  Cmd+Option+J: source -> PDF; Cmd+click in the PDF:
  PDF -> source.  Without VS Code: `latexmk -pdf -pvc paper.tex` in Paper/ and Skim.
- paper.tex has ONE SENTENCE PER LINE (Paper/sentence_per_line.py did it, whitespace only, and checks
  that every paragraph is identical after collapsing whitespace; run it after a large paste:
  `python Paper/sentence_per_line.py Paper/paper.tex`).  Equations, tables, figures and comments are
  left alone.  New text is written that way, so `git diff` shows which sentence changed.
- Small corrections: edit the text directly.  Notes for Claude: \claude{...} (magenta; \notesfalse
  hides them).  Before asking for work, commit your own edits (`git commit -am "my edits"`) or tell
  Claude, who commits them first so that the changes stay separate.
- What changed between two versions: `git diff <old> -- Paper/paper.tex`, or `latexdiff` (a coloured PDF
  with insertions and deletions) on `git show <old>:Stationary/Paper/paper.tex` against the current file.
- The latest PDF is committed (Paper/paper.pdf) and can be opened from GitHub.

## The author's \claude{} notes of 2026-11-01, and what was done

1. "This paragraph should go on the specific solution section" (the free constants lambda_0, S_1, S_2,
   r_in; the scaling lambda -> k lambda fixed by lambda_infty = 1; M from lambda_0 and r_in): moved to
   IV.A after the datum, with the numbers of the quadrupolar problem (M = R_0 = 1/sqrt 3).
2. "This part continues ok": removed (nothing to do).
3. "We don't apply to lambda but to P": II.D rewritten.  The exponents of the asymptotic expansion
   (dh ~ rho^-2, Gamma ~ rho^-3) are imposed on h and its connection; lambda's multipolar expansion
   (-2M/rho + 2M^2/rho^2 ...) violates lambda - 1 ~ rho^-1, so the scalar condition is on the
   Geroch-Hansen potential P (P_infty = 0, i.e. lambda_infty = 1); the Schwarzschild decay rates
   2.00 / 3.005 / 0.952 are now attributed to dh / Gamma / lambda - 1.
4. "I would take away all this" (the extra outer spheres and averaged conditions that were tried):
   the passage is removed.  Four references to it were simplified: IV.A (the parenthetical on the
   mass), V.B (the run that drifted to d_z = 7.6e-5 is now "the same run with extra conditions on the
   outer spheres"), V.D (width 20 plateau: the sentence on the 1.35e-6 run is gone; "points to the
   network and not to the formulation"), appendix.  The numbers stay in VERIFICATION.md.
5. "add a note about the bigger networks": III, width 40 depth 6: 5x41x40 = 8200 hidden parameters,
   4x40 + 8200 + 41x5 = 8565 (axisymmetric), 5x40 + 8200 + 41x7 = 8687 (general); both counts checked
   by building the networks.
6. "Perhaps we can rewrite the above sentence" (sampling): interior points have uniform directions and
   log-uniform radius, i.e. density ~ rho^-3, concentrated near the inner boundary.
7. "This sentence is not clear, neither the ones that follow" (spectral approximations): rewritten;
   the agreement of two unrelated methods is not an artifact of either; a stall of the spectral
   approximation points to the problem (as for h_rA prescribed on the inner sphere); the rotated
   approximation is the spectral reference for off-axis data.
8. "is the exact solution converted into a neural form...?" (pipeline test): NO.  No network is
   trained or fitted, the closed-form fields replace the network function in the same code, and
   automatic differentiation differentiates them directly; the loss measures the residuals of the
   equations, gauge and boundary conditions, not a difference from the exact solution.  The paragraph
   now says so.

Build note: VS Code's LaTeX Workshop rebuilds on file changes; a second build running at the same time
(mine, with latexmk -g or -C) corrupted paper.aux ("File ended while scanning use of \@writefile").
Use plain `latexmk -pdf paper.tex`, never -g/-C while the editor is open; to recover, run "LaTeX
Workshop: Clean up auxiliary files" and build again.

## The author's \claude{} notes of 2026-11-02, and what was done

1. "do we say what t is": IV.A/III spectral approximations: t = ln(rho/rho_in)/ln(rho_out/rho_in) in [0,1], as for the networks.
2. "This notes on commands are not needed in the paper": the command file name is gone from the caption of
   tab:firstorder (the appendix list keeps it).
3. "we can not compare with a first-order network ...": the 2048-point first-order run is no longer cited.
   The point-count evidence is now the metric-only 3-D network on the rotated quadrupole (DSGNAR.md rounds
   20-26, older boundary-condition formulation, width 20, SSBroyden): 16384 points: loss 1.3e-6, rms dh
   1.17e-3, dlam 1.7e-5; 65536: 2.6e-7, 7.2e-4, 1.4e-5; 131072: 1.16e-7, 9.9e-4, 1.0e-5; 262144: 1.18e-7,
   7.5e-4, 1.4e-5.  No rotated-Weyl run with more than 8192 points exists.
4. "Is that M=1 in Schwarzschild?": no, M_0 = R_0 = 1/sqrt 3 (M = 1 is the Schwarzschild TEST, R_0 = 1).
5. "The above paragraph is very complicated ... r_in = rho_in only if M = 0": the problem paragraph is
   rewritten in three sentences.  r_in = rho_in is possible because the harmonic gauge leaves the radial
   map rho -> rho + c_2 F_2(rho) free (c_1 = 1 for h -> delta, c_2 fixed by the inner datum:
   exact.reference_fields_asymptotic builds exactly that chart); in the canonical chart r_h = sqrt(rho^2 -
   R_0^2) < rho.  The paragraph says so and gives M_0 = r_in (1 - lambda_0)/(2 sqrt(lambda_0)) = 1/sqrt 3.
6. "We haven't said what M_0 is ... reserve M_0 for the inner mass": notation changed.  M_0 = R_0 is the mass
   of the spherical (Schwarzschild) reference (II.B, V.B, IV.A); the Geroch-Hansen monopole, the ADM mass of
   the solution, is M_ADM everywhere (extraction formulas, tables, text, fig:mass, fig:hawking, the figure
   scripts).  The mass-scan discussion (tab:mass, fig:mass, the hard-end paragraph) moved from IV.A to the
   multipole section, as a paragraph "The mass as a function of the datum".  M_infty was the alternative.
7. "Why two figures?": 4.2 and 4.7 are two doublings, -1/24 -> -1/12 and -1/12 -> -1/6, not two signs; the
   dependence on the sign is weak (shift 6.50e-3 at -1/12 against 5.57e-3 at +1/12).  Reworded.
8. Found while moving the text: the sentence "The best networks below reach a lower loss (1.3e-11) with
   2285 parameters ..." had been stranded after the Hawking figure by my earlier rewrite; it is back in the
   spectral-approximation paragraph of IV.A.  Bare "reference" in the multipole section and abstract
   became "spectral reference".

## Evidence review (2026-11-06): which claims need more runs

Paper read against VERIFICATION.md, the run configs on the hub and the code history.

### Runs needed

1. **The axisymmetric Weyl tests were trained under the old inner condition.** weyl_prod, weyl_unequal,
   weyl_close4 and weyl_close35 (2026-09-25..28) used `--inner-bc reference` before 0773b6a (2026-10-15).
   Until then that condition imposed the FULL h from the exact solution (`pack_sym(f.h - e.h)`), h_rr and
   h_rA included. The paper (IV.C) says that only lambda and the induced metric are imposed. So the
   Weyl figure, tab:weylclose, the Weyl moments and the comparison "3-D vs axisymmetric on the same shell"
   (IV.D; weyl_rot45_hyb is in the new formulation) come from an easier problem than the one stated.
   They also predate resampling and the fresh-sample check (4096 points, Armijo). Rerun all four in the
   current code (induced metric only, resample 500, fresh/train, moments). This is cheap: minutes per run.
2. **Off-axis, width or points?** qp_rot4 changed width 20 -> 40 AND points 16384 -> 32768 at the same
   time, but V.C credits the gain to the width alone ("points to the network"). Two runs separate the two
   effects: width 40 at 16384, and width 20 at 32768 (hub_qp4.sh settings otherwise). The same pair would
   replace the point-count paragraph of IV.D, which rests on older-formulation runs (DSGNAR rounds 20-26).
3. **The spectral reference's own error at -1/12.** The far-field lambda error (~7e-9) is about the size
   of the change seen when swapping references: ab_none_w2 is 9.1e-9 against the old reference and 7.1e-9
   against the final-formulation one. One reference with a larger basis (l <= 14, J = 48, as at -1/4) and a
   regrade of ab_none_w2 / fn_nocom_w2 against it would bound how much of tab:com is the reference.
   tab:com should also be graded against the final-formulation reference.
4. **Seeds for the central claim.** Each claim in V.B rests on one run per arm: the 3900x factor, "the loss
   does not see the difference", and "the size without the condition is not fixed". With 2 more seeds of
   ab_none and fn_nocom (about 25 min each), the claim becomes: with the condition the error is
   reproducible, without it d_z scatters. Optional: 2 seeds of fn_tan. The H^1 gain in axisymmetry is
   only 1.3-1.5x, which may be within seed scatter.

### Optional runs

5. **The hard end.** tab:mass picks the width-40 networks at -1/4 (fresh/train 11) and -0.30 (fresh/train
   170, M2 4% off) as "the best". Either retrain width 40 with more points and denser redraws (e.g.
   65536, every 250), or quote ms_s2_025_w5 (fresh 1.17) at -1/4 and drop or flag the -0.30 network row.
6. **Networks at -1/8 and +1/12**, so that the sign claim has networks as well as references (25 min each).

### Text only (no runs)

7. III "Every result is checked on five fresh samples ... within a few per cent": this is false for the
   hard-end width-40 runs (11, 170) and for rotated Weyl (1.18). Restrict it to the runs that pass, and
   give the ratio where they do not.
8. II.D says Robin conditions are imposed "on h and on its connection". The quadrupolar runs have
   robin_include_G = False, with orders h: 3, P: 3; the Weyl runs have include_G = True, order 2. Neither
   the order n nor this difference is stated.
9. IV.D "the 2048 of the axisymmetric tests": the Weyl runs used 4096, the quadrupole 16384.
10. II.D: "|a| ~ 0.04 ... 2.5e-6" comes from COM_GAUGE.md (old-formulation runs, a = -0.0445). Fine as
    an illustration; VERIFICATION.md does not list it yet.
