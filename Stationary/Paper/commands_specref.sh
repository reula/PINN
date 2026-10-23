#!/usr/bin/env bash
# The fine spectral reference of the quadrupolar problem (Sec. III, "A spectral reference"), as built
# on 2026-10-17 (job 1180550, logs/slurm_specref_fine_1180550.out): modes l = 0,2,..,10, 36 Chebyshev
# modes in ln rho (1008 unknowns), 2500 collocation and 400 boundary points, the rho^4 weights and the
# pin on P; Gauss-Newton until the residual stops falling.  It is solved on the problem of the run
# directory it is given (the quadrupolar recipe; runs/ib16k_ssbroyden here) and written there as
# specref_nonlinear_<tag>.npz.  The rerun uses the tag fine_rerun, so the original is not overwritten.
ARGS="--nonlinear --ls 0,2,4,6,8,10 --n-radial 36 --n-coll 2500 --n-bnd 400 --iters 6 --scale-exps 3,4,3,4 --pin-lam-potential"
if [ "${SUBMIT:-0}" = "1" ]; then
    echo "specref_fine_rerun -> $(SPECREF_ARGS="$ARGS" SPECREF_TAG=fine_rerun sbatch --parsable -J specref_fine_rerun mendieta_specref.sbatch runs/ib16k_ssbroyden)"
fi
