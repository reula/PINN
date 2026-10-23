#!/usr/bin/env bash
# The Schwarzschild tests of the paper (Sec. IV B, Table II), with the current code.
# No averaged outer pins (--no-pin-h-robin): the original runs had none, and in this chart (c2 != 0)
# the exact solution violates the tangential pin at 4.3e-10 (tests/test_pipeline.py).
# Same problem as the original runs m2R2_asym1 / m2R3_symhybrid / m2R4_realrobin (recovered in
# runs/_restored from commit 92cb622~1): R0 = 1, the inner sphere rho = 2 with lambda_0 = 1/phi^2 on
# the areal-radius-2 sphere, Robin conditions at rho_out = 100, lambda_inf = 1, the spherical
# ansatz of width 64, depth 4, 8 Fourier modes; 20000 Adam + 2000 SSBroyden iterations.
# On Mendieta each is submitted as  sbatch -J <name> mendieta_cmd.sbatch -m stationary.train <flags>
COMMON=(--R0 1.0 --rho-in 2.0 --rho-out 100.0 --outer-bc robin --ref-solution --ref-asymptotic 1.0
        --lam0 0.3819660112501051 --lam-inf 1.0 --width 64 --depth 4 --fourier 8 --decay-feature
        --n-coll 4096 --n-bnd 256 --steps 20000 --pde-ramp-steps 1000 --reweight-every 2000
        --qn-method ssbroyden --lbfgs-steps 2000 --plateau-min-iters -1 --no-pin-h-robin)
# first order (h, Gamma, lambda) with the manufactured Robin source
FIRST=(--outdir runs/schw_first_order --arch sym --robin-source "${COMMON[@]}")
# metric only (h, lambda), manufactured Robin source
METRIC=(--outdir runs/schw_metric_manuf --arch sym_hybrid --robin-source "${COMMON[@]}")
# metric only, the real (homogeneous) Robin conditions
REAL=(--outdir runs/schw_metric_real --arch sym_hybrid "${COMMON[@]}")
if [ "${SUBMIT:-0}" = "1" ]; then
    for v in FIRST METRIC REAL; do
        eval "args=(\"\${$v[@]}\")"
        name=$(basename "${args[1]}")
        echo "$name -> $(sbatch --parsable -J "$name" mendieta_cmd.sbatch -m stationary.train "${args[@]}")"
    done
fi
