#!/usr/bin/env bash
# The Schwarzschild tests of the paper (Sec. IV B, Table II), with the current code (2026-11-09).
# R0 = 1, the inner sphere rho = 2 with lambda_0 = 1/phi^2 on the areal-radius-2 sphere, Robin
# conditions of ORDER 2 on h and P at rho_out = 100, lambda_inf = 1, the spherical ansatz of width 64,
# depth 4, 8 Fourier modes; 20000 Adam + 2000 SSBroyden iterations with the strong-Wolfe line search
# (the earlier runs used Armijo), points redrawn every 500.  No averaged outer pins (--no-pin-h-robin):
# in this chart (c2 != 0) the exact solution violates the tangential pin at 4.3e-10.
# The metric-only networks get no condition on Gamma; the first-order network, which has an equation
# for Gamma, gets the corresponding order-2 Robin condition on Gamma.
# Why order 2 and not the order 3 of the quadrupole recipe (Paper/VERIFICATION.md, "Schwarzschild with
# Wolfe"): with the residual weights p = (2,1,2) used here, order 3 leaves max errors ~2e-4 at the same
# loss (sx_wolfe_o3G, sx_armijo_o3noG, and phase K of hub_recipe.sh); with p = (4,3,4) it gives 2.9e-6
# in h (sx_o3noG_rho4), still above order 2.
# On Mendieta:  SUBMIT=1 bash Paper/commands_schwarzschild.sh   (7-9 min each on one A30)
COMMON=(--R0 1.0 --rho-in 2.0 --rho-out 100.0 --outer-bc robin --ref-solution --ref-asymptotic 1.0
        --lam0 0.3819660112501051 --lam-inf 1.0 --width 64 --depth 4 --fourier 8 --decay-feature
        --n-coll 4096 --n-bnd 256 --steps 20000 --pde-ramp-steps 1000 --reweight-every 2000
        --qn-method ssbroyden --qn-linesearch wolfe --lbfgs-steps 2000 --plateau-min-iters -1
        --no-pin-h-robin --resample-every 500 --ckpt-every 500 --no-figures)
# first order (h, Gamma, lambda) with the manufactured Robin source, order 2 on h, Gamma, P
FIRST=(--outdir runs/sx_first_o2 --arch sym --robin-source --robin-orders h=2,G=2,lam=2 "${COMMON[@]}")
# metric only (h, lambda), manufactured Robin source, order 2 on h and P
METRIC=(--outdir runs/sx_wolfe_o2noG --arch sym_hybrid --robin-source --robin-orders h=2,lam=2 --no-robin-G "${COMMON[@]}")
# metric only, the real (homogeneous) Robin conditions
REAL=(--outdir runs/sx_real_o2noG --arch sym_hybrid --robin-orders h=2,lam=2 --no-robin-G "${COMMON[@]}")
if [ "${SUBMIT:-0}" = "1" ]; then
    for v in FIRST METRIC REAL; do
        eval "args=(\"\${$v[@]}\")"
        name=$(basename "${args[1]}")
        echo "$name -> $(sbatch --parsable -J "$name" mendieta_cmd.sbatch -m stationary.train "${args[@]}")"
    done
fi
