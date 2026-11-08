#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Supervised fit to the 1D reference, then the PINN loss of the fitted network (Mac mini, CPU; 2026-11-05).
#
#   bash jobs/mini/fit_ref.sh            (resumable item by item: an item whose fit_ref_wNNN.npz exists is skipped)
#
# The question: how small can the network's error get when it is told the answer (pointwise mean|f - f_ref|^2 on
# the reference's own nodes), and what PINN loss does a network that close to the exact solution have? That loss
# is the yardstick for the PINN runs' loss floors (1d_T8_adm30_ic: held ~1e-7). Reference: the same 1D evolution as
# ref_single_m0.5_dr0.005_t20.npz (dr 0.005), saved every 0.025 (10 snapshots per window of 0.25):
#   cd reference_1d && JAX_PLATFORMS=cpu python sph_reference.py --dr 0.005 --tfinal 8 --snap_dt 0.025 \
#       --r_save 24.5 --out ref_single_m0.5_dr0.005_t8_snap0025.npz          (4 min on the mini)
# 1D: sing_1d.py with the hub runs' settings (2000 points, 4000 iterations), the constraint parts reported at
# weight 30 (as 1d_T8_adm30_ic; they do not enter the fit). 3D: sing_pinn.py, plain 3D MLP (no --radial_net),
# 32000 fit points per draw (the baseline's 128000 is too slow on the CPU), 8000 iterations.
# Sobolev variants (runs/fit_ref_*_sob, --fit_d1 1 --fit_d2 1): also r f' and r^2 f'' (1D) / r grad u and r^2 grad grad u
# (3D) against the reference's 4th-order grid derivatives (accurate to 1e-8 for r >= 0.02 on the exact t = 0 data).
# Sobolev+dt (runs/fit_ref_1d_sobt, 2026-11-08, the user's idea): also d_t f against 4th-order time differences of the
# snapshots (dt 0.025), so the fit constrains the time dependence the residual is made of.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-../Stationary/.venv/bin/python}
REFN=reference_1d/ref_single_m0.5_dr0.005_t8_snap0025.npz
export JAX_ENABLE_X64=1
mkdir -p runs/fit_ref_1d runs/fit_ref_3d runs/fit_ref_1d_sob runs/fit_ref_3d_sob logs
for k in 0 16 31; do
    f=$(printf "runs/fit_ref_1d/fit_ref_w%03d.npz" "$k")
    [ -e "$f" ] && { echo "skip 1D window $k"; continue; }
    echo "--- 1D window $k  $(date -u '+%F %T UTC')"
    "$PY" -u sing_1d.py --out runs/fit_ref_1d --hard_ic --no_template --tfinal 8 --N_window 32 \
        --fit_ref "$REFN" --fit_window "$k" --iters 4000 --w_adm 30 --w_adm_ic 30 >> logs/mini_fit_ref_1d.log 2>&1
done
for k in 16 31; do
    f=$(printf "runs/fit_ref_3d/fit_ref_w%03d.npz" "$k")
    [ -e "$f" ] && { echo "skip 3D window $k"; continue; }
    echo "--- 3D window $k  $(date -u '+%F %T UTC')"
    "$PY" -u sing_pinn.py --out runs/fit_ref_3d --template bl --hard_ic --no_template --tfinal 8 --N_window 32 \
        --fit_ref "$REFN" --fit_window "$k" --n_int 32000 --n_held 8000 --iters 8000 >> logs/mini_fit_ref_3d.log 2>&1
done
for k in 0 16 31; do
    f=$(printf "runs/fit_ref_1d_sob/fit_ref_w%03d.npz" "$k")
    [ -e "$f" ] && { echo "skip 1D Sobolev window $k"; continue; }
    echo "--- 1D Sobolev window $k  $(date -u '+%F %T UTC')"
    "$PY" -u sing_1d.py --out runs/fit_ref_1d_sob --hard_ic --no_template --tfinal 8 --N_window 32 \
        --fit_ref "$REFN" --fit_window "$k" --iters 4000 --w_adm 30 --w_adm_ic 30 --fit_d1 1 --fit_d2 1 >> logs/mini_fit_ref_1d_sob.log 2>&1
done
for k in 16 31; do
    f=$(printf "runs/fit_ref_3d_sob/fit_ref_w%03d.npz" "$k")
    [ -e "$f" ] && { echo "skip 3D Sobolev window $k"; continue; }
    echo "--- 3D Sobolev window $k  $(date -u '+%F %T UTC')"
    "$PY" -u sing_pinn.py --out runs/fit_ref_3d_sob --template bl --hard_ic --no_template --tfinal 8 --N_window 32 \
        --fit_ref "$REFN" --fit_window "$k" --n_int 32000 --n_held 8000 --iters 8000 --fit_d1 1 --fit_d2 1 >> logs/mini_fit_ref_3d_sob.log 2>&1
done
for k in 0 16 31; do
    f=$(printf "runs/fit_ref_1d_sobt/fit_ref_w%03d.npz" "$k")
    [ -e "$f" ] && { echo "skip 1D Sobolev+dt window $k"; continue; }
    echo "--- 1D Sobolev+dt window $k  $(date -u '+%F %T UTC')"
    "$PY" -u sing_1d.py --out runs/fit_ref_1d_sobt --hard_ic --no_template --tfinal 8 --N_window 32 \
        --fit_ref "$REFN" --fit_window "$k" --iters 4000 --w_adm 30 --w_adm_ic 30 --fit_d1 1 --fit_d2 1 --fit_dt 1 >> logs/mini_fit_ref_1d_sobt.log 2>&1
done
echo "=== fit_ref done $(date -u '+%F %T UTC')"
