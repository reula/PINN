#!/usr/bin/env bash
# The reference SSBroyden solve, driven to convergence.
#
#   bash scripts/run_reference.sh            # the default feature map
#   FEATURES=fourier_ic MODES=6 bash scripts/run_reference.sh
#
# Everything else is the requested configuration: u_tt - c^2 u_xx = 0 on
# [-1, 1] to T = 2, periodic, Gaussian u0 with sigma = 0.2, v0 = -c u0',
# hard-coded initial condition, 6 layers x 20 neurons.
set -eu
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
PY="$(find_python)" || { echo "no interpreter with jax; set PY=" >&2; exit 1; }
cd "$(dirname "$0")/.."

LABEL=${LABEL:-ssbroyden_ref}
FEATURES=${FEATURES:-periodic}
MODES=${MODES:-1}
STEPS=${STEPS:-12000}
NCOLL=${NCOLL:-4096}

$PY -m wave_pinn.cli --label "$LABEL" \
  --set optimizer=ssbroyden \
  --set features="$FEATURES" \
  --set n_modes="$MODES" \
  --set qn_steps="$STEPS" \
  --set qn_block=250 \
  --set qn_gtol=1e-14 \
  --set qn_initial_scale=true \
  --set plateau_tol=1e-9 \
  --set plateau_patience=4 \
  --set n_coll="$NCOLL" \
  --set sampler=random \
  --set eval_every=2500 \
  --set seed=0
