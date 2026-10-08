#!/usr/bin/env bash
# DSGNAR (the paper's optimiser) on the same problem, for the same comparison.
#
#   bash scripts/run_dsgnar.sh
#   DSGNAR_STEPS=200 SKETCH=800 bash scripts/run_dsgnar.sh
set -eu
PY=${PY:-/Users/reula/jax_env/bin/python}
cd "$(dirname "$0")/.."
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl-wazepinn}

LABEL=${LABEL:-dsgnar_ref}
FEATURES=${FEATURES:-periodic}
MODES=${MODES:-1}
STEPS=${DSGNAR_STEPS:-150}
SKETCH=${SKETCH:-0}          # 0 -> floor(d_theta/3)
NCOLL=${NCOLL:-2048}

$PY -m wave_pinn.cli --label "$LABEL" \
  --set optimizer=dsgnar \
  --set features="$FEATURES" \
  --set n_modes="$MODES" \
  --set dsgnar_steps="$STEPS" \
  --set dsgnar_sketch="$SKETCH" \
  --set dsgnar_stage1_ratio=0.15 \
  --set dsgnar_stage2_ratio=0.5 \
  --set n_coll="$NCOLL" \
  --set sampler=random \
  --set eval_every=25 \
  --set seed=0
