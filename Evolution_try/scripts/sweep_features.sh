#!/usr/bin/env bash
# Feature-map sweep: which input map lets the network represent the narrow
# Gaussian pulse at all?  Short runs (2500 SSBroyden iterations, 2048 points),
# one per feature map, sequential because JAX takes the whole machine.
#
#   bash scripts/sweep_features.sh
set -u
PY=${PY:-/Users/reula/jax_env/bin/python}
cd "$(dirname "$0")/.."
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl-wazepinn}

common="--set optimizer=ssbroyden --set qn_steps=2500 --set qn_block=250 \
        --set n_coll=2048 --set sampler=random --set plateau_tol=1e-10 \
        --set eval_every=500 --set seed=0"

for spec in "f6 features=fourier n_modes=6" \
            "f6ic features=fourier_ic n_modes=6" \
            "f12 features=fourier n_modes=12" \
            "f12ic features=fourier_ic n_modes=12"; do
  set -- $spec
  name=$1; shift
  echo "=== $name : $* ==="
  $PY -m wave_pinn.cli --label "sweep_$name" $common $(for kv in "$@"; do echo --set $kv; done) 2>&1 | tail -14
done
echo "sweep done"
