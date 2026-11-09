#!/bin/zsh
# Main GP comparison: one run directory per dataset (runs_main/<dataset>): train (GP search + final training,
# validation only) then evaluate (frozen test, all conditions). Up to 6 datasets in parallel, 2 torch threads each.
# Resume-safe: re-running skips finished steps. Logs: logs/main_<dataset>.log, completion marks in logs/main_done.log
cd "$(dirname "$0")/.."
export PY=${PY:-python}
mkdir -p logs runs_main
DATASETS=(adult aust_credit cylinderband glass heart hepatitis horse housevote mammo mushroom pima pima_zeros_missing sonar wdbc)
printf '%s\n' $DATASETS | xargs -P 6 -I{} zsh -c '
  ds={}
  $PY -m telo_tabular train --config configs/main_$ds.json --output runs_main/$ds > logs/main_$ds.log 2>&1 && \
  $PY -m telo_tabular evaluate --run runs_main/$ds >> logs/main_$ds.log 2>&1
  echo "done $ds exit=$? $(date)" >> logs/main_done.log'
echo "ALL DONE $(date)" >> logs/main_done.log
