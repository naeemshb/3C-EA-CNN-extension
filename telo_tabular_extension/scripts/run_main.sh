#!/bin/zsh
# Main GP comparison over all datasets: up to 6 datasets in parallel (2 torch threads each), each via
# scripts/run_one_dataset.sh. Logs: logs/main_<dataset>.log; completion marks: logs/main_done.log. Resume-safe.
cd "$(dirname "$0")/.."
export PY=${PY:-python}
mkdir -p logs runs_main
DATASETS=(adult aust_credit cylinderband glass heart hepatitis horse housevote mammo mushroom pima pima_zeros_missing sonar wdbc)
printf '%s\n' $DATASETS | xargs -P 6 -n 1 ./scripts/run_one_dataset.sh
echo "ALL DONE $(date)" >> logs/main_done.log
