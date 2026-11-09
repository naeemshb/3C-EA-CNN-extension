#!/bin/zsh
# Main GP comparison over all datasets: ONE dataset at a time by default (PARALLEL=n to change), each via
# scripts/run_one_dataset.sh. Logs: logs/main_<dataset>.log; completion marks: logs/main_done.log. Resume-safe.
cd "$(dirname "$0")/.."
export PY=${PY:-python}
mkdir -p logs runs_main
# sequential, smallest dataset first (rows): hepatitis 155, sonar 208, glass 214, horse 299, housevote 435, cylinderband 539,
# wdbc 569, aust_credit 690, pima 768, pima_zeros_missing 768, mammo 830, heart 1025, mushroom 8124, adult 32561
DATASETS=(hepatitis sonar glass horse housevote cylinderband wdbc aust_credit pima pima_zeros_missing mammo heart mushroom adult)
PARALLEL=${PARALLEL:-1}
printf '%s\n' $DATASETS | xargs -P $PARALLEL -n 1 ./scripts/run_one_dataset.sh
echo "ALL DONE $(date)" >> logs/main_done.log
