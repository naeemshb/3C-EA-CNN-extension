#!/bin/zsh
# GP hyper-parameter tuning on VALIDATION ONLY (declared 2026-11-09, before any tuning run).
# Reduced search budget (pop 50 x 15 gen = 750 evaluations, ~1 h each on the M3 Pro) so that
# 4 configurations x 2 evolution seeds fit in one overnight. Selection-only runs: no final
# training, the test partition is never read (final.enabled=false). Criterion, fixed in advance:
# long-protocol validation accuracy of the SELECTED formula (two-stage rule, D12) averaged over
# the two seeds. If no configuration beats the reference setting (p_mut 0.15, p_cross 0.70,
# 2 elites) by more than the seed-to-seed spread, the paper's values are kept.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
BASE=cnn_extension/configs/reference_fashion_mnist.json
# NOTE: argparse keeps only the LAST --set occurrence, so every override goes into ONE --set list.
COMMON=(out_root=runs_tune final.enabled=false search.population_size=50 search.generations=15)
run() { name=$1; shift; caffeinate -i $PY -m cnn_extension.run grid --base $BASE --method cnn_evolved_zmc \
          --pattern random --rate 0.2 --replicate 0 1 --set "${COMMON[@]}" name=$name "$@" ; }
run tune_ref                                              # reference: p_mut 0.15, p_cross 0.70, elites 2
run tune_mut30   search.mutation_rate=0.30
run tune_cross90 search.crossover_rate=0.90
run tune_elite5  search.elite_size=5
$PY -m cnn_extension.tune_summary --runs runs_tune
