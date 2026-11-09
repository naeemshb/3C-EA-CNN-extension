#!/bin/zsh
# train (GP search + final training, validation only) then evaluate (frozen test) for ONE dataset. Resume-safe.
cd "$(dirname "$0")/.."
PY=${PY:-python}; ds=$1
$PY -m telo_tabular train --config configs/main_$ds.json --output runs_main/$ds > logs/main_$ds.log 2>&1 && \
$PY -m telo_tabular evaluate --run runs_main/$ds >> logs/main_$ds.log 2>&1
echo "done $ds exit=$? $(date)" >> logs/main_done.log
