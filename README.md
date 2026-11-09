# 3C-EA: Evolving Multi-Channel Confidence-Aware Activation Functions for Missing Data

Code, logs and results for the GECCO 2026 paper *Evolving Multi-Channel Confidence-Aware
Activation Functions for Missing Data with Channel Propagation* (3C-EA) and for its extension
to convolutional networks (journal version, in progress).

## Contents

| Path | What |
|---|---|
| `Evolving_Activation_Functions_with_Missing_Data__GECCO_2026_Final_ (1).pdf` | the GECCO 2026 paper |
| `Final/<Dataset>/` | original tabular experiments (one folder per UCI dataset): GP script, logs, results, figures |
| `Final/Pima/New.py` | the released GP implementation the CNN extension builds on (kept verbatim) |
| `Final/cnn_extension/` | **CNN extension**: convolutional ChannelProp, image corruption, GP search, experiments, tests, `README.md` with equations, protocol, deviations and results so far |
| `Final/runs/` | CNN experiment runs (`results.json`, logs, checkpoints, predictions, shift evaluations), `step3_*_summary.md`, `metrics_*.md` |
| `Final/runs_pilot/`, `Final/runs_fidelity/`, `Final/runs_tune/` | engineering runs: pilot, fitness-protocol fidelity check, GP-parameter tuning (stopped) |
| `Final/scripts/` | batch scripts |
| `*.py`, `*.csv` at the top level | early single-file versions of the tabular experiments and their data |

## CNN extension: quick start

```bash
cd Final
pip install -r cnn_extension/requirements.txt
python -m pytest cnn_extension/tests            # 86 tests
python -m cnn_extension.run grid --base cnn_extension/configs/reference_fashion_mnist.json \
    --method cnn_relu cnn_evolved_zmc --pattern random --rate 0.2 --replicate 0
```

Datasets (Fashion-MNIST, MNIST, CIFAR-10 via torchvision; BloodMNIST via `medmnist`) download on first
use into `Final/data/`, which is not versioned.

## Status

See `Final/cnn_extension/README.md`, sections 5 and 8, for what has run and what the numbers mean.
All image results so far are single-replicate engineering runs on Fashion-MNIST at 20 % random
missingness; they are decision gates, not reportable results.
