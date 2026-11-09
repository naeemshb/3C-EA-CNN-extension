# Tabular TELO confidence extension

This project implements the first tabular extension of the GECCO 2026 3C-EA method: entry-specific imputation confidence, unchanged original GP primitives and ChannelProp, controlled ablations, and frozen-model missingness-shift evaluation.

**Start with the confidence pilot below.** It uses PIMA and WDBC, takes no class labels into confidence estimation, and does not evaluate the test partitions or launch GP.

## 1. Run the confidence pilot

From this folder, with the venv active (Windows: `py -3.11` instead of `python`):

~~~powershell
python -m telo_tabular confidence --config configs/pilot.json --output runs/confidence_pilot
~~~

Read `runs/confidence_pilot/confidence_report.md`. The CSV beside it contains all validation scores. Each dataset's seed folder contains a reliability plot and preparation audit.

The useful first comparison is **learned vs feature_constant**, which asks whether record-specific context improves prediction of imputation success. The original score is a missing-frequency heuristic, not a calibrated probability of this event; beating it on Brier score alone does not establish a classification improvement.

These are development results. Use them to decide whether to refine the confidence estimator or continue to GP. Avoid selecting settings using test results.

## 2. Train a bounded GP pilot when ready

~~~powershell
python -m telo_tabular train --config configs/pilot.json --output runs/confidence_pilot
~~~

This reuses the prepared confidence data when the configuration, implementation, dependency versions, and dataset hashes match.

The pilot covers both datasets and one seed. Each evolved variant gets 12 individuals x 3 generations, at most 300 seconds of search, and up to 30 fitness epochs with patience 2. It uses original full-batch Adam training and [64, 32] hidden units. Final training has at most 100 epochs with patience 15.

There are eight GP searches: two datasets x four evolved variants. Their combined search-time caps are 40 minutes, plus preprocessing, checkpoint writes, fixed baselines and final training. This is a budget cap, not an estimate of actual runtime. Numerical checks and the deadline are checked between work units, so a running operation can finish after the limit.

Output: `training_validation.csv`, activation trees, full final-training histories, per-candidate search records, checkpoints, and validation channel statistics. This command does not evaluate test performance.

## 3. Evaluate frozen models after fixing the protocol

~~~powershell
python -m telo_tabular evaluate --run runs/confidence_pilot
~~~

This is the first step that evaluates the test partitions. It keeps the classifier, imputer, confidence estimator and corruption-rule training statistics fixed.

Read `test_results.csv` and `test_summary.csv`. Predictions, labels, row indices and masks are saved per condition. Failed training runs are excluded explicitly. Standard deviations are unavailable for a single run, rather than reported as evidence of zero variability.

The preset tests matched MCAR 20%, clean inputs, MCAR 40%/60%, correlated-feature group missingness at 20%/40%, and MAR/MNAR at nominal 20%. The CSV also reports realized rates. MAR preserves feature 0 as an observed driver. MAR/MNAR rates can differ from their nominal rates because their probabilities depend on training-defined thresholds.

## What is implemented

| Variant | Confidence | Propagation |
|---|---|---|
| original | GECCO frequency rule | Original ChannelProp at every hidden layer |
| learned | Cross-fitted entry-specific imputation-success estimate | Original ChannelProp at every hidden layer |
| feature_constant | Cross-fitted feature-level imputation-success frequency | Original ChannelProp at every hidden layer |
| first_only | Learned confidence | Metadata used at the first hidden layer, then neutralized |
| shuffled, optional | Learned confidence shuffled among missing entries within each feature | Original ChannelProp |

Fixed baselines are ReLU, Swish, LeakyReLU and ELU. They use the same MLP, imputed values, masks, seeds and final-training schedule. Their formulas use only the value channel.

Each evolved variant runs its own single-objective search. No NSGA-II, archive or classifier ensemble is introduced.

## How confidence is learned

For numeric feature j, the target event is:

`abs(hidden_value - training_fold_mean_j) <= 0.5 * training_fold_observed_std_j`

The 0.5 factor is a predefined pilot setting in `configs/pilot.json`.

1. The outer training partition receives its simulated missingness mask.
2. Outer cross-fitting holds out rows when generating training confidence scores.
3. Within each outer fit fold, inner folds supply artificial hiding examples. Imputation means, scales and target tolerances are fitted without the inner held-out rows.
4. A small histogram gradient boosting classifier per feature learns the event from other features and masks. It never receives the hidden target value, its class label, or validation/test records.
5. Observed entries have confidence 1. Missing entries receive the estimated event probability.
6. A final estimator fitted only from outer training data supplies validation/test confidence.

The estimator's internal folded preprocessing is separate from the classifier's common outer-training mean imputer and standardizer. Fold-specific imputation means can differ from the final training mean; confidence generated for training is therefore an out-of-fold estimate rather than an exact guarantee about the final imputation. Validation scoring uses the actual frozen final imputer.

Probabilities are evaluated for calibration; they are not assumed calibrated. Sparse or single-class fitting targets fall back to a smoothed feature success rate. Entirely unobserved features receive a flagged fallback mean of zero and confidence prior 0.5.

The optional shuffle variant is a cohort-level diagnostic: its values depend on the batch of records and shuffle seed. It is not intended as a deployment method.

## Preserved baseline and protocol differences

The original source files are under `reference/gecco2026/`. Do not run those monolithic scripts as the pilot entry point. Their external plotting helpers and original repeated-experiment driver are not used.

`telo_tabular/gp_core.py` preserves 36 definitions from the original PIMA script, including protected operators, genetic operators, ChannelProp and the MLP. A test compares their syntax trees directly.

The experiment harness is new. It deliberately:

- Splits before fitting corruption thresholds or preprocessing.
- Uses training-only MAR/MNAR thresholds, including during shifted tests.
- Adds finite-logit, probability, gradient and parameter checks.
- Rejects numerical failures rather than treating their predictions as valid accuracy.
- Uses explicit paired seeds and separate final-training initialization.
- Stops at a fixed time/generation budget and fingerprints all reusable outputs.
- Adds a conservative computational guard of depth 12 / 256 nodes during candidate evaluation.

The inherited source uses a diversity bonus of `0.01 * D / 3`, not `0.01 * D` as written in the conference fitness equation. The harness follows the released code and documents the difference. The inherited genetic operators do not enforce the initialization depth after every variation; `init_depth` is not a hard tree-depth guarantee. The separate computational guard is an implementation safeguard, applied equally across variants.

The PIMA CSV's zeros are retained as observed values, matching the supplied GECCO preprocessing. Any future change to this convention must be evaluated as a separate preprocessing change.

This harness is a controlled extension baseline, not a claim of bit-for-bit reproduction of the published tables. See `reference/SOURCE.md`.

## Reproducibility and safe resumption

Re-run the same command to resume an interrupted search or reuse completed steps. An output folder belongs to one exact configuration, source implementation, dependency-version set and dataset snapshot. Use a new output folder if any of these change.

All resume files and joblib/checkpoint files are generated locally. Load only your own trusted run directories.

To expand the experiment, copy the pilot configuration and set multiple independent seeds. Keep splits and masks paired across methods. A single seed is a development gate, not sufficient evidence of a dependable improvement.

## Verification

~~~powershell
python -m unittest discover -s tests -v
~~~

Tests cover original-code provenance, split isolation, out-of-fold confidence, missingness generation, ChannelProp bounds and redundancy, numerical failure rejection, configuration mismatch rejection, interrupted-search resumption, and the complete pipeline on synthetic data.

A separate command-line smoke configuration uses only generated synthetic data:

~~~powershell
python -m telo_tabular confidence --config configs/smoke.json --output runs/synthetic_smoke
python -m telo_tabular train --config configs/smoke.json --output runs/synthetic_smoke
python -m telo_tabular evaluate --run runs/synthetic_smoke
~~~

## Dependencies

For a fresh environment:

~~~powershell
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
~~~

Then use `.venv/bin/python` (Windows: `.venv\Scripts\python.exe`).

## Research scope

The TELO proposal studies entry-specific reliability and how it is used by a single evolved activation. The planned GECCO 2027 work studies multi-objective search, archives and ensembles. Related methods and shared foundations must still be cited.

No improvement, novelty guarantee, or satisfaction of the journal's 30% requirement is asserted by this implementation. See `TELO_TABULAR_EXTENSION_PROPOSAL.md` for the research rationale.

## Confidence screen over the twelve GECCO datasets (2026-11-09)

`python -m telo_tabular.screen --config configs/screen.json --output runs_screen` — 13 datasets x 5 seeds, validation only,
no GP, no test data. Verdict rule fixed before running. Result (`runs_screen/screen_summary.md`): entry-specific
confidence beats the feature-constant control on every seed with mean AUC gain >= 0.05 on **5 datasets** (wdbc +0.33,
adult +0.10, mushroom +0.07, sonar +0.06, cylinderband +0.05); weak on glass, pima, heart; none on horse, housevote,
aust_credit, hepatitis, mammo. The learned estimator is slightly less calibrated than the constant control everywhere
(ECE higher by 0.01-0.1) — ranking gain, not calibration gain. Reading: per-entry reliability exists where features are
redundant (WDBC's triplets, mushroom/adult categorical structure); on low-dimensional or weakly related features it does
not. The GP comparison should therefore run on both groups and report the boundary, not only on the datasets with signal.
