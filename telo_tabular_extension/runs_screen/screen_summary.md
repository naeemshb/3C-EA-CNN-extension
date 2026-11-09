# Confidence screen (validation only; no GP, no test data)

Event: |x_true - train mean_j| <= 0.5 * train sd_j on deliberately hidden validation entries (MCAR 20 % on top of natural missingness).
learned = cross-fitted entry-specific estimate; constant = feature-level success frequency (same scale, no per-entry variation).
Verdict rule (fixed before running): 'signal' = learned beats constant in AUC on every seed with mean gain >= 0.05; 'none' = mean gain < 0.02; otherwise 'weak'.

| dataset | rows | features | natural missing | learned AUC | constant AUC | delta AUC (mean ± sd) | seeds learned wins | delta Brier | verdict |
|---|---|---|---|---|---|---|---|---|---|
| wdbc | 569 | 30 | n/a | 0.866 | 0.533 | 0.332 ± 0.039 | 5/5 | 0.094 | **signal** |
| adult | 32561 | 14 | 0.009 | 0.924 | 0.827 | 0.097 ± 0.001 | 5/5 | 0.055 | **signal** |
| glass | 214 | 9 | 0.000 | 0.841 | 0.764 | 0.077 ± 0.053 | 4/5 | 0.027 | **weak** |
| mushroom | 8124 | 22 | 0.014 | 0.977 | 0.906 | 0.070 ± 0.007 | 5/5 | 0.054 | **signal** |
| sonar | 208 | 60 | 0.000 | 0.621 | 0.563 | 0.058 ± 0.027 | 5/5 | -0.014 | **signal** |
| pima | 768 | 8 | n/a | 0.641 | 0.587 | 0.054 ± 0.052 | 4/5 | 0.000 | **weak** |
| cylinderband | 539 | 25 | 0.082 | 0.821 | 0.767 | 0.053 ± 0.021 | 5/5 | 0.013 | **signal** |
| heart | 1025 | 13 | 0.000 | 0.871 | 0.827 | 0.044 ± 0.018 | 5/5 | 0.013 | **weak** |
| horse | 299 | 27 | 0.198 | 0.907 | 0.894 | 0.013 ± 0.020 | 4/5 | 0.000 | **none** |
| housevote | 435 | 16 | 0.056 | 0.997 | 0.993 | 0.003 ± 0.002 | 4/5 | -0.001 | **none** |
| aust_credit | 690 | 14 | 0.000 | 0.871 | 0.870 | 0.002 ± 0.027 | 2/5 | -0.010 | **none** |
| hepatitis | 155 | 19 | 0.057 | 0.881 | 0.894 | -0.013 ± 0.018 | 1/5 | -0.014 | **none** |
| mammo | 830 | 5 | 0.000 | 0.823 | 0.863 | -0.040 ± 0.066 | 2/5 | -0.021 | **none** |
