# Source provenance

Repository: https://github.com/naeemshb/3C-EA-CNN-extension  
Snapshot: 1eaffb0b1107649fed5cf6b590061d9d9a6882d4  
Retrieved: 2026-11-09

| Local file | Repository source |
|---|---|
| gecco2026/pima_original.py | Final/Pima/New.py |
| gecco2026/wdbc_original.py | Final/WDBC/New.py |
| ../data/pima.csv | Final/Pima/diabetes.csv |
| ../data/wdbc.csv | Final/WDBC/wdbc_prepared.csv |

Original scripts and datasets are retained as reference copies, with text line endings normalized where needed. No claim of a new license is made for upstream material.

The GP/ChannelProp module copies the original PIMA definitions from lines 17-554 of this snapshot. The provenance test verifies all 36 extracted definitions against the retained original script. The new harness is described in the project README and does not execute the original scripts' plotting or repeated-experiment entry points.

The supplied CSVs contain 768 PIMA records with eight features and 569 WDBC records with 30 features. WDBC labels B/M are mapped in sorted order to 0/1. PIMA Outcome is mapped to 0/1. PIMA zeros remain observed by default.
