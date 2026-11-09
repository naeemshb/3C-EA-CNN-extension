"""The twelve GECCO 2026 tabular datasets (plus Heart), loaded the way the original per-dataset scripts did.

Conventions reproduced from the original scripts (see the comments per entry): '?' -> NaN; natural missing values are
kept; categorical columns are integer-factorised (sorted codes, NaN kept) where the original did so; the label is
integer-encoded in sorted order; rows with a missing label are dropped.  Files are read from the repository's
``Final/<Dataset>/`` folders, so this module needs the repository checkout.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .data import ROOT, file_hash

FINAL = ROOT.parent / "Final"
NA = ["?", " ?", "? ", "", " ", "NA", "N/A", "nan", "NaN"]

# name -> (relative csv path, label column (None = last), header row, policy)
#   policy "numeric": all feature columns coerced to numeric (non-numeric -> NaN), all-NaN columns dropped
#   policy "factorize": object columns integer-factorised (sorted), NaN kept
REGISTRY = {
    "adult":        ("Adult/adult.csv",                    "income",   0,    "factorize"),  # original: factorize categoricals
    "aust_credit":  ("Aust_Credit/Aust_credi.csv",         "Label",    0,    "numeric"),
    "cylinderband": ("Cylinderband/bands.data",            None,       None, "numeric"),    # original: to_numeric(coerce), drop all-NaN cols
    "glass":        ("Glass/glass.csv",                    "Type",     0,    "numeric"),
    "heart":        ("Heart/heart (1).csv",                "target",   0,    "numeric"),
    "hepatitis":    ("Hepatiti/hepatitis.csv",             None,       0,    "numeric"),
    "horse":        ("Horse/horse.csv",                    "outcome",  0,    "factorize"),  # original: 'outcome' target, categoricals encoded
    "housevote":    ("HouseVote/HouseVotes84 (1) (1).csv", "Class",    0,    "votes"),      # original: y/n -> 1/0, Class -> label
    "mammo":        ("Mammo/Cleaned_data.csv",             "Severity", 0,    "numeric"),
    "mushroom":     ("Mushroom/mushroom_prepared.csv",     "Label",    0,    "numeric"),
    "pima":         ("Pima/diabetes.csv",                  "Outcome",  0,    "numeric"),    # zeros retained as observed (original)
    "sonar":        ("Sonar/sonar_final_numeric.csv",      "Label",    0,    "numeric"),
    "wdbc":         ("WDBC/wdbc_prepared.csv",             "Diagnosis", 0,   "numeric"),
    # same file as pima; physiologically impossible zeros treated as missing (NOT the GECCO convention; a declared variant)
    "pima_zeros_missing": ("Pima/diabetes.csv",             "Outcome",  0,    "pima_zeros"),
}
PIMA_ZERO_COLUMNS = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")


def load_gecco_dataset(name):
    rel, label_col, header, policy = REGISTRY[name]
    path = FINAL / rel
    df = pd.read_csv(path, header=header, na_values=NA, skipinitialspace=True)
    if label_col is None:
        label_col = df.columns[-1]
    y_raw = df[label_col].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})
    keep = y_raw.notna()
    df, y_raw = df.loc[keep].copy(), y_raw.loc[keep]
    X = df.drop(columns=[label_col])
    if policy == "votes":
        X = X.replace({"y": 1, "n": 0}).apply(pd.to_numeric, errors="coerce")
    elif policy == "factorize":
        for c in X.columns:
            if X[c].dtype == object:
                s = X[c].astype("string").str.strip()
                codes, _ = pd.factorize(s, sort=True)
                codes = pd.Series(codes, index=X.index).astype(float)
                codes[s.isna()] = np.nan
                X[c] = codes
            else:
                X[c] = pd.to_numeric(X[c], errors="coerce")
    else:
        X = X.apply(pd.to_numeric, errors="coerce")
    if policy == "pima_zeros":
        for c in PIMA_ZERO_COLUMNS:
            X.loc[X[c] == 0, c] = np.nan
    all_nan = X.columns[X.isna().all()].tolist()
    X = X.drop(columns=all_nan)
    classes = sorted(y_raw.unique(), key=lambda v: (len(v), v) if not v.lstrip("-").isdigit() else (0, int(v)))
    y = np.asarray([classes.index(v) for v in y_raw], dtype=np.int64)
    x = X.to_numpy(dtype=np.float64)
    meta = {"name": name, "features": [str(c) for c in X.columns], "classes": [str(c) for c in classes], "rows": int(len(x)),
            "source_sha256": file_hash(path), "source_file": rel, "policy": policy, "dropped_all_nan_columns": [str(c) for c in all_nan],
            "dropped_missing_label_rows": int((~keep).sum()), "natural_missing_rate": float(np.isnan(x).mean())}
    if name == "pima":
        meta["pima_zero_policy"] = "retain as observed, matching supplied GECCO CSV"
    if name == "pima_zeros_missing":
        meta["pima_zero_policy"] = "zeros in " + ", ".join(PIMA_ZERO_COLUMNS) + " treated as missing (declared variant)"
    return x, y, meta
