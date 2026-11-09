# =============================================================================
# baselines_fair.py
# =============================================================================
# Fair Baselines Framework (Train/Val/Test) with consistent preprocessing
#
# Goals:
#   1) SAME split protocol as your main method: Train/Val/Test via two-step split
#   2) SAME preprocessing policy: imputer fit on TRAIN only, scaler fit on TRAIN only
#   3) Works for TWO dataset categories:
#        A) Datasets with ORIGINAL missing values (no injection)
#        B) Complete datasets where you inject MNAR (e.g., 20%) before splitting
#   4) Choose models to run (RF, SVM, LR, KNN, GNB, GB, + MLP MeanImpute)
#
# IMPORTANT:
#   - This baseline script does NOT use missing/confidence channels (values-only baselines).
#   - If you want the baselines to see (m,c) too, that is a separate option (not enabled here).
#
# Usage examples:
#   - Original-missing dataset (no injection):
#       CONFIG["dataset_key"]="hepatitis"
#
#   - No-missing dataset (inject MNAR 20%):
#       CONFIG["dataset_key"]="pima"
#       CONFIG["inject_missing"]=True
#       CONFIG["missing_mechanism"]="MNAR"
#       CONFIG["missing_rate"]=0.20
#
# =============================================================================

import os
import copy
import random
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.base import clone

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # ----------------------------
    # Dataset selection
    # ----------------------------
    # Pick from: "hepatitis", "housevotes", "credit_approval", "horse_colic", "pima", "glass", ...
    "dataset_key": "horse_colic",
    "csv_path": "horse.csv",

    # ----------------------------
    # Missingness handling
    # ----------------------------
    # If True, inject missingness BEFORE splitting.
    # Use this for datasets that are originally complete.
    "inject_missing": False,
    "missing_mechanism": "MNAR",   # "MNAR" | "MAR" | "MCAR"
    "missing_rate": 0.20,          # e.g., 0.20

    # ----------------------------
    # Splitting (match your method)
    # ----------------------------
    "test_size_total": 0.40,       # first split: train vs temp
    "val_fraction_of_temp": 0.50,  # second split: val vs test inside temp (0.5 => 20/20 when temp=40)
    "num_runs": 20,
    "base_seed": 42,

    # ----------------------------
    # Models to run (choose any subset)
    # ----------------------------
    # Available: "MLP", "RF", "GB", "SVM", "LR", "KNN", "GNB"
    "models": ["MLP", "RF", "SVM"],

    # ----------------------------
    # MLP baseline settings
    # ----------------------------
    "mlp_hidden_sizes": [64, 32],
    "mlp_lr": 1e-3,
    "mlp_weight_decay": 1e-4,
    "mlp_epochs": 100,
    "mlp_patience": 15,
    "mlp_batchnorm": True,

    # Output file
    "output_file": "baselines_fair_results.txt",
}


# =============================================================================
# Utilities: seeding, missing rate
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def missing_rate_percent(X_df: pd.DataFrame) -> float:
    """Percent missing entries in feature matrix."""
    return (X_df.isnull().sum().sum() / X_df.size) * 100.0

def load_housevotes(csv_path: str) -> tuple:
    """
    Load HouseVotes (Congressional Voting Records) dataset.

    Expected format (matches your file):
      - First column: 'Class' (already 0/1 or labels)
      - Remaining columns: V1..V16 with values in {'y','n','?'} and/or NaN

    Returns
    -------
    X : np.ndarray (float)  shape (n_samples, n_features) with NaN for missing
    y : np.ndarray (int)    shape (n_samples,) encoded as 0..C-1
    n_classes : int
    """
    try:
        df = pd.read_csv(csv_path, na_values=["?", " ? ", "NA", "N/A", "nan", "NaN", ""])

        if "Class" not in df.columns:
            raise ValueError("HouseVotes loader expects a 'Class' column as target.")

        # Target
        y_raw = df["Class"].values

        # Features
        X_df = df.drop(columns=["Class"]).copy()

        # Normalize common vote encodings:
        # y -> 1, n -> 0, missing stays NaN
        # (Works even if some entries already numeric; we'll coerce safely.)
        X_df = X_df.replace({
            "y": 1, "Y": 1, "yes": 1, "YES": 1,
            "n": 0, "N": 0, "no": 0, "NO": 0,
        })

        # Convert everything to numeric, keep non-convertible as NaN
        X_df = X_df.apply(pd.to_numeric, errors="coerce")

        # Drop rows with missing target (just in case)
        mask = ~pd.isna(y_raw)
        X_df = X_df.loc[mask].reset_index(drop=True)
        y_raw = y_raw[mask]

        # Encode target to 0..C-1 (prevents "Target 2 is out of bounds" issues)
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        n_classes = len(le.classes_)

        print(f"HouseVotes: {X_df.shape[0]} samples, {X_df.shape[1]} features, {n_classes} classes")
        print(f"Missing rate: {missing_rate_percent(X_df):.1f}%")
        print(f"Target mapping: {dict(enumerate(le.classes_))}")

        return X_df.values.astype(float), y.astype(int), n_classes

    except Exception as e:
        print(f"Error loading HouseVotes dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None



# =============================================================================
# Missingness injection (applied ONLY when inject_missing=True)
# =============================================================================

def introduce_missing_data(
    df: pd.DataFrame,
    missing_rate: float,
    mechanism: str,
    seed: int
) -> pd.DataFrame:
    """
    Inject missingness into a numeric dataframe df (features only).
    This is applied BEFORE splitting to match your described setup.
    """
    np.random.seed(seed)
    df_missing = df.copy()
    n_rows, n_cols = df_missing.shape

    if mechanism == "MCAR":
        mask = np.random.random((n_rows, n_cols)) < missing_rate
        df_missing = df_missing.mask(mask)

    elif mechanism == "MAR":
        # First column drives missingness in other columns
        first_col = df_missing.iloc[:, 0].values
        thresh = np.nanpercentile(first_col, 50)
        for j in range(1, n_cols):
            probs = np.where(first_col > thresh, missing_rate * 2.0, missing_rate * 0.5)
            probs = np.clip(probs, 0.0, 1.0)
            miss = np.random.random(n_rows) < probs
            df_missing.iloc[miss, j] = np.nan

    elif mechanism == "MNAR":
        # Extreme values more likely missing, per-column
        for j, col in enumerate(df_missing.columns):
            col_values = df_missing[col].values.astype(float)
            # If column is all-nan or constant, fall back to MCAR-like
            if np.all(np.isnan(col_values)):
                miss = np.random.random(n_rows) < missing_rate
                df_missing.loc[miss, col] = np.nan
                continue
            lo = np.nanpercentile(col_values, 25)
            hi = np.nanpercentile(col_values, 75)
            is_extreme = (col_values < lo) | (col_values > hi)
            probs = np.where(is_extreme, missing_rate * 1.5, missing_rate * 0.5)
            probs = np.clip(probs, 0.0, 1.0)
            miss = np.random.random(n_rows) < probs
            df_missing.loc[miss, col] = np.nan

    else:
        raise ValueError(f"Unknown missingness mechanism: {mechanism}")

    return df_missing


# =============================================================================
# Dataset loaders
# =============================================================================

def _encode_target(y_raw) -> Tuple[np.ndarray, int]:
    """Always return y in {0..C-1} to avoid 'Target out of bounds' in PyTorch."""
    le = LabelEncoder()
    y = le.fit_transform(np.asarray(y_raw))
    n_classes = len(le.classes_)
    return y.astype(int), int(n_classes)


def load_hepatitis(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray, int]:
    df = pd.read_csv(csv_path, na_values="?")

    # Hepatitis sometimes has target first or last; keep your robust rule
    if df.iloc[:, 0].nunique(dropna=True) <= 3:
        y_raw = df.iloc[:, 0].values
        X_df = df.iloc[:, 1:].copy()
    else:
        y_raw = df.iloc[:, -1].values
        X_df = df.iloc[:, :-1].copy()

    # Drop missing targets only
    mask = ~pd.isna(y_raw)
    X_df = X_df.loc[mask].copy()
    y_raw = y_raw[mask]

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    y, n_classes = _encode_target(y_raw)

    print(f"Hepatitis: {X_df.shape[0]} samples, {X_df.shape[1]} features, {n_classes} classes")
    print(f"Original missing rate: {missing_rate_percent(X_df):.1f}%")
    return X_df, y, n_classes



def load_credit_approval(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """
    Australian Credit Approval:
    - Many versions are already numeric with '?' for missing.
    - Your uploaded version appears numeric; we treat '?' as NaN.
    - We do NOT automatically treat zeros as missing here (that is dataset-version dependent).
      If you truly need "zero-as-missing" for specific columns, implement it in a dataset-specific hook.
    """
    df = pd.read_csv(csv_path, na_values="?")

    # Target is last column in most versions
    y_raw = df.iloc[:, -1].values
    X_df = df.iloc[:, :-1].copy()

    # Force numeric (if some columns are object)
    X_df = X_df.apply(pd.to_numeric, errors="coerce")

    # Drop missing targets only
    mask = ~pd.isna(y_raw)
    X_df = X_df.loc[mask].copy()
    y_raw = y_raw[mask]

    y, n_classes = _encode_target(y_raw)

    print(f"CreditApproval: {X_df.shape[0]} samples, {X_df.shape[1]} features, {n_classes} classes")
    print(f"Original missing rate: {missing_rate_percent(X_df):.1f}%")
    return X_df, y, n_classes


def load_horse_colic(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """
    Horse Colic:
    - expects header
    - target column: 'outcome' preferred; else fallback to a column name by position.
    - encodes categorical columns to numeric codes, leaving NaN where appropriate.
    """
    df = pd.read_csv(csv_path, na_values="?", header=0)

    # choose target
    if "outcome" in df.columns:
        target_col = "outcome"
    else:
        # conservative fallback: last column
        target_col = df.columns[-1]

    # drop missing targets only
    df = df.dropna(subset=[target_col]).copy()

    y_raw = df[target_col].values
    X_df = df.drop(columns=[target_col]).copy()

    # Optional drops to avoid IDs/leakage; keep minimal and safe
    drop_candidates = ["hospital_number", "lesion_1", "lesion_2", "lesion_3", "cp_data", "surgical_lesion"]
    for c in drop_candidates:
        if c in X_df.columns:
            X_df = X_df.drop(columns=[c])

    # Encode categorical features
    for col in X_df.select_dtypes(include=["object"]).columns:
        codes = pd.Categorical(X_df[col]).codes.astype(float)
        # pandas uses -1 for NaN -> convert back
        codes[codes == -1] = np.nan
        X_df[col] = codes

    # Force numeric
    X_df = X_df.apply(pd.to_numeric, errors="coerce")

    y, n_classes = _encode_target(y_raw)

    print(f"HorseColic: {X_df.shape[0]} samples, {X_df.shape[1]} features, {n_classes} classes")
    print(f"Original missing rate: {missing_rate_percent(X_df):.1f}%")
    return X_df, y, n_classes


def load_pima(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """
    Pima (diabetes.csv):
    - typical: last column is Outcome (0/1), features numeric.
    - Many versions use 0 as "missing" in certain columns; DO NOT do that here unless you do it in your method.
    """
    df = pd.read_csv(csv_path, na_values="?")
    y_raw = df.iloc[:, -1].values
    X_df = df.iloc[:, :-1].copy()
    X_df = X_df.apply(pd.to_numeric, errors="coerce")

    mask = ~pd.isna(y_raw)
    X_df = X_df.loc[mask].copy()
    y_raw = y_raw[mask]

    y, n_classes = _encode_target(y_raw)
    print(f"Pima: {X_df.shape[0]} samples, {X_df.shape[1]} features, {n_classes} classes")
    print(f"Original missing rate: {missing_rate_percent(X_df):.1f}%")
    return X_df, y, n_classes


DATASETS: Dict[str, Dict] = {
    "hepatitis": {"path": "hepatitis.csv", "loader": load_hepatitis, "original_missing": True},
    "housevotes": {"path": "housevotes.csv", "loader": load_housevotes, "original_missing": True},
    "credit_approval": {"path": "Aust_credi.csv", "loader": load_credit_approval, "original_missing": True},
    "horse_colic": {"path": "horse.csv", "loader": load_horse_colic, "original_missing": True},
    "pima": {"path": "diabetes.csv", "loader": load_pima, "original_missing": False},
}


# =============================================================================
# Metrics
# =============================================================================

def compute_multiclass_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    specs = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specs))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray]) -> Tuple[float, float, float, float, float, float]:
    num_classes = len(np.unique(y_true))
    avg_method = "binary" if num_classes == 2 else "macro"

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg_method, zero_division=0)
    spec = compute_multiclass_specificity(y_true, y_pred)

    auc = 0.0
    try:
        if y_prob is None:
            auc = 0.0
        else:
            if num_classes == 2:
                # Accept either shape (n,) or (n,2)
                if y_prob.ndim == 2:
                    auc = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    auc = roc_auc_score(y_true, y_prob)
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except:
        auc = 0.0

    return float(acc), float(prec), float(rec), float(spec), float(f1), float(auc)


# =============================================================================
# Preprocessing: split + impute + scale (fit on TRAIN only)
# =============================================================================

@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def split_train_val_test(X_df: pd.DataFrame, y: np.ndarray, seed: int,
                         test_size_total: float, val_fraction_of_temp: float) -> SplitData:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_df, y, test_size=test_size_total, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_fraction_of_temp, random_state=seed, stratify=y_temp
    )
    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)


@dataclass
class PreprocessedData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def preprocess_mean_impute_and_scale(split: SplitData) -> PreprocessedData:
    imputer = SimpleImputer(strategy="mean")
    X_tr_imp = imputer.fit_transform(split.X_train)
    X_va_imp = imputer.transform(split.X_val)
    X_te_imp = imputer.transform(split.X_test)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_imp)
    X_va = scaler.transform(X_va_imp)
    X_te = scaler.transform(X_te_imp)

    return PreprocessedData(
        X_train=X_tr, X_val=X_va, X_test=X_te,
        y_train=split.y_train, y_val=split.y_val, y_test=split.y_test
    )


# =============================================================================
# Models: sklearn baselines
# =============================================================================

def build_sklearn_models(n_classes: int) -> Dict[str, object]:
    # For multi-class, LR should be multinomial (lbfgs supports multinomial), and SVM uses OvR by default.
    lr = LogisticRegression(max_iter=2000, random_state=42)
    if n_classes > 2:
        lr = LogisticRegression(max_iter=2000, random_state=42, multi_class="multinomial", solver="lbfgs")

    models = {
        "RF": RandomForestClassifier(n_estimators=300, random_state=42),
        "GB": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "LR": lr,
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "GNB": GaussianNB(),
    }
    return models


def fit_eval_sklearn(model, X_tr, y_tr, X_eval, y_eval) -> Tuple[float, float, float, float, float, float]:
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_eval)

    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_eval)
    elif hasattr(model, "decision_function"):
        # fallback: decision_function can be used, but AUC handling differs; keep None to avoid misleading AUC
        y_prob = None

    return compute_metrics(y_eval, y_pred, y_prob)


# =============================================================================
# MLP (MeanImpute) baseline with early stopping on VAL
# =============================================================================

class StandardMLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, use_batchnorm: bool = True):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_eval_mlp_baseline(
    prep: PreprocessedData,
    n_classes: int,
    hidden_sizes: List[int],
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    use_batchnorm: bool,
    seed: int
) -> Dict[str, Tuple[float, float, float, float, float, float]]:
    """
    Returns:
      {
        "val": (acc, prec, rec, spec, f1, auc),
        "test": (acc, prec, rec, spec, f1, auc)
      }
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr_t = torch.tensor(prep.X_train, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(prep.y_train, dtype=torch.long, device=device)

    X_va_t = torch.tensor(prep.X_val, dtype=torch.float32, device=device)
    y_va_t = torch.tensor(prep.y_val, dtype=torch.long, device=device)

    X_te_t = torch.tensor(prep.X_test, dtype=torch.float32, device=device)

    model = StandardMLP(
        input_size=prep.X_train.shape[1],
        hidden_sizes=hidden_sizes,
        output_size=n_classes,
        use_batchnorm=use_batchnorm,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    best_state = None
    best_val_acc = -1.0
    bad = 0

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(X_tr_t)
        loss = crit(out, y_tr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_va_t)
            val_pred = val_out.argmax(1)
            val_acc = (val_pred == y_va_t).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate VAL
    model.eval()
    with torch.no_grad():
        val_out = model(X_va_t)
        val_pred = val_out.argmax(1).detach().cpu().numpy()
        val_prob = torch.softmax(val_out, dim=1).detach().cpu().numpy()
    val_metrics = compute_metrics(prep.y_val, val_pred, val_prob)

    # Evaluate TEST
    with torch.no_grad():
        te_out = model(X_te_t)
        te_pred = te_out.argmax(1).detach().cpu().numpy()
        te_prob = torch.softmax(te_out, dim=1).detach().cpu().numpy()
    test_metrics = compute_metrics(prep.y_test, te_pred, te_prob)

    return {"val": val_metrics, "test": test_metrics}


# =============================================================================
# Main evaluation: per-run, per-model, aggregated (Val and Test)
# =============================================================================

def evaluate_one_run(
    X_df: pd.DataFrame,
    y: np.ndarray,
    n_classes: int,
    seed: int,
    inject_missing: bool,
    missing_mechanism: str,
    missing_rate: float,
) -> Tuple[SplitData, PreprocessedData]:
    # IMPORTANT: Keep RNG order consistent
    set_seed(seed)

    X_features = X_df.copy()

    # Inject missingness BEFORE split (only for "complete" datasets if requested)
    if inject_missing:
        # Seed immediately before injection to minimize drift
        np.random.seed(seed)
        X_features = introduce_missing_data(X_features, missing_rate, missing_mechanism, seed)
        # print injected missing rate
        print(f"  Injected missing rate: {missing_rate_percent(X_features):.1f}%")

    split = split_train_val_test(
        X_features, y, seed=seed,
        test_size_total=CONFIG["test_size_total"],
        val_fraction_of_temp=CONFIG["val_fraction_of_temp"]
    )

    prep = preprocess_mean_impute_and_scale(split)
    return split, prep


def aggregate_results(runs: List[Tuple[float, float, float, float, float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.array(runs, dtype=float)  # shape (R, 6)
    return arr.mean(axis=0), arr.std(axis=0)


def run_experiments(config: Dict) -> None:
    dataset_key = config["dataset_key"]
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset_key '{dataset_key}'. Available: {list(DATASETS.keys())}")

    ds = DATASETS[dataset_key]
    csv_path = config["csv_path"] or ds["path"]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    loader = ds["loader"]

    # Decide injection default if user left inject_missing False but dataset is complete:
    # We do NOT auto-change it (user requested control). But we print a reminder.
    X_df, y, n_classes = loader(csv_path)

    if config["inject_missing"]:
        print(f"Missing injection enabled: {config['missing_mechanism']} @ {config['missing_rate']*100:.0f}%")
    else:
        print("Missing injection disabled: using original missingness only")

    print(f"Runs: {config['num_runs']}")
    print("-" * 80)

    # Prepare models
    sklearn_models = build_sklearn_models(n_classes)

    # Results structure:
    # results[model_name]["val"] = list of 6-tuples
    # results[model_name]["test"] = list of 6-tuples
    results: Dict[str, Dict[str, List[Tuple[float, float, float, float, float, float]]]] = {}
    for m in config["models"]:
        results[m] = {"val": [], "test": []}

    for run in range(config["num_runs"]):
        seed = config["base_seed"] + run
        print(f"\nRUN {run+1}/{config['num_runs']} (seed={seed})")

        split, prep = evaluate_one_run(
            X_df=X_df, y=y, n_classes=n_classes, seed=seed,
            inject_missing=config["inject_missing"],
            missing_mechanism=config["missing_mechanism"],
            missing_rate=config["missing_rate"]
        )

        # ---------- sklearn baselines ----------
        for m in config["models"]:
            if m in ("RF", "GB", "SVM", "LR", "KNN", "GNB"):
                model = clone(sklearn_models[m])

                # Train on TRAIN, evaluate on VAL and TEST separately
                val_metrics = fit_eval_sklearn(model, prep.X_train, prep.y_train, prep.X_val, prep.y_val)
                # Need a fresh refit for test? For parity, we keep the same trained model:
                # Fit once on train, eval val; then eval test with same trained model.
                # That matches early-stopping philosophy (no peeking at test).
                # So we DO NOT refit between val and test.
                y_pred_te = model.predict(prep.X_test)
                y_prob_te = model.predict_proba(prep.X_test) if hasattr(model, "predict_proba") else None
                test_metrics = compute_metrics(prep.y_test, y_pred_te, y_prob_te)

                results[m]["val"].append(val_metrics)
                results[m]["test"].append(test_metrics)

                print(f"  {m:<4}  ValAcc={val_metrics[0]:.4f}  TestAcc={test_metrics[0]:.4f}  TestF1={test_metrics[4]:.4f}")

        # ---------- MLP baseline ----------
        if "MLP" in config["models"]:
            mlp_out = train_eval_mlp_baseline(
                prep=prep,
                n_classes=n_classes,
                hidden_sizes=config["mlp_hidden_sizes"],
                lr=config["mlp_lr"],
                weight_decay=config["mlp_weight_decay"],
                epochs=config["mlp_epochs"],
                patience=config["mlp_patience"],
                use_batchnorm=config["mlp_batchnorm"],
                seed=seed,
            )
            results["MLP"]["val"].append(mlp_out["val"])
            results["MLP"]["test"].append(mlp_out["test"])
            print(f"  MLP  ValAcc={mlp_out['val'][0]:.4f}  TestAcc={mlp_out['test'][0]:.4f}  TestF1={mlp_out['test'][4]:.4f}")

    # ---------- Aggregated report ----------
    print("\n" + "=" * 120)
    print("AGGREGATED RESULTS (MEAN ± STD)  —  reported separately for VAL and TEST")
    print("=" * 120)
    header = f"{'Model':<8} | {'Split':<5} | {'Acc':<14} {'Prec':<14} {'Rec':<14} {'Spec':<14} {'F1':<14} {'AUC':<14}"
    print(header)
    print("-" * 120)

    for m in config["models"]:
        for split_name in ("val", "test"):
            mean, std = aggregate_results(results[m][split_name])
            # order: acc, prec, rec, spec, f1, auc
            line = (
                f"{m:<8} | {split_name:<5} | "
                f"{mean[0]:.4f}±{std[0]:.4f}  "
                f"{mean[1]:.4f}±{std[1]:.4f}  "
                f"{mean[2]:.4f}±{std[2]:.4f}  "
                f"{mean[3]:.4f}±{std[3]:.4f}  "
                f"{mean[4]:.4f}±{std[4]:.4f}  "
                f"{mean[5]:.4f}±{std[5]:.4f}"
            )
            print(line)

    print("=" * 120)


# =============================================================================
# Logger (optional)
# =============================================================================

class Logger:
    def __init__(self, filename: str):
        self.terminal = None
        self.log = None
        self.filename = filename

    def __enter__(self):
        import sys
        self.terminal = sys.stdout
        self.log = open(self.filename, "w")
        sys.stdout = self
        return self

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __exit__(self, exc_type, exc, tb):
        import sys
        sys.stdout = self.terminal
        self.log.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Optional: if user does not explicitly set inject_missing, you can auto-suggest:
    #   - for original-missing datasets -> keep False
    #   - for complete datasets -> set True
    # BUT we do NOT auto-change it (you said control matters).
    with Logger(CONFIG["output_file"]):
        print("=" * 80)
        print("FAIR BASELINES (Train/Val/Test) with consistent preprocessing")
        print("=" * 80)
        print(f"Dataset key: {CONFIG['dataset_key']}")
        print(f"CSV path:    {CONFIG['csv_path'] or DATASETS[CONFIG['dataset_key']]['path']}")
        print(f"Models:      {CONFIG['models']}")
        print(f"Inject missing: {CONFIG['inject_missing']}")
        if CONFIG["inject_missing"]:
            print(f"  Mechanism: {CONFIG['missing_mechanism']}")
            print(f"  Rate:      {CONFIG['missing_rate']*100:.0f}%")
        print(f"Runs: {CONFIG['num_runs']}  |  base_seed: {CONFIG['base_seed']}")
        print("=" * 80)

        run_experiments(CONFIG)

    print(f"Results saved to: {CONFIG['output_file']}")
