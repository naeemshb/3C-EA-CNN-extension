# =============================================================================
# ml_baselines.py - Traditional ML Baselines + MeanImpute+MLP Baseline
# =============================================================================
# Run this in the same folder as your dataset CSV files.
# It will evaluate RandomForest, GradientBoosting, SVM, LogisticRegression,
# KNN, NaiveBayes, and also:
#   - Mean Imputation + Standard MLP
# =============================================================================

import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)
from sklearn.base import clone

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_multiclass_specificity(y_true, y_pred):
    """Compute macro-averaged specificity for multi-class classification."""
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec_i)
    return float(np.mean(specificities))


def compute_metrics(y_true, y_pred, y_prob, n_classes):
    """Compute Accuracy, Precision, Recall, Specificity, F1, AUC (binary or multiclass)."""
    avg_method = "binary" if n_classes == 2 else "macro"

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=avg_method, zero_division=0
    )
    spec = compute_multiclass_specificity(y_true, y_pred)

    try:
        if n_classes == 2:
            # y_prob is expected shape (n,2) for proba models; handle shape robustly
            if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_true, y_prob)
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0

    return float(acc), float(prec), float(rec), float(spec), float(f1), float(auc)


def missing_rate_percent(X):
    """Compute missing rate (%) for numpy array or pandas DataFrame."""
    if isinstance(X, pd.DataFrame):
        total = X.size
        miss = X.isna().sum().sum()
        return 100.0 * (miss / total) if total > 0 else 0.0
    X = np.asarray(X)
    total = X.size
    miss = np.isnan(X).sum() if np.issubdtype(X.dtype, np.number) else 0
    return 100.0 * (miss / total) if total > 0 else 0.0


# =============================================================================
# DATASET LOADERS
# =============================================================================

def load_horse_colic(csv_path="horse.csv"):
    """Load Horse Colic dataset."""
    df = pd.read_csv(csv_path, na_values="?", header=0)

    target_col = "outcome"
    df = df.dropna(subset=[target_col])

    cols_to_drop = ["hospital_number", "lesion_1", "lesion_2", "lesion_3",
                    "cp_data", "surgical_lesion"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    target_data = df[target_col]
    df = df.drop(columns=[target_col])

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.Categorical(df[col]).codes
        df[col] = df[col].replace(-1, np.nan)

    # Encode target with LabelEncoder to ensure 0, 1, 2, ... labels
    le = LabelEncoder()
    y = le.fit_transform(target_data)
    n_classes = len(le.classes_)

    print(f"Horse Colic: {df.shape[0]} samples, {df.shape[1]} features, {n_classes} classes")
    print(f"Missing rate: {missing_rate_percent(df):.1f}%")

    return df.values.astype(float), y.astype(int), n_classes


def load_glass(csv_path="glass.csv"):
    """Load Glass dataset."""
    df = pd.read_csv(csv_path)

    X = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    print(f"Glass: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")
    print(f"Missing rate: {missing_rate_percent(X):.1f}%")

    return X.astype(float), y.astype(int), n_classes


def load_hepatitis(csv_path="hepatitis.csv"):
    """Load Hepatitis dataset."""
    df = pd.read_csv(csv_path, na_values="?")

    X = df.iloc[:, :-1]
    y_raw = df.iloc[:, -1].values

    # Encode target with LabelEncoder to ensure 0, 1, ... labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    # Ensure numeric features where possible
    X = X.apply(pd.to_numeric, errors="coerce")

    print(f"Hepatitis: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")
    print(f"Missing rate: {missing_rate_percent(X):.1f}%")

    return X.values.astype(float), y.astype(int), n_classes


def load_pima(csv_path="diabetes.csv"):
    """Load Pima Diabetes dataset.

    Note: Pima has ~48% 'missing' values encoded as zeros in columns:
    Glucose, BloodPressure, SkinThickness, Insulin, BMI
    These zeros are biologically impossible and represent missing data.
    """
    df = pd.read_csv(csv_path)

    X = df.iloc[:, :-1].copy()
    y_raw = df.iloc[:, -1].values

    # Replace zeros with NaN for columns where 0 is biologically impossible
    # Columns: Glucose(1), BloodPressure(2), SkinThickness(3), Insulin(4), BMI(5)
    zero_is_missing_cols = [1, 2, 3, 4, 5]  # 0-indexed
    for col_idx in zero_is_missing_cols:
        col_name = X.columns[col_idx]
        X.loc[X.iloc[:, col_idx] == 0, col_name] = np.nan

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    print(f"Pima: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")
    print(f"Missing rate: {missing_rate_percent(X):.1f}%")

    return X.values.astype(float), y.astype(int), n_classes


def load_credit_approval(csv_path="Aust_credi.csv"):
    """Load UCI Credit Approval (Australian Credit) dataset.

    Dataset has 690 samples, 15 features, binary target.
    Missing values marked with '?'.
    Mix of categorical and continuous features.
    """
    try:
        # Try loading - check if first row looks like header
        df = pd.read_csv(csv_path, na_values='?')

        print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {df.columns.tolist()}")

        # Last column is target
        target_col = df.columns[-1]

        print(f"Target column '{target_col}' unique values: {df[target_col].unique()[:10]}")  # Show first 10
        print(f"Target value counts:\n{df[target_col].value_counts()}")

        # Drop rows with missing target
        df = df.dropna(subset=[target_col])

        # Handle different target encodings
        unique_targets = df[target_col].unique()

        if set(unique_targets) <= {'+', '-'}:
            # Standard +/- encoding
            y = (df[target_col] == '+').astype(int).values
        elif set(unique_targets) <= {0, 1} or set(unique_targets) <= {'0', '1'}:
            # Already 0/1 encoded
            y = df[target_col].astype(int).values
        elif set(unique_targets) <= {1, 2} or set(unique_targets) <= {'1', '2'}:
            # 1/2 encoding (common in UCI datasets)
            y = (df[target_col].astype(int) - 1).values  # Convert to 0/1
        else:
            # Use LabelEncoder for other cases
            le = LabelEncoder()
            y = le.fit_transform(df[target_col])
            print(f"Label encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")

        n_classes = len(np.unique(y))

        # Remove target from features
        X_df = df.drop(columns=[target_col])

        # Encode categorical columns to numeric
        for col in X_df.select_dtypes(include=['object']).columns:
            X_df[col] = pd.Categorical(X_df[col]).codes
            X_df[col] = X_df[col].replace(-1, np.nan)  # -1 from NaN categories

        # Convert all to numeric
        X_df = X_df.apply(pd.to_numeric, errors='coerce')

        print(f"Credit Approval: {X_df.shape[0]} samples, {X_df.shape[1]} features, {n_classes} classes")
        print(f"Target distribution: class 0={sum(y == 0)}, class 1={sum(y == 1)}")
        print(f"Missing rate: {missing_rate_percent(X_df):.1f}%")

        return X_df.values.astype(float), y.astype(int), n_classes

    except Exception as e:
        print(f"Error loading Credit Approval dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# =============================================================================
# STANDARD MLP BASELINE (Mean Impute + Standardize + MLP)
# =============================================================================

class StandardMLP(nn.Module):
    """Simple feed-forward MLP baseline."""

    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_eval_mlp_baseline(X_train, y_train, X_test, y_test, n_classes,
                            hidden_sizes=(64, 32),
                            max_epochs=200,
                            patience=20,
                            lr=1e-3,
                            weight_decay=1e-4,
                            val_frac=0.2,
                            seed=42,
                            device=None):
    """
    Train MLP on (train) with internal validation split for early stopping,
    evaluate on test. Returns metrics dict.
    """
    # split train into train/val
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train, y_train, test_size=val_frac, random_state=seed, stratify=y_train
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tensors
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)
    y_va_t = torch.tensor(y_va, dtype=torch.long, device=device)
    X_te_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    # Use n_classes passed in, not computed from subset
    model = StandardMLP(X_train.shape[1], hidden_sizes, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    best_state = None
    best_val_acc = -1.0
    patience_ctr = 0

    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        out = model(X_tr_t)
        loss = crit(out, y_tr_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_va_t)
            val_pred = val_out.argmax(1)
            val_acc = (val_pred == y_va_t).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_out = model(X_te_t)
        test_prob = torch.softmax(test_out, dim=1).cpu().numpy()
        test_pred = test_out.argmax(1).cpu().numpy()

    acc, prec, rec, spec, f1, auc = compute_metrics(y_test, test_pred, test_prob, n_classes)
    return acc, prec, rec, spec, f1, auc


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def evaluate_baselines(X, y, n_classes, dataset_name="Dataset", num_runs=5, test_size=0.4):
    """
    Evaluate multiple ML baselines on a dataset with missing values (NaN supported).
    Includes MeanImpute+MLP baseline.
    """

    # Classical models (best performers only)
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
    }

    # Baseline name for MLP
    mlp_name = "MLP (MeanImpute)"

    # Store results
    all_results = {model: {"acc": [], "prec": [], "rec": [], "spec": [], "f1": [], "auc": []}
                   for model in list(models.keys()) + [mlp_name]}

    print(f"\n{'=' * 80}")
    print(f"EVALUATING BASELINES ON {dataset_name.upper()}")
    print(f"{'=' * 80}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {n_classes}")
    print(f"Missing rate: {missing_rate_percent(X):.1f}%")
    print(f"Runs: {num_runs}")
    print(f"{'=' * 80}\n")

    for run in range(num_runs):
        seed = 42 + run

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        # Mean imputation
        imputer = SimpleImputer(strategy="mean")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        # Classical models
        for model_name, model in models.items():
            model_clone = clone(model)
            model_clone.fit(X_train_scaled, y_train)

            y_pred = model_clone.predict(X_test_scaled)

            # Some models might not support predict_proba (SVC does since probability=True)
            if hasattr(model_clone, "predict_proba"):
                y_prob = model_clone.predict_proba(X_test_scaled)
            else:
                # Fallback: make a pseudo-probability from decision function if needed
                if hasattr(model_clone, "decision_function"):
                    scores = model_clone.decision_function(X_test_scaled)
                    # convert to 2D-like probability for binary
                    if scores.ndim == 1:
                        probs1 = 1 / (1 + np.exp(-scores))
                        y_prob = np.vstack([1 - probs1, probs1]).T
                    else:
                        # multiclass: softmax
                        ex = np.exp(scores - scores.max(axis=1, keepdims=True))
                        y_prob = ex / ex.sum(axis=1, keepdims=True)
                else:
                    # last resort
                    y_prob = np.ones((len(y_pred), n_classes)) / n_classes

            acc, prec, rec, spec, f1, auc = compute_metrics(y_test, y_pred, y_prob, n_classes)

            all_results[model_name]["acc"].append(acc)
            all_results[model_name]["prec"].append(prec)
            all_results[model_name]["rec"].append(rec)
            all_results[model_name]["spec"].append(spec)
            all_results[model_name]["f1"].append(f1)
            all_results[model_name]["auc"].append(auc)

        # MLP baseline (same impute+scale outputs)
        acc, prec, rec, spec, f1, auc = train_eval_mlp_baseline(
            X_train_scaled, y_train, X_test_scaled, y_test, n_classes,
            hidden_sizes=(64, 32),
            max_epochs=200,
            patience=20,
            lr=1e-3,
            weight_decay=1e-4,
            val_frac=0.2,
            seed=seed
        )

        all_results[mlp_name]["acc"].append(acc)
        all_results[mlp_name]["prec"].append(prec)
        all_results[mlp_name]["rec"].append(rec)
        all_results[mlp_name]["spec"].append(spec)
        all_results[mlp_name]["f1"].append(f1)
        all_results[mlp_name]["auc"].append(auc)

        print(f"  Run {run + 1}/{num_runs} completed")

    # Print results
    print(f"\n{'=' * 120}")
    print(f"RESULTS: {dataset_name.upper()} (mean ± std over {num_runs} runs)")
    print(f"{'=' * 120}")
    print(f"\n{'Model':<20} {'Accuracy':<14} {'Precision':<14} "
          f"{'Recall':<14} {'Specificity':<14} {'F1':<14} {'AUC':<14}")
    print("-" * 110)

    for model_name in list(models.keys()) + [mlp_name]:
        res = all_results[model_name]
        acc_m, acc_s = np.mean(res["acc"]), np.std(res["acc"])
        prec_m, prec_s = np.mean(res["prec"]), np.std(res["prec"])
        rec_m, rec_s = np.mean(res["rec"]), np.std(res["rec"])
        spec_m, spec_s = np.mean(res["spec"]), np.std(res["spec"])
        f1_m, f1_s = np.mean(res["f1"]), np.std(res["f1"])
        auc_m, auc_s = np.mean(res["auc"]), np.std(res["auc"])

        print(f"{model_name:<20} "
              f"{acc_m:.4f}±{acc_s:.4f}  "
              f"{prec_m:.4f}±{prec_s:.4f}  "
              f"{rec_m:.4f}±{rec_s:.4f}  "
              f"{spec_m:.4f}±{spec_s:.4f}  "
              f"{f1_m:.4f}±{f1_s:.4f}  "
              f"{auc_m:.4f}±{auc_s:.4f}")

    print("=" * 110)
    return all_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    import os

    CONFIG = {
        "num_runs": 20,
        "test_size": 0.4,
    }


    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

        def close(self):
            self.log.close()


    output_file = "ml_baselines_results.txt"
    sys.stdout = Logger(output_file)

    print("=" * 80)
    print("TRADITIONAL ML BASELINES + MLP (MEAN IMPUTE) BASELINE")
    print("=" * 80)
    print(f"Number of runs: {CONFIG['num_runs']}")
    print(f"Test size: {CONFIG['test_size']}")
    print("=" * 80)

    # Evaluate on Hepatitis (if present)
    if os.path.exists("hepatitis.csv"):
        print("\n" + "=" * 80)
        print("HEPATITIS DATASET")
        print("=" * 80)
        X, y, n_classes = load_hepatitis("hepatitis.csv")
        _ = evaluate_baselines(
            X, y, n_classes,
            dataset_name="Hepatitis",
            num_runs=CONFIG["num_runs"],
            test_size=CONFIG["test_size"]
        )
    else:
        print("\n⚠️  hepatitis.csv not found")

    # Evaluate on Horse Colic (if present)
    if os.path.exists("horse.csv"):
        print("\n" + "=" * 80)
        print("HORSE COLIC DATASET")
        print("=" * 80)
        X, y, n_classes = load_horse_colic("horse.csv")
        _ = evaluate_baselines(
            X, y, n_classes,
            dataset_name="Horse Colic",
            num_runs=CONFIG["num_runs"],
            test_size=CONFIG["test_size"]
        )
    else:
        print("\n⚠️  horse.csv not found")

    # Evaluate on Pima Diabetes (if present)
    if os.path.exists("diabetes.csv"):
        print("\n" + "=" * 80)
        print("PIMA DIABETES DATASET")
        print("=" * 80)
        X, y, n_classes = load_pima("diabetes.csv")
        _ = evaluate_baselines(
            X, y, n_classes,
            dataset_name="Pima Diabetes",
            num_runs=CONFIG["num_runs"],
            test_size=CONFIG["test_size"]
        )
    else:
        print("\n⚠️  diabetes.csv not found")

    # Evaluate on Credit Approval (if present)
    if os.path.exists("Aust_credi.csv"):
        print("\n" + "=" * 80)
        print("CREDIT APPROVAL DATASET")
        print("=" * 80)
        X, y, n_classes = load_credit_approval("Aust_credi.csv")
        if X is not None:
            _ = evaluate_baselines(
                X, y, n_classes,
                dataset_name="Credit Approval",
                num_runs=CONFIG["num_runs"],
                test_size=CONFIG["test_size"]
            )
    else:
        print("\n⚠️  Aust_credit.csv not found")

    print("\n" + "=" * 80)
    print("ALL EVALUATIONS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_file}")

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print(f"Results saved to: {output_file}")