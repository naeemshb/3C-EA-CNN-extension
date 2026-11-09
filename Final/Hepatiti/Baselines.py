# =============================================================================
# ml_baselines.py - Traditional ML Baselines for Missing Data Experiments
# =============================================================================
# Run this in the same folder as your dataset CSV files.
# It will evaluate RandomForest, GradientBoosting, SVM, and other baselines.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)
import warnings

warnings.filterwarnings('ignore')


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

    return np.mean(specificities)


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all metrics for a model."""
    num_classes = len(np.unique(y_true))
    avg_method = "binary" if num_classes == 2 else "macro"

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=avg_method, zero_division=0
    )
    spec = compute_multiclass_specificity(y_true, y_pred)

    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except:
        auc = 0.0

    return acc, prec, rec, spec, f1, auc


# =============================================================================
# DATASET LOADERS
# =============================================================================

def load_horse_colic(csv_path="horse.csv"):
    """Load Horse Colic dataset."""
    df = pd.read_csv(csv_path, na_values='?', header=0)

    # Use 'outcome' as target
    target_col = 'outcome'

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])

    # Drop non-useful columns
    cols_to_drop = ['hospital_number', 'lesion_1', 'lesion_2', 'lesion_3',
                    'cp_data', 'surgical_lesion']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Separate features and target
    target_data = df[target_col]
    df = df.drop(columns=[target_col])

    # Encode categorical features
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.Categorical(df[col]).codes
        df[col] = df[col].replace(-1, np.nan)

    # Encode target
    y = pd.Categorical(target_data).codes

    print(f"Horse Colic: {df.shape[0]} samples, {df.shape[1]} features, {len(np.unique(y))} classes")
    print(f"Missing rate: {df.isnull().sum().sum() / df.size * 100:.1f}%")

    return df.values, y


def load_glass(csv_path="glass.csv"):
    """Load Glass dataset."""
    df = pd.read_csv(csv_path)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Encode target if needed
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)

    print(f"Glass: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

    return X, y


def load_hepatitis(csv_path="hepatitis.csv"):
    """Load Hepatitis dataset."""
    df = pd.read_csv(csv_path, na_values='?')

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)

    print(f"Hepatitis: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    print(f"Missing rate: {np.isnan(X).sum() / X.size * 100:.1f}%")

    return X, y


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def evaluate_baselines(X, y, dataset_name="Dataset", num_runs=5, test_size=0.4):
    """
    Evaluate multiple ML baselines on a dataset with missing values.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix (may contain NaN)
    y : array-like, shape (n_samples,)
        Target labels
    dataset_name : str
        Name for display
    num_runs : int
        Number of runs with different random seeds
    test_size : float
        Fraction for test set
    """

    # Define models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'NaiveBayes': GaussianNB(),
    }

    # Use mean imputation (same as our neural network approach)
    imputation_strategy = 'mean'

    # Store results
    all_results = {model: {'acc': [], 'prec': [], 'rec': [], 'spec': [], 'f1': [], 'auc': []}
                   for model in models}

    print(f"\n{'=' * 80}")
    print(f"EVALUATING ML BASELINES ON {dataset_name.upper()}")
    print(f"{'=' * 80}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    print(f"Missing rate: {np.isnan(X.sum()) if isinstance(X, np.ndarray) else 0:.1f}%")
    print(f"Runs: {num_runs}")
    print(f"{'=' * 80}\n")

    for run in range(num_runs):
        seed = 42 + run

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        # Impute missing values with mean (same as our neural network approach)
        imputer = SimpleImputer(strategy='mean')
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        for model_name, model in models.items():
            # Clone model for fresh training
            from sklearn.base import clone
            model_clone = clone(model)

            # Train
            model_clone.fit(X_train_scaled, y_train)

            # Predict
            y_pred = model_clone.predict(X_test_scaled)
            y_prob = model_clone.predict_proba(X_test_scaled)

            # Compute metrics
            acc, prec, rec, spec, f1, auc = compute_metrics(y_test, y_pred, y_prob)

            # Store
            all_results[model_name]['acc'].append(acc)
            all_results[model_name]['prec'].append(prec)
            all_results[model_name]['rec'].append(rec)
            all_results[model_name]['spec'].append(spec)
            all_results[model_name]['f1'].append(f1)
            all_results[model_name]['auc'].append(auc)

        print(f"  Run {run + 1}/{num_runs} completed")

    # Print results
    print(f"\n{'=' * 120}")
    print(f"RESULTS: {dataset_name.upper()} (mean ± std over {num_runs} runs)")
    print(f"{'=' * 120}")

    # Print table
    print(f"\n{'Model':<20} {'Accuracy':<14} {'Precision':<14} "
          f"{'Recall':<14} {'Specificity':<14} {'F1':<14} {'AUC':<14}")
    print("-" * 110)

    for model_name in models:
        res = all_results[model_name]
        acc_m, acc_s = np.mean(res['acc']), np.std(res['acc'])
        prec_m, prec_s = np.mean(res['prec']), np.std(res['prec'])
        rec_m, rec_s = np.mean(res['rec']), np.std(res['rec'])
        spec_m, spec_s = np.mean(res['spec']), np.std(res['spec'])
        f1_m, f1_s = np.mean(res['f1']), np.std(res['f1'])
        auc_m, auc_s = np.mean(res['auc']), np.std(res['auc'])

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

    # Configuration
    CONFIG = {
        "num_runs": 20,
        "test_size": 0.4,
    }


    # Logger to save output
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
    print("TRADITIONAL ML BASELINES FOR MISSING DATA EXPERIMENTS")
    print("=" * 80)
    print(f"Number of runs: {CONFIG['num_runs']}")
    print(f"Test size: {CONFIG['test_size']}")
    print("=" * 80)

    # ==========================================================================
    # Evaluate on Hepatitis
    # ==========================================================================
    if os.path.exists("hepatitis.csv"):
        print("\n" + "=" * 80)
        print("HEPATITIS DATASET")
        print("=" * 80)
        X, y = load_hepatitis("hepatitis.csv")
        hepatitis_results = evaluate_baselines(
            X, y,
            dataset_name="Hepatitis",
            num_runs=CONFIG["num_runs"],
            test_size=CONFIG["test_size"]
        )
    else:
        print("\n⚠️  hepatitis.csv not found")

    print("\n" + "=" * 80)
    print("ALL EVALUATIONS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_file}")

    # Close logger
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print(f"Results saved to: {output_file}")