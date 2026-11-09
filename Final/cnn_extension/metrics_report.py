"""
Extended test metrics from SAVED predictions (no retraining, no new test-set queries).

For each run directory the matched-condition predictions (``test_predictions.npz``) and every saved
shift evaluation (``shift_<pattern>_<rate>pct_predictions.npz``) are scored with: accuracy, balanced
accuracy, macro / weighted precision, recall and F1, one-vs-rest macro ROC AUC, macro average precision,
log loss, multi-class Brier score, expected calibration error (15 bins), Cohen's kappa, Matthews
correlation and top-2 accuracy. Per-class precision / recall / F1 go to the JSON only.

    python -m cnn_extension.metrics_report --runs runs/ref_fashion_mnist_random20_*/rep0 --out runs/metrics_fmnist_random20
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, cohen_kappa_score,
                             f1_score, log_loss, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score,
                             top_k_accuracy_score)

COLUMNS = [("accuracy", "Acc"), ("balanced_accuracy", "Bal. acc"), ("precision_macro", "P (macro)"),
           ("recall_macro", "R (macro)"), ("f1_macro", "F1 (macro)"), ("f1_weighted", "F1 (wtd)"),
           ("roc_auc_macro_ovr", "AUC (OvR)"), ("average_precision_macro", "AP (macro)"), ("log_loss", "Log loss"),
           ("brier_multiclass", "Brier"), ("ece_15", "ECE"), ("cohen_kappa", "Kappa"), ("mcc", "MCC"), ("top2_accuracy", "Top-2")]


def expected_calibration_error(probs, labels, n_bins=15):
    conf = probs.max(1); pred = probs.argmax(1); correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def score(probs, labels, preds):
    probs = np.asarray(probs, dtype=np.float64); labels = np.asarray(labels); preds = np.asarray(preds)
    probs = probs / probs.sum(1, keepdims=True)
    n_cls = probs.shape[1]; classes = np.arange(n_cls)
    onehot = np.eye(n_cls)[labels]
    pm, rm, fm, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    pc, rc, fc, sup = precision_recall_fscore_support(labels, preds, labels=classes, zero_division=0)
    out = {
        "n": int(len(labels)), "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "precision_macro": float(pm), "recall_macro": float(rm), "f1_macro": float(fm),
        "precision_weighted": float(pw), "recall_weighted": float(rw), "f1_weighted": float(fw),
        "f1_micro": float(f1_score(labels, preds, average="micro")),
        "roc_auc_macro_ovr": float(roc_auc_score(labels, probs, multi_class="ovr", average="macro", labels=classes)),
        "roc_auc_weighted_ovr": float(roc_auc_score(labels, probs, multi_class="ovr", average="weighted", labels=classes)),
        "average_precision_macro": float(average_precision_score(onehot, probs, average="macro")),
        "log_loss": float(log_loss(labels, probs, labels=classes)),
        "brier_multiclass": float(((probs - onehot) ** 2).sum(1).mean()),
        "ece_15": expected_calibration_error(probs, labels),
        "cohen_kappa": float(cohen_kappa_score(labels, preds)), "mcc": float(matthews_corrcoef(labels, preds)),
        "top2_accuracy": float(top_k_accuracy_score(labels, probs, k=2, labels=classes)),
        "per_class": {int(c): {"precision": float(pc[i]), "recall": float(rc[i]), "f1": float(fc[i]), "support": int(sup[i])}
                      for i, c in enumerate(classes)},
    }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="output prefix (.json and .md are written)")
    args = ap.parse_args(argv)
    dirs = []
    for pat in args.runs:
        dirs.extend(sorted(glob.glob(pat)) or [pat])
    report = {}
    for d in dirs:
        if not os.path.exists(os.path.join(d, "test_predictions.npz")):
            continue
        res = json.load(open(os.path.join(d, "results.json")))
        name = res["name"]
        entry = {"run_dir": d, "selected_infix": res["selected_infix"], "replicate": res["replicate"], "conditions": {}}
        z = np.load(os.path.join(d, "test_predictions.npz"))
        c = res["config"]["corruption"]
        entry["conditions"][f"matched_{c['pattern']}_{int(round(100 * c['rate']))}pct"] = score(z["probs"], z["labels"], z["preds"])
        for sp in sorted(glob.glob(os.path.join(d, "shift_*_predictions.npz"))):
            tag = os.path.basename(sp)[len("shift_"):-len("_predictions.npz")]
            z = np.load(sp)
            entry["conditions"][f"shift_{tag}"] = score(z["probs"], z["labels"], z["preds"])
        report[name] = entry
    json.dump(report, open(args.out + ".json", "w"), indent=1)
    conditions = sorted({k for e in report.values() for k in e["conditions"]},
                        key=lambda k: (not k.startswith("matched"), k))
    lines = ["# Extended test metrics (from saved predictions; one replicate)", ""]
    for cond in conditions:
        lines += [f"## {cond}", "", "| model | activation | " + " | ".join(h for _, h in COLUMNS) + " |",
                  "|---|---|" + "---|" * len(COLUMNS)]
        for name, e in report.items():
            s = e["conditions"].get(cond)
            if not s:
                continue
            short = name.split("_")[-1] if not name.endswith(("_rr10", "_rr20")) else "_".join(name.split("_")[-2:])
            lines.append(f"| {name} | `{e['selected_infix'][:34]}` | " + " | ".join(f"{s[k]:.4f}" for k, _ in COLUMNS) + " |")
        lines.append("")
    lines += ["Acc = accuracy; Bal. acc = balanced accuracy (mean per-class recall); P/R/F1 macro = unweighted class means; "
              "F1 (wtd) = support-weighted; AUC (OvR) = one-vs-rest ROC AUC, macro; AP = macro average precision (PR-AUC); "
              "Brier = multi-class Brier score (lower is better); ECE = expected calibration error, 15 bins (lower is better); "
              "Kappa = Cohen's kappa; MCC = Matthews correlation; Top-2 = top-2 accuracy. Fashion-MNIST test set is balanced "
              "(1,000 per class), so accuracy, balanced accuracy and micro-F1 coincide up to rounding."]
    open(args.out + ".md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
