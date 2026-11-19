"""Extended test metrics from SAVED predictions of evaluated runs (no retraining, no new test queries).

    python -m telo_tabular.extended_metrics --root runs_shift --out runs_shift/extended_metrics
"""
import argparse, csv, glob, os, re
import numpy as np
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, log_loss, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score, average_precision_score)

METRICS = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "auc", "pr_auc",
           "log_loss", "brier", "mcc", "kappa", "sensitivity", "specificity"]


def score(labels, probs):
    probs = probs.astype(np.float64); probs = probs / probs.sum(1, keepdims=True)
    k = probs.shape[1]; classes = np.arange(k); pred = probs.argmax(1); onehot = np.eye(k)[labels]
    pm, rm, fm, _ = precision_recall_fscore_support(labels, pred, average="macro", zero_division=0, labels=classes)
    _, _, fw, _ = precision_recall_fscore_support(labels, pred, average="weighted", zero_division=0, labels=classes)
    out = {"accuracy": float((pred == labels).mean()), "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
           "precision_macro": float(pm), "recall_macro": float(rm), "f1_macro": float(fm), "f1_weighted": float(fw),
           "log_loss": float(log_loss(labels, probs, labels=classes)), "brier": float(((probs - onehot) ** 2).sum(1).mean()),
           "mcc": float(matthews_corrcoef(labels, pred)), "kappa": float(cohen_kappa_score(labels, pred)),
           "auc": np.nan, "pr_auc": np.nan, "sensitivity": np.nan, "specificity": np.nan}
    present = np.unique(labels)
    if k == 2 and len(present) == 2:
        out["auc"] = float(roc_auc_score(labels, probs[:, 1])); out["pr_auc"] = float(average_precision_score(labels, probs[:, 1]))
        out["sensitivity"] = float(((pred == 1) & (labels == 1)).sum() / max(1, (labels == 1).sum()))
        out["specificity"] = float(((pred == 0) & (labels == 0)).sum() / max(1, (labels == 0).sum()))
    elif k > 2 and len(present) == k:
        out["auc"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro", labels=classes))
        out["pr_auc"] = float(average_precision_score(onehot, probs, average="macro"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = []
    for f in sorted(glob.glob(os.path.join(args.root, "*", "*", "seed_*", "*", "predictions_*.npz"))):
        parts = f.split(os.sep); ds, seed, variant = parts[-4], int(parts[-3].split("_")[1]), parts[-2]
        m = re.match(r"predictions_\d+_([a-z]+)_([0-9.e-]+)\.npz", parts[-1]); z = np.load(f)
        rows.append({"dataset": ds, "seed": seed, "variant": variant, "mechanism": m.group(1), "rate": float(m.group(2)),
                     **score(z["labels"], z["probs"])})
    with open(args.out + "_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    agg = {}
    for r in rows:
        agg.setdefault((r["dataset"], r["mechanism"], r["rate"], r["variant"]), []).append(r)
    out = []
    for (ds, mech, rate, var), g in sorted(agg.items()):
        e = {"dataset": ds, "mechanism": mech, "rate": rate, "variant": var, "n_seeds": len(g)}
        for k in METRICS:
            v = np.array([x[k] for x in g], dtype=float)
            e[k] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan"); e[k + "_sd"] = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else float("nan")
        out.append(e)
    with open(args.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    names = {"original": "3C-EA", "relu": "ReLU", "swish": "Swish", "leaky_relu": "LeakyReLU", "elu": "ELU"}
    lines = ["# Extended test metrics (mean over seeds; from saved predictions)", ""]
    for ds in sorted({o["dataset"] for o in out}):
        for mech, rate in sorted({(o["mechanism"], o["rate"]) for o in out if o["dataset"] == ds}, key=lambda c: (["mcar", "group", "mar", "mnar"].index(c[0]), c[1])):
            lines += [f"## {ds} — {mech} {rate:g}", "", "| method | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
            for o in [o for o in out if (o["dataset"], o["mechanism"], o["rate"]) == (ds, mech, rate)]:
                lines.append(f"| {names.get(o['variant'], o['variant'])} | " + " | ".join("–" if np.isnan(o[k]) else f"{o[k]:.3f}" for k in METRICS) + " |")
            lines.append("")
    open(args.out + ".md", "w").write("\n".join(lines) + "\n")
    print(f"{len(rows)} prediction files scored -> {args.out}.md / .csv / _rows.csv")


if __name__ == "__main__":
    main()
