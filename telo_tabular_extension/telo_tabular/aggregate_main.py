"""Combine per-dataset run directories (runs_main/<dataset>/) into one results table.

    python -m telo_tabular.aggregate_main --root runs_main --out runs_main/summary
"""
import argparse, csv, glob, os
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = []
    for f in sorted(glob.glob(os.path.join(args.root, "*", "test_results.csv"))):
        rows.extend(csv.DictReader(open(f)))
    rows = [r for r in rows if r.get("accuracy") not in (None, "", "None")]
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["mechanism"], r["nominal_rate"], r["variant"]), []).append(r)
    out = []
    for (ds, mech, rate, var), g in sorted(groups.items()):
        acc = np.array([float(r["accuracy"]) for r in g]); f1 = np.array([float(r["macro_f1"]) for r in g])
        auc = np.array([float(r["auc"]) for r in g if r["auc"] not in ("", "None")])
        out.append({"dataset": ds, "mechanism": mech, "nominal_rate": rate, "variant": var, "n_seeds": len(g),
                    "accuracy_mean": acc.mean(), "accuracy_sd": acc.std(ddof=1) if len(acc) > 1 else float("nan"),
                    "macro_f1_mean": f1.mean(), "auc_mean": auc.mean() if len(auc) else float("nan")})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    # matched-condition table: rows datasets, columns variants (accuracy mean ± sd)
    variants = sorted({o["variant"] for o in out}); lines = []
    for (mech, rate) in sorted({(o["mechanism"], o["nominal_rate"]) for o in out}):
        lines += [f"## {mech} {rate} (test accuracy, mean ± sd over seeds)", "", "| dataset | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
        for ds in sorted({o["dataset"] for o in out}):
            cells = []
            for v in variants:
                o = next((o for o in out if (o["dataset"], o["mechanism"], o["nominal_rate"], o["variant"]) == (ds, mech, rate, v)), None)
                cells.append(f"{o['accuracy_mean']:.3f} ± {o['accuracy_sd']:.3f} (n={o['n_seeds']})" if o else "–")
            lines.append(f"| {ds} | " + " | ".join(cells) + " |")
        lines.append("")
    open(args.out + ".md", "w").write("\n".join(lines)); print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
