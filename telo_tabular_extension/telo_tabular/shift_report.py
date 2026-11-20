"""Frozen-model missingness-shift report: evolved 3C-EA vs fixed activations, per dataset and test condition.

    python -m telo_tabular.shift_report --root runs_shift --out runs_shift/shift_report
Paired over seeds (same splits and masks). p = two-sided Wilcoxon signed-rank, evolved vs the best fixed activation
(chosen per condition by mean, which favours the baseline). Descriptive; no multiple-comparison correction.
"""
import argparse, csv, glob, json, os
import numpy as np
from scipy.stats import wilcoxon

FIXED = ["relu", "swish", "leaky_relu", "elu"]
ORDER = ["mcar", "group", "mar", "mnar"]


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    lines, summary = ["# Missingness-shift report (models trained at MCAR 20 %, frozen at test)", ""], {}
    for f in sorted(glob.glob(os.path.join(args.root, "*", "test_results.csv"))):
        ds = os.path.basename(os.path.dirname(f))
        rows = [r for r in csv.DictReader(open(f)) if r["accuracy"] not in ("", "None")]
        conds = sorted({(r["mechanism"], float(r["nominal_rate"])) for r in rows}, key=lambda c: (ORDER.index(c[0]), c[1]))
        def get(v, c, metric):
            d = {int(r["seed"]): float(r[metric]) for r in rows if r["variant"] == v and (r["mechanism"], float(r["nominal_rate"])) == c
                 and r[metric] not in ("", "None")}
            return d
        summary[ds] = {}
        for metric in ("accuracy", "macro_f1", "auc"):
            lines += [f"## {ds} — {metric} (n_test={rows[0]['n_test']}, {len({r['seed'] for r in rows})} seeds)", "",
                      "| condition | realized | 3C-EA | ReLU | Swish | LeakyReLU | ELU | 3C-EA − best fixed | p | seeds 3C-EA > best |", "|---|---|---|---|---|---|---|---|---|---|"]
            for c in conds:
                ev = get("original", c, metric); fx = {v: get(v, c, metric) for v in FIXED}
                if not ev: continue
                seeds = sorted(set(ev) & set.intersection(*[set(d) for d in fx.values()]))
                e = np.array([ev[s] for s in seeds]); F = {v: np.array([fx[v][s] for s in seeds]) for v in FIXED}
                best = max(FIXED, key=lambda v: F[v].mean()); d = e - F[best]
                try: p = float(wilcoxon(e, F[best]).pvalue)
                except ValueError: p = float("nan")
                rr = np.mean([float(r["realized_rate"]) for r in rows if (r["mechanism"], float(r["nominal_rate"])) == c])
                summary[ds].setdefault(metric, {})[f"{c[0]}_{c[1]:g}"] = {"evolved": float(e.mean()), "evolved_sd": float(e.std(ddof=1)),
                    "best_fixed": best, "best_fixed_mean": float(F[best].mean()), "best_fixed_sd": float(F[best].std(ddof=1)), "diff": float(d.mean()), "p": p, "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "n": len(seeds)}
                lines.append(f"| {c[0]} {c[1]:g} | {rr:.2f} | {e.mean():.3f}±{e.std(ddof=1):.3f} | " + " | ".join(f"{F[v].mean():.3f}±{F[v].std(ddof=1):.3f}" for v in FIXED) +
                             f" | {d.mean():+.3f} ({best}) | {p:.3f} | {int((d > 0).sum())}/{len(seeds)} |")
            lines.append("")
        lines.append(f"Evolved formulas ({ds}): " + "; ".join("`" + json.load(open(t))["infix"][:50] + "`" for t in sorted(glob.glob(os.path.join(args.root, ds, ds, "seed_*", "original", "training.json")))))
        lines.append("")
    json.dump(summary, open(args.out + ".json", "w"), indent=1); open(args.out + ".md", "w").write("\n".join(lines) + "\n")
    # compact console view: accuracy and auc differences
    for ds, s in summary.items():
        print(f"\n== {ds}")
        for k in s["accuracy"]:
            a, u = s["accuracy"][k], s.get("auc", {}).get(k)
            print(f"  {k:10s} acc 3C-EA {a['evolved']:.3f}±{a['evolved_sd']:.3f} vs {a['best_fixed']:10s} {a['best_fixed_mean']:.3f}  d {a['diff']:+.3f} p {a['p']:.3f} wins {a['wins']}/{a['n']}"
                  + (f" | auc d {u['diff']:+.3f} p {u['p']:.3f}" if u else ""))


if __name__ == "__main__":
    main()
