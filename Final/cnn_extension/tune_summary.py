"""Summarise GP-tuning (selection-only) runs: validation-only criterion, never reads test data.

    python -m cnn_extension.tune_summary --runs runs_tune
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    groups = {}
    for path in sorted(glob.glob(os.path.join(args.runs, "*", "rep*", "results.json"))):
        r = json.load(open(path))
        s = r["selection"]
        cfg = r["config"]["search"]
        key = os.path.basename(os.path.dirname(os.path.dirname(path)))
        tbl = (s.get("rerank") or {}).get("table", [])
        win = next((row for row in tbl if row["infix"] == s["infix"]), None)
        groups.setdefault(key, []).append({
            "replicate": r["replicate"], "infix": s["infix"], "selection_rule": s["selection_rule"],
            "search_val_acc_of_search_best": s.get("search_best", {}).get("search_val_acc", s["search_val_acc"]),
            "long_val_acc_selected": win["rerank_val_acc"] if win else None,
            "long_val_acc_best_in_pool": max((row["rerank_val_acc"] or 0) for row in tbl) if tbl else None,
            "winner_search_rank": (s.get("rerank") or {}).get("winner_search_rank"),
            "mutation_rate": cfg["mutation_rate"], "crossover_rate": cfg["crossover_rate"], "elite_size": cfg["elite_size"],
            "population_size": cfg["population_size"], "generations": cfg["generations"],
            "search_runtime_s": r["cost"]["search_runtime_s"], "rerank_runtime_s": r["cost"]["rerank_runtime_s"],
            "mode": r["mode"]})
    summary = {}
    lines = ["| config | p_mut | p_cross | elites | n | long-protocol val acc of selected (mean ± sd) | per seed | selected formulas |",
             "|---|---|---|---|---|---|---|---|"]
    for key, rows in groups.items():
        vals = [x["long_val_acc_selected"] for x in rows if x["long_val_acc_selected"] is not None]
        summary[key] = {"rows": rows, "mean_long_val_acc": float(np.mean(vals)) if vals else None,
                        "sd_long_val_acc": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}
        r0 = rows[0]
        lines.append(f"| {key} | {r0['mutation_rate']} | {r0['crossover_rate']} | {r0['elite_size']} | {len(rows)} | "
                     f"{summary[key]['mean_long_val_acc'] or 0:.4f} ± {summary[key]['sd_long_val_acc'] or 0:.4f} | "
                     f"{', '.join(f'{v:.4f}' for v in vals)} | {' / '.join('`' + x['infix'][:40] + '`' for x in rows)} |")
    text = "\n".join(lines)
    print(text)
    out = args.out or os.path.join(args.runs, "tune_summary")
    json.dump(summary, open(out + ".json", "w"), indent=1)
    open(out + ".md", "w").write("# GP tuning (validation only)\n\n" + text + "\n")


if __name__ == "__main__":
    main()
