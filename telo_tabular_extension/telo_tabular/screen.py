"""Confidence screen: does ENTRY-SPECIFIC confidence predict imputation success better than the feature-constant
control?  Runs the confidence stage (no GP, no test data) for every dataset x seed and tabulates validation metrics.

    python -m telo_tabular.screen --config configs/screen.json --output runs_screen
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .experiment import read_config, prepare, _write_csv
from .storage import bind_run

KEYS = ("learned", "feature_constant", "shuffled", "original")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    cfg = read_config(args.config)
    bind_run(args.output, cfg)
    rows = []
    for dataset in cfg["datasets"]:
        for seed in cfg["seeds"]:
            d = Path(args.output) / dataset / f"seed_{seed}"
            print(f"screen: {dataset} seed {seed}", flush=True)
            pack = prepare(cfg, dataset, seed, d)
            info = json.loads((d / "preparation.json").read_text())
            for v in KEYS:
                m = pack["validation_confidence"][v]
                rows.append({"dataset": dataset, "seed": seed, "variant": v, "rows": info["dataset"]["rows"],
                             "features": len(info["dataset"]["features"]), "n_hidden": m["n"], "brier": m["brier"],
                             "event_auc": m["event_auc"], "ece": m.get("ece"),
                             "natural_missing_rate": info["dataset"].get("natural_missing_rate"),
                             "realized_train_rate": info["realized_missing_rates"]["train"],
                             "confidence_fit_runtime_s": info["confidence_fit_runtime_s"]})
            _write_csv(Path(args.output) / "screen_rows.csv", rows)
    # per-dataset summary: learned vs feature_constant, paired over seeds
    summary = []
    for dataset in cfg["datasets"]:
        per_seed = {}
        for r in rows:
            if r["dataset"] == dataset:
                per_seed.setdefault(r["seed"], {})[r["variant"]] = r
        d_auc = [s["learned"]["event_auc"] - s["feature_constant"]["event_auc"] for s in per_seed.values()
                 if s["learned"]["event_auc"] is not None and s["feature_constant"]["event_auc"] is not None]
        d_brier = [s["feature_constant"]["brier"] - s["learned"]["brier"] for s in per_seed.values()]
        any_row = next(iter(per_seed.values()))["learned"]
        learned_auc = [s["learned"]["event_auc"] for s in per_seed.values() if s["learned"]["event_auc"] is not None]
        const_auc = [s["feature_constant"]["event_auc"] for s in per_seed.values() if s["feature_constant"]["event_auc"] is not None]
        wins = sum(x > 0 for x in d_auc)
        summary.append({"dataset": dataset, "rows": any_row["rows"], "features": any_row["features"],
                        "natural_missing_rate": any_row["natural_missing_rate"], "seeds": len(per_seed),
                        "learned_auc_mean": float(np.mean(learned_auc)) if learned_auc else None,
                        "constant_auc_mean": float(np.mean(const_auc)) if const_auc else None,
                        "delta_auc_mean": float(np.mean(d_auc)) if d_auc else None,
                        "delta_auc_sd": float(np.std(d_auc, ddof=1)) if len(d_auc) > 1 else None,
                        "delta_brier_mean": float(np.mean(d_brier)) if d_brier else None,
                        "seeds_learned_wins_auc": wins,
                        "verdict": ("signal" if d_auc and wins == len(d_auc) and np.mean(d_auc) >= 0.05 else
                                    "none" if d_auc and np.mean(d_auc) < 0.02 else "weak")})
    _write_csv(Path(args.output) / "screen_summary.csv", summary)
    lines = ["# Confidence screen (validation only; no GP, no test data)", "",
             "Event: |x_true - train mean_j| <= 0.5 * train sd_j on deliberately hidden validation entries (MCAR 20 % on top of natural missingness).",
             "learned = cross-fitted entry-specific estimate; constant = feature-level success frequency (same scale, no per-entry variation).",
             "Verdict rule (fixed before running): 'signal' = learned beats constant in AUC on every seed with mean gain >= 0.05; 'none' = mean gain < 0.02; otherwise 'weak'.", "",
             "| dataset | rows | features | natural missing | learned AUC | constant AUC | delta AUC (mean ± sd) | seeds learned wins | delta Brier | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s in sorted(summary, key=lambda s: -(s["delta_auc_mean"] or -9)):
        f = lambda v: "n/a" if v is None else f"{v:.3f}"
        lines.append(f"| {s['dataset']} | {s['rows']} | {s['features']} | {f(s['natural_missing_rate'])} | {f(s['learned_auc_mean'])} | "
                     f"{f(s['constant_auc_mean'])} | {f(s['delta_auc_mean'])} ± {f(s['delta_auc_sd'])} | {s['seeds_learned_wins_auc']}/{s['seeds']} | "
                     f"{f(s['delta_brier_mean'])} | **{s['verdict']}** |")
    (Path(args.output) / "screen_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
