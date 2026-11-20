"""Channel ablation of every evolved winner of a finished shift run (trains only the final models; no search).

    python -m telo_tabular.channel_ablation --root runs_shift --out runs_shift/channel_ablation

For each dataset and split, the winner evolved on that split is retrained in four versions with the run's own seed,
partitions, masks and final-training budget: both channels (reproduces the stored model), m only (c fixed to 1),
c only (m fixed to 0) and neither (m = 0, c = 1, i.e. the same formula as a function of the value alone). The channel
is neutralised during training, validation and test alike. All versions are evaluated under every test condition of the
run and compared paired by split (two-sided Wilcoxon signed-rank). The best fixed activation is taken from the stored
results. The formula itself is not re-evolved, so this isolates what the channels contribute to a given function, not
what a search restricted to the value channel would find.
"""
import argparse, glob, json, os
from pathlib import Path
import joblib
import numpy as np
import torch
from scipy.stats import wilcoxon
from .data import load_dataset
from .models import build_model, tensors, train
from .winner_analysis import VERSIONS, FIXED, neutralise, conditions_of, key, acc


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args(argv); out, lines = {}, ["# Channel ablation of the evolved winners (same formula, retrained with channels neutralised)", ""]
    dss = args.datasets or sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(args.root, "*", "manifest.json")))
    for ds in dss:
        root = Path(args.root) / ds; cfg = json.loads((root / "manifest.json").read_text())["config"]
        torch.set_num_threads(cfg["torch_threads"]); conds = conditions_of(cfg); x_full, y_full, _ = load_dataset(ds)
        runs = {key(c): {**{v: [] for v, _ in VERSIONS}, "fixed": []} for c in conds}; uses = []
        for seed in cfg["seeds"]:
            d = root / ds / f"seed_{seed}"; pk = joblib.load(d / "prepared.joblib"); res = json.loads((d / "original" / "training.json").read_text())
            st = {v: {f"{r['mechanism']}_{r['nominal_rate']:g}": r["accuracy"] for r in json.loads((d / v / "evaluation.json").read_text())["rows"]}
                  for v in ["original"] + FIXED}
            models = {}
            for v, _ in VERSIONS:
                tr = (pk["train"][0], *neutralise(pk["train"][1], pk["train"][2], v), pk["train"][3])
                va = (pk["val"][0], *neutralise(pk["val"][1], pk["val"][2], v), pk["val"][3])
                torch.manual_seed(seed + 10000)
                mdl = build_model("three_channel", tr[0].shape[1], len(pk["metadata"]["classes"]), cfg["hidden_sizes"], res["tree"], "full")
                info = train(mdl, tensors(tr, "cpu"), tensors(va, "cpu"), cfg["final"], final=True)
                models[v] = mdl if info["valid"] else None
            te = pk["split_indices"]["test"]
            for i, cond in enumerate(conds):
                masked, _ = pk["mask_generator"].corrupt(x_full[te], cond["mechanism"], cond["rate"], seed + 20000 + i)
                z, m, c = pk["preprocessor"].transform(masked); y = y_full[te]; k = key(cond)
                for v, _ in VERSIONS:
                    try: a = acc(models[v], z, *neutralise(m, c, v), y) if models[v] is not None else float("nan")
                    except FloatingPointError: a = float("nan")
                    runs[k][v].append(a)
                runs[k]["fixed"].append({f: st[f][k] for f in FIXED})
                if i == 0:
                    assert abs(runs[k]["both"][-1] - st["original"][k]) < 1e-9, (ds, seed, "stored model not reproduced")
            uses.append({"seed": seed, "infix": res["infix"]})
        summary = {}
        for k, row in runs.items():
            fx = {f: float(np.mean([r[f] for r in row["fixed"]])) for f in FIXED}; best = max(fx, key=fx.get)
            bf = np.array([r[best] for r in row["fixed"]]); arr = {v: np.array(row[v]) for v, _ in VERSIONS}
            ok = np.isfinite(np.column_stack(list(arr.values()))).all(axis=1)
            def p(a, b):
                try: return float(wilcoxon(a[ok], b[ok]).pvalue)
                except ValueError: return None
            summary[k] = {**{v: {"mean": float(a[ok].mean()), "sd": float(a[ok].std(ddof=1))} for v, a in arr.items()}, "n": int(ok.sum()),
                          "best_fixed": best, "best_fixed_mean": float(bf[ok].mean()), "p_both_vs_neither": p(arr["both"], arr["neither"]),
                          "both_minus_neither": float((arr["both"][ok] - arr["neither"][ok]).mean()),
                          "wins_both": int((arr["both"][ok] > arr["neither"][ok]).sum()), "wins_neither": int((arr["both"][ok] < arr["neither"][ok]).sum()),
                          "p_neither_vs_best_fixed": p(arr["neither"], bf), "neither_minus_best_fixed": float((arr["neither"][ok] - bf[ok]).mean())}
        out[ds] = {"summary": summary, "runs": {k: {v: row[v] for v, _ in VERSIONS} for k, row in runs.items()}, "winners": uses}
        lines += [f"## {ds} ({len(cfg['seeds'])} splits)", "",
                  "| condition | both | m only | c only | neither | both − neither | p | both wins / neither wins | best fixed | neither − best fixed | p |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in [key(c) for c in conds]:
            s = summary[k]; fmt = lambda q: "n/a" if q is None else f"{q:.4f}"
            lines.append(f"| {k} | " + " | ".join(f"{s[v]['mean']:.3f}" for v, _ in VERSIONS) + f" | {s['both_minus_neither'] * 100:+.1f} | {fmt(s['p_both_vs_neither'])} | "
                         f"{s['wins_both']} / {s['wins_neither']} | {s['best_fixed_mean']:.3f} ({s['best_fixed']}) | {s['neither_minus_best_fixed'] * 100:+.1f} | {fmt(s['p_neither_vs_best_fixed'])} |")
        lines.append("")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out + ".json", "w"), indent=1); open(args.out + ".md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
