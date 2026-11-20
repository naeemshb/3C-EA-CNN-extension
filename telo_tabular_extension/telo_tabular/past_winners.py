"""Do the winners of the original runs need their reliability channels? Screening followed by independent confirmation.

    python -m telo_tabular.past_winners screen  --datasets hepatitis housevote horse cylinderband --out runs_past_winners
    python -m telo_tabular.past_winners confirm --out runs_past_winners

Source of candidates: the winner logs written by the original scripts (Final/<Dataset>/**/evo_logs/winner_run_0pct_*.json),
i.e. runs on the datasets with natural missingness and no injected missingness. Only winners whose formula contains both
m and c are candidates; identical formulas are counted once.

Protocol (fixed before any confirmation data were looked at):
  screen   every candidate is retrained on SCREEN_SEEDS splits in two versions - with both channels and with the channels
           neutralised (m = 0, c = 1) in training, validation and test - together with the four fixed activations.
           Candidates are ranked by the mean paired difference "both minus neither"; a candidate is eligible only if its
           mean accuracy with both channels is at least that of the best fixed activation on the screening splits.
  confirm  the TOP_K eligible candidates per dataset are retrained on CONFIRM_SEEDS, which are disjoint from the screening
           seeds, in four versions (both, m only, c only, neither) plus the fixed activations. Reported: mean accuracy,
           paired differences, two-sided Wilcoxon signed-rank p-values and their Holm correction over all confirmed candidates.
Splits, mean imputation, the confidence rule, the network, and the final-training budget are those of the package
(60/20/20 stratified split, hidden sizes 64-32, full-batch Adam, at most 100 epochs with patience 15).
"""
import argparse, ast, glob, json, os, re
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wilcoxon
from . import gp_core as g
from .data import load_dataset, split_indices, MeanPreprocessor
from .datasets_gecco import REGISTRY
from .models import build_model, tensors, train, predict, scores, fixed_tree
from .winner_analysis import VERSIONS, FIXED, neutralise

FINAL_ROOT = Path(__file__).resolve().parents[2] / "Final"
FINAL = {"lr": 0.001, "weight_decay": 0.0001, "grad_clip": 1, "epochs": 100, "patience": 15}
HIDDEN = [64, 32]
SCREEN_SEEDS = list(range(1000, 1012))
CONFIRM_SEEDS = list(range(5000, 5100))
TOP_K = 3
INPUTS = {"value", "missing", "confidence"}


def parse(text):
    """Tree text written by the original tree_to_string -> gp_core tree."""
    tokens = re.findall(r"[A-Za-z_]+|-?\d+\.?\d*(?:[eE][-+]?\d+)?|[(),]", text); pos = 0
    def node():
        nonlocal pos
        tok = tokens[pos]; pos += 1
        if pos < len(tokens) and tokens[pos] == "(":
            pos += 1; kids = [node()]
            while tokens[pos] == ",":
                pos += 1; kids.append(node())
            assert tokens[pos] == ")"; pos += 1
            return g.create_unary_node(tok, kids[0]) if len(kids) == 1 else g.create_binary_node(tok, kids[0], kids[1])
        return g.create_input_node(tok) if tok in INPUTS else g.create_constant_node(float(tok))
    tree = node(); assert pos == len(tokens), "unparsed tail"
    return tree


def candidates(ds):
    folder = FINAL_ROOT / REGISTRY[ds][0].split("/")[0]; found = {}
    for f in sorted(glob.glob(str(folder / "**" / "evo_logs" / "winner_run_0pct_*.json"), recursive=True)):
        try:
            best = json.load(open(f))["best"]; best = ast.literal_eval(best) if isinstance(best, str) else best
            tree = parse(best["tree_str"]); infix = g.tree_to_infix(tree)
        except Exception:
            continue
        if re.search(r"\bm\b", infix) and re.search(r"\bc\b", infix):
            found.setdefault(infix, {"tree": tree, "infix": infix, "logs": 0, "first_log": os.path.relpath(f, FINAL_ROOT)})["logs"] += 1
    return list(found.values())


class Split:
    def __init__(self, x, y, seed):
        idx = split_indices(y, seed); pre = MeanPreprocessor.fit(x[idx["train"]]); self.seed = seed
        self.parts = {k: (*pre.transform(x[idx[k]]), y[idx[k]]) for k in ("train", "val", "test")}; self.classes = len(np.unique(y))

    def run(self, tree, version="both"):
        tr, va, te = ((p[0], *neutralise(p[1], p[2], version), p[3]) for p in (self.parts[k] for k in ("train", "val", "test")))
        torch.manual_seed(self.seed + 10000)
        model = build_model("three_channel", tr[0].shape[1], self.classes, HIDDEN, tree, "full")
        info = train(model, tensors(tr, "cpu"), tensors(va, "cpu"), FINAL, final=True)
        if not info["valid"]:
            return float("nan")
        try:
            return scores(te[3], predict(model, tensors(te, "cpu")))["accuracy"]
        except FloatingPointError:
            return float("nan")


def pval(a, b):
    try: return float(wilcoxon(a, b).pvalue)
    except ValueError: return None


def screen(args):
    os.makedirs(args.out, exist_ok=True); torch.set_num_threads(2); result = {}
    for ds in args.datasets:
        x, y, _ = load_dataset(ds); cands = candidates(ds); splits = [Split(x, y, s) for s in SCREEN_SEEDS]
        fixed = {f: [sp.run(fixed_tree(f)) for sp in splits] for f in FIXED}; best_fixed = max(FIXED, key=lambda f: np.mean(fixed[f]))
        rows = []
        for i, cnd in enumerate(cands):
            both = np.array([sp.run(cnd["tree"], "both") for sp in splits]); nei = np.array([sp.run(cnd["tree"], "neither") for sp in splits])
            ok = bool(np.isfinite(both).all() and np.isfinite(nei).all())
            rows.append({"infix": cnd["infix"], "logs": cnd["logs"], "first_log": cnd["first_log"], "valid": ok, "both": float(np.nanmean(both)),
                         "neither": float(np.nanmean(nei)), "diff": float(np.nanmean(both - nei)) if ok else None,
                         "eligible": ok and float(both.mean()) >= float(np.mean(fixed[best_fixed])), "tree": cnd["tree"]})
            if (i + 1) % 20 == 0: print(f"{ds}: {i + 1}/{len(cands)} candidates screened", flush=True)
        rows.sort(key=lambda r: (-(r["diff"] if r["eligible"] else -9), r["infix"]))
        result[ds] = {"n_candidates": len(cands), "screen_seeds": SCREEN_SEEDS, "fixed": {f: float(np.mean(v)) for f, v in fixed.items()},
                      "best_fixed": best_fixed, "rows": rows}
        print(f"== {ds}: {len(cands)} candidates with both channels; best fixed {best_fixed} {np.mean(fixed[best_fixed]):.3f}; "
              f"valid {sum(r['valid'] for r in rows)}, eligible {sum(r['eligible'] for r in rows)}, "
              f"both>neither in {sum(1 for r in rows if r['valid'] and r['diff'] > 0)}", flush=True)
        for r in rows[:6]:
            print(f"   both {r['both']:.3f} neither {r['neither']:.3f} diff {0 if r['diff'] is None else r['diff'] * 100:+.1f} eligible {r['eligible']} | {r['infix'][:90]}", flush=True)
        json.dump(result, open(os.path.join(args.out, "screen.json"), "w"), indent=1)


def confirm(args):
    torch.set_num_threads(2); scr = json.load(open(os.path.join(args.out, "screen.json"))); out, tests = {}, []
    for ds, block in scr.items():
        chosen = [r for r in block["rows"] if r["eligible"]][:TOP_K]
        if not chosen: continue
        x, y, _ = load_dataset(ds); splits = [Split(x, y, s) for s in CONFIRM_SEEDS]
        fixed = {f: np.array([sp.run(fixed_tree(f)) for sp in splits]) for f in FIXED}; best = max(FIXED, key=lambda f: fixed[f].mean()); out[ds] = []
        for r in chosen:
            acc = {v: np.array([sp.run(r["tree"], v) for sp in splits]) for v, _ in VERSIONS}
            ok = np.isfinite(np.column_stack(list(acc.values()))).all(axis=1); a = {v: q[ok] for v, q in acc.items()}; bf = fixed[best][ok]
            row = {"infix": r["infix"], "first_log": r["first_log"], "logs": r["logs"], "screen_diff": r["diff"], "n": int(ok.sum()),
                   **{v: {"mean": float(q.mean()), "sd": float(q.std(ddof=1))} for v, q in a.items()},
                   "fixed": {f: float(fixed[f][ok].mean()) for f in FIXED}, "best_fixed": best,
                   "both_minus_neither": float((a["both"] - a["neither"]).mean()), "p_both_vs_neither": pval(a["both"], a["neither"]),
                   "wins": int((a["both"] > a["neither"]).sum()), "losses": int((a["both"] < a["neither"]).sum()),
                   "both_minus_m_only": float((a["both"] - a["m_only"]).mean()), "p_both_vs_m_only": pval(a["both"], a["m_only"]),
                   "both_minus_c_only": float((a["both"] - a["c_only"]).mean()), "p_both_vs_c_only": pval(a["both"], a["c_only"]),
                   "both_minus_best_fixed": float((a["both"] - bf).mean()), "p_both_vs_best_fixed": pval(a["both"], bf),
                   "runs": {v: [float(t) for t in q] for v, q in acc.items()}, "runs_best_fixed": [float(t) for t in fixed[best]], "tree": r["tree"]}
            out[ds].append(row); tests.append(row)
            print(f"{ds}: both {row['both']['mean']:.3f} m {row['m_only']['mean']:.3f} c {row['c_only']['mean']:.3f} neither {row['neither']['mean']:.3f} "
                  f"best fixed {row['fixed'][best]:.3f} | both-neither {row['both_minus_neither'] * 100:+.1f} p {row['p_both_vs_neither']} | {r['infix'][:70]}", flush=True)
    order = sorted(range(len(tests)), key=lambda i: 1 if tests[i]["p_both_vs_neither"] is None else tests[i]["p_both_vs_neither"]); running = 0.
    for rank, i in enumerate(order):      # Holm step-down over all confirmed candidates
        p = tests[i]["p_both_vs_neither"]; adj = None if p is None else min(1., p * (len(tests) - rank))
        if adj is not None: running = max(running, adj); adj = running
        tests[i]["p_holm_both_vs_neither"] = adj
    json.dump({"confirm_seeds": CONFIRM_SEEDS, "top_k": TOP_K, "datasets": out}, open(os.path.join(args.out, "confirm.json"), "w"), indent=1)
    L = ["# Winners of the original runs: do they need their channels?", "",
         f"Screening on {len(SCREEN_SEEDS)} splits, confirmation of the top {TOP_K} per dataset on {len(CONFIRM_SEEDS)} disjoint splits. "
         "Accuracy, mean over splits; p = two-sided Wilcoxon signed-rank, paired by split; Holm over all confirmed candidates.", ""]
    for ds, rows in out.items():
        blk = scr[ds]; valid = [r for r in blk["rows"] if r["valid"]]
        L += [f"## {ds}", "", f"Screening: {blk['n_candidates']} distinct winners with both channels, {len(valid)} numerically valid, "
              f"{sum(r['eligible'] for r in blk['rows'])} at least as accurate as the best fixed activation, "
              f"{sum(1 for r in valid if r['diff'] > 0)} more accurate with channels than without; median difference {np.median([r['diff'] for r in valid]) * 100:+.1f} points.", "",
              "| formula | both | m only | c only | neither | best fixed | both − neither | p | Holm p | wins/losses | both − best fixed | p |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        f4 = lambda q: "n/a" if q is None else f"{q:.4f}"
        for r in rows:
            L.append(f"| `{r['infix'][:80]}` | {r['both']['mean']:.3f} | {r['m_only']['mean']:.3f} | {r['c_only']['mean']:.3f} | {r['neither']['mean']:.3f} | "
                     f"{r['fixed'][r['best_fixed']]:.3f} ({r['best_fixed']}) | {r['both_minus_neither'] * 100:+.1f} | {f4(r['p_both_vs_neither'])} | {f4(r['p_holm_both_vs_neither'])} | "
                     f"{r['wins']}/{r['losses']} | {r['both_minus_best_fixed'] * 100:+.1f} | {f4(r['p_both_vs_best_fixed'])} |")
        L.append("")
    open(os.path.join(args.out, "confirm.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["screen", "confirm"]); ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", default=["hepatitis", "housevote", "horse", "cylinderband"])
    args = ap.parse_args(argv); (screen if args.stage == "screen" else confirm)(args)


if __name__ == "__main__":
    main()
