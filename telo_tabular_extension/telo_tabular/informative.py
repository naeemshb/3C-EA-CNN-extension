"""Controlled informative missingness: does the benefit of the reliability channels grow with how much the missingness
pattern tells about the class?

    python -m telo_tabular.informative design --out runs_informative          # no model: realised rates and mask-only accuracy
    python -m telo_tabular.informative run    --out runs_informative [--datasets ...] [--levels ...] [--seeds ...]
    python -m telo_tabular.informative report --out runs_informative

PROTOCOL (fixed on 2026-11-20, before any network of this experiment was trained; see README)

Data      Every dataset of the package that is complete and has two classes: wdbc, pima, sonar, heart, aust_credit, mammo.
          Excluded by this rule: glass (six classes; the mechanism below is defined for two), mushroom and adult (their
          files contain natural missing values, 30.5 % of one mushroom feature) and the datasets with natural missingness.
Mechanism Every entry of a record of class k is deleted independently with probability p_k, in the training, validation
          and test partitions alike (the mechanism is a property of the data source; labels are never given to a model).
          With overall rate r = 0.2, d features, pi = share of class 1 and strength s:
              delta = s * sqrt(r (1 - r) / d),   p_1 = r + (1 - pi) delta,   p_0 = r - pi delta.
          The expected overall rate is r for every s. s is the distance between the classes, in standard deviations, of the
          number of missing features in a record; s = 0 is MCAR at 20 %. Levels: s in {0, 0.5, 1, 2}.
          How informative the pattern is in fact is measured, not assumed: test accuracy of a logistic regression that
          sees only the mask ("mask-only"), reported next to the majority rate.
Splits    20 splits per dataset and level, seeds 42-61, package split (60/20/20, stratified); mean imputation and the
          confidence rule c = max(0.1, 1 - r_j) fitted on the masked training partition. Mask seeds: seed + 100 (train),
          seed + 200 (validation), seed + 20000 (test).
Methods   on        3C-EA: the search and final training of the package with the propagated channels (m, c).
          off       the identical search (same seed, same initial population, same fitness) and final training with the
                    channels neutralised everywhere (m = 0, c = 1 in training, validation and test).
          m_only, c_only, neither   the winner of "on" retrained with one or both channels neutralised (no new search).
          relu, swish, leaky_relu, elu   fixed activations, same network.
          mask_concat_relu   ReLU network that receives the mask as additional inputs.
          Search: population 100, 10 generations, remaining settings and the final-training budget as in the shift runs.
Outcome   Test accuracy (macro-F1 and AUC are stored).
Primary   Per dataset and level: paired difference on - off over the 20 splits, mean with 95 % t-interval and two-sided
          Wilcoxon signed-rank test. Hypothesis: the difference is about zero at s = 0 and grows with s. Test of the
          hypothesis per dataset: Wilcoxon on the per-split change of the difference between s = 2 and s = 0
          (same split seeds), Holm-corrected over the six datasets.
Secondary on - mask_concat_relu, on - best fixed activation (best by mean per dataset and level, which favours the
          baseline), off - best fixed activation, and the three ablations of the winner; uncorrected, labelled as such.
Everything is reported whichever way it comes out.

Each (dataset, level, seed) cell is written to its own json and skipped when present, so runs can be split over
processes and resumed. The search state of a cell is removed once the cell is complete; the cell keeps the winner, the
ten best candidates and the search time.
"""
import argparse, hashlib, json, os, shutil
from pathlib import Path
import numpy as np
import torch
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from . import gp_core as g
from .data import load_dataset, split_indices, MeanPreprocessor
from .models import build_model, tensors, train, predict, scores, fixed_tree
from .search import search
from .winner_analysis import neutralise

DATASETS = ["wdbc", "pima", "sonar", "heart", "aust_credit", "mammo"]
LEVELS = [0.0, 0.5, 1.0, 2.0]
SEEDS = list(range(42, 62))
RATE = 0.2
HIDDEN = [64, 32]
FIXED = ["relu", "swish", "leaky_relu", "elu"]
ABLATIONS = ["m_only", "c_only", "neither"]
SEARCH = {"population": 100, "generations": 10, "init_depth": 3, "mutation_rate": 0.15, "crossover_rate": 0.7, "elite_size": 2,
          "max_seconds": 36000, "safety_max_depth": 12, "safety_max_nodes": 256,
          "training": {"lr": 0.001, "weight_decay": 0.0001, "grad_clip": 1, "epochs": 30, "patience": 2}}
FINAL = {"lr": 0.001, "weight_decay": 0.0001, "grad_clip": 1, "epochs": 100, "patience": 15}
FINGERPRINT = hashlib.sha256(json.dumps([RATE, HIDDEN, SEARCH, FINAL], sort_keys=True).encode()).hexdigest()
METHODS = ["on", "off"] + ABLATIONS + FIXED + ["mask_concat_relu"]


def class_rates(y, level, d, rate=RATE):
    pi = float((y == 1).mean()); delta = level * np.sqrt(rate * (1 - rate) / d)
    p1, p0 = rate + (1 - pi) * delta, rate - pi * delta
    if not (0 <= p0 <= 1 and 0 <= p1 <= 1): raise ValueError(f"strength {level} is not feasible for d = {d}, pi = {pi:.3f}")
    return p0, p1


def informative_mask(y, d, rates, seed):
    p0, p1 = rates
    return np.random.default_rng(seed).random((len(y), d)) < np.where(y == 1, p1, p0)[:, None]


def build(ds, level, seed):
    """Masked, imputed partitions of one cell and the measured informativeness of the mask."""
    x, y, _ = load_dataset(ds)
    if np.isnan(x).any() or len(np.unique(y)) != 2: raise ValueError(ds + " must be complete and two-class")
    idx = split_indices(y, seed); rates = class_rates(y, level, x.shape[1]); masks, masked = {}, {}
    for k, off in (("train", 100), ("val", 200), ("test", 20000)):
        masks[k] = informative_mask(y[idx[k]], x.shape[1], rates, seed + off); masked[k] = np.where(masks[k], np.nan, x[idx[k]])
    pre = MeanPreprocessor.fit(masked["train"]); parts = {k: (*pre.transform(masked[k]), y[idx[k]]) for k in masked}
    lr = LogisticRegression(max_iter=2000).fit(masks["train"].astype(float), y[idx["train"]])
    info = {"p_class0": rates[0], "p_class1": rates[1], "share_class1": float((y == 1).mean()),
            "realised_rate": {k: float(m.mean()) for k, m in masks.items()},
            "realised_rate_by_class_test": [float(masks["test"][y[idx["test"]] == c].mean()) for c in (0, 1)],
            "mask_only_accuracy": float((lr.predict(masks["test"].astype(float)) == y[idx["test"]]).mean()),
            "majority_rate_test": float(np.bincount(y[idx["test"]]).max() / len(idx["test"]))}
    return parts, info


def fit_eval(model_type, tree, parts, version, seed):
    tr, va, te = ((p[0], *neutralise(p[1], p[2], version), p[3]) for p in (parts[k] for k in ("train", "val", "test")))
    torch.manual_seed(seed + 10000); model = build_model(model_type, tr[0].shape[1], 2, HIDDEN, tree, "full")
    info = train(model, tensors(tr, "cpu"), tensors(va, "cpu"), FINAL, final=True)
    if not info["valid"]: return {"valid": False, "failure": info["failure"]}
    try: return {"valid": True, **scores(te[3], predict(model, tensors(te, "cpu"))), "best_val_accuracy": info["best_val_accuracy"]}
    except FloatingPointError as exc: return {"valid": False, "failure": str(exc)}


def run_cell(ds, level, seed, out):
    folder = Path(out) / ds / f"level_{level:g}"; path = folder / f"seed_{seed}.json"
    if path.exists(): return
    parts, info = build(ds, level, seed); res = {"dataset": ds, "level": level, "seed": seed, "mechanism": info, "methods": {}, "search": {}}
    for mode, version in (("on", "both"), ("off", "neither")):
        tr, va = ((p[0], *neutralise(p[1], p[2], version), p[3]) for p in (parts[k] for k in ("train", "val")))
        work = folder / f"_search_{seed}_{mode}"
        st = search(lambda tree: build_model("three_channel", tr[0].shape[1], 2, HIDDEN, tree, "full"), tensors(tr, "cpu"), tensors(va, "cpu"),
                    SEARCH, seed + 700, work, FINGERPRINT)
        top = sorted((r for r in st["records"] if r["valid"]), key=lambda r: -r["fitness"])[:10]
        res["search"][mode] = {"seconds": st["elapsed_s"], "evaluations": len(st["records"]), "stop_reason": st["stop_reason"],
                               "top10": [{k: r[k] for k in ("infix", "fitness", "best_val_accuracy", "generation", "nodes", "depth")} for r in top]}
        if st["best"] is None:
            res["methods"][mode] = {"valid": False, "failure": "no numerically valid candidate"}; continue
        tree = st["best"]["tree"]; res["methods"][mode] = {**fit_eval("three_channel", tree, parts, version, seed), "infix": g.tree_to_infix(tree), "tree": tree}
        if mode == "on":
            for ab in ABLATIONS: res["methods"][ab] = fit_eval("three_channel", tree, parts, ab, seed)
    for ab in ABLATIONS: res["methods"].setdefault(ab, {"valid": False, "failure": "no winner to ablate"})
    for f in FIXED: res["methods"][f] = fit_eval("three_channel", fixed_tree(f), parts, "both", seed)
    res["methods"]["mask_concat_relu"] = fit_eval("concat_mlp", fixed_tree("mask_concat_relu"), parts, "both", seed)
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(res, indent=1)); tmp.replace(path)
    for mode in ("on", "off"): shutil.rmtree(folder / f"_search_{seed}_{mode}", ignore_errors=True)
    a = {k: v.get("accuracy", float("nan")) for k, v in res["methods"].items()}
    print(f"{ds} s={level:g} seed {seed}: on {a['on']:.3f} off {a['off']:.3f} concat {a['mask_concat_relu']:.3f} relu {a['relu']:.3f} | mask-only {info['mask_only_accuracy']:.3f} | {res['methods']['on'].get('infix', '')[:60]}", flush=True)


def design(args):
    """Realised class rates and mask-only accuracy of every cell; trains no network."""
    rows = {}
    for ds in args.datasets:
        for level in args.levels:
            infos = [build(ds, level, s)[1] for s in args.seeds]
            rows[f"{ds}|{level:g}"] = {"p_class0": infos[0]["p_class0"], "p_class1": infos[0]["p_class1"],
                                       "realised_rate_test": float(np.mean([i["realised_rate"]["test"] for i in infos])),
                                       "mask_only_accuracy": float(np.mean([i["mask_only_accuracy"] for i in infos])),
                                       "majority_rate": float(np.mean([i["majority_rate_test"] for i in infos]))}
            r = rows[f"{ds}|{level:g}"]; print(f"{ds:9s} s={level:<4g} p0 {r['p_class0']:.3f} p1 {r['p_class1']:.3f} rate {r['realised_rate_test']:.3f} "
                                               f"mask-only {r['mask_only_accuracy']:.3f} majority {r['majority_rate']:.3f}", flush=True)
    os.makedirs(args.out, exist_ok=True); json.dump(rows, open(os.path.join(args.out, "design.json"), "w"), indent=1)


def run(args):
    torch.set_num_threads(args.threads)
    for ds in args.datasets:
        for level in args.levels:
            for seed in args.seeds: run_cell(ds, level, seed, args.out)


def _diff(a, b):
    ok = np.isfinite(a) & np.isfinite(b); d = (a[ok] - b[ok]) * 100; n = len(d)
    if n < 3: return {"n": n, "mean": None}
    h = stats.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    try: p = float(wilcoxon(d).pvalue)
    except ValueError: p = None
    return {"n": n, "mean": float(d.mean()), "ci_low": float(d.mean() - h), "ci_high": float(d.mean() + h), "p": p, "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}


def report(args):
    out, L = {}, ["# Informative missingness: channels on vs. off", "",
                  "Test accuracy, mean over splits. Differences in points with 95 % t-intervals; p = two-sided Wilcoxon signed-rank, paired by split.", ""]
    for ds in args.datasets:
        acc, mech = {}, {}
        for level in args.levels:
            cells = [json.load(open(f)) for f in sorted((Path(args.out) / ds / f"level_{level:g}").glob("seed_*.json"))]
            if not cells: continue
            acc[level] = {m: np.array([c["methods"][m].get("accuracy", np.nan) for c in cells]) for m in METHODS}; acc[level]["seeds"] = [c["seed"] for c in cells]
            mech[level] = {"mask_only": float(np.mean([c["mechanism"]["mask_only_accuracy"] for c in cells])), "majority": float(np.mean([c["mechanism"]["majority_rate_test"] for c in cells])),
                           "search_s_on": float(np.mean([c["search"]["on"]["seconds"] for c in cells])), "n": len(cells)}
        if not acc: continue
        out[ds] = {"levels": {}}
        L += [f"## {ds}", "", "| s | n | mask-only | on | off | m only | c only | neither | best fixed | mask-concat | on − off | p | on − mask-concat | p | on − best fixed | p | off − best fixed | p |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for level, a in acc.items():
            best = max(FIXED, key=lambda f: np.nanmean(a[f])); d = {"on_off": _diff(a["on"], a["off"]), "on_concat": _diff(a["on"], a["mask_concat_relu"]),
                                                                  "on_best_fixed": _diff(a["on"], a[best]), "off_best_fixed": _diff(a["off"], a[best])}
            out[ds]["levels"][f"{level:g}"] = {**mech[level], "means": {m: float(np.nanmean(a[m])) for m in METHODS}, "sds": {m: float(np.nanstd(a[m], ddof=1)) for m in METHODS},
                                               "invalid": {m: int((~np.isfinite(a[m])).sum()) for m in METHODS}, "best_fixed": best, "differences": d,
                                               "runs": {m: a[m].tolist() for m in METHODS}, "seeds": a["seeds"]}
            f = lambda q: "n/a" if q.get("mean") is None else f"{q['mean']:+.1f} [{q['ci_low']:+.1f}, {q['ci_high']:+.1f}]"; pp = lambda q: "n/a" if q.get("p") is None else f"{q['p']:.4f}"
            mm = out[ds]["levels"][f"{level:g}"]["means"]
            L.append(f"| {level:g} | {mech[level]['n']} | {mech[level]['mask_only']:.3f} | {mm['on']:.3f} | {mm['off']:.3f} | {mm['m_only']:.3f} | {mm['c_only']:.3f} | {mm['neither']:.3f} | "
                     f"{mm[best]:.3f} ({best}) | {mm['mask_concat_relu']:.3f} | {f(d['on_off'])} | {pp(d['on_off'])} | {f(d['on_concat'])} | {pp(d['on_concat'])} | "
                     f"{f(d['on_best_fixed'])} | {pp(d['on_best_fixed'])} | {f(d['off_best_fixed'])} | {pp(d['off_best_fixed'])} |")
        lo, hi = min(args.levels), max(args.levels)
        if lo in acc and hi in acc and acc[lo]["seeds"] == acc[hi]["seeds"]:
            out[ds]["primary"] = _diff(acc[hi]["on"] - acc[hi]["off"], acc[lo]["on"] - acc[lo]["off"])
        L.append("")
    prim = [(ds, v["primary"]) for ds, v in out.items() if v.get("primary", {}).get("p") is not None]; running = 0.
    for rank, (ds, q) in enumerate(sorted(prim, key=lambda t: t[1]["p"])):
        running = max(running, min(1., q["p"] * (len(prim) - rank))); q["p_holm"] = running
    L += ["## Primary test: change of (on − off) from the lowest to the highest strength", "", "| dataset | change (points) | 95 % interval | p | Holm p |", "|---|---|---|---|---|"]
    L += [f"| {ds} | {q['mean']:+.1f} | [{q['ci_low']:+.1f}, {q['ci_high']:+.1f}] | {q['p']:.4f} | {q['p_holm']:.4f} |" for ds, q in prim]
    json.dump(out, open(os.path.join(args.out, "report.json"), "w"), indent=1); open(os.path.join(args.out, "report.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["design", "run", "report"]); ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", default=DATASETS); ap.add_argument("--levels", nargs="+", type=float, default=LEVELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS); ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv); {"design": design, "run": run, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
