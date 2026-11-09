"""
Fitness-protocol fidelity check.

Scores the SAME random activation trees under two fitness protocols and
reports how well the cheap protocol ranks them like the full one:

  cheap : the ``search`` section of the base config (e.g. 5 epochs, patience 2,
          10k-sample training subset, batch 256)
  full  : 30 epochs, patience 2, the full training partition, batch 128
          (the original evaluator's epoch / patience values on all data)

Both protocols use the same data, masks, preprocessing, per-tree
initialisation seed and minibatch seed, so the only difference is the
training schedule.  Trees are drawn like the GP's initial population
(``generate_random_tree`` at the configured max depth; depth-1 trees skipped
because both protocols give them fitness 0 without training).

Outputs <out_root>/<name>/fidelity.json and fidelity.md with Spearman and
Kendall correlations of fitness and validation accuracy, computed over all
trees and over the trees that reference the preactivation ``x`` (trees
without ``x`` sit at chance under any protocol and would inflate agreement).

    python -m cnn_extension.fidelity --base cnn_extension/configs/reference_fashion_mnist.json --n-trees 30
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
from scipy.stats import kendalltau, spearmanr

from . import gp_core as g
from .data import ensure_dir, fingerprint
from .evolution import FitnessEvaluator
from .experiment import Logger, derive_seeds, load_config, model_factory_for, prepare_data
from .training import resolve_device

FULL_PROTOCOL = {"fitness_epochs": 30, "fitness_patience": 2, "fitness_batch_size": 128, "fitness_train_subset": None}


def sample_trees(n: int, max_depth: int, seed: int):
    random.seed(seed)
    trees, n_skipped = [], 0
    while len(trees) < n:
        t = g.generate_random_tree(max_depth)
        if g.get_tree_depth(t) == 1:
            n_skipped += 1
            continue
        trees.append(t)
    return trees, n_skipped


def _uses_x(tree):
    return any(n["type"] == "input" and n["input_name"] == "value" for n in g.get_all_nodes(tree))


def _evaluator(cfg, protocol, data, seeds, meta, device, log):
    train = data["train"]
    sub = protocol.get("fitness_train_subset")
    if sub and sub < len(train):
        rng = np.random.default_rng(seeds["split"] + 7)       # same subset rule as experiment.select_activation
        train = train.subset(np.sort(rng.choice(len(train), size=int(sub), replace=False)))
    opt = cfg["optimizer"]
    return FitnessEvaluator(model_factory_for(cfg, meta, device), train, data["val"],
                            epochs=protocol["fitness_epochs"], patience=protocol["fitness_patience"],
                            batch_size=protocol["fitness_batch_size"], init_seed=seeds["init"],
                            batch_seed=seeds["batch"], lr=opt["lr"], weight_decay=opt["weight_decay"],
                            grad_clip=opt["grad_clip"], log=log)


def _agreement(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return {"n": int(len(a)), "spearman": None, "kendall": None}
    return {"n": int(len(a)), "spearman": float(spearmanr(a, b).correlation), "kendall": float(kendalltau(a, b).correlation)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--n-trees", type=int, default=30)
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--out-root", default="runs_fidelity")
    ap.add_argument("--tree-seed", type=int, default=777)
    args = ap.parse_args(argv)

    cfg = load_config(args.base)
    cheap = {k: cfg["search"][k] for k in FULL_PROTOCOL}
    name = f"fidelity_{cfg['dataset']}_{cfg['corruption']['pattern']}{int(round(100 * cfg['corruption']['rate']))}"
    out_dir = ensure_dir(os.path.join(args.out_root, name))
    log = Logger(os.path.join(out_dir, "log.txt"))
    device = resolve_device(cfg["device"])
    seeds = derive_seeds(cfg["seeds"], args.replicate)
    log(f"=== fidelity check {name}  cheap={cheap}  full={FULL_PROTOCOL}  device={device}")
    data, data_info, meta = prepare_data(cfg, seeds, out_dir, device, log)

    trees, n_skipped = sample_trees(args.n_trees, cfg["search"]["max_tree_depth"], args.tree_seed)
    log(f"  {len(trees)} trees sampled (depth-1 skipped: {n_skipped}); {sum(_uses_x(t) for t in trees)} reference x")
    evaluators = {"cheap": _evaluator(cfg, cheap, data, seeds, meta, device, log),
                  "full": _evaluator(cfg, FULL_PROTOCOL, data, seeds, meta, device, log)}
    rows = []
    for i, tree in enumerate(trees):
        row = {"index": i, "infix": g.tree_to_infix(tree), "uses_x": _uses_x(tree), "nodes": g.count_nodes(tree),
               "depth": g.get_tree_depth(tree)}
        for proto, ev in evaluators.items():
            rec = ev.evaluate(tree, i)
            row[proto] = {k: rec[k] for k in ("fitness", "val_acc", "epochs_run", "runtime_s", "valid", "failure")}
        rows.append(row)
        log(f"  tree {i:2d}: cheap fit {row['cheap']['fitness']:.4f} ({row['cheap']['runtime_s']:.0f}s)  "
            f"full fit {row['full']['fitness']:.4f} ({row['full']['runtime_s']:.0f}s, {row['full']['epochs_run']} ep)  {row['infix'][:60]}")
        with open(os.path.join(out_dir, "fidelity_partial.json"), "w") as f:
            json.dump(rows, f, indent=1)

    def stats(subset):
        fit = _agreement([r["cheap"]["fitness"] for r in subset], [r["full"]["fitness"] for r in subset])
        acc = _agreement([r["cheap"]["val_acc"] or 0 for r in subset], [r["full"]["val_acc"] or 0 for r in subset])
        k = max(1, len(subset) // 5)
        top_cheap = {r["index"] for r in sorted(subset, key=lambda r: -r["cheap"]["fitness"])[:k]}
        top_full = {r["index"] for r in sorted(subset, key=lambda r: -r["full"]["fitness"])[:k]}
        best_cheap = max(subset, key=lambda r: r["cheap"]["fitness"])["index"]
        full_rank_of_best_cheap = sorted(subset, key=lambda r: -r["full"]["fitness"]).index(
            next(r for r in subset if r["index"] == best_cheap)) + 1
        return {"fitness": fit, "val_acc": acc, "top_k": k, "top_k_overlap": len(top_cheap & top_full),
                "full_rank_of_cheap_winner": full_rank_of_best_cheap}

    valid = [r for r in rows if r["cheap"]["valid"] and r["full"]["valid"]]
    with_x = [r for r in valid if r["uses_x"]]
    summary = {
        "name": name, "config_base": args.base, "replicate": args.replicate, "seeds": seeds, "tree_seed": args.tree_seed,
        "cheap_protocol": cheap, "full_protocol": FULL_PROTOCOL, "device": str(device), "data": data_info,
        "n_trees": len(rows), "n_valid_both": len(valid), "n_with_x": len(with_x),
        "agreement_all_valid": stats(valid), "agreement_trees_with_x": stats(with_x) if len(with_x) >= 3 else None,
        "runtime_s": {"cheap_total": sum(r["cheap"]["runtime_s"] for r in rows),
                      "full_total": sum(r["full"]["runtime_s"] for r in rows),
                      "cheap_mean": float(np.mean([r["cheap"]["runtime_s"] for r in rows])),
                      "full_mean": float(np.mean([r["full"]["runtime_s"] for r in rows]))},
        "trees": rows, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "fidelity.json"), "w") as f:
        json.dump(summary, f, indent=1)
    os.remove(os.path.join(out_dir, "fidelity_partial.json"))

    a, b = summary["agreement_all_valid"], summary["agreement_trees_with_x"]
    lines = [f"# Fitness-protocol fidelity: {name}", "",
             f"cheap = {cheap}", f"full  = {FULL_PROTOCOL}", "",
             f"{len(rows)} random trees (depth <= {cfg['search']['max_tree_depth']}); {len(valid)} valid under both; {len(with_x)} reference x.", "",
             "| set | n | Spearman(fitness) | Kendall(fitness) | Spearman(val acc) | top-k overlap | full-rank of cheap winner |",
             "|---|---|---|---|---|---|---|",
             f"| all valid | {a['fitness']['n']} | {a['fitness']['spearman']:.3f} | {a['fitness']['kendall']:.3f} | {a['val_acc']['spearman']:.3f} | {a['top_k_overlap']}/{a['top_k']} | {a['full_rank_of_cheap_winner']} |"]
    if b:
        lines.append(f"| trees with x | {b['fitness']['n']} | {b['fitness']['spearman']:.3f} | {b['fitness']['kendall']:.3f} | {b['val_acc']['spearman']:.3f} | {b['top_k_overlap']}/{b['top_k']} | {b['full_rank_of_cheap_winner']} |")
    lines += ["", f"Mean runtime per tree: cheap {summary['runtime_s']['cheap_mean']:.1f} s, full {summary['runtime_s']['full_mean']:.1f} s "
              f"(speed-up x{summary['runtime_s']['full_mean'] / max(summary['runtime_s']['cheap_mean'], 1e-9):.1f}).", "",
              "| # | x | cheap fitness | full fitness | cheap val | full val | full epochs | expression |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["full"]["fitness"]):
        lines.append(f"| {r['index']} | {'y' if r['uses_x'] else '-'} | {r['cheap']['fitness']:.4f} | {r['full']['fitness']:.4f} | "
                     f"{(r['cheap']['val_acc'] or 0):.4f} | {(r['full']['val_acc'] or 0):.4f} | {r['full']['epochs_run']} | `{r['infix'][:70]}` |")
    with open(os.path.join(out_dir, "fidelity.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
