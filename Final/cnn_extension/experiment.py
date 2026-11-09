"""
Experiment runner: data -> (search | fixed tree) -> fresh retraining -> test.

Two experiment modes
--------------------
* matched conditions   (``run_experiment``): the model is evolved / trained at
  one (pattern, rate) and tested on independently corrupted test images with
  the same pattern and rate.
* changing conditions  (``evaluate_shift``): a finished run is loaded -- its
  preprocessor state, selected expression and checkpoint are frozen -- and
  evaluated on the official test partition corrupted with another pattern /
  rate (or clean).  Nothing is refitted, re-selected or retrained.

Seeds (all recorded in results.json): split, corruption (spawned per
partition), evolution, init (network initialisation), batch (minibatch
order).  A replicate index r is added to every seed, so methods sharing a
replicate see identical partitions, masks and initialisation protocol.
"""
from __future__ import annotations

import copy
import json
import os
import platform
import shutil
import time

import numpy as np
import torch

from . import gp_core as g
from .baselines import fixed_activation_tree
from .data import (DATASETS, ImputationPreprocessor, ensure_dir, fingerprint, generate_masks, load_dataset,
                   load_masks, save_masks, spawn_corruption_seeds)
from .evolution import ALL_TERMINALS, FitnessEvaluator, rerank_candidates, restricted_terminals, run_search
from .models import build_model, count_parameters, first_layer_parameter_count
from .training import TensorData, evaluate_model, resolve_device, train_model

DEFAULT_CONFIG = {
    "name": "unnamed",
    "label": "",
    "dataset": "fashion_mnist",
    "data_root": "data",
    "download": True,
    "val_fraction": 0.1,
    "synthetic": {},
    "corruption": {"pattern": "random", "rate": 0.2},
    "method": {"type": "evolved", "terminals": list(ALL_TERMINALS)},
    "backbone": "cnn",
    "metadata_mode": "full",
    "architecture": {"conv_channels": [16, 32], "kernel_size": 3, "pool_size": 2, "fc_hidden": 64,
                     "mlp_hidden_sizes": [64, 32]},
    "search": {"population_size": 100, "generations": 30, "max_tree_depth": 3, "mutation_rate": 0.15,
               "crossover_rate": 0.70, "elite_size": 2, "fitness_epochs": 30, "fitness_patience": 2,
               "fitness_batch_size": 128, "fitness_train_subset": None,
               # two-stage selection (deviation D12): None = original rule (best of the search);
               # e.g. {"top_k": None, "epochs": 30, "patience": 2, "batch_size": 128, "train_subset": None}
               # top_k None/0/"all" = every distinct valid finalist (reference setting); an int limits the pool
               "rerank": None, "reuse_search_dir": None},
    "final": {"epochs": 100, "patience": 15, "batch_size": 128},
    "optimizer": {"lr": 0.001, "weight_decay": 0.0001, "grad_clip": 1.0},
    "seeds": {"split": 1000, "corruption": 2000, "evolution": 3000, "init": 4000, "batch": 5000},
    "device": "auto",
    "out_root": "runs",
}


def _deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str, overrides: dict | None = None) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    cfg = _deep_update(DEFAULT_CONFIG, cfg)
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg


def derive_seeds(seed_cfg: dict, replicate: int) -> dict:
    seeds = {k: int(v) + int(replicate) for k, v in seed_cfg.items()}
    seeds["replicate"] = int(replicate)
    seeds["corruption_per_split"] = spawn_corruption_seeds(seeds["corruption"])
    return seeds


class Logger:
    def __init__(self, path: str | None):
        self.path = path

    def __call__(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.path:
            with open(self.path, "a") as f:
                f.write(line + "\n")


# --------------------------------------------------------------------------
# data preparation
# --------------------------------------------------------------------------

def prepare_data(cfg: dict, seeds: dict, out_dir: str, device, log=print, corruption: dict | None = None):
    corruption = corruption or cfg["corruption"]
    splits, meta = load_dataset(cfg["dataset"], cfg["data_root"], cfg["val_fraction"], seeds["split"],
                                cfg.get("download", True), cfg.get("synthetic"))
    C, H, W = meta["in_channels"], meta["height"], meta["width"]
    masks, mask_info = {}, {}
    for split in ("train", "val", "test"):
        n = len(splits[split])
        seed = seeds["corruption_per_split"][split]
        mk = generate_masks(n, C, H, W, corruption["pattern"], float(corruption["rate"]), seed)
        masks[split] = mk
        mask_info[split] = {"pattern": corruption["pattern"], "nominal_rate": float(corruption["rate"]),
                            "seed": int(seed), "realized_rate": float(mk.mean()), "n": int(n),
                            "fingerprint": fingerprint(np.packbits(mk.reshape(-1)))}
        save_masks(os.path.join(out_dir, f"masks_{split}.npz"), mk, mask_info[split])
    pre = ImputationPreprocessor().fit(splits["train"].x, masks["train"])
    pre.save(os.path.join(out_dir, "preprocessor.npz"))
    data = {}
    for split in ("train", "val", "test"):
        x, m, c = pre.transform(splits[split].x, masks[split])
        data[split] = TensorData.from_numpy(x, m, c, splits[split].y).to(device)
    log(f"  data: {cfg['dataset']} {meta['sizes']}  pattern={corruption['pattern']} nominal={corruption['rate']}  "
        f"realized train/val/test = {mask_info['train']['realized_rate']:.4f}/{mask_info['val']['realized_rate']:.4f}/{mask_info['test']['realized_rate']:.4f}")
    info = {"dataset": meta, "masks": mask_info, "preprocessor": {"fingerprint": pre.fingerprint(),
                                                                   "fallbacks": pre.fallback_counts, "tau": pre.tau}}
    return data, info, meta


def model_factory_for(cfg: dict, meta: dict, device):
    def factory(tree):
        return build_model(cfg["backbone"], meta["in_channels"], meta["height"], meta["width"], meta["num_classes"],
                           tree, cfg["architecture"], cfg["metadata_mode"]).to(device)
    return factory


# --------------------------------------------------------------------------
# activation selection
# --------------------------------------------------------------------------

def select_activation(cfg: dict, data: dict, seeds: dict, meta: dict, device, out_dir: str, resume: bool, log):
    method = cfg["method"]
    if method["type"] == "fixed":
        tree = fixed_activation_tree(method["activation"])
        return tree, {"type": "fixed", "activation": method["activation"], "infix": g.tree_to_infix(tree),
                      "search_runtime_s": 0.0, "n_evaluations": 0}
    if method["type"] != "evolved":
        raise ValueError("method.type must be 'fixed' or 'evolved'")
    terminals = list(method.get("terminals", ALL_TERMINALS))
    s = cfg["search"]
    train = data["train"]
    subset_info = None
    if s.get("fitness_train_subset"):
        n_sub = int(s["fitness_train_subset"])
        if n_sub < len(train):
            rng = np.random.default_rng(seeds["split"] + 7)
            idx = np.sort(rng.choice(len(train), size=n_sub, replace=False))
            train = train.subset(idx)
            subset_info = {"n": n_sub, "fingerprint": fingerprint(idx)}
            log(f"  fitness evaluation uses a {n_sub}-sample training subset (pilot-style setting)")
    opt = cfg["optimizer"]
    evaluator = FitnessEvaluator(model_factory_for(cfg, meta, device), train, data["val"],
                                 epochs=s["fitness_epochs"], patience=s["fitness_patience"],
                                 batch_size=s["fitness_batch_size"], init_seed=seeds["init"],
                                 batch_seed=seeds["batch"], lr=opt["lr"], weight_decay=opt["weight_decay"],
                                 grad_clip=opt["grad_clip"], log=log)
    search_dir = os.path.join(out_dir, "search")
    reuse = s.get("reuse_search_dir")
    if reuse and not os.path.exists(os.path.join(search_dir, "search_state.pkl")):
        # reuse a finished search of an identical configuration (same data, masks, seeds, budget):
        # the search artefacts are copied so this run stays self-contained; provenance is recorded below
        ensure_dir(search_dir)
        for fname in ("search_state.pkl", "search_history.json"):
            shutil.copy2(os.path.join(reuse, fname), os.path.join(search_dir, fname))
        log(f"  search artefacts copied from {reuse} (no new search)")
    log(f"  GP search: pop {s['population_size']} x gen {s['generations']}  depth {s['max_tree_depth']}  terminals {terminals}")
    with restricted_terminals(terminals):
        res = run_search(evaluator, population_size=s["population_size"], generations=s["generations"],
                         max_tree_depth=s["max_tree_depth"], mutation_rate=s["mutation_rate"],
                         crossover_rate=s["crossover_rate"], elite_size=s["elite_size"],
                         evolution_seed=seeds["evolution"], out_dir=search_dir,
                         resume=resume or bool(reuse), log=log)
    best = res["best"]
    g.validate_tree(best["tree"])
    info = {"type": "evolved", "terminals": terminals, "selection_rule": "search_best",
            "infix": best["infix"], "fitness": best["fitness"],
            "search_val_acc": best["val_acc"], "nodes": best["nodes"], "depth": best["depth"],
            "uses_inputs": best["diversity"], "found_in_generation": best["generation"],
            "n_evaluations": res["n_evaluations"], "search_runtime_s": res["runtime_s"], "rerank_runtime_s": 0.0,
            "search_copied_from": reuse, "fitness_train_subset": subset_info, "per_generation": res["history"]}
    rr = s.get("rerank")
    if not rr:
        return best["tree"], info
    # ---- two-stage selection (deviation D12): re-score the top-k distinct candidates under the long protocol
    long_train = data["train"]
    if rr.get("train_subset") and int(rr["train_subset"]) < len(long_train):
        rng = np.random.default_rng(seeds["split"] + 11)
        long_train = long_train.subset(np.sort(rng.choice(len(long_train), size=int(rr["train_subset"]), replace=False)))
    long_eval = FitnessEvaluator(model_factory_for(cfg, meta, device), long_train, data["val"],
                                 epochs=int(rr["epochs"]), patience=int(rr["patience"]), batch_size=int(rr["batch_size"]),
                                 init_seed=seeds["init"], batch_seed=seeds["batch"], lr=opt["lr"],
                                 weight_decay=opt["weight_decay"], grad_clip=opt["grad_clip"], log=log)
    log(f"  two-stage selection: re-ranking {'ALL' if rr.get('top_k') in (None, 0, 'all') else 'top-' + str(rr['top_k'])} distinct finalists under epochs<= {rr['epochs']} "
        f"patience {rr['patience']} batch {rr['batch_size']} train_subset {rr.get('train_subset')}")
    with restricted_terminals(terminals):
        out = rerank_candidates(long_eval, res["population"], best,
                                top_k=None if rr.get("top_k") in (None, 0, "all") else int(rr["top_k"]),
                                out_path=os.path.join(search_dir, "rerank.json"), resume=resume, log=log)
    w = out["winner"]
    g.validate_tree(w["tree"])
    info.update({"selection_rule": "two_stage_rerank", "infix": w["infix"], "fitness": w["fitness"],
                 "search_val_acc": w["val_acc"], "nodes": w["nodes"], "depth": w["depth"], "uses_inputs": w["diversity"],
                 "found_in_generation": w.get("generation"), "rerank_runtime_s": out["runtime_s"],
                 "search_best": {"infix": best["infix"], "fitness": best["fitness"], "search_val_acc": best["val_acc"]},
                 "rerank": {"protocol": dict(rr), "top_k": out["top_k"], "pool_size": out["pool_size"],
                            "winner_search_rank": out["winner_search_rank"],
                            "table": out["table"]}})
    log(f"  two-stage selection picked search-rank {out['winner_search_rank']}: {w['infix']}")
    return w["tree"], info


# --------------------------------------------------------------------------
# final training + test
# --------------------------------------------------------------------------

def final_train_and_test(cfg: dict, tree: dict, data: dict, seeds: dict, meta: dict, device, out_dir: str, log):
    f, opt = cfg["final"], cfg["optimizer"]
    torch.manual_seed(seeds["init"] + 10_000)         # fresh initialisation, distinct from search candidates
    model = model_factory_for(cfg, meta, device)(tree)
    log(f"  final training: {cfg['backbone']} params {count_parameters(model)}  epochs<= {f['epochs']} patience {f['patience']}")
    out = train_model(model, data["train"], data["val"], epochs=f["epochs"], patience=f["patience"],
                      batch_size=f["batch_size"], batch_seed=seeds["batch"] + 10_000, lr=opt["lr"],
                      weight_decay=opt["weight_decay"], grad_clip=opt["grad_clip"], plateau_scheduler=True,
                      keep_best_state=True, log=log)
    if out["failure"]:
        log(f"  WARNING: final training failed: {out['failure']}")
    t0 = time.time()
    test = evaluate_model(model, data["test"])
    val = evaluate_model(model, data["val"])
    test_runtime = time.time() - t0
    torch.save({"state_dict": model.state_dict(), "tree": tree, "backbone": cfg["backbone"],
                "architecture": cfg["architecture"], "metadata_mode": cfg["metadata_mode"], "dataset": meta},
               os.path.join(out_dir, "checkpoint.pt"))
    np.savez_compressed(os.path.join(out_dir, "test_predictions.npz"), probs=test["probs"], labels=test["labels"],
                        preds=test["preds"])
    return {
        "val_accuracy_selected": val["accuracy"], "best_val_accuracy": out["best_val_acc"],
        "best_epoch": out["best_epoch"], "epochs_run": out["epochs_run"], "training_failure": out["failure"],
        "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"],
        "final_train_runtime_s": out["runtime_s"], "test_eval_runtime_s": test_runtime,
        "parameters": count_parameters(model), "first_layer_parameters": first_layer_parameter_count(model),
        "history": out["history"],
    }


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def run_dir_for(cfg: dict, replicate: int) -> str:
    return os.path.join(cfg["out_root"], cfg["name"], f"rep{replicate}")


def run_experiment(cfg: dict, replicate: int = 0, resume: bool = True) -> dict:
    out_dir = ensure_dir(run_dir_for(cfg, replicate))
    results_path = os.path.join(out_dir, "results.json")
    if resume and os.path.exists(results_path):
        with open(results_path) as f:
            res = json.load(f)
        acc = res["final"]["test_accuracy"] if res.get("final") else float("nan")
        print(f"[skip] {out_dir} already has results.json (test_accuracy={acc:.4f})")
        return res
    log = Logger(os.path.join(out_dir, "log.txt"))
    device = resolve_device(cfg["device"])
    seeds = derive_seeds(cfg["seeds"], replicate)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"config": cfg, "replicate": replicate, "seeds": seeds}, f, indent=2)
    log(f"=== {cfg['name']} rep{replicate}  {cfg.get('label', '')}  device={device}")
    t0 = time.time()
    data, data_info, meta = prepare_data(cfg, seeds, out_dir, device, log)
    tree, selection = select_activation(cfg, data, seeds, meta, device, out_dir, resume, log)
    log(f"  selected activation: {g.tree_to_infix(tree)}")
    if cfg["final"].get("enabled", True) is False:
        # selection-only run (GP hyper-parameter tuning): no final training, the test partition is never touched
        results = {
            "name": cfg["name"], "label": cfg.get("label", ""), "replicate": replicate, "mode": "selection_only",
            "config": cfg, "seeds": seeds, "device": str(device), "platform": platform.platform(),
            "torch": torch.__version__, "data": data_info, "selection": selection,
            "selected_tree": tree, "selected_infix": g.tree_to_infix(tree), "final": None,
            "cost": {"search_runtime_s": selection["search_runtime_s"], "search_evaluations": selection["n_evaluations"],
                     "rerank_runtime_s": selection.get("rerank_runtime_s", 0.0), "final_train_runtime_s": 0.0,
                     "total_runtime_s": time.time() - t0},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        log(f"  DONE (selection only) {g.tree_to_infix(tree)}  search {selection['search_runtime_s']:.0f}s")
        return results
    final = final_train_and_test(cfg, tree, data, seeds, meta, device, out_dir, log)
    results = {
        "name": cfg["name"], "label": cfg.get("label", ""), "replicate": replicate, "mode": "matched_conditions",
        "config": cfg, "seeds": seeds, "device": str(device), "platform": platform.platform(),
        "torch": torch.__version__, "data": data_info, "selection": selection,
        "selected_tree": tree, "selected_infix": g.tree_to_infix(tree), "final": final,
        "cost": {"search_runtime_s": selection["search_runtime_s"], "search_evaluations": selection["n_evaluations"],
                 "rerank_runtime_s": selection.get("rerank_runtime_s", 0.0),
                 "final_train_runtime_s": final["final_train_runtime_s"], "total_runtime_s": time.time() - t0},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"  DONE test_accuracy {final['test_accuracy']:.4f}  macro_f1 {final['test_macro_f1']:.4f}  "
        f"search {selection['search_runtime_s']:.0f}s  final {final['final_train_runtime_s']:.0f}s")
    return results


def load_run(run_dir: str, device=None):
    """Rebuild the trained model of a finished run from its checkpoint (no training)."""
    device = device or resolve_device("auto")
    with open(os.path.join(run_dir, "config.json")) as f:
        saved = json.load(f)
    ckpt = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location="cpu", weights_only=False)
    meta = ckpt["dataset"]
    model = build_model(ckpt["backbone"], meta["in_channels"], meta["height"], meta["width"], meta["num_classes"],
                        ckpt["tree"], ckpt["architecture"], ckpt["metadata_mode"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    pre = ImputationPreprocessor.load(os.path.join(run_dir, "preprocessor.npz"))
    return model, pre, ckpt["tree"], saved, meta, device


def evaluate_shift(run_dir: str, pattern: str, rate: float, corruption_seed: int | None = None,
                   device=None) -> dict:
    """Changing test conditions: frozen model + frozen preprocessing, new test corruption."""
    model, pre, tree, saved, meta, device = load_run(run_dir, device)
    cfg, seeds = saved["config"], saved["seeds"]
    splits, _ = load_dataset(cfg["dataset"], cfg["data_root"], cfg["val_fraction"], seeds["split"],
                             cfg.get("download", True), cfg.get("synthetic"))
    test = splits["test"]
    seed = int(corruption_seed) if corruption_seed is not None else int(seeds["corruption_per_split"]["test"])
    mk = generate_masks(len(test), meta["in_channels"], meta["height"], meta["width"], pattern, float(rate), seed)
    x, m, c = pre.transform(test.x, mk)
    data = TensorData.from_numpy(x, m, c, test.y).to(device)
    ev = evaluate_model(model, data)
    tag = f"{pattern}_{int(round(float(rate) * 100))}pct"
    out = {"run_dir": run_dir, "mode": "changing_test_conditions", "trained_on": cfg["corruption"],
           "tested_on": {"pattern": pattern, "nominal_rate": float(rate), "realized_rate": float(mk.mean()),
                         "seed": seed, "mask_fingerprint": fingerprint(np.packbits(mk.reshape(-1)))},
           "selected_infix": g.tree_to_infix(tree), "test_accuracy": ev["accuracy"], "test_macro_f1": ev["macro_f1"],
           "n_test": int(len(test)), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(run_dir, f"shift_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(os.path.join(run_dir, f"shift_{tag}_predictions.npz"), probs=ev["probs"],
                        labels=ev["labels"], preds=ev["preds"])
    return out


def reproduce_predictions(run_dir: str, device=None) -> dict:
    """Reload masks, preprocessing, expression and checkpoint; recompute test predictions; compare with saved."""
    model, pre, tree, saved, meta, device = load_run(run_dir, device)
    cfg, seeds = saved["config"], saved["seeds"]
    splits, _ = load_dataset(cfg["dataset"], cfg["data_root"], cfg["val_fraction"], seeds["split"],
                             cfg.get("download", True), cfg.get("synthetic"))
    mk, info = load_masks(os.path.join(run_dir, "masks_test.npz"))
    x, m, c = pre.transform(splits["test"].x, mk)
    ev = evaluate_model(model, TensorData.from_numpy(x, m, c, splits["test"].y).to(device))
    saved_pred = np.load(os.path.join(run_dir, "test_predictions.npz"))
    return {"preds_equal": bool(np.array_equal(ev["preds"], saved_pred["preds"])),
            "max_prob_abs_diff": float(np.abs(ev["probs"] - saved_pred["probs"]).max()),
            "accuracy": ev["accuracy"]}


def aggregate(run_dirs: list[str]) -> dict:
    """Mean / std over independent runs, keeping every individual result."""
    rows = []
    for d in run_dirs:
        p = os.path.join(d, "results.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            r = json.load(f)
        rows.append({"run_dir": d, "name": r["name"], "replicate": r["replicate"], "backbone": r["config"]["backbone"],
                     "method": r["config"]["method"], "metadata_mode": r["config"]["metadata_mode"],
                     "corruption": r["config"]["corruption"], "selected_infix": r["selected_infix"],
                     "test_accuracy": r["final"]["test_accuracy"], "test_macro_f1": r["final"]["test_macro_f1"],
                     "parameters": r["final"]["parameters"], "search_runtime_s": r["cost"]["search_runtime_s"],
                     "final_train_runtime_s": r["cost"]["final_train_runtime_s"]})
    groups = {}
    for row in rows:
        groups.setdefault(row["name"], []).append(row)
    summary = {}
    for name, rs in groups.items():
        acc = np.array([r["test_accuracy"] for r in rs])
        f1 = np.array([r["test_macro_f1"] for r in rs])
        summary[name] = {"n_runs": len(rs), "test_accuracy_mean": float(acc.mean()),
                         "test_accuracy_std": float(acc.std(ddof=1)) if len(rs) > 1 else 0.0,
                         "test_macro_f1_mean": float(f1.mean()),
                         "test_macro_f1_std": float(f1.std(ddof=1)) if len(rs) > 1 else 0.0,
                         "search_runtime_s_mean": float(np.mean([r["search_runtime_s"] for r in rs])),
                         "final_train_runtime_s_mean": float(np.mean([r["final_train_runtime_s"] for r in rs])),
                         "parameters": rs[0]["parameters"], "runs": rs}
    return summary
