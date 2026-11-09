"""
Backbone-agnostic GP fitness evaluation and the evolutionary search loop.

The search reuses the ORIGINAL genetic machinery unchanged
(``gp_core.create_initial_population`` / ``evolve_population`` / softmax
fitness-proportional selection / subtree crossover / mutation) and the
ORIGINAL scalar fitness:

    F(T) = A_val(T) + 0.01 * D(T) / 3 - 0.0001 * N(T) - 0.0002 * (H(T) - 1)

NOTE (README "Discrepancies"): the paper prints the diversity term as
0.01 * D with D in {0,1,2,3}; the released code divides D by 3
(``diversity = count_input_diversity(tree) / 3.0``).  The code's version is
kept here so that the extension inherits the published baseline exactly.

Original treatment preserved:
  * trees of depth 1 (a bare terminal) get fitness 0.0 without training;
  * a non-finite loss or any exception during training gives fitness 0.0.
Additionally (new, and required by the brief) such candidates are flagged
``valid = False`` and can never be selected as the winning expression.

The evaluator receives training and validation data only; there is no code
path through which a test partition can reach it.
"""
from __future__ import annotations

import copy
import json
import os
import pickle
import random
import time
from contextlib import contextmanager

import numpy as np
import torch

from . import gp_core as g
from .training import TensorData, train_model

LAMBDA_D, LAMBDA_S, LAMBDA_H = 0.01, 0.0001, 0.0002
DIVERSITY_NORMALIZER = 3.0  # original code: count_input_diversity(tree) / 3.0
ALL_TERMINALS = ("value", "missing", "confidence")


def fitness_components(tree: dict, val_acc: float) -> dict:
    n_nodes = g.count_nodes(tree)
    depth = g.get_tree_depth(tree)
    diversity = g.count_input_diversity(tree)          # counts the terminals actually referenced
    fitness = (val_acc + LAMBDA_D * diversity / DIVERSITY_NORMALIZER
               - LAMBDA_S * n_nodes - LAMBDA_H * (depth - 1))
    return {"val_acc": float(val_acc), "nodes": int(n_nodes), "depth": int(depth),
            "diversity": int(diversity), "fitness": float(fitness)}


@contextmanager
def restricted_terminals(allowed):
    """Temporarily restrict the GP terminal set (for the f(z,m) / f(z,c) / f(z) ablations).

    gp_core's generators and mutation read the module-level INPUTS at call time, so
    rebinding it is sufficient; nothing else in gp_core is touched.
    """
    allowed = list(allowed)
    for t in allowed:
        if t not in ALL_TERMINALS:
            raise ValueError(f"unknown terminal '{t}'")
    if "value" not in allowed:
        raise ValueError("the 'value' terminal (the preactivation) can not be removed")
    old = g.INPUTS
    g.INPUTS = allowed
    try:
        yield
    finally:
        g.INPUTS = old


class FitnessEvaluator:
    """Trains a fresh network for each candidate tree on (train, val) and returns the original scalar fitness."""

    def __init__(self, model_factory, train: TensorData, val: TensorData, *, epochs: int, patience: int,
                 batch_size: int, init_seed: int, batch_seed: int, lr: float = 1e-3, weight_decay: float = 1e-4,
                 grad_clip: float = 1.0, log=None):
        self.model_factory = model_factory  # callable(tree) -> nn.Module on the right device
        self.train = train
        self.val = val
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.init_seed = int(init_seed)
        self.batch_seed = int(batch_seed)
        self.lr, self.weight_decay, self.grad_clip = lr, weight_decay, grad_clip
        self.log = log
        self.n_evaluated = 0

    def evaluate(self, tree: dict, candidate_id: int) -> dict:
        """Returns a record: fitness, components, validation score, runtime, failure, valid."""
        t0 = time.time()
        rec = {"candidate_id": int(candidate_id), "infix": g.tree_to_infix(tree), "tree": copy.deepcopy(tree),
               "nodes": g.count_nodes(tree), "depth": g.get_tree_depth(tree),
               "diversity": g.count_input_diversity(tree), "val_acc": None, "fitness": 0.0,
               "epochs_run": 0, "failure": None, "valid": False, "runtime_s": 0.0,
               "init_seed": self.init_seed + candidate_id, "batch_seed": self.batch_seed + candidate_id}
        if rec["depth"] == 1:
            rec["failure"] = "terminal_only_tree"   # original: fitness 0.0 without training
            rec["runtime_s"] = time.time() - t0
            return rec
        try:
            torch.manual_seed(rec["init_seed"])
            model = self.model_factory(tree)
            out = train_model(model, self.train, self.val, epochs=self.epochs, patience=self.patience,
                              batch_size=self.batch_size, batch_seed=rec["batch_seed"], lr=self.lr,
                              weight_decay=self.weight_decay, grad_clip=self.grad_clip,
                              plateau_scheduler=False, keep_best_state=False)
            rec["epochs_run"] = out["epochs_run"]
            if out["failure"]:
                rec["failure"] = out["failure"]       # original: fitness 0.0
            else:
                comp = fitness_components(tree, out["best_val_acc"])
                rec.update(comp)
                rec["valid"] = bool(np.isfinite(comp["fitness"]))
                if not rec["valid"]:
                    rec["fitness"], rec["failure"] = 0.0, "non-finite fitness"
            del model
        except Exception as e:  # original: print + fitness 0.0
            rec["failure"] = f"{type(e).__name__}: {e}"
            rec["fitness"] = 0.0
        rec["runtime_s"] = time.time() - t0
        self.n_evaluated += 1
        return rec


def _strip(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k != "tree"}


def run_search(evaluator: FitnessEvaluator, *, population_size: int, generations: int, max_tree_depth: int,
               mutation_rate: float, crossover_rate: float, elite_size: int, evolution_seed: int,
               out_dir: str, resume: bool = True, log=print) -> dict:
    """Generational GP exactly as in the original run_evolution_with_nn, with per-candidate checkpointing.

    Returns {"best": record (with tree), "history": per-generation summaries, "n_evaluations", "runtime_s"}.
    """
    os.makedirs(out_dir, exist_ok=True)
    state_path = os.path.join(out_dir, "search_state.pkl")
    history_path = os.path.join(out_dir, "search_history.json")

    if resume and os.path.exists(state_path):
        with open(state_path, "rb") as f:
            state = pickle.load(f)
        random.setstate(state["random_state"])
        log(f"  resuming search at generation {state['generation'] + 1}, candidate {state['candidate_counter']}")
    else:
        random.seed(int(evolution_seed))
        state = {"generation": 0, "population": g.create_initial_population(population_size, max_tree_depth),
                 "best": None, "history": [], "candidate_counter": 0, "runtime_s": 0.0,
                 "evaluations": [], "random_state": random.getstate(), "config": {
                     "population_size": population_size, "generations": generations, "max_tree_depth": max_tree_depth,
                     "mutation_rate": mutation_rate, "crossover_rate": crossover_rate, "elite_size": elite_size,
                     "evolution_seed": evolution_seed}}
        for ind in state["population"]:
            ind["record"] = None

    def save_state():
        state["random_state"] = random.getstate()
        tmp = state_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(state, f)
        os.replace(tmp, state_path)

    t_start = time.time() - state["runtime_s"]
    while state["generation"] < generations:
        gen = state["generation"]
        population = state["population"]
        t_gen = time.time()
        for ind in population:
            if ind["fitness"] is None:
                rec = evaluator.evaluate(ind["tree"], state["candidate_counter"])
                rec["generation"] = gen + 1
                ind["fitness"] = rec["fitness"]
                ind["record"] = rec
                state["evaluations"].append(_strip(rec))
                state["candidate_counter"] += 1
                state["runtime_s"] = time.time() - t_start
                save_state()
        population.sort(key=lambda ind: ind["fitness"], reverse=True)
        valid = [ind for ind in population if ind["record"] and ind["record"]["valid"]]
        gen_best = valid[0] if valid else None
        if gen_best is not None and (state["best"] is None or gen_best["fitness"] > state["best"]["fitness"]):
            state["best"] = copy.deepcopy(gen_best["record"])
            log(f"  gen {gen + 1:3d}: NEW BEST fitness {state['best']['fitness']:.4f}  val_acc {state['best']['val_acc']:.4f}  {state['best']['infix']}")
        fits = [ind["fitness"] for ind in population]
        state["history"].append({
            "generation": gen + 1,
            "best_fitness": float(gen_best["fitness"]) if gen_best else None,
            "best_val_acc": float(gen_best["record"]["val_acc"]) if gen_best else None,
            "best_infix": gen_best["record"]["infix"] if gen_best else None,
            "best_nodes": gen_best["record"]["nodes"] if gen_best else None,
            "best_depth": gen_best["record"]["depth"] if gen_best else None,
            "best_uses_inputs": gen_best["record"]["diversity"] if gen_best else None,
            "mean_fitness": float(np.mean(fits)), "n_invalid": int(len(population) - len(valid)),
            "runtime_s": time.time() - t_gen,
        })
        log(f"  gen {gen + 1:3d}/{generations}: best {state['history'][-1]['best_fitness']}  mean {np.mean(fits):.4f}  invalid {len(population) - len(valid)}  ({time.time() - t_gen:.1f}s)")
        if gen < generations - 1:
            # original: elites are re-inserted with fitness None and therefore re-trained next generation
            nxt = g.evolve_population(population, mutation_rate=mutation_rate, crossover_rate=crossover_rate,
                                      elite_size=elite_size)
            for ind in nxt:
                ind["record"] = None
            state["population"] = nxt
        state["generation"] = gen + 1
        state["runtime_s"] = time.time() - t_start
        save_state()
        with open(history_path, "w") as f:
            json.dump({"config": state["config"], "per_generation": state["history"],
                       "best": _strip(state["best"]) if state["best"] else None,
                       "n_evaluations": state["candidate_counter"], "runtime_s": state["runtime_s"],
                       "evaluations": state["evaluations"]}, f, indent=1)

    if state["best"] is None:
        raise RuntimeError("search produced no valid candidate (all evaluations failed)")
    return {"best": state["best"], "history": state["history"], "n_evaluations": state["candidate_counter"],
            "runtime_s": state["runtime_s"], "evaluations": state["evaluations"],
            "population": state["population"]}


# --------------------------------------------------------------------------
# two-stage selection: re-rank the search's top candidates under a longer protocol
# --------------------------------------------------------------------------

def rerank_candidates(evaluator: FitnessEvaluator, population: list, best: dict, *, top_k: int | None,
                      out_path: str | None = None, resume: bool = True, log=print) -> dict:
    """DEVIATION from the released selection rule (README section 6, D12).

    The original rule returns the best candidate of the search.  Here the ``top_k`` DISTINCT valid
    formulas of the final population (plus the search's all-time best; ``top_k=None`` = ALL of them, the
    reference setting -- no pool-size parameter) are re-scored with ``evaluator``
    -- intended to be the long fitness protocol -- and the formula with the highest re-scored fitness is
    selected.  Every candidate is re-scored with the same candidate_id (0), i.e. identical initialisation
    and minibatch seeds: a paired comparison.  Only validation data is involved; the test partition is
    never touched.  Returns {"winner": record (with tree), "table": [...], "runtime_s", "top_k"}.
    """
    pool, seen = [], set()
    cands = [ind["record"] | {"tree": ind["tree"]} for ind in population if ind.get("record") and ind["record"]["valid"]]
    if best is not None and best.get("valid", True):
        cands.append(best)
    for rec in sorted(cands, key=lambda r: -r["fitness"]):
        if rec["infix"] in seen:
            continue
        seen.add(rec["infix"])
        pool.append(rec)
        if top_k is not None and len(pool) == int(top_k):
            break
    if not pool:
        raise RuntimeError("re-ranking pool is empty (no valid candidates)")

    done = {}
    if resume and out_path and os.path.exists(out_path):
        with open(out_path) as f:
            saved = json.load(f)
        if saved.get("top_k", top_k) == top_k:
            done = {row["infix"]: row for row in saved.get("table", []) if row.get("rerank_fitness") is not None}
        if done:
            log(f"  re-ranking: {len(done)} candidates already scored, resuming")
    t0 = time.time()
    table = []
    for rank, rec in enumerate(pool, start=1):
        if rec["infix"] in done:
            row = done[rec["infix"]]
        else:
            r = evaluator.evaluate(rec["tree"], 0)
            row = {"search_rank": rank, "infix": rec["infix"], "nodes": rec["nodes"], "depth": rec["depth"],
                   "search_fitness": rec["fitness"], "search_val_acc": rec["val_acc"],
                   "rerank_fitness": r["fitness"], "rerank_val_acc": r["val_acc"], "rerank_valid": r["valid"],
                   "rerank_failure": r["failure"], "rerank_epochs": r["epochs_run"], "rerank_runtime_s": r["runtime_s"]}
        table.append(row)
        log(f"  re-rank {rank:2d}/{len(pool)}: search {row['search_fitness']:.4f} -> long {row['rerank_fitness']:.4f} "
            f"({row['rerank_epochs']} ep)  {row['infix'][:70]}")
        if out_path:
            with open(out_path, "w") as f:
                json.dump({"top_k": top_k, "table": table, "complete": False}, f, indent=1)
    valid_rows = [row for row in table if row["rerank_valid"]]
    if not valid_rows:
        raise RuntimeError("every re-ranked candidate failed under the long protocol")
    win_row = max(valid_rows, key=lambda row: row["rerank_fitness"])
    winner = next(rec for rec in pool if rec["infix"] == win_row["infix"])
    out = {"top_k": top_k, "pool_size": len(pool), "table": table, "runtime_s": sum(row["rerank_runtime_s"] for row in table),
           "wall_runtime_s": time.time() - t0, "winner_infix": winner["infix"],
           "winner_search_rank": win_row["search_rank"], "complete": True}
    if out_path:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1)
    return out | {"winner": winner}
