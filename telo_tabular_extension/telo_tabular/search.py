"""Bounded, resumable single-objective GP using the original genetic operators."""
import copy
import json
import random
import time
from pathlib import Path
import torch
from . import gp_core as g
from .models import train
from .storage import write_json


def _tuples(x):
    return tuple(_tuples(v) for v in x) if isinstance(x, list) else x


def search(factory, train_data, val_data, cfg, seed, output, fingerprint):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "search_state.json"
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["fingerprint"] != fingerprint:
            raise ValueError("Search state belongs to another configuration")
        random.setstate(_tuples(state["random_state"]))
    else:
        random.seed(seed)
        state = {"fingerprint": fingerprint, "generation": 0,
                 "population": g.create_initial_population(cfg["population"], cfg["init_depth"]),
                 "best": None, "records": [], "elapsed_s": 0., "complete": False,
                 "random_state": random.getstate()}
    if state["complete"]:
        return state
    started = time.monotonic()
    previous_elapsed = state["elapsed_s"]
    remaining = cfg["max_seconds"] - state["elapsed_s"]
    deadline = started + max(0, remaining)
    exhausted = False
    while state["generation"] < cfg["generations"]:
        for individual in state["population"]:
            if individual["fitness"] is not None:
                continue
            if time.monotonic() >= deadline:
                exhausted = True
                break
            tree = individual["tree"]
            num = len(state["records"])
            torch.manual_seed(seed + 1000 + num)
            depth, nodes = g.get_tree_depth(tree), g.count_nodes(tree)
            if depth == 1 or depth > cfg["safety_max_depth"] or nodes > cfg["safety_max_nodes"]:
                record = {"valid": False, "failure": "Trivial tree or computational safety limit",
                          "best_val_accuracy": 0., "epochs_run": 0, "runtime_s": 0.}
            else:
                record = train(factory(tree), train_data, val_data, cfg["training"],
                               deadline=deadline)
            fit = (record["best_val_accuracy"] + .01 * g.count_input_diversity(tree) / 3.
                   - .0001 * nodes - .0002 * (depth - 1)) if record["valid"] else 0.
            record.update({"candidate": num, "generation": state["generation"],
                           "fitness": fit, "tree": tree, "infix": g.tree_to_infix(tree),
                           "nodes": nodes, "depth": depth})
            record.pop("history", None)
            individual["fitness"] = fit
            state["records"].append(record)
            if record["valid"] and (state["best"] is None or fit > state["best"]["fitness"]):
                state["best"] = copy.deepcopy(record)
            state["random_state"] = random.getstate()
            state["elapsed_s"] = previous_elapsed + time.monotonic() - started
            write_json(path, state)
        if exhausted:
            break
        state["generation"] += 1
        if state["generation"] < cfg["generations"]:
            state["population"] = g.evolve_population(
                state["population"], mutation_rate=cfg["mutation_rate"],
                crossover_rate=cfg["crossover_rate"], elite_size=cfg["elite_size"])
    state["elapsed_s"] = previous_elapsed + time.monotonic() - started
    state["complete"] = True
    state["stop_reason"] = "time_budget" if exhausted else "generation_budget"
    state["random_state"] = random.getstate()
    write_json(path, state)
    return state
