"""Two-stage selection (deviation D12): re-ranking the search's top candidates under a longer protocol."""
import json
import os

import numpy as np
import pytest

from cnn_extension import gp_core as g
from cnn_extension.evolution import FitnessEvaluator, rerank_candidates, run_search
from cnn_extension.experiment import DEFAULT_CONFIG, _deep_update, run_experiment
from cnn_extension.methods import compose
from cnn_extension.models import build_model
from cnn_extension.training import TensorData


def tiny_data(n=48, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n, 1, 6, 6)).astype(np.float32)
    y = (x[:, 0, :3, :].mean((1, 2)) > x[:, 0, 3:, :].mean((1, 2))).astype(np.int64)
    m = (rng.random(x.shape) < 0.2).astype(np.float32)
    c = np.where(m > 0, 0.8, 1.0).astype(np.float32)
    return TensorData.from_numpy(x, m, c, y)


def factory(tree):
    return build_model("mlp", 1, 6, 6, 2, tree, {"mlp_hidden_sizes": [8, 4]})


def test_rerank_pool_is_distinct_and_winner_maximises_long_fitness(tmp_path):
    train, val = tiny_data(48, 0), tiny_data(24, 1)
    cheap = FitnessEvaluator(factory, train, val, epochs=1, patience=1, batch_size=16, init_seed=1, batch_seed=2)
    res = run_search(cheap, population_size=6, generations=2, max_tree_depth=3, mutation_rate=0.3, crossover_rate=0.7,
                     elite_size=1, evolution_seed=5, out_dir=str(tmp_path / "search"), log=lambda *_: None)
    assert "population" in res and len(res["population"]) == 6
    long = FitnessEvaluator(factory, train, val, epochs=3, patience=3, batch_size=16, init_seed=1, batch_seed=2)
    out_path = str(tmp_path / "search" / "rerank.json")
    out = rerank_candidates(long, res["population"], res["best"], top_k=3, out_path=out_path, log=lambda *_: None)
    infixes = [row["infix"] for row in out["table"]]
    assert len(infixes) == len(set(infixes)) and 1 <= len(infixes) <= 3
    # the pool is the search's best-ranked distinct formulas
    assert out["table"][0]["search_fitness"] >= out["table"][-1]["search_fitness"]
    assert out["table"][0]["infix"] == res["best"]["infix"]
    valid = [row for row in out["table"] if row["rerank_valid"]]
    assert out["winner_infix"] == max(valid, key=lambda r: r["rerank_fitness"])["infix"]
    assert g.tree_to_infix(out["winner"]["tree"]) == out["winner_infix"]
    g.validate_tree(out["winner"]["tree"])
    saved = json.load(open(out_path))
    assert saved["complete"] and len(saved["table"]) == len(out["table"])
    # resume: a second call re-uses the saved scores instead of re-training
    calls = []
    long.evaluate = lambda tree, cid: calls.append(1) or (_ for _ in ()).throw(AssertionError("must not retrain"))
    again = rerank_candidates(long, res["population"], res["best"], top_k=3, out_path=out_path, log=lambda *_: None)
    assert again["winner_infix"] == out["winner_infix"] and not calls


def test_rerank_never_sees_test_data_and_is_paired():
    """All re-ranked candidates are scored with candidate_id 0 -> identical init / minibatch seeds."""
    train, val = tiny_data(32, 0), tiny_data(16, 1)
    ev = FitnessEvaluator(factory, train, val, epochs=1, patience=1, batch_size=16, init_seed=7, batch_seed=9)
    seen = []
    orig = ev.evaluate
    ev.evaluate = lambda tree, cid: seen.append(cid) or orig(tree, cid)
    pop = []
    for k, tree in enumerate((g.create_unary_node("relu", g.create_input_node("value")),
                              g.create_binary_node("multiply", g.create_input_node("value"), g.create_input_node("confidence")))):
        rec = orig(tree, 100 + k) | {"tree": tree}
        pop.append({"tree": tree, "fitness": rec["fitness"], "record": rec})
    out = rerank_candidates(ev, pop, None, top_k=5, log=lambda *_: None)
    assert seen == [0, 0] and len(out["table"]) == 2


def test_end_to_end_two_stage_selection_names_run_and_records_rule(tmp_path):
    cfg = _deep_update(DEFAULT_CONFIG, {
        "name": "smoke", "dataset": "synthetic",
        "synthetic": {"n_train": 96, "n_val": 32, "n_test": 32, "seed": 0},
        "corruption": {"pattern": "random", "rate": 0.2},
        "search": {"population_size": 4, "generations": 2, "fitness_epochs": 1, "fitness_patience": 2,
                   "fitness_batch_size": 32, "fitness_train_subset": 64,
                   "rerank": {"top_k": 2, "epochs": 2, "patience": 2, "batch_size": 32, "train_subset": None}},
        "final": {"epochs": 2, "patience": 2, "batch_size": 32},
        "device": "cpu", "out_root": str(tmp_path / "runs"), "data_root": str(tmp_path / "data"),
    })
    base = cfg
    cfg = compose(base, "cnn_evolved_zmc")
    assert cfg["name"] == "smoke_random20_cnn_evolved_zmc_rr2"
    assert compose(base, "cnn_relu")["name"] == "smoke_random20_cnn_relu"     # fixed methods: no suffix
    res = run_experiment(cfg, replicate=0)
    sel = res["selection"]
    assert sel["selection_rule"] == "two_stage_rerank" and sel["rerank"]["top_k"] == 2
    assert sel["infix"] == res["selected_infix"] and "search_best" in sel
    assert res["cost"]["rerank_runtime_s"] > 0
    assert os.path.exists(os.path.join(cfg["out_root"], cfg["name"], "rep0", "search", "rerank.json"))
    # reuse_search_dir: a second run name re-uses the finished search and re-ranks without searching again
    cfg2 = dict(cfg, name=cfg["name"] + "_reuse")
    cfg2["search"] = dict(cfg["search"], reuse_search_dir=os.path.join(cfg["out_root"], cfg["name"], "rep0", "search"))
    res2 = run_experiment(cfg2, replicate=0)
    assert res2["selection"]["search_copied_from"] == cfg2["search"]["reuse_search_dir"]
    assert res2["selection"]["n_evaluations"] == res["selection"]["n_evaluations"]
    assert res2["selected_infix"] == res["selected_infix"]


def test_selection_only_run_never_trains_final_model_or_touches_test(tmp_path):
    cfg = _deep_update(DEFAULT_CONFIG, {
        "name": "tune", "dataset": "synthetic",
        "synthetic": {"n_train": 96, "n_val": 32, "n_test": 32, "seed": 0},
        "corruption": {"pattern": "random", "rate": 0.2},
        "search": {"population_size": 4, "generations": 1, "fitness_epochs": 1, "fitness_patience": 2,
                   "fitness_batch_size": 32, "fitness_train_subset": 64,
                   "rerank": {"top_k": 2, "epochs": 1, "patience": 1, "batch_size": 32, "train_subset": None}},
        "final": {"epochs": 2, "patience": 2, "batch_size": 32, "enabled": False},
        "device": "cpu", "out_root": str(tmp_path / "runs"), "data_root": str(tmp_path / "data"),
    })
    cfg = compose(cfg, "cnn_evolved_zmc")
    res = run_experiment(cfg, replicate=0)
    assert res["mode"] == "selection_only" and res["final"] is None
    assert res["selection"]["selection_rule"] == "two_stage_rerank"
    run_dir = os.path.join(cfg["out_root"], cfg["name"], "rep0")
    assert not os.path.exists(os.path.join(run_dir, "checkpoint.pt"))
    assert not os.path.exists(os.path.join(run_dir, "test_predictions.npz"))
    assert run_experiment(cfg, replicate=0)["mode"] == "selection_only"     # resume path tolerates final=None


def test_rerank_all_finalists_when_top_k_is_none(tmp_path):
    train, val = tiny_data(48, 0), tiny_data(24, 1)
    cheap = FitnessEvaluator(factory, train, val, epochs=1, patience=1, batch_size=16, init_seed=1, batch_seed=2)
    res = run_search(cheap, population_size=6, generations=2, max_tree_depth=3, mutation_rate=0.3, crossover_rate=0.7,
                     elite_size=1, evolution_seed=11, out_dir=str(tmp_path / "search"), log=lambda *_: None)
    long = FitnessEvaluator(factory, train, val, epochs=1, patience=1, batch_size=16, init_seed=1, batch_seed=2)
    out = rerank_candidates(long, res["population"], res["best"], top_k=None, log=lambda *_: None)
    distinct_valid = {ind["record"]["infix"] for ind in res["population"] if ind["record"] and ind["record"]["valid"]}
    distinct_valid.add(res["best"]["infix"])
    assert out["pool_size"] == len(distinct_valid) == len(out["table"])
    base = _deep_update(DEFAULT_CONFIG, {"name": "smoke", "search": {"rerank": {"top_k": None, "epochs": 1, "patience": 1,
                                                                                "batch_size": 32, "train_subset": None}}})
    assert compose(base, "cnn_evolved_zmc")["name"] == "smoke_random20_cnn_evolved_zmc_rrall"
    base["search"]["rerank"]["top_k"] = "all"
    assert compose(base, "cnn_evolved_zmc")["name"] == "smoke_random20_cnn_evolved_zmc_rrall"
