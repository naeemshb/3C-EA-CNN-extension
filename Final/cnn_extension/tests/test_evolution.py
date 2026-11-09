import inspect
import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from cnn_extension import gp_core as g
from cnn_extension.evolution import (ALL_TERMINALS, DIVERSITY_NORMALIZER, FitnessEvaluator, fitness_components,
                                     restricted_terminals, run_search)
from cnn_extension.models import build_model
from cnn_extension.training import TensorData


def tiny_data(n=48, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n, 1, 6, 6)).astype(np.float32)
    y = (x[:, 0, :3, :].mean((1, 2)) > x[:, 0, 3:, :].mean((1, 2))).astype(np.int64)
    m = (rng.random(x.shape) < 0.2).astype(np.float32)
    c = np.where(m > 0, 0.8, 1.0).astype(np.float32)
    return TensorData.from_numpy(x, m, c, y)


def mlp_factory(tree):
    return build_model("mlp", 1, 6, 6, 2, tree, {"mlp_hidden_sizes": [8, 4]})


def test_fitness_formula_matches_original_code():
    tree = g.create_binary_node("add", g.create_input_node("value"), g.create_input_node("missing"))
    comp = fitness_components(tree, 0.8)
    assert comp["nodes"] == 3 and comp["depth"] == 2 and comp["diversity"] == 2
    expected = 0.8 + 0.01 * 2 / DIVERSITY_NORMALIZER - 0.0001 * 3 - 0.0002 * (2 - 1)
    assert comp["fitness"] == pytest.approx(expected)
    assert DIVERSITY_NORMALIZER == 3.0        # the released code divides D by 3 (documented discrepancy)


def test_terminal_only_tree_gets_zero_fitness_without_training():
    def factory(tree):
        raise AssertionError("must not be called for a depth-1 tree")

    ev = FitnessEvaluator(factory, tiny_data(), tiny_data(seed=1), epochs=1, patience=2, batch_size=16,
                          init_seed=0, batch_seed=0)
    rec = ev.evaluate(g.create_input_node("value"), 0)
    assert rec["fitness"] == 0.0 and rec["valid"] is False and rec["failure"] == "terminal_only_tree"


def test_numerical_failure_gets_zero_fitness_and_is_invalid():
    class NaNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(36, 2)

        def forward(self, x, m, c):
            return self.lin(x.flatten(1)) * float("nan")

    ev = FitnessEvaluator(lambda tree: NaNModel(), tiny_data(), tiny_data(seed=1), epochs=2, patience=2,
                          batch_size=16, init_seed=0, batch_seed=0)
    rec = ev.evaluate(g.create_unary_node("tanh", g.create_input_node("value")), 3)
    assert rec["fitness"] == 0.0 and rec["valid"] is False and "non-finite" in rec["failure"]

    def exploding(tree):
        raise RuntimeError("boom")

    ev2 = FitnessEvaluator(exploding, tiny_data(), tiny_data(seed=1), epochs=1, patience=2, batch_size=16,
                           init_seed=0, batch_seed=0)
    rec2 = ev2.evaluate(g.create_unary_node("tanh", g.create_input_node("value")), 4)
    assert rec2["fitness"] == 0.0 and rec2["valid"] is False and "boom" in rec2["failure"]


def test_valid_candidate_records_components_and_runtime():
    ev = FitnessEvaluator(mlp_factory, tiny_data(), tiny_data(seed=1), epochs=2, patience=2, batch_size=16,
                          init_seed=0, batch_seed=0)
    tree = g.create_binary_node("multiply", g.create_unary_node("tanh", g.create_input_node("value")),
                                g.create_input_node("confidence"))
    rec = ev.evaluate(tree, 5)
    assert rec["valid"] and 0 <= rec["val_acc"] <= 1 and rec["runtime_s"] > 0 and rec["epochs_run"] >= 1
    assert rec["fitness"] == pytest.approx(fitness_components(tree, rec["val_acc"])["fitness"])
    assert rec["init_seed"] == 5 and rec["batch_seed"] == 5


def test_evaluator_receives_no_test_data():
    params = set(inspect.signature(FitnessEvaluator.__init__).parameters) - {"self"}
    assert not any("test" in p for p in params)
    ev = FitnessEvaluator(mlp_factory, tiny_data(), tiny_data(seed=1), epochs=1, patience=1, batch_size=16,
                          init_seed=0, batch_seed=0)
    assert not any("test" in k for k in vars(ev))


def test_restricted_terminals_limit_generation_and_mutation_and_restore():
    random.seed(0)
    with restricted_terminals(["value", "missing"]):
        assert g.INPUTS == ["value", "missing"]
        pop = g.create_initial_population(200, 3)
        for ind in pop:
            for node in g.get_all_nodes(ind["tree"]):
                assert node["type"] != "input" or node["input_name"] != "confidence"
            mutated = g.mutate_tree(ind["tree"], 0.9, 3)
            for node in g.get_all_nodes(mutated):
                assert node["type"] != "input" or node["input_name"] != "confidence"
    assert g.INPUTS == list(ALL_TERMINALS)
    with pytest.raises(ValueError):
        with restricted_terminals(["missing", "confidence"]):
            pass
    assert g.INPUTS == list(ALL_TERMINALS)
    # D is computed from the terminals actually referenced, whatever the restriction
    tree = g.create_binary_node("add", g.create_input_node("value"), g.create_input_node("value"))
    assert fitness_components(tree, 0.5)["diversity"] == 1


def test_run_search_reuses_original_operators_checkpoints_and_resumes(tmp_path):
    ev = FitnessEvaluator(mlp_factory, tiny_data(), tiny_data(seed=1), epochs=1, patience=2, batch_size=16,
                          init_seed=0, batch_seed=0)
    kw = dict(population_size=6, generations=3, max_tree_depth=3, mutation_rate=0.15, crossover_rate=0.7,
              elite_size=2, evolution_seed=42, out_dir=str(tmp_path / "search"), log=lambda s: None)
    res = run_search(ev, **kw)
    assert len(res["history"]) == 3
    # elites re-enter with fitness None (original behaviour) -> every generation evaluates the full population
    assert res["n_evaluations"] == 18 and ev.n_evaluated <= 18
    best = res["best"]
    assert best["valid"] and best["depth"] >= 2
    assert (tmp_path / "search" / "search_history.json").exists()
    n_before = ev.n_evaluated
    res2 = run_search(ev, **kw)                  # resume on a finished search: nothing re-evaluated
    assert ev.n_evaluated == n_before and res2["best"]["infix"] == best["infix"]


def test_winner_is_never_an_invalid_candidate(tmp_path):
    class RejectConfidence(FitnessEvaluator):
        def evaluate(self, tree, candidate_id):
            rec = super().evaluate(tree, candidate_id)
            if "confidence" in rec["infix"]:
                rec.update({"fitness": 0.0, "valid": False, "failure": "forced"})
            else:
                rec["fitness"] = 5.0 if rec["valid"] else rec["fitness"]   # dominated only by invalid ones
            return rec

    ev = RejectConfidence(mlp_factory, tiny_data(), tiny_data(seed=1), epochs=1, patience=2, batch_size=16,
                          init_seed=0, batch_seed=0)
    res = run_search(ev, population_size=8, generations=2, max_tree_depth=3, mutation_rate=0.15,
                     crossover_rate=0.7, elite_size=2, evolution_seed=1, out_dir=str(tmp_path / "s"),
                     log=lambda s: None)
    assert res["best"]["valid"] and "confidence" not in res["best"]["infix"]
    assert all(not e["valid"] for e in res["evaluations"] if "confidence" in e["infix"])
