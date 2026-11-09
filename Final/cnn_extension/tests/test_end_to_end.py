"""Tiny synthetic end-to-end run: search -> selection -> fresh retraining -> saving -> loading -> evaluation."""
import json
import os

import numpy as np
import pytest

from cnn_extension import gp_core as g
from cnn_extension.experiment import (DEFAULT_CONFIG, _deep_update, aggregate, evaluate_shift, reproduce_predictions,
                                      run_experiment)
from cnn_extension.methods import METHODS, compose


def smoke_config(tmp_path, method="cnn_evolved_zmc", name=None):
    cfg = _deep_update(DEFAULT_CONFIG, {
        "name": name or "smoke", "label": "SMOKE TEST", "dataset": "synthetic",
        "synthetic": {"n_train": 96, "n_val": 32, "n_test": 32, "seed": 0},
        "corruption": {"pattern": "random", "rate": 0.2},
        "search": {"population_size": 4, "generations": 2, "fitness_epochs": 1, "fitness_patience": 2,
                   "fitness_batch_size": 32, "fitness_train_subset": 64},
        "final": {"epochs": 2, "patience": 2, "batch_size": 32},
        "device": "cpu", "out_root": str(tmp_path / "runs"), "data_root": str(tmp_path / "data"),
    })
    return compose(cfg, method)


def test_end_to_end_evolved_cnn(tmp_path):
    cfg = smoke_config(tmp_path)
    res = run_experiment(cfg, replicate=0)
    run_dir = os.path.join(cfg["out_root"], cfg["name"], "rep0")
    for fname in ("config.json", "results.json", "checkpoint.pt", "preprocessor.npz", "test_predictions.npz",
                  "masks_train.npz", "masks_val.npz", "masks_test.npz", "log.txt",
                  os.path.join("search", "search_history.json"), os.path.join("search", "search_state.pkl")):
        assert os.path.exists(os.path.join(run_dir, fname)), fname
    g.validate_tree(res["selected_tree"])
    assert res["selection"]["type"] == "evolved" and res["selection"]["n_evaluations"] == 8
    assert res["selection"]["fitness_train_subset"]["n"] == 64
    assert 0 <= res["final"]["test_accuracy"] <= 1 and 0 <= res["final"]["test_macro_f1"] <= 1
    assert res["final"]["parameters"] > 0 and res["cost"]["search_runtime_s"] > 0
    assert set(res["seeds"]) >= {"split", "corruption", "evolution", "init", "batch", "corruption_per_split"}
    assert res["data"]["masks"]["test"]["realized_rate"] == pytest.approx(0.2, abs=0.05)
    with open(os.path.join(run_dir, "search", "search_history.json")) as f:
        hist = json.load(f)
    assert len(hist["per_generation"]) == 2 and len(hist["evaluations"]) == 8
    # saved artefacts reproduce the saved predictions exactly
    rep = reproduce_predictions(run_dir, device="cpu")
    assert rep["preds_equal"] and rep["max_prob_abs_diff"] < 1e-5
    # changing test conditions: frozen model + preprocessing, new corruption
    out = evaluate_shift(run_dir, "block", 0.4, device="cpu")
    assert out["mode"] == "changing_test_conditions" and out["tested_on"]["pattern"] == "block"
    assert os.path.exists(os.path.join(run_dir, "shift_block_40pct.json"))
    clean = evaluate_shift(run_dir, "random", 0.0, device="cpu")
    assert clean["tested_on"]["realized_rate"] == 0.0
    # re-running resumes from results.json
    again = run_experiment(cfg, replicate=0)
    assert again["final"]["test_accuracy"] == res["final"]["test_accuracy"]


def test_end_to_end_fixed_and_mlp_and_concat(tmp_path):
    for method in ("cnn_relu", "mlp_evolved_zmc", "concat_cnn_relu", "cnn_evolved_zm", "cnn_evolved_zmc_first_only"):
        cfg = smoke_config(tmp_path, method)
        res = run_experiment(cfg, replicate=1)
        if cfg["method"]["type"] == "fixed":
            assert res["cost"]["search_runtime_s"] == 0.0 and res["selection"]["n_evaluations"] == 0
        if method == "cnn_evolved_zm":
            assert "confidence" not in res["selected_infix"]
        if method == "cnn_evolved_zmc_first_only":
            assert res["config"]["metadata_mode"] == "first_only"
    summ = aggregate([os.path.join(cfg["out_root"], f"smoke_random20_{m}", "rep1")
                      for m in ("cnn_relu", "mlp_evolved_zmc", "concat_cnn_relu")])
    assert len(summ) == 3 and all(s["n_runs"] == 1 for s in summ.values())


def test_methods_registry_is_complete():
    assert {"cnn_relu", "cnn_swish", "cnn_leakyrelu", "cnn_elu", "cnn_evolved_z", "cnn_evolved_zmc",
            "cnn_evolved_zm", "cnn_evolved_zc", "cnn_evolved_zmc_first_only", "concat_cnn_relu",
            "concat_cnn_evolved_z", "mlp_relu", "mlp_evolved_zmc"} <= set(METHODS)


def test_paired_runs_share_masks_and_split(tmp_path):
    a = run_experiment(smoke_config(tmp_path, "cnn_relu"), replicate=3)
    b = run_experiment(smoke_config(tmp_path, "cnn_swish"), replicate=3)
    assert a["data"]["masks"] == b["data"]["masks"]
    assert a["data"]["preprocessor"]["fingerprint"] == b["data"]["preprocessor"]["fingerprint"]
    assert a["seeds"] == b["seeds"]
    c = run_experiment(smoke_config(tmp_path, "cnn_relu"), replicate=4)
    assert c["data"]["masks"]["test"]["fingerprint"] != a["data"]["masks"]["test"]["fingerprint"]
