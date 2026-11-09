"""Separate development/training and frozen test evaluation commands."""
from pathlib import Path
import csv
import json
import time
import joblib
import numpy as np
from .data import ROOT, load_dataset, split_indices, MeanPreprocessor, MaskGenerator, array_hash
from .confidence import cross_fit, confidence_metrics, shuffle_missing
from .storage import bind_run, write_json

VARIANTS = ("original", "learned", "feature_constant", "first_only", "shuffled", "regression_imputation")
FIXED = ("relu", "swish", "leaky_relu", "elu", "mask_concat_relu")
# regression_imputation: IterativeImputer (train-only) replaces mean imputation; ORIGINAL confidence rule; evolved
#   activation -> isolates "better imputed value" from "better confidence".
# mask_concat_relu: standard ReLU MLP on [values, mask] (fixed baseline; no ChannelProp, no confidence).


def read_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    from .datasets_gecco import REGISTRY
    if not cfg["datasets"] or any(d not in ("pima", "wdbc", "synthetic", *REGISTRY) for d in cfg["datasets"]):
        raise ValueError("Unknown dataset; choose from pima, wdbc, synthetic or " + ", ".join(sorted(REGISTRY)))
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Seeds must be nonempty and unique")
    if not cfg["variants"] or any(v not in VARIANTS for v in cfg["variants"]):
        raise ValueError("Invalid variant")
    if any(v not in FIXED for v in cfg["fixed_baselines"]):
        raise ValueError("Invalid fixed activation")
    s = cfg["search"]
    if s["population"] < 2 or not 0 <= s["elite_size"] < s["population"]:
        raise ValueError("Require population >= 2 and elite_size < population")
    if s["generations"] < 1 or s["max_seconds"] <= 0:
        raise ValueError("Search budgets must be positive")
    c = cfg["confidence"]
    if c["outer_folds"] < 2 or c["inner_folds"] < 2 or c["tolerance_std"] <= 0:
        raise ValueError("Invalid cross-fitting or tolerance configuration")
    if c["mask_repeats"] < 1 or c["max_iter"] < 1:
        raise ValueError("Confidence estimator budget must be positive")
    if len(set(cfg["variants"])) != len(cfg["variants"]):
        raise ValueError("Duplicate variant")
    for training in (s["training"], cfg["final"]):
        if training["epochs"] < 1 or training["patience"] < 1:
            raise ValueError("Training budgets must be positive")
    return cfg


def _values_for(pack, split, variant):
    if variant == "regression_imputation":
        return pack["regimp_" + split]
    return pack[split][0]


def _confidence_for(pack, split, variant, seed):
    x, m, original, _ = pack[split]
    if variant in ("original", "regression_imputation") or variant in FIXED:
        return original        # regression_imputation changes the VALUES only; the confidence rule stays original
    learned = pack["learned_" + split]
    if variant == "feature_constant":
        return pack["constant_" + split]
    if variant == "shuffled":
        return shuffle_missing(learned, m, seed)
    return learned


def variant_arrays(pack, split, variant, seed):
    x, m, _, y = pack[split]
    return _values_for(pack, split, variant), m, _confidence_for(pack, split, variant, seed), y


def regression_impute_standardized(imputer, pre, masked):
    """IterativeImputer values on the preprocessor's (train-fitted) standardisation scale."""
    filled = imputer.transform(masked)
    z = (filled - pre.scale_mean) / pre.scale
    if not np.isfinite(z).all():
        raise FloatingPointError("Non-finite regression imputation")
    return z.astype(np.float32)


def prepare(cfg, dataset, seed, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / "prepared.joblib"
    if cache.exists():
        return joblib.load(cache)
    start = time.monotonic()
    x, y, meta = load_dataset(dataset)
    sub_n = (cfg.get("subsample") or {}).get(dataset)
    subsample_info = None
    if sub_n and sub_n < len(y):
        # declared stratified subsample (large datasets only), drawn once per seed before any split
        from sklearn.model_selection import train_test_split
        keep, _ = train_test_split(np.arange(len(y)), train_size=int(sub_n), stratify=y, random_state=seed + 50)
        keep = np.sort(keep)
        x, y = x[keep], y[keep]
        subsample_info = {"n": int(sub_n), "of": int(len(keep) and meta["rows"]), "indices_sha256": array_hash(keep)}
        meta = dict(meta, rows=int(len(y)), subsample=subsample_info)
    splits = split_indices(y, seed)
    corruption = MaskGenerator.fit(x[splits["train"]])
    mech, rate = cfg["corruption"]["mechanism"], cfg["corruption"]["rate"]
    masked = {}
    for split, offset in (("train", 100), ("val", 200)):
        masked[split], _ = corruption.corrupt(x[splits[split]], mech, rate, seed + offset)
    pre = MeanPreprocessor.fit(masked["train"])
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    t_imp = time.monotonic()
    imputer = IterativeImputer(max_iter=10, random_state=seed + 77, keep_empty_features=True).fit(masked["train"])
    regimp = {s: regression_impute_standardized(imputer, pre, masked[s]) for s in ("train", "val")}
    imputer_runtime = time.monotonic() - t_imp
    learned, constant, estimator, audit, elapsed = cross_fit(
        masked["train"], cfg["confidence"], seed + 300)
    pack = {"metadata": meta, "split_indices": splits, "preprocessor": pre,
            "mask_generator": corruption, "estimator": estimator, "imputer": imputer,
            "regimp_train": regimp["train"], "regimp_val": regimp["val"], "imputer_runtime_s": imputer_runtime,
            "learned_train": learned, "constant_train": constant,
            "confidence_fit_runtime_s": elapsed}
    for split in ("train", "val"):
        z, m, c = pre.transform(masked[split])
        pack[split] = (z, m, c, y[splits[split]])
        if split == "val":
            pack["learned_val"] = estimator.predict(masked[split])
            pack["constant_val"] = np.where(m, estimator.feature_priors, 1.).astype(np.float32)
    val_scores = {}
    for v in ("original", "learned", "feature_constant", "shuffled"):
        val_scores[v] = confidence_metrics(
            x[splits["val"]], masked["val"],
            _confidence_for(pack, "val", v, seed + 400), pre,
            cfg["confidence"]["tolerance_std"])
    pack["validation_confidence"] = val_scores
    pack["prepare_runtime_s"] = time.monotonic() - start
    info = {"dataset": meta, "seed": seed,
            "split_indices": splits,
            "split_fingerprints": {k: array_hash(v) for k, v in splits.items()},
            "mask_fingerprints": {k: array_hash(~np.isfinite(v)) for k, v in masked.items()},
            "realized_missing_rates": {k: float((~np.isfinite(v)).mean()) for k, v in masked.items()},
            "confidence_outer_folds": audit,
            "confidence_fit_runtime_s": elapsed, "imputer_runtime_s": imputer_runtime, "subsample": subsample_info,
            "prepare_runtime_s": pack["prepare_runtime_s"],
            "empty_training_features": pre.empty_features,
            "validation_confidence": val_scores,
            "test_evaluated": False}
    tmp = cache.with_suffix(".tmp")
    joblib.dump(pack, tmp)
    tmp.replace(cache)
    write_json(directory / "preparation.json", info)
    _plot_reliability(val_scores, directory / "validation_reliability.png")
    return pack


def _plot_reliability(scores, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4.2), layout="constrained")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    for name in ("original", "learned", "feature_constant"):
        b = scores[name]["bins"]
        ax.plot([v["predicted"] for v in b], [v["observed"] for v in b],
                marker="o", label=name.replace("_", " "))
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted imputation-success probability",
           ylabel="Observed success fraction", title="Validation: deliberately hidden entries")
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def confidence_stage(cfg, output):
    bind_run(output, cfg)
    rows = []
    for dataset in cfg["datasets"]:
        for seed in cfg["seeds"]:
            directory = Path(output) / dataset / ("seed_" + str(seed))
            print(f"Confidence preparation: {dataset}, seed {seed}", flush=True)
            pack = prepare(cfg, dataset, seed, directory)
            for variant, metrics in pack["validation_confidence"].items():
                rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                             "n_hidden": metrics["n"], "brier": metrics["brier"],
                             "event_auc": metrics["event_auc"], "ece": metrics.get("ece"),
                             "confidence_fit_runtime_s": pack["confidence_fit_runtime_s"]})
    _write_csv(Path(output) / "confidence_validation.csv", rows)
    lines = ["# Confidence pilot", "", "Validation only. No GP search or test evaluation.",
             "Original confidence is a missing-frequency heuristic; Brier scores assess how well each score predicts the chosen imputation-success event.",
             "", "| Dataset | Seed | Variant | Brier (lower is better) | Event AUC |",
             "|---|---:|---|---:|---:|"]
    for r in rows:
        brier = "n/a" if r["brier"] is None else f'{r["brier"]:.4f}'
        auc = "n/a" if r["event_auc"] is None else f'{r["event_auc"]:.4f}'
        lines.append(f'| {r["dataset"]} | {r["seed"]} | {r["variant"]} | {brier} | {auc} |')
    (Path(output) / "confidence_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def _device(cfg):
    import torch
    if cfg["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])


def train_stage(cfg, output):
    import torch
    from .models import build_model, tensors, train, scores, predict, fixed_tree, channel_trace
    from .search import search
    from . import gp_core as g
    manifest = bind_run(output, cfg)
    torch.set_num_threads(cfg["torch_threads"])
    device = _device(cfg)
    rows = []
    for dataset in cfg["datasets"]:
        for seed in cfg["seeds"]:
            directory = Path(output) / dataset / ("seed_" + str(seed))
            pack = prepare(cfg, dataset, seed, directory)
            for variant in cfg["variants"] + cfg["fixed_baselines"]:
                method_dir = directory / variant
                method_dir.mkdir(parents=True, exist_ok=True)
                result_path = method_dir / "training.json"
                if result_path.exists():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    rows.append(_training_row(dataset, seed, variant, result))
                    continue
                print(f"Training: {dataset}, seed {seed}, {variant}", flush=True)
                tr = tensors(variant_arrays(pack, "train", variant, seed + 500), device)
                va = tensors(variant_arrays(pack, "val", variant, seed + 600), device)
                propagation = "first_only" if variant == "first_only" else "full"
                model_type = "concat_mlp" if variant == "mask_concat_relu" else "three_channel"
                def factory(tree):
                    return build_model(model_type, tr[0].shape[1], len(pack["metadata"]["classes"]),
                                       cfg["hidden_sizes"], tree, propagation).to(device)
                search_info = None
                if variant in FIXED:
                    tree = fixed_tree(variant)
                else:
                    search_info = search(factory, tr, va, cfg["search"], seed + 700,
                                         method_dir, manifest["fingerprint"])
                    if search_info["best"] is None:
                        result = {"status": "failed", "failure": "No numerically valid GP candidate",
                                  "search_runtime_s": search_info["elapsed_s"],
                                  "search_stop_reason": search_info["stop_reason"]}
                        write_json(result_path, result)
                        rows.append(_training_row(dataset, seed, variant, result))
                        continue
                    tree = search_info["best"]["tree"]
                torch.manual_seed(seed + 10000)
                model = factory(tree)
                final = train(model, tr, va, cfg["final"], final=True)
                result = {"status": "trained" if final["valid"] else "failed",
                          "tree": tree, "infix": g.tree_to_infix(tree), "propagation": propagation, "model_type": model_type,
                          "training": final, "failure": final["failure"],
                          "search_runtime_s": search_info["elapsed_s"] if search_info else 0.,
                          "search_evaluations": len(search_info["records"]) if search_info else 0,
                          "search_stop_reason": search_info["stop_reason"] if search_info else None,
                          "torch": torch.__version__, "device": str(device),
                          "test_evaluated": False}
                if final["valid"]:
                    result["validation"] = scores(pack["val"][3], predict(model, va))
                    result["validation_channel_trace"] = channel_trace(model, va) if model_type == "three_channel" else None
                    checkpoint = method_dir / "checkpoint.pt"
                    tmp = checkpoint.with_suffix(".tmp")
                    torch.save(model.state_dict(), tmp)
                    tmp.replace(checkpoint)
                write_json(result_path, result)
                rows.append(_training_row(dataset, seed, variant, result))
    _write_csv(Path(output) / "training_validation.csv", rows)
    return rows


def _training_row(dataset, seed, variant, result):
    metrics = result.get("validation", {})
    return {"dataset": dataset, "seed": seed, "variant": variant, "status": result["status"],
            "validation_accuracy": metrics.get("accuracy"), "validation_auc": metrics.get("auc"),
            "validation_macro_f1": metrics.get("macro_f1"),
            "search_runtime_s": result.get("search_runtime_s"),
            "final_runtime_s": result.get("training", {}).get("runtime_s"),
            "infix": result.get("infix"), "failure": result.get("failure")}


def evaluate_stage(output):
    import torch
    from .models import build_model, tensors, predict, scores
    output = Path(output)
    saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    cfg = saved["config"]
    bind_run(output, cfg)
    torch.set_num_threads(cfg["torch_threads"])
    device = _device(cfg)
    conditions = [cfg["corruption"]] + cfg["test_conditions"]
    # Stable removal of duplicate matched conditions.
    conditions = list({(v["mechanism"], v["rate"]): v for v in conditions}.values())
    rows = []
    for dataset in cfg["datasets"]:
        x_full, y_full, _ = load_dataset(dataset)
        for seed in cfg["seeds"]:
            directory = output / dataset / ("seed_" + str(seed))
            cache = directory / "prepared.joblib"
            if not cache.exists():
                raise ValueError("Train the run before test evaluation")
            pack = joblib.load(cache)
            x, y = x_full, y_full
            sub_n = (cfg.get("subsample") or {}).get(dataset)
            if sub_n and sub_n < len(y_full):   # same declared subsample as prepare()
                from sklearn.model_selection import train_test_split
                keep, _ = train_test_split(np.arange(len(y_full)), train_size=int(sub_n), stratify=y_full, random_state=seed + 50)
                keep = np.sort(keep)
                assert array_hash(keep) == pack["metadata"]["subsample"]["indices_sha256"]
                x, y = x_full[keep], y_full[keep]
            test_idx = pack["split_indices"]["test"]
            for variant in cfg["variants"] + cfg["fixed_baselines"]:
                method_dir = directory / variant
                path = method_dir / "training.json"
                if not path.exists():
                    raise ValueError("Training incomplete: " + str(path))
                result = json.loads(path.read_text(encoding="utf-8"))
                if result["status"] != "trained":
                    print(f"Skip failed model: {dataset}/{seed}/{variant}", flush=True)
                    continue
                eval_path = method_dir / "evaluation.json"
                if eval_path.exists():
                    rows.extend(json.loads(eval_path.read_text(encoding="utf-8"))["rows"])
                    continue
                model = build_model(result.get("model_type", "three_channel"), pack["train"][0].shape[1],
                                    len(pack["metadata"]["classes"]), cfg["hidden_sizes"],
                                    result["tree"], result["propagation"]).to(device)
                model.load_state_dict(torch.load(method_dir / "checkpoint.pt",
                                                map_location=device, weights_only=True))
                method_rows = []
                for i, condition in enumerate(conditions):
                    mechanism, rate = condition["mechanism"], condition["rate"]
                    masked, mask = pack["mask_generator"].corrupt(
                        x[test_idx], mechanism, rate, seed + 20000 + i)
                    z, m, original = pack["preprocessor"].transform(masked)
                    learned = pack["estimator"].predict(masked)
                    tiny = {"test": (z, m, original, y[test_idx]), "learned_test": learned,
                            "regimp_test": regression_impute_standardized(pack["imputer"], pack["preprocessor"], masked),
                            "constant_test": np.where(m, pack["estimator"].feature_priors, 1.).astype(np.float32)}
                    arrays = variant_arrays(tiny, "test", variant, seed + 30000 + i)
                    start = time.monotonic()
                    probs = predict(model, tensors(arrays, device))
                    runtime = time.monotonic() - start
                    metrics = scores(y[test_idx], probs)
                    cm = confidence_metrics(x[test_idx], masked, arrays[2], pack["preprocessor"],
                                            cfg["confidence"]["tolerance_std"])
                    tag = f"{i}_{mechanism}_{rate:g}"
                    np.savez_compressed(method_dir / ("predictions_" + tag + ".npz"),
                                        indices=test_idx, labels=y[test_idx], probs=probs, mask=mask)
                    row = {"dataset": dataset, "seed": seed, "variant": variant,
                           "mechanism": mechanism, "nominal_rate": rate,
                           "realized_rate": float(mask.mean()), "n_test": len(test_idx),
                           "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                           "auc": metrics["auc"], "confidence_brier": cm["brier"],
                           "confidence_event_auc": cm["event_auc"],
                           "inference_runtime_s": runtime, "mask_fingerprint": array_hash(mask)}
                    method_rows.append(row)
                write_json(eval_path, {"fingerprint": saved["fingerprint"], "rows": method_rows})
                rows.extend(method_rows)
    _write_csv(output / "test_results.csv", rows)
    groups = {}
    for row in rows:
        key = (row["dataset"], row["variant"], row["mechanism"], row["nominal_rate"])
        groups.setdefault(key, []).append(row)
    aggregates = []
    for key, group in groups.items():
        entry = dict(zip(("dataset", "variant", "mechanism", "nominal_rate"), key))
        entry["n_runs"] = len(group)
        for metric in ("accuracy", "macro_f1", "auc"):
            vals = [r[metric] for r in group if r[metric] is not None]
            entry[metric + "_mean"] = float(np.mean(vals)) if vals else None
            entry[metric + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        aggregates.append(entry)
    _write_csv(output / "test_summary.csv", aggregates)
    return rows
