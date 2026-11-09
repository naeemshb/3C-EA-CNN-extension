"""
Method registry: the comparisons and ablations of the brief as config fragments.

Compose a base config (dataset, corruption, search budget, seeds) with one of
these names; the experiment name becomes ``<base name>_<method>``.
"""
from __future__ import annotations

import copy

ALL = ["value", "missing", "confidence"]

METHODS = {
    # --- main CNN comparison: identical architecture / training protocol ---
    "cnn_relu":        {"backbone": "cnn", "method": {"type": "fixed", "activation": "relu"}},
    "cnn_swish":       {"backbone": "cnn", "method": {"type": "fixed", "activation": "swish"}},
    "cnn_leakyrelu":   {"backbone": "cnn", "method": {"type": "fixed", "activation": "leakyrelu"}},
    "cnn_elu":         {"backbone": "cnn", "method": {"type": "fixed", "activation": "elu"}},
    "cnn_evolved_z":   {"backbone": "cnn", "method": {"type": "evolved", "terminals": ["value"]}},
    "cnn_evolved_zmc": {"backbone": "cnn", "method": {"type": "evolved", "terminals": ALL}},
    # --- ablations ---
    "cnn_evolved_zm":  {"backbone": "cnn", "method": {"type": "evolved", "terminals": ["value", "missing"]}},
    "cnn_evolved_zc":  {"backbone": "cnn", "method": {"type": "evolved", "terminals": ["value", "confidence"]}},
    "cnn_evolved_zmc_first_only": {"backbone": "cnn", "metadata_mode": "first_only",
                                   "method": {"type": "evolved", "terminals": ALL}},
    # --- conventional CNN fed cat([X, M, C]) ---
    "concat_cnn_relu":      {"backbone": "concat_cnn", "method": {"type": "fixed", "activation": "relu"}},
    "concat_cnn_evolved_z": {"backbone": "concat_cnn", "method": {"type": "evolved", "terminals": ["value"]}},
    # --- MLP vs CNN on the same images ---
    "mlp_relu":        {"backbone": "mlp", "method": {"type": "fixed", "activation": "relu"}},
    "mlp_swish":       {"backbone": "mlp", "method": {"type": "fixed", "activation": "swish"}},
    "mlp_evolved_zmc": {"backbone": "mlp", "method": {"type": "evolved", "terminals": ALL}},
}


def compose(base_cfg: dict, method: str, pattern: str | None = None, rate: float | None = None) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown method '{method}'; choose from {sorted(METHODS)}")
    cfg = copy.deepcopy(base_cfg)
    cfg.update(copy.deepcopy(METHODS[method]))
    cfg.setdefault("metadata_mode", "full")
    if method not in ("cnn_evolved_zmc_first_only",):
        cfg["metadata_mode"] = "full"
    if pattern is not None:
        cfg["corruption"]["pattern"] = pattern
    if rate is not None:
        cfg["corruption"]["rate"] = float(rate)
    p, r = cfg["corruption"]["pattern"], int(round(100 * float(cfg["corruption"]["rate"])))
    cfg["name"] = f"{base_cfg['name']}_{p}{r}_{method}"
    if cfg["method"]["type"] == "evolved" and cfg.get("search", {}).get("rerank"):
        k = cfg["search"]["rerank"].get("top_k")
        cfg["name"] += "_rrall" if k in (None, 0, "all") else f"_rr{int(k)}"   # two-stage selection, D12
    return cfg
