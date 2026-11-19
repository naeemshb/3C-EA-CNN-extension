"""Atomic files and strict configuration/data/code fingerprints."""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import importlib.metadata
import numpy as np
import sklearn
from .data import ROOT, file_hash, load_dataset


def _default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_default, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def manifest(cfg):
    files = sorted((ROOT / "telo_tabular").glob("*.py"))
    payload = {"config": cfg, "code": {p.name: file_hash(p) for p in files},
               "data": {d: load_dataset(d)[2]["source_sha256"] for d in cfg["datasets"]},
               "versions": {name: importlib.metadata.version(name)
                            for name in ("numpy", "scikit-learn", "torch", "joblib")}}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {"fingerprint": fingerprint, **payload,
            "environment": {"python": sys.version, "platform": platform.platform(),
                            "numpy": np.__version__, "sklearn": sklearn.__version__}}


# Modules that determine results. Reporting / plotting scripts (screen, aggregate_main, shift_report,
# extended_metrics, make_figures) and this file only read or label finished runs, so adding or editing them
# must not invalidate a run directory.
RESULT_MODULES = ("confidence.py", "data.py", "datasets_gecco.py", "experiment.py", "gp_core.py", "models.py", "search.py")


def _same_run(existing, current):
    if existing["fingerprint"] == current["fingerprint"]:
        return True
    return (existing["config"] == json.loads(json.dumps(current["config"], default=_default))
            and existing["data"] == current["data"] and existing["versions"] == current["versions"]
            and all(existing["code"].get(name) == current["code"].get(name) for name in RESULT_MODULES))


def bind_run(output, cfg):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    current = manifest(cfg)
    path = output / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not _same_run(existing, current):
            raise ValueError("Configuration, dataset or code changed. Use a new output directory.")
        current = existing          # keep the fingerprint the run was started with (search states are bound to it)
    else:
        write_json(path, current)
    return current
