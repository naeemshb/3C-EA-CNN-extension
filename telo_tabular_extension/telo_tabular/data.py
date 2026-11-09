"""Numeric PIMA/WDBC loading, train-only preprocessing and controlled masks."""
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(x):
    a = np.ascontiguousarray(x)
    return hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()


def load_dataset(name):
    if name == "synthetic":
        rng = np.random.default_rng(2026)
        latent = rng.normal(size=180)
        x = np.column_stack([latent + rng.normal(scale=.3, size=180) for _ in range(6)])
        y = (latent + rng.normal(scale=.2, size=180) > 0).astype(np.int64)
        return x, y, {"name": name, "features": [f"x{j}" for j in range(6)],
                      "classes": ["0", "1"], "rows": len(x),
                      "source_sha256": array_hash(x), "synthetic": True}
    if name not in ("pima", "wdbc"):
        from .datasets_gecco import REGISTRY, load_gecco_dataset
        if name in REGISTRY:
            return load_gecco_dataset(name)
        raise ValueError("Unknown dataset: " + name)
    path = ROOT / "data" / (name + ".csv")
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    features = rows[0][:-1]
    x = np.asarray([[float(v) if v.strip() else np.nan for v in row[:-1]]
                    for row in rows[1:]], dtype=np.float64)
    raw_labels = [row[-1] for row in rows[1:]]
    classes = sorted(set(raw_labels))
    y = np.asarray([classes.index(v) for v in raw_labels], dtype=np.int64)
    return x, y, {"name": name, "features": features, "classes": classes,
                  "rows": len(x), "source_sha256": file_hash(path),
                  "pima_zero_policy": "retain as observed, matching supplied GECCO CSV"}


def split_indices(y, seed):
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=.4, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=.5, stratify=y[rest], random_state=seed + 1)
    return {"train": tr, "val": va, "test": te}


@dataclass
class MeanPreprocessor:
    mean: np.ndarray
    scale_mean: np.ndarray
    scale: np.ndarray
    observed_std: np.ndarray
    missing_rate: np.ndarray
    empty_features: list

    @classmethod
    def fit(cls, x):
        x = np.asarray(x, dtype=float)
        obs = np.isfinite(x)
        n = obs.sum(axis=0)
        mean = np.where(obs, x, 0).sum(axis=0) / np.maximum(n, 1)
        filled = np.where(obs, x, mean)
        center = filled.mean(axis=0)
        scale = filled.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.)
        variance = np.where(obs, (x - mean) ** 2, 0).sum(axis=0) / np.maximum(n, 1)
        std = np.sqrt(variance)
        std = np.where(std > 1e-12, std, 1.)
        return cls(mean, center, scale, std, (~obs).mean(axis=0),
                   np.flatnonzero(n == 0).tolist())

    def transform(self, x):
        m = ~np.isfinite(x)
        z = (np.where(m, self.mean, x) - self.scale_mean) / self.scale
        c = np.where(m, np.maximum(.1, 1 - self.missing_rate), 1.)
        return z.astype(np.float32), m.astype(np.float32), c.astype(np.float32)


@dataclass
class MaskGenerator:
    median: np.ndarray
    q1: np.ndarray
    q3: np.ndarray
    neighbors: np.ndarray
    mar_driver: int = 0

    @classmethod
    def fit(cls, train_x):
        pre = MeanPreprocessor.fit(train_x)
        filled = np.where(np.isfinite(train_x), train_x, pre.mean)
        corr = np.corrcoef(filled, rowvar=False)
        corr = np.nan_to_num(corr)
        neighbors = np.argsort(-np.abs(corr), axis=1, kind="stable")
        # MAR driver = the first feature fully observed in the training partition (feature 0 when it is complete)
        complete = np.flatnonzero(np.isfinite(train_x).all(axis=0))
        driver = int(complete[0]) if len(complete) else -1
        return cls(np.median(filled, axis=0), np.quantile(filled, .25, axis=0),
                   np.quantile(filled, .75, axis=0), neighbors, driver)

    def mask(self, x, mechanism, rate, seed):
        if not 0 <= rate <= 1:
            raise ValueError("Missing rate must be in [0, 1]")
        rng = np.random.default_rng(seed)
        n, d = x.shape
        natural = ~np.isfinite(x)
        mechanism = mechanism.lower()
        if mechanism == "mcar":
            added = np.zeros(n * d, dtype=bool)
            available = np.flatnonzero(~natural.reshape(-1))
            count = min(len(available), int(n * d * rate))
            added[rng.choice(available, count, replace=False)] = True
            added = added.reshape(n, d)
        elif mechanism == "mar":
            # The driver (first training-complete feature) stays observed and receives no artificial mask.
            j = self.mar_driver
            if j < 0 or natural[:, j].any():
                raise ValueError("MAR simulation requires a fully observed driver feature")
            p = np.where(x[:, j] > self.median[j], rate * 2, rate * .5)
            added = rng.random((n, d)) < np.clip(p[:, None], 0, 1)
            added[:, j] = False
        elif mechanism == "mnar":
            extreme = (x < self.q1) | (x > self.q3)
            p = np.where(extreme, rate * 1.5, rate * .5)
            added = rng.random((n, d)) < np.clip(p, 0, 1)
        elif mechanism == "group":
            count = int(round(d * rate))
            added = np.zeros((n, d), dtype=bool)
            anchors = rng.integers(0, d, size=n)
            for i, anchor in enumerate(anchors):
                added[i, self.neighbors[anchor, :count]] = True
        else:
            raise ValueError("Unknown mechanism: " + mechanism)
        return natural | added

    def corrupt(self, x, mechanism, rate, seed):
        mask = self.mask(x, mechanism, rate, seed)
        return np.where(mask, np.nan, x), mask
