"""
Datasets, stratified validation splits, reproducible spatial missingness and
train-only preprocessing (mean imputation + confidence).

Conventions
-----------
* Images are float32 in [0, 1], shape [N, C, H, W].  No augmentation.
* Masks are bool, shape [N, C, H, W], True = missing.  A mask is generated
  from (pattern, nominal rate, seed) only -- never from pixel values or
  labels -- so a genuinely black pixel with mask False is observed.
* For RGB images every selected spatial location hides all colour channels.
* The split-specific corruption seeds are spawned deterministically from one
  corruption seed with ``numpy.random.SeedSequence``; the realised rate of
  each mask is recorded because blocks / rows / columns are quantised.

Preprocessing (fitted on the masked TRAINING partition only)
------------------------------------------------------------
For every feature (channel, row, col):
    mean_f   = mean of the observed training entries of that feature
    r_f      = fraction of training entries of that feature that are missing
Missing entries are replaced by mean_f.  Fallbacks when a feature has no
observed training entry: the observed training mean of its colour channel;
if the whole channel has no observed entry, the constant 0.5 (midpoint of the
[0, 1] intensity scale).  Fallback usage is counted and stored.
Confidence:  C = 1 for observed entries, C = max(tau, 1 - r_f) for missing
entries, tau = 0.1 (the original method's lower bound).  Confidence is a
heuristic score, not a calibrated probability.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

DATASETS = {
    "mnist":         {"in_channels": 1, "height": 28, "width": 28, "num_classes": 10},
    "fashion_mnist": {"in_channels": 1, "height": 28, "width": 28, "num_classes": 10},
    "cifar10":       {"in_channels": 3, "height": 32, "width": 32, "num_classes": 10},
    "bloodmnist":    {"in_channels": 3, "height": 28, "width": 28, "num_classes": 8},
    "synthetic":     {"in_channels": 1, "height": 28, "width": 28, "num_classes": 2},
    "synthetic_rgb": {"in_channels": 3, "height": 32, "width": 32, "num_classes": 2},
}

PATTERNS = ("random", "block", "rows", "cols")
CONFIDENCE_TAU = 0.1
CHANNEL_FALLBACK_MEAN = 0.5


@dataclass
class ImageSplit:
    x: np.ndarray  # float32 [N, C, H, W] in [0, 1]
    y: np.ndarray  # int64 [N]

    def __len__(self):
        return int(self.x.shape[0])


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _to_nchw(u8: np.ndarray) -> np.ndarray:
    a = np.asarray(u8)
    if a.ndim == 3:            # [N, H, W]
        a = a[:, None, :, :]
    elif a.ndim == 4:          # [N, H, W, C]
        a = a.transpose(0, 3, 1, 2)
    else:
        raise ValueError(a.shape)
    return np.ascontiguousarray(a.astype(np.float32) / 255.0)


def _load_torchvision(name: str, root: str, download: bool):
    import torchvision

    cls = {"mnist": torchvision.datasets.MNIST,
           "fashion_mnist": torchvision.datasets.FashionMNIST,
           "cifar10": torchvision.datasets.CIFAR10}[name]
    out = {}
    for split, flag in (("train", True), ("test", False)):
        ds = cls(root=root, train=flag, download=download)
        data = ds.data.numpy() if hasattr(ds.data, "numpy") else np.asarray(ds.data)
        targets = np.asarray(ds.targets, dtype=np.int64)
        out[split] = ImageSplit(_to_nchw(data), targets)
    return out


def _load_bloodmnist(root: str, download: bool):
    try:
        import medmnist
    except ImportError as e:  # pragma: no cover
        raise ImportError("BloodMNIST needs the 'medmnist' package: pip install medmnist") from e
    out = {}
    for split in ("train", "val", "test"):
        ds = medmnist.BloodMNIST(split=split, download=download, root=root, size=28)
        out[split] = ImageSplit(_to_nchw(ds.imgs), np.asarray(ds.labels).reshape(-1).astype(np.int64))
    return out


def make_synthetic(n_train=160, n_val=40, n_test=40, seed=0, rgb=False):
    """Tiny two-class image problem: class 0 bright patch top-left, class 1 bottom-right (plus noise)."""
    meta = DATASETS["synthetic_rgb" if rgb else "synthetic"]
    C, H, W = meta["in_channels"], meta["height"], meta["width"]
    rng = np.random.default_rng(seed)

    def gen(n):
        x = rng.uniform(0.0, 0.3, size=(n, C, H, W)).astype(np.float32)
        y = rng.integers(0, 2, size=n).astype(np.int64)
        for i in range(n):
            base_r, base_c = (3, 3) if y[i] == 0 else (H - 13, W - 13)
            r = base_r + rng.integers(-2, 3)
            c = base_c + rng.integers(-2, 3)
            x[i, :, r:r + 10, c:c + 10] = rng.uniform(0.7, 1.0)
        return ImageSplit(np.clip(x, 0, 1), y)

    return {"train": gen(n_train), "val": gen(n_val), "test": gen(n_test)}


def stratified_split(y: np.ndarray, val_fraction: float, seed: int):
    """Per-class shuffled split of the official training partition; returns (train_idx, val_idx)."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        idx = idx[rng.permutation(len(idx))]
        n_val = int(round(val_fraction * len(idx)))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    train_idx = np.sort(np.concatenate(train_idx))
    val_idx = np.sort(np.concatenate(val_idx))
    return train_idx, val_idx


def fingerprint(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def load_dataset(name: str, root: str = "data", val_fraction: float = 0.1, split_seed: int = 0,
                 download: bool = True, synthetic_kwargs: dict | None = None):
    """Returns (splits: dict[train/val/test -> ImageSplit], meta: dict with split identifiers)."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset '{name}'; choose from {sorted(DATASETS)}")
    meta = dict(DATASETS[name])
    meta["name"] = name
    if name.startswith("synthetic"):
        kw = dict(synthetic_kwargs or {})
        kw.setdefault("seed", split_seed)
        splits = make_synthetic(rgb=(name == "synthetic_rgb"), **kw)
        meta["split"] = {"kind": "synthetic", "seed": kw["seed"]}
    elif name == "bloodmnist":
        splits = _load_bloodmnist(root, download)
        meta["split"] = {"kind": "official_train_val_test"}
    else:
        raw = _load_torchvision(name, root, download)
        tr_idx, va_idx = stratified_split(raw["train"].y, val_fraction, split_seed)
        splits = {
            "train": ImageSplit(raw["train"].x[tr_idx], raw["train"].y[tr_idx]),
            "val": ImageSplit(raw["train"].x[va_idx], raw["train"].y[va_idx]),
            "test": raw["test"],
        }
        meta["split"] = {"kind": "stratified_from_official_train", "val_fraction": val_fraction,
                         "seed": split_seed, "val_index_fingerprint": fingerprint(va_idx),
                         "n_train": int(len(tr_idx)), "n_val": int(len(va_idx))}
    meta["sizes"] = {k: len(v) for k, v in splits.items()}
    return splits, meta


# --------------------------------------------------------------------------
# missingness
# --------------------------------------------------------------------------

def spawn_corruption_seeds(corruption_seed: int) -> dict:
    """One independent child seed per partition, derived deterministically."""
    children = np.random.SeedSequence(int(corruption_seed)).spawn(3)
    return {name: int(child.generate_state(1, dtype=np.uint32)[0])
            for name, child in zip(("train", "val", "test"), children)}


def generate_masks(n: int, channels: int, height: int, width: int, pattern: str, rate: float, seed: int) -> np.ndarray:
    """bool [n, C, H, W]; True = missing.  Spatial pattern shared by all colour channels."""
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pattern '{pattern}'; choose from {PATTERNS}")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    rng = np.random.default_rng(int(seed))
    spatial = np.zeros((n, height, width), dtype=bool)
    if rate > 0.0:
        if pattern == "random":
            spatial = rng.random((n, height, width)) < rate
        elif pattern == "block":
            area = rate * height * width
            bh = int(np.clip(round(np.sqrt(rate) * height), 1, height))
            bw = int(np.clip(round(area / bh), 1, width))
            tops = rng.integers(0, height - bh + 1, size=n)
            lefts = rng.integers(0, width - bw + 1, size=n)
            for i in range(n):
                spatial[i, tops[i]:tops[i] + bh, lefts[i]:lefts[i] + bw] = True
        elif pattern == "rows":
            k = int(np.clip(round(rate * height), 0, height))
            order = np.argsort(rng.random((n, height)), axis=1)[:, :k]
            for i in range(n):
                spatial[i, order[i], :] = True
        elif pattern == "cols":
            k = int(np.clip(round(rate * width), 0, width))
            order = np.argsort(rng.random((n, width)), axis=1)[:, :k]
            for i in range(n):
                spatial[i, :, order[i]] = True
    return np.repeat(spatial[:, None, :, :], channels, axis=1)


def save_masks(path: str, masks: np.ndarray, info: dict) -> None:
    np.savez_compressed(path, packed=np.packbits(masks.reshape(-1)), shape=np.asarray(masks.shape),
                        info=json.dumps(info))


def load_masks(path: str):
    z = np.load(path, allow_pickle=False)
    shape = tuple(int(s) for s in z["shape"])
    flat = np.unpackbits(z["packed"])[: int(np.prod(shape))].astype(bool)
    return flat.reshape(shape), json.loads(str(z["info"]))


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

class ImputationPreprocessor:
    """Train-only mean imputation + confidence construction; picklable via state_dict/save/load."""

    def __init__(self, tau: float = CONFIDENCE_TAU, channel_fallback_mean: float = CHANNEL_FALLBACK_MEAN):
        self.tau = float(tau)
        self.channel_fallback_mean = float(channel_fallback_mean)
        self.feature_mean = None      # [C, H, W]
        self.missing_rate = None      # [C, H, W]
        self.conf_missing = None      # [C, H, W]
        self.fallback_counts = None

    @property
    def fitted(self) -> bool:
        return self.feature_mean is not None

    def fit(self, x_train: np.ndarray, m_train: np.ndarray):
        x = np.asarray(x_train, dtype=np.float64)
        m = np.asarray(m_train, dtype=bool)
        if x.shape != m.shape:
            raise ValueError("x_train and m_train shapes differ")
        obs = ~m
        count = obs.sum(axis=0)                       # [C, H, W]
        total = (x * obs).sum(axis=0)                 # masked entries contribute nothing
        feature_mean = np.divide(total, np.maximum(count, 1))
        n_feature_fallback = int((count == 0).sum())
        n_channel_fallback = 0
        channel_count = count.sum(axis=(1, 2))        # [C]
        channel_total = total.sum(axis=(1, 2))
        for ch in range(x.shape[1]):
            missing_feats = count[ch] == 0
            if not missing_feats.any():
                continue
            if channel_count[ch] > 0:
                feature_mean[ch][missing_feats] = channel_total[ch] / channel_count[ch]
            else:
                feature_mean[ch][missing_feats] = self.channel_fallback_mean
                n_channel_fallback += int(missing_feats.sum())
        self.feature_mean = feature_mean.astype(np.float32)
        self.missing_rate = m.mean(axis=0).astype(np.float32)
        self.conf_missing = np.maximum(self.tau, 1.0 - self.missing_rate).astype(np.float32)
        self.fallback_counts = {"features_without_observed_train_entries": n_feature_fallback,
                                "features_using_channel_fallback_constant": n_channel_fallback}
        return self

    def transform(self, x: np.ndarray, m: np.ndarray):
        """Returns (X_imputed, M as float32, C) -- all float32 [N, C, H, W]."""
        if not self.fitted:
            raise RuntimeError("preprocessor not fitted")
        m = np.asarray(m, dtype=bool)
        x = np.asarray(x, dtype=np.float32)
        x_imp = np.where(m, self.feature_mean[None], x).astype(np.float32)
        conf = np.where(m, self.conf_missing[None], np.float32(1.0)).astype(np.float32)
        return x_imp, m.astype(np.float32), conf

    def state_dict(self) -> dict:
        return {"tau": self.tau, "channel_fallback_mean": self.channel_fallback_mean,
                "feature_mean": self.feature_mean, "missing_rate": self.missing_rate,
                "conf_missing": self.conf_missing, "fallback_counts": self.fallback_counts}

    def save(self, path: str) -> None:
        np.savez(path, tau=self.tau, channel_fallback_mean=self.channel_fallback_mean,
                 feature_mean=self.feature_mean, missing_rate=self.missing_rate,
                 conf_missing=self.conf_missing, fallback_counts=json.dumps(self.fallback_counts))

    @classmethod
    def load(cls, path: str) -> "ImputationPreprocessor":
        z = np.load(path, allow_pickle=False)
        p = cls(tau=float(z["tau"]), channel_fallback_mean=float(z["channel_fallback_mean"]))
        p.feature_mean = z["feature_mean"]
        p.missing_rate = z["missing_rate"]
        p.conf_missing = z["conf_missing"]
        p.fallback_counts = json.loads(str(z["fallback_counts"]))
        return p

    def fingerprint(self) -> str:
        return fingerprint(np.concatenate([self.feature_mean.ravel(), self.conf_missing.ravel()]))


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
