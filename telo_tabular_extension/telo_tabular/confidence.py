"""Cross-fitted imputation-success probabilities. Never accepts class labels."""
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from .data import MeanPreprocessor


class ConstantProbability:
    def __init__(self, p):
        self.p = float(p)

    def predict(self, x):
        return np.full(len(x), self.p)


class ReliabilityEstimator:
    def __init__(self, cfg, seed=42):
        self.cfg = dict(cfg)
        self.seed = int(seed)

    def _context(self, x, pre, feature):
        hidden = x.copy()
        hidden[:, feature] = np.nan
        z, m, _ = pre.transform(hidden)
        # Drop both target-value and target-mask columns. Target is always missing.
        other = np.arange(x.shape[1]) != feature
        return np.concatenate([z[:, other], m[:, other]], axis=1)

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        if len(x) < 6:
            raise ValueError("At least six rows are needed for confidence fitting")
        self.pre = MeanPreprocessor.fit(x)
        self.models, self.feature_priors = [], []
        self.fit_rows = len(x)
        inner = KFold(n_splits=self.cfg["inner_folds"], shuffle=True,
                      random_state=self.seed)
        folds = list(inner.split(x))
        rng = np.random.default_rng(self.seed + 913)
        examples = [[] for _ in range(x.shape[1])]
        targets = [[] for _ in range(x.shape[1])]
        self.inner_audit = []
        for fit_idx, hold_idx in folds:
            # Imputation targets and feature scales are derived without held-out rows.
            prep = MeanPreprocessor.fit(x[fit_idx])
            self.inner_audit.append({"fit": fit_idx.tolist(), "held_out": hold_idx.tolist()})
            for _ in range(self.cfg["mask_repeats"]):
                held = x[hold_idx]
                extra = rng.random(held.shape) < self.cfg["context_mask_rate"]
                context = np.where(extra, np.nan, held)
                for j in range(x.shape[1]):
                    known = np.isfinite(held[:, j])
                    if not known.any():
                        continue
                    success = (np.abs(held[known, j] - prep.mean[j]) <=
                               self.cfg["tolerance_std"] * prep.observed_std[j])
                    examples[j].append(self._context(context[known], prep, j))
                    targets[j].append(success.astype(int))
        for j in range(x.shape[1]):
            if not targets[j]:
                self.models.append(ConstantProbability(.5))
                self.feature_priors.append(.5)
                continue
            xx, yy = np.concatenate(examples[j]), np.concatenate(targets[j])
            prior = (float(yy.sum()) + 1) / (len(yy) + 2)
            self.feature_priors.append(prior)
            if len(np.unique(yy)) < 2 or len(yy) < self.cfg["min_examples"]:
                model = ConstantProbability(prior)
            else:
                model = HistGradientBoostingClassifier(
                    max_iter=self.cfg["max_iter"], max_leaf_nodes=self.cfg["max_leaf_nodes"],
                    min_samples_leaf=self.cfg["min_samples_leaf"],
                    l2_regularization=self.cfg["l2_regularization"],
                    learning_rate=.08, early_stopping=False, random_state=self.seed + j)
                with threadpool_limits(limits=1):
                    model.fit(xx, yy)
            self.models.append(model)
        self.feature_priors = np.asarray(self.feature_priors)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        out = np.ones(x.shape, dtype=np.float64)
        for j, model in enumerate(self.models):
            idx = np.flatnonzero(~np.isfinite(x[:, j]))
            if len(idx):
                context = self._context(x[idx], self.pre, j)
                with threadpool_limits(limits=1):
                    out[idx, j] = (model.predict(context) if isinstance(model, ConstantProbability)
                                   else model.predict_proba(context)[:, 1])
        if not np.isfinite(out).all() or np.any((out < 0) | (out > 1)):
            raise FloatingPointError("Invalid confidence probabilities")
        return out.astype(np.float32)


def cross_fit(x, cfg, seed):
    start = time.monotonic()
    out = np.ones(x.shape, dtype=np.float32)
    # Feature-constant controls are also fitted without each held-out row.
    feature_control = np.ones(x.shape, dtype=np.float32)
    audit = []
    kfold = KFold(n_splits=cfg["outer_folds"], shuffle=True, random_state=seed)
    for fold, (fit_idx, hold_idx) in enumerate(kfold.split(x)):
        estimator = ReliabilityEstimator(cfg, seed + fold * 101).fit(x[fit_idx])
        out[hold_idx] = estimator.predict(x[hold_idx])
        feature_control[hold_idx] = np.where(
            np.isfinite(x[hold_idx]), 1., estimator.feature_priors)
        audit.append({"fold": fold, "fit": fit_idx.tolist(), "held_out": hold_idx.tolist()})
    final = ReliabilityEstimator(cfg, seed + 10001).fit(x)
    return out, feature_control, final, audit, time.monotonic() - start


def shuffle_missing(c, m, seed):
    rng = np.random.default_rng(seed)
    out = c.copy()
    for j in range(c.shape[1]):
        idx = np.flatnonzero(m[:, j] > .5)
        out[idx, j] = rng.permutation(out[idx, j])
    return out


def confidence_metrics(clean, masked, c, pre, tolerance_std):
    """Scores only artificially hidden entries with known ground truth."""
    eligible = ~np.isfinite(masked) & np.isfinite(clean)
    if not eligible.any():
        return {"n": 0, "brier": None, "event_auc": None, "bins": []}
    error = np.abs(clean - pre.mean) / pre.observed_std
    target = (error[eligible] <= tolerance_std).astype(int)
    prob = c[eligible].astype(float)
    if not np.isfinite(prob).all():
        raise FloatingPointError("Non-finite confidence")
    bins = []
    for lo in np.linspace(0, .9, 10):
        hi = lo + .1
        take = (prob >= lo) & ((prob < hi) if hi < .999 else (prob <= 1))
        if take.any():
            bins.append({"lower": float(lo), "upper": float(hi), "n": int(take.sum()),
                         "predicted": float(prob[take].mean()), "observed": float(target[take].mean())})
    ece = sum(b["n"] * abs(b["predicted"] - b["observed"]) for b in bins) / len(prob)
    auc = float(roc_auc_score(target, prob)) if len(np.unique(target)) == 2 else None
    return {"n": len(prob), "brier": float(np.mean((prob - target) ** 2)),
            "event_auc": auc, "ece": float(ece),
            "mean_normalized_absolute_error": float(error[eligible].mean()), "bins": bins}
