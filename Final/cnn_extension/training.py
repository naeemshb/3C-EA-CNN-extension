"""
Minibatch training and evaluation for the three-stream backbones.

Optimiser and safeguards follow the original scripts: Adam(lr=0.001,
weight_decay=1e-4), gradient-norm clipping at 1.0, validation accuracy
tracked after every epoch, early stopping on validation accuracy, and
(final training only) ReduceLROnPlateau(mode="max", factor=0.5, patience=5)
with the best-validation checkpoint restored.

Deviation from the original (documented in README): the original scripts
train tabular data with ONE full-batch step per epoch.  Images do not fit
that regime, so an epoch here is one pass over shuffled minibatches whose
order is drawn from a dedicated, recorded generator (``batch_seed``).
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TensorData:
    x: torch.Tensor
    m: torch.Tensor
    c: torch.Tensor
    y: torch.Tensor

    def __len__(self):
        return int(self.y.shape[0])

    def to(self, device):
        return TensorData(self.x.to(device), self.m.to(device), self.c.to(device), self.y.to(device))

    def subset(self, idx):
        idx = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=self.y.device)
        return TensorData(self.x[idx], self.m[idx], self.c[idx], self.y[idx])

    @classmethod
    def from_numpy(cls, x, m, c, y):
        return cls(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(m, dtype=torch.float32),
                   torch.as_tensor(c, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model: nn.Module, data: TensorData, batch_size: int = 1024):
    model.eval()
    probs = []
    for start in range(0, len(data), batch_size):
        sl = slice(start, start + batch_size)
        logits = model(data.x[sl], data.m[sl], data.c[sl])
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
    return torch.cat(probs, dim=0).numpy()


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 over the classes present in y_true or y_pred (sklearn convention), zero_division=0."""
    labels = np.union1d(y_true, y_pred)
    f1s = []
    for lab in labels:
        tp = np.sum((y_pred == lab) & (y_true == lab))
        fp = np.sum((y_pred == lab) & (y_true != lab))
        fn = np.sum((y_pred != lab) & (y_true == lab))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def evaluate_model(model: nn.Module, data: TensorData, batch_size: int = 1024) -> dict:
    probs = predict(model, data, batch_size)
    preds = probs.argmax(axis=1)
    labels = data.y.cpu().numpy()
    return {"accuracy": float((preds == labels).mean()), "macro_f1": macro_f1(labels, preds),
            "probs": probs, "preds": preds, "labels": labels}


def train_model(model: nn.Module, train: TensorData, val: TensorData, *, epochs: int, patience: int,
                batch_size: int, batch_seed: int, lr: float = 1e-3, weight_decay: float = 1e-4,
                grad_clip: float = 1.0, plateau_scheduler: bool = False, keep_best_state: bool = True,
                log=None) -> dict:
    """Train with early stopping on validation accuracy.  Returns a dict with best_val_acc, best_state, history, failure."""
    device = train.y.device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
                 if plateau_scheduler else None)
    gen = torch.Generator(device="cpu").manual_seed(int(batch_seed))

    best_val_acc, best_epoch, best_state = 0.0, -1, None
    patience_counter, failure, history = 0, None, []
    n = len(train)
    t0 = time.time()
    epochs_run = 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        loss_sum, seen = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(train.x[idx], train.m[idx], train.c[idx])
            loss = criterion(logits, train.y[idx])
            if not torch.isfinite(loss):
                failure = f"non-finite loss at epoch {epoch + 1}"
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(idx)
            seen += len(idx)
        epochs_run = epoch + 1
        if failure:
            break
        val_acc = evaluate_model(model, val)["accuracy"]
        if not np.isfinite(val_acc):
            failure = f"non-finite validation accuracy at epoch {epoch + 1}"
            break
        if scheduler is not None:
            scheduler.step(val_acc)
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / max(seen, 1), "val_acc": val_acc})
        if log and ((epoch + 1) % 10 == 0 or epoch == 0):
            log(f"    epoch {epoch + 1:3d}/{epochs}  loss {loss_sum / max(seen, 1):.4f}  val_acc {val_acc:.4f}  best {max(best_val_acc, val_acc):.4f}")
        if val_acc > best_val_acc:
            best_val_acc, best_epoch, patience_counter = val_acc, epoch + 1, 0
            if keep_best_state:
                best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    if keep_best_state and best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_acc": float(best_val_acc), "best_epoch": best_epoch, "epochs_run": epochs_run,
            "history": history, "failure": failure, "runtime_s": time.time() - t0}
