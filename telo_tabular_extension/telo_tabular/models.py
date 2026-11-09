"""Original MLP/GP components with explicit numerical checks and metadata ablations."""
import copy
import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from . import gp_core as g


class PilotMLP(nn.Module):
    def __init__(self, inputs, classes, hidden, tree, propagation="full"):
        super().__init__()
        if propagation not in ("full", "first_only"):
            raise ValueError("Unknown propagation mode")
        self.network = g.ThreeChannelMLP(inputs, hidden, classes, tree)
        self.propagation = propagation

    def forward(self, x, m, c):
        for i, block in enumerate(self.network.blocks):
            if i and self.propagation == "first_only":
                m, c = torch.zeros_like(m), torch.ones_like(c)
            x, m, c = block(x, m, c)
        return self.network.out(x)


def fixed_tree(name):
    x = g.create_input_node("value")
    if name == "relu":
        return g.create_unary_node("relu", x)
    if name == "swish":
        return g.create_binary_node("multiply", x, g.create_unary_node("sigmoid", x))
    if name in ("elu", "leaky_relu"):
        return g.create_unary_node(name, x)
    raise ValueError(name)


def tensors(arrays, device):
    x, m, c, y = arrays
    return tuple(torch.as_tensor(a, dtype=torch.long if i == 3 else torch.float32,
                                 device=device) for i, a in enumerate((x, m, c, y)))


@torch.no_grad()
def predict(model, data):
    model.eval()
    logits = model(*data[:3])
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite logits")
    probs = torch.softmax(logits, dim=1)
    if not torch.isfinite(probs).all():
        raise FloatingPointError("Non-finite prediction probabilities")
    return probs.cpu().numpy()


def scores(y, probs):
    if not np.isfinite(probs).all():
        raise FloatingPointError("Non-finite predictions cannot be scored")
    pred = probs.argmax(axis=1)
    auc = None
    if len(np.unique(y)) == probs.shape[1]:
        auc = float(roc_auc_score(y, probs[:, 1])) if probs.shape[1] == 2 else float(
            roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)), "auc": auc}


def train(model, train_data, val_data, cfg, *, final=False, deadline=None):
    start = time.monotonic()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=.5, patience=5) if final else None
    criterion = nn.CrossEntropyLoss()
    best, best_epoch, best_state, stale = -1., 0, None, 0
    history, failure = [], None
    try:
        for epoch in range(cfg["epochs"]):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Search time budget reached during candidate")
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(*train_data[:3]), train_data[3])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError("Non-finite gradient")
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip"],
                                     error_if_nonfinite=True)
            opt.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Non-finite parameter after optimizer step")
            val = scores(val_data[3].cpu().numpy(), predict(model, val_data))["accuracy"]
            if sched:
                sched.step(val)
            history.append({"epoch": epoch + 1, "loss": float(loss.detach()), "val_accuracy": val})
            if val > best:
                best, best_epoch, stale = val, epoch + 1, 0
                if final:
                    best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= cfg["patience"]:
                break
    except (FloatingPointError, TimeoutError, RuntimeError) as exc:
        failure = str(exc)
    if final and best_state is not None:
        model.load_state_dict(best_state)
    return {"valid": failure is None and best >= 0, "failure": failure,
            "best_val_accuracy": max(best, 0.), "best_epoch": best_epoch,
            "epochs_run": len(history), "runtime_s": time.monotonic() - start, "history": history}


@torch.no_grad()
def channel_trace(model, data):
    x, m, c = data[:3]
    rows = []
    for i, block in enumerate(model.network.blocks):
        if i and model.propagation == "first_only":
            m, c = torch.zeros_like(m), torch.ones_like(c)
        z, m, c = block.prop(x, m, c)
        x = block.act(z, m, c)
        rows.append({"layer": i + 1, "confidence_min": float(c.min()),
                     "confidence_max": float(c.max()),
                     "confidence_mean": float(c.mean()),
                     "confidence_between_rows_std": float(c.std(dim=0, unbiased=False).mean()),
                     "missingness_between_rows_std": float(m.std(dim=0, unbiased=False).mean())})
    return rows
