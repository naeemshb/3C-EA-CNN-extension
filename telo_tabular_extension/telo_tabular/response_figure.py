"""Response of four Hepatitis winners of the original runs, drawn at the channel values their hidden units receive.

    python -m telo_tabular.response_figure --out figures --json runs_past_winners/response_states.json

The four functions are runs 1, 19, 24 and 23 of the published table of Hepatitis winners. Each is trained on SPLITS
splits of Hepatitis (natural missingness only; package split, network and final-training budget), and the pre-activation z
and the propagated channels (m, c) entering the activation of every hidden unit are recorded for every record.
The curves are drawn at four reliability states of the first hidden layer, pooled over the four functions:
  - no input of the unit is missing (m = 0, c = 1);
  - the 75th, 95th and 99th percentile of m over all unit-record pairs, each with the mean c of the 1 % of pairs nearest
    to that m (m and c are coupled: c can fall below 1 only where m is above 0).
The grey band in each panel is the central 98 % of the pre-activations that function's first layer receives.
The json holds the per-function, per-layer distributions, the states, the largest change of each function between the
fully observed state and every other state, and how often the values of a fixed-slice plot (m = 0.8, c <= 0.9) occur.
"""
import argparse, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from . import gp_core as g
from .data import load_dataset
from .models import build_model, tensors, train
from .make_figures import INK2, style_axis, save
from .past_winners import Split, FINAL, HIDDEN, CONFIRM_SEEDS

SPLITS = CONFIRM_SEEDS[:30]
QUANTILES = [75, 95, 99]
X, M, C = (g.create_input_node(k) for k in ("value", "missing", "confidence"))
B, U = g.create_binary_node, g.create_unary_node
RUNS = [(1, r"$x^{2}c$", B("multiply", U("square", X), C)),
        (19, r"$x/c$", B("divide", X, C)),
        (24, r"$x+m$", B("add", M, X)),
        (23, r"$x\,(x-(x-c)\,m)-x$", B("subtract", B("multiply", X, B("subtract", X, B("multiply", B("subtract", X, U("identity", C)), M))), X))]
STYLE = [(INK2, "-", 2.0), ("#5d9be6", (0, (5, 1.6)), 1.5), ("#2a78d6", (0, (4, 1.4, 1, 1.4)), 1.5), ("#0f3f7d", (0, (1, 1.1)), 1.7)]
BANDC = "#f0efec"


def record(tree, x, y):
    """z, m, c entering the activation of each hidden layer, all records of every split; number of invalid trainings."""
    layers, bad = {1: [[], [], []], 2: [[], [], []]}, 0
    for seed in SPLITS:
        sp = Split(x, y, seed); tr, va = sp.parts["train"], sp.parts["val"]
        torch.manual_seed(seed + 10000); model = build_model("three_channel", tr[0].shape[1], sp.classes, HIDDEN, tree, "full")
        if not train(model, tensors(tr, "cpu"), tensors(va, "cpu"), FINAL, final=True)["valid"]:
            bad += 1; continue
        with torch.no_grad():
            h, m, c = (torch.tensor(np.concatenate([sp.parts[k][i] for k in ("train", "val", "test")])) for i in range(3))
            for i, blk in enumerate(model.network.blocks):
                z, m, c = blk.prop(h, m, c); h = blk.act(z, m, c)
                for store, v in zip(layers[i + 1], (z, m, c)): store.append(v.numpy().ravel())
    return {k: [np.concatenate(v) for v in vs] for k, vs in layers.items()}, bad


def describe(z, m, c):
    q = [1, 5, 25, 50, 75, 95, 99]
    return {"n_pairs": int(len(m)), "z_percentiles": dict(zip(map(str, q), np.percentile(z, q).round(4).tolist())),
            "m_percentiles": dict(zip(map(str, q), np.percentile(m, q).round(4).tolist())), "m_max": float(m.max()),
            "c_percentiles": dict(zip(map(str, q), np.percentile(c, q).round(4).tolist())), "c_min": float(c.min()),
            "share_m_zero": float((m < 1e-6).mean()), "share_m_at_least_0.8": float((m >= .8).mean()),
            "share_c_at_most_0.9": float((c <= .9).mean()), "share_c_at_most_0.7": float((c <= .7).mean())}


def states_of(m, c):
    out = [(0.0, 1.0)]
    for qq in np.percentile(m, QUANTILES):
        near = np.argsort(np.abs(m - qq))[: max(1, len(m) // 100)]; out.append((float(qq), float(c[near].mean())))
    return out


def f(tree, z, m, c):
    t = torch.tensor(z, dtype=torch.float32)
    return g.evaluate_tree(tree, t, torch.full_like(t, m), torch.full_like(t, c)).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--json", required=True)
    args = ap.parse_args(argv); torch.set_num_threads(2); x, y, meta = load_dataset("hepatitis"); miss = np.isnan(x)
    rec = {run: record(tree, x, y) for run, _, tree in RUNS}
    m1 = np.concatenate([rec[run][0][1][1] for run, *_ in RUNS]); c1 = np.concatenate([rec[run][0][1][2] for run, *_ in RUNS]); states = states_of(m1, c1)
    out = {"dataset": {"records": int(len(x)), "features": int(x.shape[1]), "classes": meta["classes"], "class_counts": np.bincount(y).tolist(),
                       "missing_rate": float(miss.mean()), "feature_missing_rates": miss.mean(0).round(4).tolist(),
                       "share_records_without_missing": float((miss.sum(1) == 0).mean()), "lowest_input_confidence": float(1 - miss.mean(0).max())},
           "splits": SPLITS, "state_quantiles_of_m": QUANTILES, "states_m_c": states, "layer1_pooled": describe(np.concatenate([rec[r][0][1][0] for r, *_ in RUNS]), m1, c1), "runs": {}}
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.0)); zz = np.linspace(-2, 2, 401)
    for (run, tex, tree), ax, letter in zip(RUNS, axes.ravel(), "abcd"):
        layers, bad = rec[run]; lo, hi = np.percentile(layers[1][0], [1, 99]); zin = np.linspace(lo, hi, 401)
        ax.axvspan(lo, hi, color=BANDC, zorder=0); ax.axhline(0, color=INK2, linewidth=0.5, zorder=1)
        ref = f(tree, zin, *states[0]); change = []
        for (mm, cc), (col, ls, lw) in zip(states, STYLE):
            ax.plot(zz, f(tree, zz, mm, cc), color=col, linestyle=ls, linewidth=lw, zorder=3)
            d = np.abs(f(tree, zin, mm, cc) - ref); change.append({"m": mm, "c": cc, "max_abs_change": float(d.max()), "relative_to_max_abs_reference": float(d.max() / np.abs(ref).max())})
        ax.set_xlim(-2, 2); ax.set_xlabel("Pre-activation $x$"); ax.set_ylabel("$f(x, m, c)$"); style_axis(ax)
        ax.set_title(f"({letter}) Run {run}: {tex}", loc="left")
        out["runs"][str(run)] = {"infix": g.tree_to_infix(tree), "invalid_trainings": bad, "own_states_m_c": states_of(layers[1][1], layers[1][2]),
                                 "change_from_fully_observed_within_received_range": change[1:],
                                 "layer1": describe(*layers[1]), "layer2": describe(*layers[2])}
    names = ["no input missing"] + [f"{q}th percentile of $m$" for q in QUANTILES]
    handles = [Line2D([], [], color=col, linestyle=ls, linewidth=lw, label=f"$m={mm:.2f}$, $c={cc:.2f}$: {nm}") for (mm, cc), (col, ls, lw), nm in zip(states, STYLE, names)]
    handles.append(Patch(facecolor=BANDC, edgecolor="none", label="Pre-activations received (central 98%)"))
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.085), handlelength=2.8, columnspacing=1.6, fontsize=7.2)
    fig.tight_layout(w_pad=1.3, h_pad=1.2); save(fig, args.out, "fig_response_examples")
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True); json.dump(out, open(args.json, "w"), indent=1)
    short = {k: v for k, v in out.items() if k != "runs"}; short["change"] = {r: v["change_from_fully_observed_within_received_range"] for r, v in out["runs"].items()}
    short["own_states"] = {r: v["own_states_m_c"] for r, v in out["runs"].items()}; print(json.dumps(short, indent=1))


if __name__ == "__main__":
    main()
