"""Evolved functions drawn at the channel values their hidden units receive, with the evidence for the confirmed one.

    python -m telo_tabular.response_figure --confirm runs_past_winners/confirm.json --out figures --json runs_past_winners/response_states.json

Top row: the Cylinder Bands winner that was confirmed to need both channels (third confirmed candidate in confirm.json),
  (a) its response at four reliability states and (b) the paired accuracy differences of the confirmation (100 fresh splits)
  between the function with both channels and the same function with one or both channels neutralised, and the best fixed
  activation; 95 % paired t-intervals.
Bottom row: runs 24 and 23 of the published table of Hepatitis winners, (c) and (d), at four reliability states.
Runs 1 and 19 of that table (functions of the value and the confidence only) are measured too and reported in the json.

How the states are obtained: every function is trained on SPLITS splits of its dataset (natural missingness only; package
split, network and final-training budget) and the pre-activation z and the propagated channels (m, c) entering the
activation of every hidden unit are recorded for every record. A state is a percentile of m over all unit-record pairs of
the first hidden layer together with the mean c of the 1 % of pairs nearest to that m (m and c are coupled: c can fall
below 1 only where m is above 0). Hepatitis: m = 0, c = 1 (no input of the unit missing) and the 75th, 95th and 99th
percentiles, pooled over the four functions. Cylinder Bands: the 5th, 50th, 95th and 99th percentiles (no unit is free of
missing inputs there). The grey band in a panel is the central 98 % of the pre-activations that function's first layer receives.
"""
import argparse, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from . import gp_core as g
from .data import load_dataset
from .models import build_model, tensors, train
from .make_figures import BLUE, INK2, style_axis, save
from .past_winners import Split, FINAL, HIDDEN, CONFIRM_SEEDS

SPLITS = CONFIRM_SEEDS[:30]
X, M, C = (g.create_input_node(k) for k in ("value", "missing", "confidence"))
B, U = g.create_binary_node, g.create_unary_node
RUNS = [(1, r"$x^{2}c$", B("multiply", U("square", X), C)),
        (19, r"$x/c$", B("divide", X, C)),
        (24, r"$x+m$", B("add", M, X)),
        (23, r"$x\,(x-(x-c)\,m)-x$", B("subtract", B("multiply", X, B("subtract", X, B("multiply", B("subtract", X, U("identity", C)), M))), X))]
DRAWN = (24, 23)
STYLE = [(INK2, "-", 2.0), ("#5d9be6", (0, (5, 1.6)), 1.5), ("#2a78d6", (0, (4, 1.4, 1, 1.4)), 1.5), ("#0f3f7d", (0, (1, 1.1)), 1.7)]
BANDC = "#f0efec"
CONTRASTS = [("m_only", "Both vs. $m$ only"), ("c_only", "Both vs. $c$ only"), ("neither", "Both vs. no channels"), (None, "Both vs. best fixed")]


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
            "m_percentiles": dict(zip(map(str, q), np.percentile(m, q).round(4).tolist())), "m_min": float(m.min()), "m_max": float(m.max()),
            "c_percentiles": dict(zip(map(str, q), np.percentile(c, q).round(4).tolist())), "c_min": float(c.min()),
            "share_m_zero": float((m < 1e-6).mean()), "share_m_at_least_0.8": float((m >= .8).mean()),
            "share_c_at_most_0.9": float((c <= .9).mean()), "share_c_at_most_0.7": float((c <= .7).mean()),
            "share_c_minus_m_below_0.2": float((c - m < .2).mean())}


def states_of(m, c, quantiles, zero_state):
    out = [(0.0, 1.0)] if zero_state else []
    for qq in np.percentile(m, quantiles):
        near = np.argsort(np.abs(m - qq))[: max(1, len(m) // 100)]; out.append((float(qq), float(c[near].mean())))
    return out


def f(tree, z, m, c):
    t = torch.tensor(z, dtype=torch.float32)
    return g.evaluate_tree(tree, t, torch.full_like(t, m), torch.full_like(t, c)).numpy()


def dataset_facts(x, y, meta):
    miss = np.isnan(x)
    return {"records": int(len(x)), "features": int(x.shape[1]), "classes": meta["classes"], "class_counts": np.bincount(y).tolist(),
            "missing_rate": float(miss.mean()), "feature_missing_rates": miss.mean(0).round(4).tolist(),
            "share_records_without_missing": float((miss.sum(1) == 0).mean()), "lowest_input_confidence": float(max(.1, 1 - miss.mean(0).max()))}


def curves(ax, tree, layers, states, zz):
    """Draw the function at the states; return the largest change from the first state within the received pre-activations."""
    lo, hi = np.percentile(layers[1][0], [1, 99]); zin = np.linspace(lo, hi, 401)
    ax.axvspan(lo, hi, color=BANDC, zorder=0); ax.axhline(0, color=INK2, linewidth=0.5, zorder=1)
    ref = f(tree, zin, *states[0]); change = []
    for (mm, cc), (col, ls, lw) in zip(states, STYLE):
        ax.plot(zz, f(tree, zz, mm, cc), color=col, linestyle=ls, linewidth=lw, zorder=3)
        d = np.abs(f(tree, zin, mm, cc) - ref); change.append({"m": mm, "c": cc, "max_abs_change": float(d.max()), "relative_to_max_abs_first_state": float(d.max() / np.abs(ref).max())})
    ax.set_xlim(zz[0], zz[-1]); ax.set_xlabel("Pre-activation $x$"); ax.set_ylabel("$f(x, m, c)$"); style_axis(ax)
    return change[1:]


def handles(states, names):
    return [Line2D([], [], color=col, linestyle=ls, linewidth=lw, label=f"$m={mm:.2f}$, $c={cc:.2f}$ ({nm})") for (mm, cc), (col, ls, lw), nm in zip(states, STYLE, names)]


def row_legend(fig, ax, hs):
    """One legend line above a row of panels, outside the plot area (the curves fill the panels)."""
    fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, ax.get_position().y1 + 0.052), fontsize=6.6, handlelength=2.4, columnspacing=1.2, handletextpad=0.5)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--confirm", required=True); ap.add_argument("--out", required=True); ap.add_argument("--json", required=True)
    args = ap.parse_args(argv); torch.set_num_threads(2); zz = np.linspace(-2, 2, 401)
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.3)); a, b, c, d = axes.ravel(); out = {"splits": SPLITS}

    # (a) Cylinder Bands winner B at its measured states
    rowB = json.load(open(args.confirm))["datasets"]["cylinderband"][2]; x, y, meta = load_dataset("cylinderband"); layers, bad = record(rowB["tree"], x, y)
    qB = [5, 50, 95, 99]; sB = states_of(layers[1][1], layers[1][2], qB, False); changeB = curves(a, rowB["tree"], layers, sB, zz)
    a.set_title(r"(a) Cylinder Bands, winner B: $\frac{m-x}{c\,(c-m)}$", loc="left")
    grid = [torch.tensor(v, dtype=torch.float32) for v in np.meshgrid(np.linspace(-3, 3, 61), np.linspace(0, .6, 31), np.linspace(.85, 1, 16))]
    closed = (grid[1] - grid[0]) / (grid[2] * (grid[2] - grid[1])); err = float((g.evaluate_tree(rowB["tree"], *grid) - closed).abs().max())

    # (b) the confirmation: paired differences on 100 fresh splits
    runs = {k: np.array(v) for k, v in rowB["runs"].items()}; runs[None] = np.array(rowB["runs_best_fixed"]); ok = np.isfinite(np.column_stack(list(runs.values()))).all(axis=1); diffs = []
    for i, (v, lab) in enumerate(CONTRASTS):
        dd = (runs["both"][ok] - runs[v][ok]) * 100; mean = dd.mean(); h = stats.t.ppf(.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd)); sig = mean - h > 0 or mean + h < 0
        b.plot([mean - h, mean + h], [-i, -i], color=BLUE, linewidth=1.3, solid_capstyle="round", zorder=2)
        b.plot(mean, -i, marker="o", markersize=5.2, color=BLUE, markerfacecolor=BLUE if sig else "white", markeredgewidth=1.1, zorder=3)
        diffs.append({"contrast": lab.replace("$", ""), "mean_points": float(mean), "ci95_low": float(mean - h), "ci95_high": float(mean + h), "n": int(len(dd)),
                      "p_wilcoxon": rowB["p_both_vs_" + (v if v else "best_fixed")]})
    b.axvline(0, color=INK2, linewidth=0.7, zorder=1); b.set_yticks([-i for i in range(len(CONTRASTS))])
    b.set_yticklabels([lab if v else f"{lab} ({rowB['best_fixed'].upper()})" for v, lab in CONTRASTS], fontsize=7); b.tick_params(axis="y", length=0)
    b.set_ylim(-len(CONTRASTS) + 0.4, 0.6); b.set_xlabel("Paired accuracy difference (points)"); style_axis(b, "x"); b.set_title("(b) Winner B on 100 fresh splits", loc="left")
    out["cylinderband_winner_B"] = {"infix": rowB["infix"], "closed_form": "(m - x) / (c * (c - m))", "max_abs_error_of_closed_form_on_grid": err, "dataset": dataset_facts(x, y, meta),
                                    "invalid_trainings": bad, "state_percentiles_of_m": qB, "states_m_c": sB, "change_from_first_state_within_received_range": changeB,
                                    "layer1": describe(*layers[1]), "layer2": describe(*layers[2]), "paired_differences": diffs,
                                    "accuracy": {k: rowB[k]["mean"] for k in ("both", "m_only", "c_only", "neither")} | {"best_fixed": rowB["fixed"][rowB["best_fixed"]]}}

    # (c), (d) Hepatitis runs 24 and 23 at the states pooled over the four published functions
    x, y, meta = load_dataset("hepatitis"); rec = {run: record(tree, x, y) for run, _, tree in RUNS}; qH = [75, 95, 99]
    m1 = np.concatenate([rec[run][0][1][1] for run, *_ in RUNS]); c1 = np.concatenate([rec[run][0][1][2] for run, *_ in RUNS]); sH = states_of(m1, c1, qH, True)
    out["hepatitis"] = {"dataset": dataset_facts(x, y, meta), "state_percentiles_of_m": qH, "states_m_c": sH,
                        "layer1_pooled": describe(np.concatenate([rec[r][0][1][0] for r, *_ in RUNS]), m1, c1), "runs": {}}
    panels = dict(zip(DRAWN, ((c, "c"), (d, "d")))); scratch = plt.figure().subplots()
    for run, tex, tree in RUNS:
        layers, bad = rec[run]; ax, letter = panels.get(run, (scratch, None)); change = curves(ax, tree, layers, sH, zz)
        if letter: ax.set_title(f"({letter}) Hepatitis, run {run}: {tex}", loc="left")
        out["hepatitis"]["runs"][str(run)] = {"infix": g.tree_to_infix(tree), "drawn": letter is not None, "invalid_trainings": bad,
                                              "own_states_m_c": states_of(layers[1][1], layers[1][2], qH, True), "change_from_first_state_within_received_range": change,
                                              "layer1": describe(*layers[1]), "layer2": describe(*layers[2])}
    d.text(0.25, 0.93, "central 98% of the\npre-activations received", transform=d.get_xaxis_transform(), fontsize=6.6, color=INK2, ha="center", va="top", linespacing=1.25)
    fig.tight_layout(w_pad=1.3, h_pad=3.6, rect=(0, 0, 1, 0.955))
    row_legend(fig, a, handles(sB, [f"{q}th pct." for q in qB])); row_legend(fig, c, handles(sH, ["none missing"] + [f"{q}th pct." for q in qH]))
    save(fig, args.out, "fig_response_examples")
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True); json.dump(out, open(args.json, "w"), indent=1)
    wb = out["cylinderband_winner_B"]; print(json.dumps({k: wb[k] for k in ("closed_form", "max_abs_error_of_closed_form_on_grid", "invalid_trainings", "states_m_c",
          "change_from_first_state_within_received_range", "layer1", "paired_differences", "accuracy")}, indent=1))
    print(json.dumps({"hepatitis_states": sH, "change": {r: v["change_from_first_state_within_received_range"] for r, v in out["hepatitis"]["runs"].items()}}, indent=1))


if __name__ == "__main__":
    main()
