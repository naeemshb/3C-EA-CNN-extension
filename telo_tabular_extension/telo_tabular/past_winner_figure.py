"""Figure for the confirmed winners of the original runs (reads runs_past_winners/confirm.json; retrains only for panel c).

    python -m telo_tabular.past_winner_figure --confirm runs_past_winners/confirm.json --out figures

(a) the Cylinder Bands winner that needs both channels, f = (m - x) / (c (c - m)), at three reliability states taken from
    the propagated channel values the first hidden layer actually receives (5 %, 50 %, 95 % quantiles of m)
(b) the three confirmed winners retrained on 100 fresh splits with both channels, m only, c only and neither, and the best
    fixed activation; mean accuracy with 95 % t-intervals
(c) accuracy against the number of missing features in a record for the winner with the largest channel effect
Also writes past_winner_case.json with the numbers behind panels (a) and (c).
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
from .data import load_dataset, split_indices
from .models import build_model, tensors, train, predict, fixed_tree
from .make_figures import BLUE, ORANGE, AQUA, INK2, GRID, style_axis, save
from .past_winners import Split, FINAL, HIDDEN, CONFIRM_SEEDS
from .winner_analysis import neutralise

RAMP = ["#0f3f7d", "#2a78d6", "#9cc3f2"]
PICKS = [("cylinderband", 2, "Cylinder Bands, winner B"), ("cylinderband", 0, "Cylinder Bands, winner A"), ("horse", 2, "Horse Colic, winner C")]
VERS = [("both", "Both channels", BLUE, "o"), ("m_only", "m only", BLUE, "v"), ("c_only", "c only", BLUE, "D"), ("neither", "No channels", ORANGE, "s"), (None, "Best fixed", AQUA, "^")]


def fit(split, tree, version):
    tr, va = ((p[0], *neutralise(p[1], p[2], version), p[3]) for p in (split.parts[k] for k in ("train", "val")))
    torch.manual_seed(split.seed + 10000); model = build_model("three_channel", tr[0].shape[1], split.classes, HIDDEN, tree, "full")
    train(model, tensors(tr, "cpu"), tensors(va, "cpu"), FINAL, final=True); return model


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--confirm", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(argv); conf = json.load(open(args.confirm))["datasets"]; torch.set_num_threads(2); case = {}
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.5), gridspec_kw={"width_ratios": [1, 1.3, 1]}); a, b, c = axes

    # (a) winner B: closed form and the channel values layer 1 receives
    rowB = conf["cylinderband"][2]; tree = rowB["tree"]; x_all, y_all, _ = load_dataset("cylinderband")
    rng = np.random.default_rng(0); gx, gm = rng.uniform(-3, 3, 5000), rng.uniform(0, .4, 5000); gc = rng.uniform(gm + .2, 1, 5000)
    tx, tm, tc = (torch.tensor(v, dtype=torch.float32) for v in (gx, gm, gc))
    err = float((g.evaluate_tree(tree, tx, tm, tc) - (tm - tx) / (tc * (tc - tm))).abs().max())
    sp = Split(x_all, y_all, CONFIRM_SEEDS[0]); model = fit(sp, tree, "both"); te = sp.parts["test"]
    with torch.no_grad():
        _, m1, c1 = model.network.blocks[0].prop(*(torch.tensor(v) for v in te[:3]))
    m1, c1 = m1.numpy().ravel(), c1.numpy().ravel(); q = np.quantile(m1, [.05, .5, .95]); states = []
    for qq in q:      # the confidence that accompanies each missingness level (mean of c among the nearest 5 % of units)
        near = np.argsort(np.abs(m1 - qq))[: max(1, len(m1) // 20)]; states.append((float(qq), float(c1[near].mean())))
    z = np.linspace(-2, 2, 200)
    for (mm, cc), col in zip(states, RAMP):
        a.plot(z, (mm - z) / (cc * (cc - mm)), color=col, linewidth=1.6, label=f"$m={mm:.2f}$, $c={cc:.2f}$")
    a.axhline(0, color=INK2, linewidth=0.5); a.set_xlabel("Pre-activation $z$"); a.set_ylabel(r"$f(z, m, c)$"); style_axis(a)
    a.set_title("(a) Winner B", loc="left"); a.legend(loc="upper right", frameon=False, fontsize=6.6, handlelength=1.5, title="Reliability state", title_fontsize=6.6)
    case["winner_B"] = {"infix": rowB["infix"], "closed_form": "(m - x) / (c * (c - m))", "max_abs_error_on_grid": err,
                        "layer1_m_quantiles_5_50_95": [float(v) for v in q], "states_m_c": states,
                        "share_units_c_minus_m_below_0.2": float((c1 - m1 < .2).mean())}

    # (b) ablation on 100 fresh splits
    ypos, labels, y0 = [], [], 0.
    for ds, i, title in PICKS:
        r = conf[ds][i]; ok = np.isfinite(np.column_stack([r["runs"][v] for v, *_ in VERS[:4]])).all(axis=1)
        for k, (v, lab, col, mk) in enumerate(VERS):
            arr = np.array(r["runs"][v] if v else r["runs_best_fixed"])[ok] * 100; mean = arr.mean()
            h = stats.t.ppf(.975, len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr)); yy = y0 - k
            b.plot([mean - h, mean + h], [yy, yy], color=col, linewidth=1.1, solid_capstyle="round", zorder=2)
            b.plot(mean, yy, marker=mk, markersize=4.6, color=col, markerfacecolor=col if v in ("both", "neither", None) else "white", markeredgewidth=1.0, zorder=3)
            ypos.append(yy); labels.append(lab)
        b.text(0.02, y0 + 1.15, title, transform=b.get_yaxis_transform(), fontsize=7, color=INK2, ha="left", va="center", fontweight="bold")
        y0 -= len(VERS) + 2.2
    b.set_yticks(ypos); b.set_yticklabels(labels, fontsize=6.6); b.tick_params(axis="y", length=0); style_axis(b, "x")
    b.set_xlabel("Test accuracy (%), 100 fresh splits"); b.set_title("(b) Channel ablation", loc="left"); b.set_ylim(min(ypos) - 0.8, 2.0)

    # (c) winner A by number of missing features
    rowA = conf["cylinderband"][0]; bins = [(1, 1), (2, 2), (3, 6), (7, 25)]; hit = {v: {bn: [0, 0] for bn in bins} for v in ("both", "neither", "fixed")}
    for seed in CONFIRM_SEEDS:
        sp = Split(x_all, y_all, seed); te = sp.parts["test"]; n_miss = te[1].sum(axis=1)
        for v, tr_, ver in (("both", rowA["tree"], "both"), ("neither", rowA["tree"], "neither"), ("fixed", fixed_tree(rowA["best_fixed"]), "both")):
            mdl = fit(sp, tr_, ver); pr = predict(mdl, tensors((te[0], *neutralise(te[1], te[2], ver), te[3]), "cpu")).argmax(1) == te[3]
            for bn in bins:
                sel = (n_miss >= bn[0]) & (n_miss <= bn[1]); hit[v][bn][0] += int(pr[sel].sum()); hit[v][bn][1] += int(sel.sum())
    xs = np.arange(len(bins)); names = ["1", "2", "3–6", "7+"]
    for v, lab, col, mk, ls in (("both", "Both channels", BLUE, "o", "-"), ("neither", "No channels", ORANGE, "s", (0, (4, 1.5))), ("fixed", "Best fixed", AQUA, "^", (0, (1, 1.2)))):
        c.plot(xs, [100 * hit[v][bn][0] / hit[v][bn][1] for bn in bins], color=col, linestyle=ls, linewidth=1.6, marker=mk, markersize=4.4, markeredgecolor="white", markeredgewidth=0.6, label=lab)
    c.set_xticks(xs); c.set_xticklabels(names); c.set_xlabel("Missing features per record"); c.set_ylabel("Test accuracy (%)"); style_axis(c)
    c.set_title("(c) Winner A, by record", loc="left"); c.legend(loc="lower right", frameon=False, fontsize=6.6)
    case["winner_A_by_missing_count"] = {names[i]: {v: {"correct": hit[v][bn][0], "n": hit[v][bn][1]} for v in hit} for i, bn in enumerate(bins)}
    fig.tight_layout(w_pad=1.0); save(fig, args.out, "fig_past_winners")
    json.dump(case, open(os.path.join(os.path.dirname(args.confirm), "past_winner_case.json"), "w"), indent=1); print(json.dumps(case, indent=1))


if __name__ == "__main__":
    main()
