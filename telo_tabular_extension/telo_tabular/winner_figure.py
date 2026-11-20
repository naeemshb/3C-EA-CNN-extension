"""Four-panel figure for the winner case study (reads winner_analysis.json and the stored test results only).

    python -m telo_tabular.winner_figure --analysis runs_shift/winner_analysis/winner_analysis.json --root runs_shift --out figures

(a) the closed-form activation for four values of the reliability offset s = c - m
(b) the offset the first hidden layer actually receives as the test missing rate grows, against the range seen in training
(c) accuracy of the formula retrained with both channels and with neither, and of the best fixed activation, vs test rate
(d) the same three, by the number of missing features in a record
Colours follow make_figures (blue / orange / aqua, each with its own marker and line style); panel (a) uses one hue from
light to dark because the offset is an ordered quantity.
"""
import argparse, csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .make_figures import BLUE, ORANGE, AQUA, INK, INK2, GRID, BAND, style_axis, save

RAMP = ["#9cc3f2", "#5d9be6", "#2a78d6", "#0f3f7d"]
FIXED_NAME = {"relu": "ReLU", "swish": "Swish", "leaky_relu": "LeakyReLU", "elu": "ELU"}


def draw(out, root, figdir, name="fig_winner_case"):
    ds, kappa = out["dataset"], out["closed_form"]["kappa"]
    rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; keys = [f"mcar_{r:g}" for r in rates]; x = np.array(rates) * 100
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.7)); a, b, c, d = axes.ravel()

    # (a) activation shape
    z = np.linspace(-1.6, 2.4, 300)
    for s, col, lab in zip([1.0, 0.74, 0.25, -0.2], RAMP[::-1], ["all inputs observed", "20% missing (training)", "60% missing", "all inputs missing"]):
        a.plot(z, kappa * (z - s) ** 2, color=col, linewidth=1.5); a.plot(s, 0, marker="o", markersize=3.6, color=col, markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    a.set_ylim(-0.3, 8.5); a.set_xlim(-1.6, 2.4); a.set_xlabel("Pre-activation $z$"); a.set_ylabel(r"$f(z, m, c)$"); style_axis(a)
    a.set_title("(a) Evolved function", loc="left")
    a.legend(handles=[Line2D([], [], color=col, linewidth=1.5, label=f"$s={s:g}$: {lab}") for s, col, lab in
                      zip([1.0, 0.74, 0.25, -0.2], RAMP[::-1], ["none missing", "training", "60% missing", "all missing"])],
             loc="upper center", frameon=True, facecolor="white", edgecolor="none", framealpha=0.92, fontsize=6.8, handlelength=1.6, labelspacing=0.3, borderaxespad=0.3)

    # (b) offset received by layer 1 vs test rate
    tr = [out["channel_trace"][k][0] for k in keys]; ref = out["channel_trace"]["mcar_0.2"][0]
    b.axhspan(ref["offset_p05"], ref["offset_p95"], color="#f0efec", zorder=0)
    b.fill_between(x, [t["offset_p05"] for t in tr], [t["offset_p95"] for t in tr], color=BLUE, alpha=0.16, linewidth=0, zorder=2)
    b.plot(x, [t["offset_mean"] for t in tr], color=BLUE, linewidth=1.8, marker="o", markersize=4.2, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    b.axvline(20, color=INK2, linewidth=0.6, linestyle=(0, (1, 2)), zorder=1)
    b.set_xlabel("Test missing rate, MCAR (%)"); b.set_ylabel("Offset $s = c - m$ in layer 1"); b.set_xticks([0, 20, 40, 60]); style_axis(b)
    b.set_title("(b) What the unit receives", loc="left")
    b.legend(handles=[Line2D([], [], color=BLUE, linewidth=1.8, marker="o", markersize=4.2, markeredgecolor="white", label="Mean, 5–95% band"),
                      Patch(facecolor="#f0efec", edgecolor="none", label="Range seen in training")], loc="lower left", frameon=False, fontsize=6.8, borderaxespad=0.3)

    # (c) accuracy vs test rate, 20 splits
    runs = out["transplant_runs"]; best = out["transplant"]["mcar_0.2"]["best_fixed"]
    rows = [r for r in csv.DictReader(open(os.path.join(root, ds, "test_results.csv"))) if r["variant"] == best and r["mechanism"] == "mcar"]
    fx = {k: np.array([float(r["accuracy"]) for r in sorted(rows, key=lambda r: int(r["seed"])) if abs(float(r["nominal_rate"]) - rt) < 1e-9]) for k, rt in zip(keys, rates)}
    series = [("both", "Both channels", BLUE, "o", "-", 1.8), ("neither", "Same formula, no channels", ORANGE, "s", (0, (4, 1.5)), 1.4), (None, FIXED_NAME[best], AQUA, "^", (0, (1, 1.2)), 1.4)]
    for v, lab, col, mk, ls, lw in series:
        arr = np.array([runs[k][v] if v else fx[k] for k in keys]) * 100; mean, se = arr.mean(1), arr.std(1, ddof=1) / np.sqrt(arr.shape[1])
        c.fill_between(x, mean - se, mean + se, color=col, alpha=0.16, linewidth=0, zorder=2)
        c.plot(x, mean, color=col, linewidth=lw, linestyle=ls, marker=mk, markersize=4, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    c.axvline(20, color=INK2, linewidth=0.6, linestyle=(0, (1, 2)), zorder=1)
    c.set_xlabel("Test missing rate, MCAR (%)"); c.set_ylabel("Test accuracy (%)"); c.set_xticks([0, 20, 40, 60]); style_axis(c)
    c.set_title("(c) Retrained on 20 splits", loc="left")

    # (d) accuracy by number of missing features
    bc = {int(n): row for n, row in out["by_missing_count"].items() if row["both"]["n"] >= 200}; ns = sorted(bc)
    for v, lab, col, mk, ls, lw in series:
        kk = v or best; d.plot(ns, [100 * bc[n][kk]["correct"] / bc[n][kk]["n"] for n in ns], color=col, linewidth=lw, linestyle=ls, marker=mk, markersize=3.4,
                               markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    d.set_xlabel("Missing features per record (of 22)"); d.set_ylabel("Test accuracy (%)"); style_axis(d); d.set_xticks([0, 5, 10, 15, 20])
    d.set_title("(d) By record", loc="left")
    fig.legend(handles=[Line2D([], [], color=col, linewidth=lw, linestyle=ls, marker=mk, markersize=4, markeredgecolor="white", label=lab) for _, lab, col, mk, ls, lw in series] +
               [Line2D([], [], color=INK2, linewidth=0.6, linestyle=(0, (1, 2)), label="Training rate")],
               loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.045), handlelength=2.6, columnspacing=1.4)
    fig.tight_layout(w_pad=1.2, h_pad=1.2); save(fig, figdir, name)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--analysis", required=True); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(argv); draw(json.load(open(args.analysis)), args.root, args.out); print("wrote", os.path.join(args.out, "fig_winner_case"))


if __name__ == "__main__":
    main()
