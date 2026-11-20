"""Paper figures for the frozen-model missingness-shift experiment (reads saved results only).

    python -m telo_tabular.make_figures --root runs_shift --out figures
    python -m telo_tabular.make_figures --root runs_shift --out paper_subsection/figures --datasets glass wdbc pima mushroom

fig_shift_curves       accuracy vs test MCAR rate, one panel per dataset (models trained at 20 % MCAR, frozen)
fig_shift_differences  paired accuracy difference 3C-EA minus the best fixed activation, all test conditions, 95 % CI
fig_matched_metrics    six metrics at the matched condition, 3C-EA vs ELU vs ReLU
Colours: three categorical slots (blue, orange, aqua) documented as colour-blind safe for all pairs; every series
also differs by marker shape and line style, so the figures survive greyscale printing.
"""
import argparse, csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"
FIXED = ["relu", "swish", "leaky_relu", "elu"]
TITLE = {"wdbc": "WDBC", "sonar": "Sonar", "glass": "Glass", "pima": "PIMA", "mushroom": "Mushroom"}
DS_ORDER = ["sonar", "glass", "wdbc", "pima", "mushroom"]
COND = [("mcar", 0.0, "Clean"), ("mcar", 0.1, "MCAR 10%"), ("mcar", 0.2, "MCAR 20% (training)"), ("mcar", 0.3, "MCAR 30%"),
        ("mcar", 0.4, "MCAR 40%"), ("mcar", 0.5, "MCAR 50%"), ("mcar", 0.6, "MCAR 60%"), ("group", 0.2, "Grouped 20%"),
        ("group", 0.4, "Grouped 40%"), ("mar", 0.2, "MAR 20%"), ("mar", 0.4, "MAR 40%"), ("mnar", 0.2, "MNAR 20%"), ("mnar", 0.4, "MNAR 40%")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})


def load(root, order=None):
    data = {}
    for f in glob.glob(os.path.join(root, "*", "test_results.csv")):
        ds = os.path.basename(os.path.dirname(f))
        rows = [r for r in csv.DictReader(open(f)) if r["accuracy"] not in ("", "None")]
        d = {}
        for r in rows:
            d.setdefault((r["variant"], r["mechanism"], round(float(r["nominal_rate"]), 2)), {})[int(r["seed"])] = float(r["accuracy"])
        data[ds] = d
    return {ds: data[ds] for ds in (order or DS_ORDER) if ds in data}


def series(d, variant, mech, rate):
    s = d.get((variant, mech, rate), {})
    return np.array([s[k] for k in sorted(s)])


def style_axis(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=2.5, width=0.5)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf")); fig.savefig(os.path.join(out, name + ".png"), dpi=300); plt.close(fig)


def fig_curves(data, out):
    rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(1.72 * n + 0.3, 2.25), sharex=True)
    for ax, (ds, d) in zip(np.atleast_1d(axes), data.items()):
        ev = np.array([series(d, "original", "mcar", r).mean() for r in rates]) * 100
        se = np.array([series(d, "original", "mcar", r).std(ddof=1) / np.sqrt(len(series(d, "original", "mcar", r))) for r in rates]) * 100
        fx = np.array([[series(d, v, "mcar", r).mean() for r in rates] for v in FIXED]) * 100
        x = np.array(rates) * 100
        ax.axvline(20, color=INK2, linewidth=0.6, linestyle=(0, (1, 2)), zorder=1)
        ax.fill_between(x, fx.min(0), fx.max(0), color=BAND, linewidth=0, zorder=2)
        ax.plot(x, fx[FIXED.index("elu")], color=ORANGE, linewidth=1.4, linestyle=(0, (4, 1.5)), marker="s", markersize=3.6,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        ax.fill_between(x, ev - se, ev + se, color=BLUE, alpha=0.16, linewidth=0, zorder=3)
        ax.plot(x, ev, color=BLUE, linewidth=1.8, marker="o", markersize=4.2, markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        ax.set_title(TITLE.get(ds, ds)); ax.set_xticks([0, 20, 40, 60]); style_axis(ax)
        lo, hi = min(fx.min(), (ev - se).min()), max(fx.max(), (ev + se).max()); pad = 0.12 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
    np.atleast_1d(axes)[0].set_ylabel("Test accuracy (%)")
    fig.supxlabel("Test missing rate, MCAR (%)", fontsize=8, y=0.02)
    handles = [Line2D([], [], color=BLUE, linewidth=1.8, marker="o", markersize=4.2, markeredgecolor="white", label="3C-EA (mean ± s.e.)"),
               Line2D([], [], color=ORANGE, linewidth=1.4, linestyle=(0, (4, 1.5)), marker="s", markersize=3.6, markeredgecolor="white", label="ELU"),
               Patch(facecolor=BAND, edgecolor="none", label="Range of fixed activations"),
               Line2D([], [], color=INK2, linewidth=0.6, linestyle=(0, (1, 2)), label="Training rate")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08), handlelength=2.4, columnspacing=1.6)
    fig.tight_layout(w_pad=0.9); save(fig, out, "fig_shift_curves")


def fig_differences(data, out):
    n = len(data); fig, axes = plt.subplots(1, n, figsize=(1.55 * n + 1.5, 3.3), sharey=True)
    ypos = np.arange(len(COND))[::-1]
    for ax, (ds, d) in zip(np.atleast_1d(axes), data.items()):
        ax.axhspan(ypos[2] - 0.5, ypos[2] + 0.5, color="#f0efec", zorder=0)
        ax.axvline(0, color=INK2, linewidth=0.7, zorder=1)
        lim = 0
        for y, (mech, rate, _) in zip(ypos, COND):
            e = series(d, "original", mech, rate)
            if len(e) == 0: continue
            best = max(FIXED, key=lambda v: series(d, v, mech, rate).mean())
            diff = (e - series(d, best, mech, rate)) * 100
            m = diff.mean(); h = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
            sig = (m - h > 0) or (m + h < 0)
            ax.plot([m - h, m + h], [y, y], color=BLUE, linewidth=1.1, solid_capstyle="round", zorder=2)
            ax.plot(m, y, marker="o", markersize=4.6, color=BLUE, markerfacecolor=BLUE if sig else "white", markeredgewidth=1.1, zorder=3)
            lim = max(lim, abs(m - h), abs(m + h))
        lim = np.ceil(lim / 2) * 2; ax.set_xlim(-lim, lim); ax.set_title(TITLE.get(ds, ds)); style_axis(ax, "x")
        ax.tick_params(axis="y", length=0)
        for yy in (ypos[6] - 0.5, ypos[8] - 0.5, ypos[10] - 0.5):
            ax.axhline(yy, color=GRID, linewidth=0.6)
    a0 = np.atleast_1d(axes)[0]; a0.set_yticks(ypos); a0.set_yticklabels([c[2] for c in COND]); a0.set_ylim(-0.6, len(COND) - 0.4)
    fig.supxlabel("Accuracy difference, 3C-EA − best fixed activation (percentage points)", fontsize=8, y=0.01)
    handles = [Line2D([], [], color=BLUE, linewidth=1.1, marker="o", markersize=4.6, markerfacecolor="white", markeredgewidth=1.1, label="Mean paired difference, 95% CI includes 0"),
               Line2D([], [], color=BLUE, linewidth=1.1, marker="o", markersize=4.6, markerfacecolor=BLUE, label="95% CI excludes 0")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.05), columnspacing=1.8)
    fig.tight_layout(w_pad=0.8); save(fig, out, "fig_shift_differences")


def fig_metrics(root, data, out):
    path = os.path.join(root, "extended_metrics.csv")
    if not os.path.exists(path): return
    rows = [r for r in csv.DictReader(open(path)) if r["mechanism"] == "mcar" and abs(float(r["rate"]) - 0.2) < 1e-9]
    metrics = [("accuracy", "Accuracy"), ("precision_macro", "Precision"), ("recall_macro", "Recall"), ("f1_macro", "F1"), ("auc", "AUC"), ("mcc", "MCC")]
    meth = [("original", "3C-EA", BLUE, "o", 5.0), ("elu", "ELU", ORANGE, "s", 4.4), ("relu", "ReLU", AQUA, "^", 4.8)]
    dss = [ds for ds in data if any(r["dataset"] == ds for r in rows)]; n = len(dss)
    fig, axes = plt.subplots(1, n, figsize=(1.62 * n + 0.75, 2.9), sharey=True); ypos = np.arange(len(metrics))[::-1]
    for ax, ds in zip(np.atleast_1d(axes), dss):
        allv = []
        for y, (key, _) in zip(ypos, metrics):
            vals = [float(next(r for r in rows if r["dataset"] == ds and r["variant"] == v)[key]) for v, *_ in meth]
            allv += vals
            # one thin lane per method inside each metric row, so coinciding values stay visible
            for (v, _, col, mk, ms), val, dy in zip(meth, vals, (0.24, 0.0, -0.24)):
                ax.plot([min(vals), val], [y + dy, y + dy], color=BAND, linewidth=0.9, solid_capstyle="round", zorder=1)
                ax.plot(val, y + dy, marker=mk, markersize=ms, color=col, markeredgecolor="white", markeredgewidth=0.7, linestyle="none", zorder=3)
            ax.axhline(y - 0.5, color=GRID, linewidth=0.5, zorder=0)
        pad = 0.08 * (max(allv) - min(allv)); ax.set_xlim(min(allv) - pad, max(allv) + pad); ax.set_title(TITLE.get(ds, ds)); style_axis(ax, "x")
        ax.grid(axis="x", visible=False); ax.tick_params(axis="y", length=0); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    a0 = np.atleast_1d(axes)[0]; a0.set_yticks(ypos); a0.set_yticklabels([m[1] for m in metrics]); a0.set_ylim(-0.6, len(metrics) - 0.4)
    fig.supxlabel("Score at the matched condition (20% MCAR), mean over runs", fontsize=8, y=0.01)
    handles = [Line2D([], [], color=c, marker=mk, markersize=ms, linestyle="none", markeredgecolor="white", label=lab) for _, lab, c, mk, ms in meth]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 1.06), columnspacing=2.0, handletextpad=0.3)
    fig.tight_layout(w_pad=0.8); save(fig, out, "fig_matched_metrics")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", default=None, help="datasets to draw, in panel order (default: all five)")
    args = ap.parse_args(argv); data = load(args.root, args.datasets)
    fig_curves(data, args.out); fig_differences(data, args.out); fig_metrics(args.root, data, args.out)
    print("figures for", list(data), "->", args.out)


if __name__ == "__main__":
    main()
