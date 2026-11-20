"""Case study of one evolved activation that uses both reliability channels (reads a finished shift run; trains only tiny ablation twins).

    python -m telo_tabular.winner_analysis --root runs_shift --dataset mushroom --seed 44 --out runs_shift/winner_analysis --fig figures

What it does, for the winner evolved in <dataset>/seed_<seed>:
  1. closed form     checks the simplified formula against the stored tree on a grid
  2. frozen knock-out the stored model is evaluated under every test condition with m set to 0, c set to 1, or both
                     (input-level intervention; ChannelProp then carries the neutral value through every layer)
  3. channel trace   what the activation actually receives in each hidden layer (propagated m, c and the offset c - m)
  4. transplant      the same formula is retrained on every split of the run in four versions - both channels, m only,
                     c only, neither - with the run's own seeds, masks and final-training budget, and compared paired by split
  5. by record       accuracy against the number of missing features in a record (pooled over splits)
Nothing here selects or tunes anything on test data: the winner and all training settings come from the finished run.
"""
import argparse, json, os
from pathlib import Path
import joblib
import numpy as np
import torch
from scipy.stats import wilcoxon
from . import gp_core as g
from .data import load_dataset
from .models import build_model, tensors, predict, scores, train

VERSIONS = [("both", "Both channels"), ("m_only", "m only (c = 1)"), ("c_only", "c only (m = 0)"), ("neither", "Neither (m = 0, c = 1)")]
FIXED = ["relu", "swish", "leaky_relu", "elu"]


def neutralise(m, c, version):
    return (m if version in ("both", "m_only") else np.zeros_like(m)), (c if version in ("both", "c_only") else np.ones_like(c))


def conditions_of(cfg):
    conds = [cfg["corruption"]] + cfg["test_conditions"]
    return list({(v["mechanism"], v["rate"]): v for v in conds}.values())


def key(cond):
    return f"{cond['mechanism']}_{cond['rate']:g}"


def acc(model, z, m, c, y):
    return scores(y, predict(model, tensors((z, m, c, y), "cpu")))["accuracy"]


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True); ap.add_argument("--fig", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root) / args.dataset; cfg = json.loads((root / "manifest.json").read_text())["config"]
    torch.set_num_threads(cfg["torch_threads"]); conds = conditions_of(cfg); ds = args.dataset
    x_full, y_full, _ = load_dataset(ds)
    home = root / ds / f"seed_{args.seed}"; res = json.loads((home / "original" / "training.json").read_text()); tree = res["tree"]
    out = {"dataset": ds, "seed": args.seed, "infix": res["infix"], "conditions": [key(c) for c in conds]}

    # 1. closed form: kappa * (x + m - c)^2 with kappa = softplus(1)^2  (only asserted for this formula)
    rng = np.random.default_rng(0)
    gx, gm, gc = (torch.tensor(a, dtype=torch.float32) for a in (rng.uniform(-3, 3, 4000), rng.uniform(0, 1, 4000), rng.uniform(.1, 1, 4000)))
    kappa = float(torch.nn.functional.softplus(torch.tensor(1.)) ** 2)
    closed = kappa * (gx + gm - gc) ** 2
    out["closed_form"] = {"expression": "kappa * (x + m - c)^2", "kappa": kappa,
                          "max_abs_error_on_grid": float((g.evaluate_tree(tree, gx, gm, gc) - closed).abs().max())}

    def build(pack):
        return build_model("three_channel", pack["train"][0].shape[1], len(pack["metadata"]["classes"]), cfg["hidden_sizes"], tree, "full")

    def test_arrays(pack, seed, i):
        te = pack["split_indices"]["test"]
        masked, mask = pack["mask_generator"].corrupt(x_full[te], conds[i]["mechanism"], conds[i]["rate"], seed + 20000 + i)
        z, m, c = pack["preprocessor"].transform(masked)
        return z, m, c, y_full[te], mask

    # 2. frozen knock-out + 3. channel trace on the stored model
    pack = joblib.load(home / "prepared.joblib"); model = build(pack)
    model.load_state_dict(torch.load(home / "original" / "checkpoint.pt", map_location="cpu", weights_only=True))
    stored = {v: {f"{r['mechanism']}_{r['nominal_rate']:g}": r["accuracy"] for r in json.loads((home / v / "evaluation.json").read_text())["rows"]}
              for v in ["original"] + FIXED}
    frozen, trace = {}, {}
    for i, cond in enumerate(conds):
        z, m, c, y, _ = test_arrays(pack, args.seed, i)
        frozen[key(cond)] = {v: acc(model, z, *neutralise(m, c, v), y) for v, _ in VERSIONS}
        frozen[key(cond)]["stored_3cea"] = stored["original"][key(cond)]
        frozen[key(cond)].update({f"fixed_{v}": stored[v][key(cond)] for v in FIXED})
        with torch.no_grad():
            h, mm, cc = (torch.tensor(a) for a in (z, m, c)); layers = []
            for block in model.network.blocks:
                pre, mm, cc = block.prop(h, mm, cc); h = block.act(pre, mm, cc)
                s = (cc - mm).numpy().ravel(); mmf, ccf = mm.numpy().ravel(), cc.numpy().ravel()
                r2 = float(np.corrcoef(mmf, ccf)[0, 1] ** 2) if mmf.std() > 0 and ccf.std() > 0 else None
                layers.append({"m_mean": float(mmf.mean()), "m_max": float(mmf.max()), "c_mean": float(ccf.mean()), "c_min": float(ccf.min()),
                               "offset_mean": float(s.mean()), "offset_p05": float(np.quantile(s, .05)), "offset_p95": float(np.quantile(s, .95)),
                               "offset_min": float(s.min()), "offset_max": float(s.max()), "r2_c_vs_m": r2,
                               "share_left_of_vertex": float((pre + mm - cc < 0).float().mean()),
                               "offset_quantiles": [float(q) for q in np.quantile(s, [.01, .25, .5, .75, .99])]})
        trace[key(cond)] = layers
    assert abs(frozen[key(conds[0])]["both"] - frozen[key(conds[0])]["stored_3cea"]) < 1e-9, "stored model not reproduced"
    out["frozen_knockout"], out["channel_trace"] = frozen, trace

    # 4. transplant on every split, four versions, paired; 5. accuracy by number of missing features
    transplant = {key(c): {v: [] for v, _ in VERSIONS} for c in conds}
    for c in conds:
        transplant[key(c)].update({"best_fixed": [], "evolved_own": []})
    by_count = {}
    for seed in cfg["seeds"]:
        d = root / ds / f"seed_{seed}"; pk = joblib.load(d / "prepared.joblib")
        st = {v: {f"{r['mechanism']}_{r['nominal_rate']:g}": r["accuracy"] for r in json.loads((d / v / "evaluation.json").read_text())["rows"]}
              for v in ["original"] + FIXED}
        models = {}
        for v, _ in VERSIONS:
            tr = tuple(pk["train"][k] for k in (0, 1, 2, 3)); va = tuple(pk["val"][k] for k in (0, 1, 2, 3))
            tr = (tr[0], *neutralise(tr[1], tr[2], v), tr[3]); va = (va[0], *neutralise(va[1], va[2], v), va[3])
            torch.manual_seed(seed + 10000); mdl = build(pk)
            info = train(mdl, tensors(tr, "cpu"), tensors(va, "cpu"), cfg["final"], final=True)
            assert info["valid"], (seed, v, info["failure"]); models[v] = mdl
        for i, cond in enumerate(conds):
            z, m, c, y, mask = test_arrays(pk, seed, i); k = key(cond)
            for v, _ in VERSIONS:
                transplant[k][v].append(acc(models[v], z, *neutralise(m, c, v), y))
            transplant[k]["best_fixed"].append({f: st[f][k] for f in FIXED}); transplant[k]["evolved_own"].append(st["original"][k])
            if cond["mechanism"] == "mcar" and cond["rate"] > 0:
                n_miss = mask.sum(axis=1).astype(int)
                preds = {v: predict(models[v], tensors((z, *neutralise(m, c, v), y), "cpu")).argmax(1) == y for v in ("both", "neither")}
                for f in FIXED:
                    npz = np.load(d / f / f"predictions_{i}_{k}.npz"); assert (npz["mask"] == mask).all()
                    preds[f] = npz["probs"].argmax(1) == y
                for n, *hit in zip(n_miss, *[preds[v] for v in ("both", "neither", *FIXED)]):
                    row = by_count.setdefault(int(n), {v: [0, 0] for v in ("both", "neither", *FIXED)})
                    for v, h in zip(("both", "neither", *FIXED), hit):
                        row[v][0] += int(h); row[v][1] += 1
        if seed == args.seed:      # the transplant with both channels on the winner's own split is the stored model again
            assert abs(transplant[key(conds[0])]["both"][-1] - st["original"][key(conds[0])]) < 1e-9, "own-split retraining differs from the stored model"
    summary = {}
    for k, row in transplant.items():
        fx = {f: float(np.mean([r[f] for r in row["best_fixed"]])) for f in FIXED}; best = max(fx, key=fx.get)
        arr = {v: np.array(row[v]) for v, _ in VERSIONS}; bf = np.array([r[best] for r in row["best_fixed"]])
        def p(a, b):
            try: return float(wilcoxon(a, b).pvalue)
            except ValueError: return None
        summary[k] = {**{v: {"mean": float(a.mean()), "sd": float(a.std(ddof=1))} for v, a in arr.items()},
                      "best_fixed": best, "best_fixed_mean": float(bf.mean()), "evolved_own_mean": float(np.mean(row["evolved_own"])),
                      "p_both_vs_neither": p(arr["both"], arr["neither"]), "p_both_vs_m_only": p(arr["both"], arr["m_only"]),
                      "p_both_vs_c_only": p(arr["both"], arr["c_only"]), "p_both_vs_best_fixed": p(arr["both"], bf),
                      "p_neither_vs_best_fixed": p(arr["neither"], bf), "wins_both_over_neither": int((arr["both"] > arr["neither"]).sum()),
                      "n": len(bf)}
    out["transplant"] = summary
    out["transplant_runs"] = {k: {v: row[v] for v, _ in VERSIONS} for k, row in transplant.items()}
    out["by_missing_count"] = {str(n): {v: {"correct": c0, "n": n0} for v, (c0, n0) in row.items()} for n, row in sorted(by_count.items())}

    os.makedirs(args.out, exist_ok=True)
    json.dump(out, open(os.path.join(args.out, "winner_analysis.json"), "w"), indent=1)
    L = [f"# Case study: winner of {ds}, seed {args.seed}", "", f"Stored formula: `{res['infix']}`  ",
         f"Closed form: `{out['closed_form']['expression']}`, kappa = {kappa:.4f} (max abs error on grid {out['closed_form']['max_abs_error_on_grid']:.2e})", "",
         "## Frozen stored model, channels switched off at test time (accuracy)", "",
         "| condition | intact | m = 0 | c = 1 | both off | best fixed, same split |", "|---|---|---|---|---|---|"]
    for k in out["conditions"]:
        r = frozen[k]; bf = max(FIXED, key=lambda f: r["fixed_" + f])
        L.append(f"| {k} | {r['both']:.3f} | {r['c_only']:.3f} | {r['m_only']:.3f} | {r['neither']:.3f} | {r['fixed_' + bf]:.3f} ({bf}) |")
    L += ["", f"## Same formula retrained on all {len(cfg['seeds'])} splits (accuracy, mean ± sd; p = two-sided Wilcoxon, paired by split)", "",
          "| condition | both channels | m only | c only | neither | best fixed | own winners | p both vs neither | both > neither | p neither vs best fixed |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k in out["conditions"]:
        s = summary[k]
        L.append(f"| {k} | " + " | ".join(f"{s[v]['mean']:.3f}±{s[v]['sd']:.3f}" for v, _ in VERSIONS) +
                 f" | {s['best_fixed_mean']:.3f} ({s['best_fixed']}) | {s['evolved_own_mean']:.3f} | {s['p_both_vs_neither']:.4f} | {s['wins_both_over_neither']}/{s['n']} | {s['p_neither_vs_best_fixed']:.4f} |")
    L += ["", "## What the activation receives (stored model; offset = c - m after propagation)", "",
          "| condition | layer | mean m | mean c | offset mean | offset 5%..95% | R² of c on m | share of pre-activations left of the vertex |", "|---|---|---|---|---|---|---|---|"]
    for k in out["conditions"]:
        for li, t in enumerate(trace[k]):
            r2 = "n/a" if t["r2_c_vs_m"] is None else f"{t['r2_c_vs_m']:.4f}"
            L.append(f"| {k} | {li + 1} | {t['m_mean']:.3f} | {t['c_mean']:.3f} | {t['offset_mean']:.3f} | {t['offset_p05']:.3f}..{t['offset_p95']:.3f} | {r2} | {t['share_left_of_vertex']:.3f} |")
    L += ["", "## Accuracy by number of missing features in a record (MCAR test conditions with rate > 0, pooled over splits)", "",
          "| missing features | records | both channels | neither | " + " | ".join(FIXED) + " |", "|---|---|---|---|" + "---|" * len(FIXED)]
    for n, row in out["by_missing_count"].items():
        L.append(f"| {n} | {row['both']['n']} | " + " | ".join(f"{row[v]['correct'] / row[v]['n']:.3f}" for v in ("both", "neither", *FIXED)) + " |")
    open(os.path.join(args.out, "winner_analysis.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    if args.fig:
        from .winner_figure import draw
        draw(out, args.root, args.fig)


if __name__ == "__main__":
    main()
