"""
Command line entry point.

    python -m cnn_extension.run train     --config cnn_extension/configs/pilot_fmnist.json [--replicate 0] [--set key=value ...]
    python -m cnn_extension.run shift     --run-dir runs/<name>/rep0 --pattern block --rate 0.4 [--rate 0.6 ...]
    python -m cnn_extension.run aggregate --runs runs/<name>/rep* [--out summary.json]
    python -m cnn_extension.run reproduce --run-dir runs/<name>/rep0

``--set`` overrides nested keys with dots, e.g. ``--set search.generations=2 --set corruption.rate=0.4``
(JSON-parsed values; plain strings otherwise).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from .experiment import aggregate, evaluate_shift, load_config, reproduce_predictions, run_experiment
from .methods import METHODS, compose


def _parse_overrides(items):
    out = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        d = out
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cnn_extension.run")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="matched-conditions run (search if evolved, fresh retrain, test)")
    t.add_argument("--config", required=True)
    t.add_argument("--replicate", type=int, nargs="+", default=[0])
    t.add_argument("--set", nargs="*", default=[], help="override config keys: a.b=value")
    t.add_argument("--no-resume", action="store_true")

    s = sub.add_parser("shift", help="evaluate a finished run under other test corruption (frozen model)")
    s.add_argument("--run-dir", required=True, nargs="+")
    s.add_argument("--pattern", required=True, choices=["random", "block", "rows", "cols"])
    s.add_argument("--rate", type=float, nargs="+", required=True)
    s.add_argument("--corruption-seed", type=int, default=None)

    a = sub.add_parser("aggregate", help="mean/std over run directories (grouped by experiment name)")
    a.add_argument("--runs", nargs="+", required=True)
    a.add_argument("--out", default=None)

    r = sub.add_parser("reproduce", help="check that saved artefacts reproduce the saved test predictions")
    r.add_argument("--run-dir", required=True)

    gr = sub.add_parser("grid", help="compose a base config with registered methods / patterns / rates and run them")
    gr.add_argument("--base", required=True, help="base config (dataset, search budget, seeds, out_root)")
    gr.add_argument("--method", nargs="+", required=True, choices=sorted(METHODS), metavar="METHOD")
    gr.add_argument("--pattern", nargs="+", default=[None], choices=["random", "block", "rows", "cols", None])
    gr.add_argument("--rate", type=float, nargs="+", default=[None])
    gr.add_argument("--replicate", type=int, nargs="+", default=[0])
    gr.add_argument("--set", nargs="*", default=[], help="override base config keys: a.b=value")
    gr.add_argument("--dry-run", action="store_true", help="print the composed run names and exit")
    gr.add_argument("--no-resume", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "train":
        cfg = load_config(args.config, _parse_overrides(args.set))
        for rep in args.replicate:
            run_experiment(cfg, replicate=rep, resume=not args.no_resume)
    elif args.cmd == "grid":
        base = load_config(args.base, _parse_overrides(args.set))
        for pattern in args.pattern:
            for rate in args.rate:
                for method in args.method:
                    cfg = compose(base, method, pattern, rate)
                    for rep in args.replicate:
                        if args.dry_run:
                            print(f"{cfg['name']}  rep{rep}  backbone={cfg['backbone']}  method={cfg['method']}  "
                                  f"metadata_mode={cfg['metadata_mode']}  corruption={cfg['corruption']}")
                        else:
                            run_experiment(cfg, replicate=rep, resume=not args.no_resume)
    elif args.cmd == "shift":
        for d in args.run_dir:
            for rate in args.rate:
                out = evaluate_shift(d, args.pattern, rate, args.corruption_seed)
                print(f"{d}: {args.pattern} {rate:.2f} (realized {out['tested_on']['realized_rate']:.4f}) -> "
                      f"acc {out['test_accuracy']:.4f}  macro-F1 {out['test_macro_f1']:.4f}")
    elif args.cmd == "aggregate":
        dirs = []
        for pat in args.runs:
            dirs.extend(sorted(glob.glob(pat)) or [pat])
        summ = aggregate([d for d in dirs if os.path.isdir(d)])
        for name, s_ in summ.items():
            print(f"{name}: n={s_['n_runs']}  acc {s_['test_accuracy_mean']:.4f} ± {s_['test_accuracy_std']:.4f}  "
                  f"macro-F1 {s_['test_macro_f1_mean']:.4f} ± {s_['test_macro_f1_std']:.4f}  "
                  f"search {s_['search_runtime_s_mean']:.0f}s  final {s_['final_train_runtime_s_mean']:.0f}s")
        if args.out:
            with open(args.out, "w") as f:
                json.dump(summ, f, indent=2)
    elif args.cmd == "reproduce":
        print(json.dumps(reproduce_predictions(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
