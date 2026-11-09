"""CLI: confidence (development), train (validation only), evaluate (frozen test)."""
import argparse
from pathlib import Path
from .data import ROOT
from .experiment import read_config, confidence_stage, train_stage, evaluate_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("confidence", "train"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=ROOT / "configs" / "pilot.json")
        p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        rows = evaluate_stage(args.run)
    else:
        cfg = read_config(args.config)
        fn = confidence_stage if args.command == "confidence" else train_stage
        rows = fn(cfg, args.output)
    print(f"Completed {args.command}: {len(rows)} result rows.", flush=True)


if __name__ == "__main__":
    main()
