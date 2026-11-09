"""Score the four fixed baseline trees and the step-3 winner under the cheap and the full fitness protocol
(same data, masks, subset, seeds as the search) to test whether the fitness protocol mis-ranks them."""
import json, sys, os
from cnn_extension import gp_core as g
from cnn_extension.baselines import fixed_activation_tree
from cnn_extension.evolution import FitnessEvaluator
from cnn_extension.experiment import Logger, derive_seeds, load_config, model_factory_for, prepare_data
from cnn_extension.fidelity import FULL_PROTOCOL, _evaluator
from cnn_extension.training import resolve_device

out_dir = sys.argv[1]
cfg = load_config("cnn_extension/configs/reference_fashion_mnist.json")
cfg["corruption"] = {"pattern": "random", "rate": 0.2}
cheap = {k: cfg["search"][k] for k in FULL_PROTOCOL}
device = resolve_device(cfg["device"]); seeds = derive_seeds(cfg["seeds"], 0)
log = Logger(os.path.join(out_dir, "log.txt"))
data, info, meta = prepare_data(cfg, seeds, out_dir, device, log)
winner = json.load(open("runs/ref_fashion_mnist_random20_cnn_evolved_zmc/rep0/results.json"))["selected_tree"]
trees = {n: fixed_activation_tree(n) for n in ("relu", "swish", "leakyrelu", "elu")}
trees["evolved_winner"] = winner
evs = {"cheap": _evaluator(cfg, cheap, data, seeds, meta, device, log), "full": _evaluator(cfg, FULL_PROTOCOL, data, seeds, meta, device, log)}
rows = {}
for name, t in trees.items():
    rows[name] = {"infix": g.tree_to_infix(t)}
    for p, ev in evs.items():
        r = ev.evaluate(t, 0)
        rows[name][p] = {k: r[k] for k in ("fitness", "val_acc", "epochs_run", "runtime_s")}
    print(f"{name:15s} {rows[name]['infix']:35s} cheap fit {rows[name]['cheap']['fitness']:.4f} val {rows[name]['cheap']['val_acc']:.4f} ({rows[name]['cheap']['epochs_run']} ep) | "
          f"full fit {rows[name]['full']['fitness']:.4f} val {rows[name]['full']['val_acc']:.4f} ({rows[name]['full']['epochs_run']} ep)", flush=True)
json.dump(rows, open(os.path.join(out_dir, "baselines_under_protocols.json"), "w"), indent=1)
