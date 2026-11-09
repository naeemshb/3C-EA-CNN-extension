"""Score relu / elu / the step-3 winner under several intermediate fitness protocols: which protocol ranks
ReLU above the fast-learning sine formula, and at what per-tree cost."""
import json, sys, os, time
from cnn_extension import gp_core as g
from cnn_extension.baselines import fixed_activation_tree
from cnn_extension.experiment import Logger, derive_seeds, load_config, prepare_data
from cnn_extension.fidelity import _evaluator
from cnn_extension.training import resolve_device

out_dir = sys.argv[1]
cfg = load_config("cnn_extension/configs/reference_fashion_mnist.json")
cfg["corruption"] = {"pattern": "random", "rate": 0.2}
device = resolve_device(cfg["device"]); seeds = derive_seeds(cfg["seeds"], 0)
log = Logger(os.path.join(out_dir, "log_sweep.txt"))
data, info, meta = prepare_data(cfg, seeds, out_dir, device, log)
winner = json.load(open("runs/ref_fashion_mnist_random20_cnn_evolved_zmc/rep0/results.json"))["selected_tree"]
trees = {"relu": fixed_activation_tree("relu"), "elu": fixed_activation_tree("elu"), "winner": winner}
protocols = {
    "sub10k_e5_p2_b256 (current cheap)": dict(fitness_epochs=5, fitness_patience=2, fitness_batch_size=256, fitness_train_subset=10000),
    "sub10k_e10_p3_b128": dict(fitness_epochs=10, fitness_patience=3, fitness_batch_size=128, fitness_train_subset=10000),
    "sub10k_e15_p3_b128": dict(fitness_epochs=15, fitness_patience=3, fitness_batch_size=128, fitness_train_subset=10000),
    "sub20k_e10_p3_b128": dict(fitness_epochs=10, fitness_patience=3, fitness_batch_size=128, fitness_train_subset=20000),
    "sub20k_e15_p3_b128": dict(fitness_epochs=15, fitness_patience=3, fitness_batch_size=128, fitness_train_subset=20000),
    "full_e10_p3_b128": dict(fitness_epochs=10, fitness_patience=3, fitness_batch_size=128, fitness_train_subset=None),
}
rows = {}
for pname, proto in protocols.items():
    ev = _evaluator(cfg, proto, data, seeds, meta, device, log)
    rows[pname] = {}
    for tname, t in trees.items():
        r = ev.evaluate(t, 0)
        rows[pname][tname] = {k: r[k] for k in ("fitness", "val_acc", "epochs_run", "runtime_s")}
    rr = rows[pname]
    print(f"{pname:36s} relu {rr['relu']['val_acc']:.4f}({rr['relu']['epochs_run']}ep,{rr['relu']['runtime_s']:.0f}s)  "
          f"elu {rr['elu']['val_acc']:.4f}({rr['elu']['epochs_run']}ep)  winner {rr['winner']['val_acc']:.4f}({rr['winner']['epochs_run']}ep,{rr['winner']['runtime_s']:.0f}s)  "
          f"relu>winner: {rr['relu']['val_acc'] > rr['winner']['val_acc']}", flush=True)
json.dump(rows, open(os.path.join(out_dir, "protocol_sweep.json"), "w"), indent=1)
