"""DIAGNOSTIC: final-train (full final protocol, same seeds as the real run) the top-3 formulas of the step-3 final
population when re-ranked by the FULL fitness protocol, and report test accuracy next to the committed runs."""
import json, os, pickle, sys
from cnn_extension import gp_core as g
from cnn_extension.experiment import Logger, derive_seeds, final_train_and_test, load_config, prepare_data
from cnn_extension.training import resolve_device
out_dir = sys.argv[1]
cfg = load_config("cnn_extension/configs/reference_fashion_mnist.json"); cfg["corruption"] = {"pattern": "random", "rate": 0.2}
device = resolve_device(cfg["device"]); seeds = derive_seeds(cfg["seeds"], 0)
log = Logger(os.path.join(out_dir, "log_final_reranked.txt"))
data, info, meta = prepare_data(cfg, seeds, out_dir, device, log)
st = pickle.load(open("runs/ref_fashion_mnist_random20_cnn_evolved_zmc/rep0/search/search_state.pkl", "rb"))
by_infix = {g.tree_to_infix(ind["tree"]): ind["tree"] for ind in st["population"]}
resc = json.load(open(os.path.join(out_dir, "rescore_top20_full.json")))
top = sorted([r for r in resc if r["full_val"]], key=lambda r: -r["full_val"])[:3]
rows = []
for r in top:
    tree = by_infix[r["infix"]]
    res = final_train_and_test(cfg, tree, data, seeds, meta, device, out_dir, log)
    row = {"infix": r["infix"], "full_fitness_val": r["full_val"], "cheap_val": r["cheap_val"],
           "test_accuracy": res["test_accuracy"], "test_macro_f1": res["test_macro_f1"],
           "best_epoch": res.get("best_epoch"), "epochs_run": res.get("epochs_run")}
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(os.path.join(out_dir, "final_train_reranked.json"), "w"), indent=1)
print("committed: relu 0.8861  leakyrelu 0.8871  swish 0.8778  elu 0.8753  evolved(cheap winner) 0.8700")
