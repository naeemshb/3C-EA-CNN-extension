"""Re-score the 20 best DISTINCT formulas of the step-3 search under the full 30-epoch fitness protocol."""
import json, sys, os
from cnn_extension import gp_core as g
from cnn_extension.experiment import Logger, derive_seeds, load_config, prepare_data
from cnn_extension.fidelity import FULL_PROTOCOL, _evaluator
from cnn_extension.training import resolve_device

out_dir = sys.argv[1]
cfg = load_config("cnn_extension/configs/reference_fashion_mnist.json")
cfg["corruption"] = {"pattern": "random", "rate": 0.2}
device = resolve_device(cfg["device"]); seeds = derive_seeds(cfg["seeds"], 0)
log = Logger(os.path.join(out_dir, "log_rescore.txt"))
data, info, meta = prepare_data(cfg, seeds, out_dir, device, log)
import pickle; st = pickle.load(open("runs/ref_fashion_mnist_random20_cnn_evolved_zmc/rep0/search/search_state.pkl", "rb"))
seen, cands = set(), []
for e in sorted([dict(ind["record"], tree=ind["tree"]) for ind in st["population"] if ind.get("record") and ind["record"].get("valid")], key=lambda e: -e["fitness"]):
    if e["infix"] in seen: continue
    seen.add(e["infix"]); cands.append(e)
    if len(cands) == 20: break
ev = _evaluator(cfg, FULL_PROTOCOL, data, seeds, meta, device, log)
rows = []
for e in cands:
    r = ev.evaluate(e["tree"], 0)
    rows.append({"infix": e["infix"], "cheap_val": e["val_acc"], "cheap_fitness": e["fitness"], "full_val": r["val_acc"], "full_fitness": r["fitness"], "full_epochs": r["epochs_run"]})
    print(f"cheap {e['val_acc']:.4f} -> full {r['val_acc'] or 0:.4f} ({r['epochs_run']} ep)  {e['infix']}", flush=True)
json.dump(rows, open(os.path.join(out_dir, "rescore_top20_full.json"), "w"), indent=1)
print("reference: relu full val 0.8785 (20 ep)")
