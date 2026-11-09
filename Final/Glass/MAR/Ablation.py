import math
import random
import copy
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Dict, Optional, Tuple


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_multiclass_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec_i)
    return np.mean(specificities)


def compute_metrics(y_true, y_pred, y_probs):
    num_classes = len(np.unique(y_true))
    avg_method = "binary" if num_classes == 2 else "macro"
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=avg_method, zero_division=0
    )
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_probs[:, 1] if y_probs.ndim > 1 else y_probs)
        else:
            auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = 0.0
    spec = compute_multiclass_specificity(y_true, y_pred)
    return {"precision": prec, "recall": rec, "specificity": spec, "f1": f1, "auc": auc}


INPUTS_FULL = ["value", "missing", "confidence"]
INPUTS_NO_CONF = ["value", "missing"]
INPUTS_NO_MISS = ["value", "confidence"]


CONSTANTS = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1]


def create_input_node(input_name: str) -> dict:
    return {"type": "input", "input_name": input_name, "children": []}


def create_constant_node(value: float) -> dict:
    return {"type": "constant", "value": float(value), "children": []}


def create_unary_node(operator: str, child: dict) -> dict:
    return {"type": "unary", "operator": operator, "children": [child]}


def create_binary_node(operator: str, left: dict, right: dict) -> dict:
    return {"type": "binary", "operator": operator, "children": [left, right]}


# Safe math operations
def safe_divide(x, y):
    denom = torch.where(torch.abs(y) < 1e-3, torch.sign(y) * 1e-3 + (y == 0).float() * 1e-3, y)
    return x / denom

def safe_sqrt(x):
    return torch.sqrt(torch.clamp(x, min=0.0))

def safe_log(x):
    return torch.log(torch.clamp(torch.abs(x), min=1e-8))

def clip_result(x, min_val=-1000.0, max_val=1000.0):
    return torch.clamp(x, min=min_val, max=max_val)


UNARY_OPERATORS = {
    "identity": lambda x: x,
    "negate": lambda x: -x,
    "abs": lambda x: torch.abs(x),
    "square": lambda x: clip_result(x * x),
    "cube": lambda x: clip_result(x * x * x),
    "sqrt": lambda x: safe_sqrt(x),
    "exp": lambda x: clip_result(torch.exp(torch.clamp(x, min=-10, max=10))),
    "log": lambda x: safe_log(x),
    "sin": lambda x: torch.sin(x),
    "cos": lambda x: torch.cos(x),
    "tanh": lambda x: torch.tanh(x),
    "sigmoid": lambda x: torch.sigmoid(x),
    "relu": lambda x: torch.clamp(x, min=0),
    "softplus": lambda x: nn.functional.softplus(x),
    "leaky_relu": lambda x: nn.functional.leaky_relu(x, negative_slope=0.01),
    "elu": lambda x: nn.functional.elu(x, alpha=1.0),
}

BINARY_OPERATORS = {
    "add": lambda x, y: x + y,
    "subtract": lambda x, y: x - y,
    "multiply": lambda x, y: clip_result(x * y),
    "divide": lambda x, y: safe_divide(x, y),
    "max": lambda x, y: torch.maximum(x, y),
    "min": lambda x, y: torch.minimum(x, y),
}


def evaluate_tree(node, value, missing, confidence):

    ntype = node["type"]

    if ntype == "input":
        mapping = {"value": value, "missing": missing, "confidence": confidence}
        return mapping[node["input_name"]]
    if ntype == "constant":
        return torch.full_like(value, node["value"])
    if ntype == "unary":
        child = evaluate_tree(node["children"][0], value, missing, confidence)
        return clip_result(UNARY_OPERATORS[node["operator"]](child))
    if ntype == "binary":
        left = evaluate_tree(node["children"][0], value, missing, confidence)
        right = evaluate_tree(node["children"][1], value, missing, confidence)
        return clip_result(BINARY_OPERATORS[node["operator"]](left, right))
    raise ValueError(f"Unknown node type: {ntype}")


def tree_to_infix(node):
    t = node["type"]
    if t == "input":
        return {"value": "x", "missing": "m", "confidence": "c"}[node["input_name"]]
    if t == "constant":
        v = node["value"]
        return str(int(v)) if float(v).is_integer() else f"{v}"
    if t == "unary":
        op = node["operator"]
        s = tree_to_infix(node["children"][0])
        mapping = {
            "identity": s, "negate": f"(-{s})", "abs": f"|{s}|",
            "square": f"({s})^2", "cube": f"({s})^3", "sqrt": f"sqrt({s})",
            "exp": f"exp({s})", "log": f"log(|{s}|)", "sin": f"sin({s})",
            "cos": f"cos({s})", "tanh": f"tanh({s})", "sigmoid": f"σ({s})",
            "relu": f"relu({s})", "softplus": f"softplus({s})",
            "leaky_relu": f"leakyrelu({s})", "elu": f"elu({s})",
        }
        return mapping.get(op, f"{op}({s})")
    if t == "binary":
        op = node["operator"]
        a, b = tree_to_infix(node["children"][0]), tree_to_infix(node["children"][1])
        mapping = {
            "add": f"({a} + {b})", "subtract": f"({a} - {b})",
            "multiply": f"({a} * {b})", "divide": f"({a} / {b})",
            "max": f"max({a}, {b})", "min": f"min({a}, {b})",
        }
        return mapping.get(op, f"{op}({a}, {b})")
    return "<?>"


def get_tree_depth(node):
    t = node["type"]
    if t in ["input", "constant"]:
        return 1
    if t == "unary":
        return 1 + get_tree_depth(node["children"][0])
    if t == "binary":
        return 1 + max(get_tree_depth(node["children"][0]), get_tree_depth(node["children"][1]))
    return 1


def count_nodes(node):
    t = node["type"]
    if t in ["input", "constant"]:
        return 1
    if t == "unary":
        return 1 + count_nodes(node["children"][0])
    if t == "binary":
        return 1 + count_nodes(node["children"][0]) + count_nodes(node["children"][1])
    return 1


def count_input_diversity(node):
    found = set()
    def walk(n):
        if n.get("type") == "input":
            found.add(n["input_name"])
        for ch in n.get("children", []):
            walk(ch)
    walk(node)
    return len(found)


def get_all_nodes(tree):
    nodes = [tree]
    if tree["type"] in ["unary", "binary"]:
        for ch in tree["children"]:
            nodes.extend(get_all_nodes(ch))
    return nodes


def validate_tree(node, allowed_inputs):
    t = node["type"]
    if t == "input":
        assert node["input_name"] in allowed_inputs, f"Invalid input: {node['input_name']}"
        assert len(node["children"]) == 0
    elif t == "constant":
        assert isinstance(node["value"], (int, float))
        assert len(node["children"]) == 0
    elif t == "unary":
        assert node["operator"] in UNARY_OPERATORS
        assert len(node["children"]) == 1
        validate_tree(node["children"][0], allowed_inputs)
    elif t == "binary":
        assert node["operator"] in BINARY_OPERATORS
        assert len(node["children"]) == 2
        validate_tree(node["children"][0], allowed_inputs)
        validate_tree(node["children"][1], allowed_inputs)
    else:
        raise ValueError(f"Invalid node type: {t}")


def generate_terminal_node(allowed_inputs):
    if random.random() < 0.7:
        return create_input_node(random.choice(allowed_inputs))
    return create_constant_node(random.choice(CONSTANTS))


def generate_random_tree(max_depth, current_depth=0, allowed_inputs=INPUTS_FULL):
    if current_depth >= max_depth:
        return generate_terminal_node(allowed_inputs)

    terminal_prob = 0.3 + 0.4 * (current_depth / max_depth)
    if random.random() < terminal_prob:
        return generate_terminal_node(allowed_inputs)

    if random.random() < 0.4:
        op = random.choice(list(UNARY_OPERATORS.keys()))
        child = generate_random_tree(max_depth, current_depth + 1, allowed_inputs)
        return create_unary_node(op, child)

    op = random.choice(list(BINARY_OPERATORS.keys()))
    left = generate_random_tree(max_depth, current_depth + 1, allowed_inputs)
    right = generate_random_tree(max_depth, current_depth + 1, allowed_inputs)
    return create_binary_node(op, left, right)



def mutate_tree(tree, mutation_rate, max_depth, allowed_inputs):
    mutated = copy.deepcopy(tree)
    all_nodes = get_all_nodes(mutated)

    for node in all_nodes:
        if random.random() < mutation_rate:
            t = node["type"]
            if t == "input":
                candidates = [inp for inp in allowed_inputs if inp != node["input_name"]]
                if candidates:
                    node["input_name"] = random.choice(candidates)
            elif t == "constant":
                node["value"] = random.choice(CONSTANTS)
            elif t == "unary":
                if random.random() < 0.7:
                    candidates = [op for op in UNARY_OPERATORS if op != node["operator"]]
                    if candidates:
                        node["operator"] = random.choice(candidates)
                else:
                    new_subtree = generate_random_tree(min(3, max_depth), 0, allowed_inputs)
                    node.clear()
                    node.update(new_subtree)
            elif t == "binary":
                if random.random() < 0.7:
                    candidates = [op for op in BINARY_OPERATORS if op != node["operator"]]
                    if candidates:
                        node["operator"] = random.choice(candidates)
                else:
                    new_subtree = generate_random_tree(min(3, max_depth), 0, allowed_inputs)
                    node.clear()
                    node.update(new_subtree)

    validate_tree(mutated, allowed_inputs)
    return mutated


def crossover_trees(parent1, parent2, allowed_inputs):
    offspring1 = copy.deepcopy(parent1)
    offspring2 = copy.deepcopy(parent2)

    nodes1 = get_all_nodes(offspring1)
    nodes2 = get_all_nodes(offspring2)

    n1 = random.choice(nodes1)
    n2 = random.choice(nodes2)

    temp = copy.deepcopy(n1)
    n1.clear()
    n1.update(copy.deepcopy(n2))
    n2.clear()
    n2.update(temp)

    try:
        validate_tree(offspring1, allowed_inputs)
        validate_tree(offspring2, allowed_inputs)
    except:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    return offspring1, offspring2


def create_initial_population(pop_size, max_depth, allowed_inputs):
    return [{"tree": generate_random_tree(max_depth, 0, allowed_inputs), "fitness": None}
            for _ in range(pop_size)]


def fitness_proportional_selection(population, num_selected):
    fitness_values = [ind["fitness"] for ind in population]
    min_f = min(fitness_values)
    shifted = [f - min_f + 1.0 for f in fitness_values] if min_f < 0 else fitness_values
    max_f = max(shifted)
    exp_f = [math.exp(f - max_f) for f in shifted]
    total = sum(exp_f)
    probs = [e / total for e in exp_f]

    selected = []
    for _ in range(num_selected):
        r = random.random()
        cumulative = 0.0
        for ind, p in zip(population, probs):
            cumulative += p
            if r <= cumulative:
                selected.append(ind)
                break
        else:
            selected.append(population[-1])
    return selected


def evolve_population(population, mutation_rate, crossover_rate, elite_size, max_depth, allowed_inputs):
    pop_size = len(population)
    sorted_pop = sorted(population, key=lambda x: x["fitness"], reverse=True)

    new_pop = [{"tree": copy.deepcopy(sorted_pop[i]["tree"]), "fitness": None}
               for i in range(elite_size)]

    while len(new_pop) < pop_size:
        if random.random() < crossover_rate and len(new_pop) < pop_size - 1:
            p1 = fitness_proportional_selection(population, 1)[0]
            p2 = fitness_proportional_selection(population, 1)[0]
            o1, o2 = crossover_trees(p1["tree"], p2["tree"], allowed_inputs)
            new_pop.append({"tree": o1, "fitness": None})
            if len(new_pop) < pop_size:
                new_pop.append({"tree": o2, "fitness": None})
        else:
            parent = fitness_proportional_selection(population, 1)[0]
            mutated = mutate_tree(parent["tree"], mutation_rate, max_depth, allowed_inputs)
            new_pop.append({"tree": mutated, "fitness": None})

    return new_pop


class ChannelProp(nn.Module):

    def __init__(self, linear, eps=1e-8):
        super().__init__()
        self.linear = linear
        self.eps = eps

    def forward(self, x, m, c):
        y = self.linear(x)
        W = self.linear.weight
        W_abs = W.abs() + self.eps
        col_sum = W_abs.sum(dim=1, keepdim=True)
        W_norm = W_abs / col_sum
        W_norm_T = W_norm.t()

        c_out = (c @ W_norm_T).clamp(0.0, 1.0)
        obs_out = ((1.0 - m) @ W_norm_T)
        m_out = (1.0 - obs_out).clamp(0.0, 1.0)

        return y, m_out, c_out


class NoChannelProp(nn.Module):

    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x, m, c):
        y = self.linear(x)
        out_dim = y.shape[1]

        m_avg = m.mean(dim=1, keepdim=True)
        c_avg = c.mean(dim=1, keepdim=True)

        m_out = m_avg.expand(-1, out_dim)
        c_out = c_avg.expand(-1, out_dim)

        return y, m_out, c_out


class AdaptiveLayer(nn.Module):

    def __init__(self, activation_tree):
        super().__init__()
        self.activation_tree = activation_tree

    def forward(self, x, m, c):
        return evaluate_tree(self.activation_tree, x, m, c)


class ThreeChannelBlock(nn.Module):

    def __init__(self, in_dim, out_dim, activation_tree, use_channel_prop=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.prop = ChannelProp(self.lin) if use_channel_prop else NoChannelProp(self.lin)
        self.act = AdaptiveLayer(activation_tree)

    def forward(self, x, m, c):
        x, m, c = self.prop(x, m, c)
        x = self.act(x, m, c)
        return x, m, c


class ThreeChannelMLP(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, activation_tree, use_channel_prop=True):
        super().__init__()
        sizes = [input_size] + list(hidden_sizes)
        self.blocks = nn.ModuleList([
            ThreeChannelBlock(sizes[i], sizes[i+1], activation_tree, use_channel_prop)
            for i in range(len(hidden_sizes))
        ])
        self.out = nn.Linear(sizes[-1], output_size)

    def forward(self, values, missing_flags, confidence_scores):
        x, m, c = values, missing_flags, confidence_scores
        for block in self.blocks:
            x, m, c = block(x, m, c)
        return self.out(x)



def evaluate_fitness(tree, X_train, y_train, X_val, y_val,
                     missing_train, missing_val, conf_train, conf_val,
                     num_epochs=50, hidden_sizes=None, use_channel_prop=True,
                     allowed_inputs=INPUTS_FULL):

    if hidden_sizes is None:
        hidden_sizes = [32, 16]

    if get_tree_depth(tree) == 1:
        return 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        missing_train = missing_train.to(device)
        missing_val = missing_val.to(device)
        conf_train = conf_train.to(device)
        conf_val = conf_val.to(device)

        model = ThreeChannelMLP(
            input_size=X_train.shape[1],
            hidden_sizes=hidden_sizes,
            output_size=len(torch.unique(y_train)),
            activation_tree=tree,
            use_channel_prop=use_channel_prop,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        patience, patience_counter = 2, 0

        for _ in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train, missing_train, conf_train)
            loss = criterion(outputs, y_train)

            if not torch.isfinite(loss):
                return 0.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val, missing_val, conf_val)
                _, preds = torch.max(val_outputs, 1)
                val_acc = (preds == y_val).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        size_penalty = 0.001 * count_nodes(tree)
        depth_penalty = 0.002 * (get_tree_depth(tree) - 1)
        diversity = count_input_diversity(tree) / len(allowed_inputs)

        return best_val_acc + 0.01 * diversity - size_penalty - depth_penalty

    except Exception as e:
        print(f"Error: {e}")
        return 0.0


def run_evolution(X_train, y_train, X_val, y_val,
                  missing_train, missing_val, conf_train, conf_val,
                  population_size=30, generations=20, max_depth=4,
                  mutation_rate=0.15, crossover_rate=0.7, elite_size=2,
                  hidden_sizes=None, num_epochs_fitness=30,
                  use_channel_prop=True, allowed_inputs=INPUTS_FULL,
                  verbose=True):

    if hidden_sizes is None:
        hidden_sizes = [32, 16]

    population = create_initial_population(population_size, max_depth, allowed_inputs)
    best_ever = None
    best_ever_fitness = -float("inf")

    for gen in range(generations):
        for ind in population:
            if ind["fitness"] is None:
                ind["fitness"] = evaluate_fitness(
                    ind["tree"], X_train, y_train, X_val, y_val,
                    missing_train, missing_val, conf_train, conf_val,
                    num_epochs=num_epochs_fitness, hidden_sizes=hidden_sizes,
                    use_channel_prop=use_channel_prop, allowed_inputs=allowed_inputs,
                )

        population.sort(key=lambda x: x["fitness"], reverse=True)
        best_ind = population[0]

        if best_ind["fitness"] > best_ever_fitness:
            best_ever_fitness = best_ind["fitness"]
            best_ever = copy.deepcopy(best_ind)
            if verbose:
                print(f"  Gen {gen+1}: New best fitness = {best_ever_fitness:.4f}")

        if gen < generations - 1:
            population = evolve_population(
                population, mutation_rate, crossover_rate, elite_size,
                max_depth, allowed_inputs
            )

    return best_ever


def train_final_model(activation_tree, X_train, y_train, X_val, y_val,
                      missing_train, missing_val, conf_train, conf_val,
                      hidden_sizes=None, epochs=100, patience=15,
                      use_channel_prop=True):

    if hidden_sizes is None:
        hidden_sizes = [64, 32]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    missing_train = missing_train.to(device)
    missing_val = missing_val.to(device)
    conf_train = conf_train.to(device)
    conf_val = conf_val.to(device)

    model = ThreeChannelMLP(
        input_size=X_train.shape[1],
        hidden_sizes=hidden_sizes,
        output_size=len(torch.unique(y_train)),
        activation_tree=activation_tree,
        use_channel_prop=use_channel_prop,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train, missing_train, conf_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val, missing_val, conf_val)
            _, val_pred = torch.max(val_outputs, 1)
            val_acc = (val_pred == y_val).float().mean().item()

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, best_val_acc


def load_glass_dataset(filepath="glass.csv"):
    df = pd.read_csv(filepath)
    if df.columns[0].lower() in ["id", "index"]:
        df = df.iloc[:, 1:]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    print(f"Loaded Glass: {df.shape[0]} samples, {df.shape[1]-1} features, {df.iloc[:,-1].nunique()} classes")
    return df


def introduce_missing_data(df, missing_rate=0.10, mechanism="MCAR", seed=42):
    np.random.seed(seed)
    df_missing = df.copy()
    n_rows, n_cols = df.shape
    missing_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    if mechanism == "MCAR":
        total = n_rows * n_cols
        n_missing = int(total * missing_rate)
        positions = np.random.choice(total, size=n_missing, replace=False)
        for pos in positions:
            df_missing.iloc[pos // n_cols, pos % n_cols] = np.nan
            missing_mask.iloc[pos // n_cols, pos % n_cols] = True
    elif mechanism == "MAR":
        first_col = df.iloc[:, 0].values
        threshold = np.percentile(first_col, 50)
        for col_idx in range(1, n_cols):
            col = df.columns[col_idx]
            probs = np.where(first_col > threshold, missing_rate * 2, missing_rate * 0.5)
            mask = np.random.random(n_rows) < probs
            df_missing.loc[mask, col] = np.nan
            missing_mask.loc[mask, col] = True
    elif mechanism == "MNAR":
        for col in df.columns:
            vals = df[col].values
            lower, upper = np.percentile(vals, 25), np.percentile(vals, 75)
            is_extreme = (vals < lower) | (vals > upper)
            probs = np.where(is_extreme, missing_rate * 1.5, missing_rate * 0.5)
            mask = np.random.random(n_rows) < probs
            df_missing.loc[mask, col] = np.nan
            missing_mask.loc[mask, col] = True

    return df_missing, missing_mask


def prepare_data(df, missing_rate, mechanism, seed):
    X_full = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_missing, _ = introduce_missing_data(X_full, missing_rate, mechanism, seed)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_missing, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    train_means = X_train.mean()
    train_missing_rates = {}

    def process_split(X, train_means, train_missing_rates, fit=False):
        values = X.copy()
        missing_flags = X.isnull().astype(float)
        confidence = (~X.isnull()).astype(float)
        values = values.fillna(train_means)

        for col in X.columns:
            if fit:
                mr = missing_flags[col].mean()
                train_missing_rates[col] = mr
            else:
                mr = train_missing_rates[col]
            imputed_conf = max(0.1, 1.0 - mr)
            confidence.loc[missing_flags[col] == 1, col] = imputed_conf

        return values, missing_flags, confidence

    val_train, miss_train, conf_train = process_split(X_train, train_means, train_missing_rates, fit=True)
    val_val, miss_val, conf_val = process_split(X_val, train_means, train_missing_rates)
    val_test, miss_test, conf_test = process_split(X_test, train_means, train_missing_rates)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(val_train)
    X_val_scaled = scaler.transform(val_val)
    X_test_scaled = scaler.transform(val_test)

    data = {
        "X_train": torch.tensor(X_train_scaled, dtype=torch.float32),
        "X_val": torch.tensor(X_val_scaled, dtype=torch.float32),
        "X_test": torch.tensor(X_test_scaled, dtype=torch.float32),
        "y_train": torch.tensor(y_train_enc, dtype=torch.long),
        "y_val": torch.tensor(y_val_enc, dtype=torch.long),
        "y_test": torch.tensor(y_test_enc, dtype=torch.long),
        "miss_train": torch.tensor(miss_train.values, dtype=torch.float32),
        "miss_val": torch.tensor(miss_val.values, dtype=torch.float32),
        "miss_test": torch.tensor(miss_test.values, dtype=torch.float32),
        "conf_train": torch.tensor(conf_train.values, dtype=torch.float32),
        "conf_val": torch.tensor(conf_val.values, dtype=torch.float32),
        "conf_test": torch.tensor(conf_test.values, dtype=torch.float32),
    }

    return data



def run_single_ablation(data, variant_name, allowed_inputs, use_channel_prop,
                        use_evolution, fixed_tree=None, config=None):


    print(f"\n{'='*60}")
    print(f"ABLATION: {variant_name}")
    print(f"{'='*60}")

    miss_train = data["miss_train"]
    miss_val = data["miss_val"]
    miss_test = data["miss_test"]
    conf_train = data["conf_train"]
    conf_val = data["conf_val"]
    conf_test = data["conf_test"]

    if "missing" not in allowed_inputs:
        miss_train = torch.zeros_like(miss_train)
        miss_val = torch.zeros_like(miss_val)
        miss_test = torch.zeros_like(miss_test)

    if "confidence" not in allowed_inputs:
        conf_train = torch.ones_like(conf_train)
        conf_val = torch.ones_like(conf_val)
        conf_test = torch.ones_like(conf_test)

    if use_evolution:
        print("  Running evolution...")
        best_activation = run_evolution(
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
            miss_train, miss_val,
            conf_train, conf_val,
            population_size=config["population_size"],
            generations=config["generations"],
            max_depth=config["max_depth"],
            mutation_rate=config["mutation_rate"],
            crossover_rate=config["crossover_rate"],
            elite_size=config["elite_size"],
            hidden_sizes=config["hidden_sizes_fitness"],
            num_epochs_fitness=config["num_epochs_fitness"],
            use_channel_prop=use_channel_prop,
            allowed_inputs=allowed_inputs,
            verbose=True,
        )
        tree = best_activation["tree"]
        formula = tree_to_infix(tree)
    else:
        tree = fixed_tree
        formula = tree_to_infix(tree) if tree else "N/A"

    print(f"  Formula: f(x,m,c) = {formula}")
    print("  Training final model...")

    model, val_acc = train_final_model(
        tree,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        miss_train, miss_val,
        conf_train, conf_val,
        hidden_sizes=config["hidden_sizes_final"],
        epochs=config["final_epochs"],
        patience=config["final_patience"],
        use_channel_prop=use_channel_prop,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        test_out = model(
            data["X_test"].to(device),
            miss_test.to(device),
            conf_test.to(device),
        )
        _, test_pred = torch.max(test_out, 1)
        test_acc = (test_pred == data["y_test"].to(device)).float().mean().item()

        y_true = data["y_test"].cpu().numpy()
        y_pred = test_pred.cpu().numpy()
        y_probs = torch.softmax(test_out, dim=1).cpu().numpy()

    metrics = compute_metrics(y_true, y_pred, y_probs)

    print(f"  Val Acc:  {val_acc:.4f}")
    print(f"  Test Acc: {test_acc:.4f}")
    print(f"  F1:       {metrics['f1']:.4f}")
    print(f"  AUC:      {metrics['auc']:.4f}")

    return {
        "variant": variant_name,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "specificity": metrics["specificity"],
        "f1": metrics["f1"],
        "auc": metrics["auc"],
        "formula": formula,
    }


def run_ablation_study(df, missing_rate, mechanism, num_runs, config):
    variants = [
        {
            "name": "Full System (x,m,c)",
            "allowed_inputs": INPUTS_FULL,
            "use_channel_prop": True,
            "use_evolution": True,
            "fixed_tree": None,
        },
        {
            "name": "No Confidence (x,m)",
            "allowed_inputs": INPUTS_NO_CONF,
            "use_channel_prop": True,
            "use_evolution": True,
            "fixed_tree": None,
        },
        {
            "name": "No Missing Flag (x,c)",
            "allowed_inputs": INPUTS_NO_MISS,
            "use_channel_prop": True,
            "use_evolution": True,
            "fixed_tree": None,
        },
        {
            "name": "No Channel Prop",
            "allowed_inputs": INPUTS_FULL,
            "use_channel_prop": False,
            "use_evolution": True,
            "fixed_tree": None,
        },
    ]

    all_results = {v["name"]: [] for v in variants}

    for run_idx in range(num_runs):
        run_seed = config["base_seed"] + run_idx
        print(f"\n{'#'*70}")
        print(f"# RUN {run_idx + 1}/{num_runs} (base seed={run_seed})")
        print(f"{'#'*70}")


        set_seed(run_seed)
        data = prepare_data(df, missing_rate, mechanism, run_seed)

        for variant_idx, variant in enumerate(variants):

            variant_seed = run_seed + 1000 * (variant_idx + 1)
            set_seed(variant_seed)

            result = run_single_ablation(
                data,
                variant["name"],
                variant["allowed_inputs"],
                variant["use_channel_prop"],
                variant["use_evolution"],
                variant["fixed_tree"],
                config,
            )
            all_results[variant["name"]].append(result)

    return all_results, variants


def print_ablation_summary(all_results, variants):


    print("\n" + "=" * 100)
    print("ABLATION STUDY RESULTS (MEAN ± STD)")
    print("=" * 100)
    print(f"{'Variant':<25} {'Test Acc':<15} {'F1':<15} {'AUC':<15} {'Δ Acc':<10}")
    print("-" * 100)

    full_results = all_results["Full System (x,m,c)"]
    full_test_mean = np.mean([r["test_acc"] for r in full_results])

    for variant in variants:
        name = variant["name"]
        results = all_results[name]

        test_accs = [r["test_acc"] for r in results]
        f1s = [r["f1"] for r in results]
        aucs = [r["auc"] for r in results]

        test_mean, test_std = np.mean(test_accs), np.std(test_accs)
        f1_mean, f1_std = np.mean(f1s), np.std(f1s)
        auc_mean, auc_std = np.mean(aucs), np.std(aucs)

        delta = test_mean - full_test_mean
        delta_str = f"{delta:+.2%}" if name != "Full System (x,m,c)" else "—"

        print(f"{name:<25} "
              f"{test_mean:.4f}±{test_std:.4f}   "
              f"{f1_mean:.4f}±{f1_std:.4f}   "
              f"{auc_mean:.4f}±{auc_std:.4f}   "
              f"{delta_str}")

    print("=" * 100)


# MAIN


if __name__ == "__main__":

    # Configuration
    CONFIG = {
        # Data settings
        "missing_rate": 0.40,
        "mechanism": "MNAR",

        # Evolution settings
        "population_size": 50,
        "generations": 15,
        "max_depth": 3,
        "mutation_rate": 0.15,
        "crossover_rate": 0.70,
        "elite_size": 2,

        # NN settings
        "hidden_sizes_fitness": [32, 16],
        "hidden_sizes_final": [64, 32],
        "num_epochs_fitness": 30,
        "final_epochs": 100,
        "final_patience": 15,

        # Experiment settings
        "base_seed": 42,
        "num_runs": 2,
    }

    print("=" * 70)
    print("ABLATION STUDY: Evolved Activation Functions for Missing Data")
    print("Dataset: Glass | Missing: 20% MAR")
    print("=" * 70)

    # Load dataset
    df = load_glass_dataset("glass.csv")


    all_results, variants = run_ablation_study(
        df,
        CONFIG["missing_rate"],
        CONFIG["mechanism"],
        CONFIG["num_runs"],
        CONFIG,
    )

    print_ablation_summary(all_results, variants)

    # Save results
    os.makedirs("ablation_results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")


    save_data = {
        "config": CONFIG,
        "results": {
            name: [
                {k: v for k, v in r.items() if k != "model"}
                for r in results
            ]
            for name, results in all_results.items()
        }
    }

    with open(f"ablation_results/ablation_glass_{timestamp}.json", "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n Results saved to ablation_results/ablation_glass_{timestamp}.json")