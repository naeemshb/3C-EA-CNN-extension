# =============================================================================
# ablation_study.py - Ablation Study for Three-Channel Activation Functions
# =============================================================================
# CORRECTED VERSION
#
# This script tests the contribution of each channel (value, missing, confidence)
# by systematically removing them and measuring performance drop.
#
# Strategy:
#   1. Evolve activation function with FULL channels (x, m, c)
#   2. Train model with full channels
#   3. Evaluate SAME trained model but with ablated channel inputs
#   This tests: "Does the model actually USE m and c channels?"
#
# Run in the same folder as hepatitis.csv
# =============================================================================

import math
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "dataset": "horse.csv",
    "num_runs": 5,
    "test_size": 0.4,
    "base_seed": 42,

    # Evolution parameters
    "population_size": 50,
    "generations": 15,
    "max_tree_depth": 3,
    "mutation_rate": 0.15,
    "crossover_rate": 0.70,
    "elite_size": 2,

    # Neural network parameters
    "hidden_sizes": [64, 32],
    "num_epochs_fitness": 30,
    "final_epochs": 100,
    "final_patience": 15,
}

# =============================================================================
# TREE STRUCTURE COMPONENTS
# =============================================================================

INPUTS = ["value", "missing", "confidence"]
CONSTANTS = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1]


def create_input_node(input_name):
    return {"type": "input", "input_name": input_name, "children": []}


def create_constant_node(value):
    return {"type": "constant", "value": float(value), "children": []}


def create_unary_node(operator, child):
    return {"type": "unary", "operator": operator, "children": [child]}


def create_binary_node(operator, left_child, right_child):
    return {"type": "binary", "operator": operator, "children": [left_child, right_child]}


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
    "softplus": lambda x: torch.nn.functional.softplus(x),
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
        }
        return mapping.get(op, f"{op}({s})")
    if t == "binary":
        op = node["operator"]
        a = tree_to_infix(node["children"][0])
        b = tree_to_infix(node["children"][1])
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


def generate_terminal_node():
    if random.random() < 0.7:
        return create_input_node(random.choice(INPUTS))
    return create_constant_node(random.choice(CONSTANTS))


def generate_random_tree(max_depth, current_depth=0):
    if current_depth >= max_depth:
        return generate_terminal_node()
    terminal_prob = 0.3 + 0.4 * (current_depth / max_depth)
    if random.random() < terminal_prob:
        return generate_terminal_node()
    if random.random() < 0.4:
        op = random.choice(list(UNARY_OPERATORS.keys()))
        child = generate_random_tree(max_depth, current_depth + 1)
        return create_unary_node(op, child)
    op = random.choice(list(BINARY_OPERATORS.keys()))
    left = generate_random_tree(max_depth, current_depth + 1)
    right = generate_random_tree(max_depth, current_depth + 1)
    return create_binary_node(op, left, right)


def validate_tree(node):
    t = node["type"]
    if t == "input":
        assert node["input_name"] in INPUTS
    elif t == "constant":
        assert isinstance(node["value"], (int, float))
    elif t == "unary":
        assert node["operator"] in UNARY_OPERATORS
        validate_tree(node["children"][0])
    elif t == "binary":
        assert node["operator"] in BINARY_OPERATORS
        validate_tree(node["children"][0])
        validate_tree(node["children"][1])


def get_all_nodes(tree):
    nodes = [tree]
    if tree["type"] in ["unary", "binary"]:
        for ch in tree["children"]:
            nodes.extend(get_all_nodes(ch))
    return nodes


def mutate_tree(tree, mutation_rate=0.1):
    mutated = copy.deepcopy(tree)
    for node in get_all_nodes(mutated):
        if random.random() < mutation_rate:
            t = node["type"]
            if t == "input":
                node["input_name"] = random.choice(INPUTS)
            elif t == "constant":
                node["value"] = random.choice(CONSTANTS)
            elif t == "unary":
                node["operator"] = random.choice(list(UNARY_OPERATORS.keys()))
            elif t == "binary":
                node["operator"] = random.choice(list(BINARY_OPERATORS.keys()))
    validate_tree(mutated)
    return mutated


def crossover_trees(parent1, parent2):
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
        validate_tree(offspring1)
        validate_tree(offspring2)
    except:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)
    return offspring1, offspring2


# =============================================================================
# NEURAL NETWORK
# =============================================================================

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
        c_out = c @ W_norm_T
        obs_in = 1.0 - m
        obs_out = obs_in @ W_norm_T
        m_out = 1.0 - obs_out
        return y, m_out.clamp(0, 1), c_out.clamp(0, 1)


class AdaptiveLayer(nn.Module):
    def __init__(self, activation_tree):
        super().__init__()
        self.activation_tree = activation_tree

    def forward(self, x, m, c):
        return evaluate_tree(self.activation_tree, x, m, c)


class ThreeChannelBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation_tree):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.prop = ChannelProp(self.lin)
        self.act = AdaptiveLayer(activation_tree)

    def forward(self, x, m, c):
        x, m, c = self.prop(x, m, c)
        x = self.act(x, m, c)
        return x, m, c


class ThreeChannelMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_tree):
        super().__init__()
        sizes = [input_size] + list(hidden_sizes)
        self.blocks = nn.ModuleList([
            ThreeChannelBlock(sizes[i], sizes[i + 1], activation_tree)
            for i in range(len(hidden_sizes))
        ])
        self.out = nn.Linear(sizes[-1], output_size)

    def forward(self, values, missing_flags, confidence_scores):
        x, m, c = values, missing_flags, confidence_scores
        for block in self.blocks:
            x, m, c = block(x, m, c)
        return self.out(x)


# =============================================================================
# METRICS
# =============================================================================

def compute_multiclass_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec_i)
    return np.mean(specificities)


def compute_metrics(y_true, y_pred, y_prob):
    num_classes = len(np.unique(y_true))
    avg_method = "binary" if num_classes == 2 else "macro"

    acc = (y_pred == y_true).mean()
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=avg_method, zero_division=0
    )
    spec = compute_multiclass_specificity(y_true, y_pred)

    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except:
        auc = 0.0

    return acc, prec, rec, spec, f1, auc


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_horse_colic(csv_path="horse.csv"):
    """Load Horse Colic dataset with proper handling."""
    df = pd.read_csv(csv_path, na_values='?', header=0)
    print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")

    # Use 'outcome' as target (died/euthanized/lived)
    if 'outcome' in df.columns:
        target_col = 'outcome'
    else:
        target_col = df.columns[22]  # Fallback to column index

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])

    # Drop non-useful columns
    cols_to_drop = ['hospital_number', 'lesion_1', 'lesion_2', 'lesion_3',
                    'cp_data', 'surgical_lesion']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # Separate features and target
    target_data = df[target_col]
    X = df.drop(columns=[target_col])

    # Encode categorical features
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.Categorical(X[col]).codes
        X[col] = X[col].replace(-1, np.nan)

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(target_data)

    missing_rate = X.isnull().sum().sum() / X.size * 100
    print(f"Samples: {len(y)}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    print(f"Missing rate: {missing_rate:.1f}%")
    print(f"Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y


def load_dataset(csv_path):
    """Load dataset based on filename."""
    if 'horse' in csv_path.lower():
        return load_horse_colic(csv_path)
    elif 'hepatitis' in csv_path.lower():
        return load_hepatitis(csv_path)
    else:
        # Generic loader
        df = pd.read_csv(csv_path, na_values='?')
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        y = LabelEncoder().fit_transform(y)
        return X, y


def load_hepatitis(csv_path="hepatitis.csv"):
    df = pd.read_csv(csv_path, na_values='?')
    print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")

    # Target is first or last column
    if df.iloc[:, 0].nunique() <= 3:
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

    mask = ~y.isnull()
    X = X[mask]
    y = y[mask]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X = X.apply(pd.to_numeric, errors='coerce')

    missing_rate = X.isnull().sum().sum() / X.size * 100
    print(f"Samples: {len(y)}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    print(f"Missing rate: {missing_rate:.1f}%")

    return X, y


# =============================================================================
# EVOLUTION
# =============================================================================

def evaluate_fitness(tree, X_train, y_train, X_val, y_val,
                     missing_train, missing_val, conf_train, conf_val,
                     num_epochs=30, hidden_sizes=[32, 16]):
    if get_tree_depth(tree) == 1:
        return 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = ThreeChannelMLP(
            input_size=X_train.shape[1],
            hidden_sizes=hidden_sizes,
            output_size=len(torch.unique(y_train)),
            activation_tree=tree,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        X_tr = X_train.to(device)
        y_tr = y_train.to(device)
        X_v = X_val.to(device)
        y_v = y_val.to(device)
        m_tr = missing_train.to(device)
        m_v = missing_val.to(device)
        c_tr = conf_train.to(device)
        c_v = conf_val.to(device)

        best_val_acc = 0.0

        for _ in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_tr, m_tr, c_tr)
            loss = criterion(outputs, y_tr)
            if not torch.isfinite(loss):
                return 0.0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_v, m_v, c_v)
                val_acc = (val_outputs.argmax(1) == y_v).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        return best_val_acc - 0.0001 * count_nodes(tree)

    except:
        return 0.0


def run_evolution(X_train, y_train, X_val, y_val,
                  missing_train, missing_val, conf_train, conf_val, config):
    population = [{"tree": generate_random_tree(config["max_tree_depth"]), "fitness": None}
                  for _ in range(config["population_size"])]

    best_ever = None
    best_ever_fitness = -float("inf")

    for gen in range(config["generations"]):
        for ind in population:
            if ind["fitness"] is None:
                ind["fitness"] = evaluate_fitness(
                    ind["tree"], X_train, y_train, X_val, y_val,
                    missing_train, missing_val, conf_train, conf_val,
                    num_epochs=config["num_epochs_fitness"],
                    hidden_sizes=config["hidden_sizes"]
                )

        population.sort(key=lambda x: x["fitness"], reverse=True)

        if population[0]["fitness"] > best_ever_fitness:
            best_ever_fitness = population[0]["fitness"]
            best_ever = copy.deepcopy(population[0])

        new_population = []
        for i in range(config["elite_size"]):
            new_population.append({"tree": copy.deepcopy(population[i]["tree"]), "fitness": None})

        while len(new_population) < config["population_size"]:
            if random.random() < config["crossover_rate"]:
                p1 = random.choice(population[:config["population_size"] // 2])
                p2 = random.choice(population[:config["population_size"] // 2])
                off1, off2 = crossover_trees(p1["tree"], p2["tree"])
                new_population.append({"tree": off1, "fitness": None})
                if len(new_population) < config["population_size"]:
                    new_population.append({"tree": off2, "fitness": None})
            else:
                parent = random.choice(population[:config["population_size"] // 2])
                mutated = mutate_tree(parent["tree"], config["mutation_rate"])
                new_population.append({"tree": mutated, "fitness": None})

        population = new_population
        print(f"    Gen {gen + 1}/{config['generations']} - Best: {best_ever_fitness:.4f}")

    return best_ever


def train_final_model(activation_tree, X_train, y_train, X_val, y_val,
                      missing_train, missing_val, conf_train, conf_val, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ThreeChannelMLP(
        input_size=X_train.shape[1],
        hidden_sizes=config["hidden_sizes"],
        output_size=len(torch.unique(y_train)),
        activation_tree=activation_tree,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    X_tr = X_train.to(device)
    y_tr = y_train.to(device)
    X_v = X_val.to(device)
    y_v = y_val.to(device)
    m_tr = missing_train.to(device)
    m_v = missing_val.to(device)
    c_tr = conf_train.to(device)
    c_v = conf_val.to(device)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(config["final_epochs"]):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tr, m_tr, c_tr)
        loss = criterion(outputs, y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_v, m_v, c_v)
            val_acc = (val_outputs.argmax(1) == y_v).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["final_patience"]:
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, best_val_acc


def evaluate_model(model, X_test, y_test, missing_test, conf_test, device):
    """Evaluate model and return metrics."""
    model.eval()
    model = model.to(device)

    X_t = X_test.to(device)
    m_t = missing_test.to(device)
    c_t = conf_test.to(device)

    with torch.no_grad():
        outputs = model(X_t, m_t, c_t)
        y_pred = outputs.argmax(1).cpu().numpy()
        y_prob = torch.softmax(outputs, dim=1).cpu().numpy()

    acc, prec, rec, spec, f1, auc = compute_metrics(y_test, y_pred, y_prob)
    return acc, prec, rec, spec, f1, auc


# =============================================================================
# ABLATION STUDY
# =============================================================================

def run_ablation_experiment(X, y, seed, config):
    """
    Run ablation experiment.

    Strategy:
    1. Evolve activation function with FULL channels (x, m, c)
    2. Train final model with full channels
    3. Re-evaluate the SAME trained model but with ablated channels

    This tests: "Does the trained model actually USE m and c channels?"
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    # Imputation from training set only
    train_means = X_train.mean()

    # TRAIN channels
    missing_train = X_train.isnull().astype(float)
    confidence_train = (~X_train.isnull()).astype(float)
    values_train = X_train.fillna(train_means)

    for col in X_train.columns:
        mr = missing_train[col].mean()
        imputed_conf = max(0.1, 1.0 - mr)
        confidence_train.loc[missing_train[col] == 1, col] = imputed_conf

    # VAL channels
    missing_val = X_val.isnull().astype(float)
    confidence_val = (~X_val.isnull()).astype(float)
    values_val = X_val.fillna(train_means)

    for col in X_val.columns:
        mr = X_train[col].isnull().mean()
        imputed_conf = max(0.1, 1.0 - mr)
        confidence_val.loc[missing_val[col] == 1, col] = imputed_conf

    # TEST channels
    missing_test = X_test.isnull().astype(float)
    confidence_test = (~X_test.isnull()).astype(float)
    values_test = X_test.fillna(train_means)

    for col in X_test.columns:
        mr = X_train[col].isnull().mean()
        imputed_conf = max(0.1, 1.0 - mr)
        confidence_test.loc[missing_test[col] == 1, col] = imputed_conf

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(values_train)
    X_val_scaled = scaler.transform(values_val)
    X_test_scaled = scaler.transform(values_test)

    # To tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    missing_train_t = torch.tensor(missing_train.values, dtype=torch.float32)
    missing_val_t = torch.tensor(missing_val.values, dtype=torch.float32)
    missing_test_t = torch.tensor(missing_test.values, dtype=torch.float32)

    conf_train_t = torch.tensor(confidence_train.values, dtype=torch.float32)
    conf_val_t = torch.tensor(confidence_val.values, dtype=torch.float32)
    conf_test_t = torch.tensor(confidence_test.values, dtype=torch.float32)

    # Ablated versions
    zeros_test = torch.zeros_like(missing_test_t)
    ones_test = torch.ones_like(conf_test_t)

    results = {}

    # Step 1: Evolve with FULL channels
    print("\n  Evolving activation with full channels (x, m, c)...")

    best_activation = run_evolution(
        X_train_t, y_train_t, X_val_t, y_val_t,
        missing_train_t, missing_val_t, conf_train_t, conf_val_t, config
    )

    evolved_formula = tree_to_infix(best_activation["tree"])
    uses_m = "m" in evolved_formula
    uses_c = "c" in evolved_formula
    print(f"\n  Evolved: {evolved_formula}")
    print(f"  Uses m: {uses_m}, Uses c: {uses_c}")

    # Step 2: Train final model
    print("\n  Training final model...")

    model, val_acc = train_final_model(
        best_activation["tree"],
        X_train_t, y_train_t, X_val_t, y_val_t,
        missing_train_t, missing_val_t, conf_train_t, conf_val_t, config
    )

    # Step 3: Evaluate with different channel configs
    print("\n  Evaluating ablations...")

    # Full (x, m, c)
    acc, prec, rec, spec, f1, auc = evaluate_model(
        model, X_test_t, y_test, missing_test_t, conf_test_t, device
    )
    results["Full (x,m,c)"] = {"acc": acc, "prec": prec, "rec": rec, "spec": spec, "f1": f1, "auc": auc}
    print(f"  Full (x,m,c):      Acc={acc:.4f}, F1={f1:.4f}")

    # No Confidence (x, m, 1)
    acc, prec, rec, spec, f1, auc = evaluate_model(
        model, X_test_t, y_test, missing_test_t, ones_test, device
    )
    results["No Conf (x,m,1)"] = {"acc": acc, "prec": prec, "rec": rec, "spec": spec, "f1": f1, "auc": auc}
    print(f"  No Conf (x,m,1):   Acc={acc:.4f}, F1={f1:.4f}")

    # No Missing (x, 0, c)
    acc, prec, rec, spec, f1, auc = evaluate_model(
        model, X_test_t, y_test, zeros_test, conf_test_t, device
    )
    results["No Miss (x,0,c)"] = {"acc": acc, "prec": prec, "rec": rec, "spec": spec, "f1": f1, "auc": auc}
    print(f"  No Miss (x,0,c):   Acc={acc:.4f}, F1={f1:.4f}")

    # Value Only (x, 0, 1)
    acc, prec, rec, spec, f1, auc = evaluate_model(
        model, X_test_t, y_test, zeros_test, ones_test, device
    )
    results["Value Only (x,0,1)"] = {"acc": acc, "prec": prec, "rec": rec, "spec": spec, "f1": f1, "auc": auc}
    print(f"  Value Only (x,0,1): Acc={acc:.4f}, F1={f1:.4f}")

    return results, evolved_formula, uses_m, uses_c


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

        def close(self):
            self.log.close()


    output_file = "ablation_study_horsecolic_results.txt"
    sys.stdout = Logger(output_file)

    print("=" * 80)
    print("ABLATION STUDY: Three-Channel Activation Functions")
    print("=" * 80)
    print(f"Dataset: {CONFIG['dataset']}")
    print(f"Number of runs: {CONFIG['num_runs']}")
    print("=" * 80)

    print("\nStrategy:")
    print("  1. Evolve activation with FULL channels (x, m, c)")
    print("  2. Train model with full channels")
    print("  3. Evaluate SAME model with ablated inputs")
    print("  This tests if the model actually USES m and c.")
    print("=" * 80)

    # Load dataset
    X, y = load_dataset(CONFIG["dataset"])

    # Store results
    all_results = {
        "Full (x,m,c)": {"acc": [], "prec": [], "rec": [], "spec": [], "f1": [], "auc": []},
        "No Conf (x,m,1)": {"acc": [], "prec": [], "rec": [], "spec": [], "f1": [], "auc": []},
        "No Miss (x,0,c)": {"acc": [], "prec": [], "rec": [], "spec": [], "f1": [], "auc": []},
        "Value Only (x,0,1)": {"acc": [], "prec": [], "rec": [], "spec": [], "f1": [], "auc": []},
    }

    evolved_formulas = []
    uses_m_list = []
    uses_c_list = []

    # Run experiments
    for run in range(CONFIG["num_runs"]):
        seed = CONFIG["base_seed"] + run
        print(f"\n{'=' * 80}")
        print(f"RUN {run + 1}/{CONFIG['num_runs']} (seed={seed})")
        print("=" * 80)

        results, formula, uses_m, uses_c = run_ablation_experiment(X, y, seed, CONFIG)

        evolved_formulas.append(formula)
        uses_m_list.append(uses_m)
        uses_c_list.append(uses_c)

        for variant, metrics in results.items():
            for metric, value in metrics.items():
                all_results[variant][metric].append(value)

    # Print formulas
    print("\n" + "=" * 80)
    print("EVOLVED FORMULAS")
    print("=" * 80)
    for i, formula in enumerate(evolved_formulas):
        print(f"Run {i + 1}: {formula}")
        print(f"       Uses m: {uses_m_list[i]}, Uses c: {uses_c_list[i]}")

    # Print results
    print("\n" + "=" * 100)
    print("ABLATION RESULTS (mean ± std)")
    print("=" * 100)
    print(
        f"{'Variant':<20} {'Accuracy':<14} {'Precision':<14} {'Recall':<14} {'Specificity':<14} {'F1':<14} {'AUC':<14}")
    print("-" * 100)

    for variant in all_results:
        res = all_results[variant]
        print(f"{variant:<20} "
              f"{np.mean(res['acc']):.4f}±{np.std(res['acc']):.4f}  "
              f"{np.mean(res['prec']):.4f}±{np.std(res['prec']):.4f}  "
              f"{np.mean(res['rec']):.4f}±{np.std(res['rec']):.4f}  "
              f"{np.mean(res['spec']):.4f}±{np.std(res['spec']):.4f}  "
              f"{np.mean(res['f1']):.4f}±{np.std(res['f1']):.4f}  "
              f"{np.mean(res['auc']):.4f}±{np.std(res['auc']):.4f}")

    print("=" * 100)

    # Performance drop
    print("\n" + "=" * 60)
    print("PERFORMANCE DROP ANALYSIS")
    print("=" * 60)

    full_acc = np.mean(all_results["Full (x,m,c)"]["acc"])
    full_f1 = np.mean(all_results["Full (x,m,c)"]["f1"])

    for variant in ["No Conf (x,m,1)", "No Miss (x,0,c)", "Value Only (x,0,1)"]:
        var_acc = np.mean(all_results[variant]["acc"])
        var_f1 = np.mean(all_results[variant]["f1"])

        print(f"\n{variant}:")
        print(f"  Accuracy drop: {(full_acc - var_acc) * 100:+.2f}% points")
        print(f"  F1 drop:       {(full_f1 - var_f1) * 100:+.2f}% points")

    print("\n" + "=" * 60)
    print("CHANNEL USAGE")
    print("=" * 60)
    print(f"Evolved functions using 'm': {sum(uses_m_list)}/{len(uses_m_list)}")
    print(f"Evolved functions using 'c': {sum(uses_c_list)}/{len(uses_c_list)}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_file}")

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print(f"Results saved to: {output_file}")