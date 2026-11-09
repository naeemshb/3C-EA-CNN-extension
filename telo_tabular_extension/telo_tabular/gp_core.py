"""GP primitives and ChannelProp copied unchanged from GECCO 2026 PIMA source.
See reference/SOURCE.md for provenance and documented protocol differences.
"""
import math
import random
import copy
import torch
import torch.nn as nn


# ============================================================================
# TREE STRUCTURE COMPONENTS
# ============================================================================

INPUTS = ["value", "missing", "confidence"]
CONSTANTS = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1]


def create_input_node(input_name: str) -> dict:
    assert input_name in INPUTS, f"Invalid input name: {input_name}"
    return {"type": "input", "input_name": input_name, "children": []}


def create_constant_node(value: float) -> dict:
    return {"type": "constant", "value": float(value), "children": []}


def create_unary_node(operator: str, child: dict) -> dict:
    return {"type": "unary", "operator": operator, "children": [child]}


def create_binary_node(operator: str, left_child: dict, right_child: dict) -> dict:
    return {"type": "binary", "operator": operator, "children": [left_child, right_child]}


def safe_divide(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Safe division with small epsilon to avoid division by zero."""
    denom = torch.where(
        torch.abs(y) < 1e-3,
        torch.sign(y) * 1e-3 + (y == 0).float() * 1e-3,
        y,
    )
    return x / denom


def safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    """Safe square root that clamps negative inputs to zero."""
    return torch.sqrt(torch.clamp(x, min=0.0))


def safe_log(x: torch.Tensor) -> torch.Tensor:
    """Safe log that avoids log(0) and handles negative values via abs."""
    return torch.log(torch.clamp(torch.abs(x), min=1e-8))


def safe_pow(x: torch.Tensor, power: float) -> torch.Tensor:
    """Safe power with clamping to avoid numerical blow-ups."""
    return torch.pow(torch.clamp(torch.abs(x), min=1e-8, max=1000.0), power)


def clip_result(x: torch.Tensor, min_val: float = -1000.0, max_val: float = 1000.0) -> torch.Tensor:
    """Clip results to keep values in a safe numeric range."""
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
    "leaky_relu": lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.01),
    "elu": lambda x: torch.nn.functional.elu(x, alpha=1.0),
}

BINARY_OPERATORS = {
    "add": lambda x, y: x + y,
    "subtract": lambda x, y: x - y,
    "multiply": lambda x, y: clip_result(x * y),
    "divide": lambda x, y: safe_divide(x, y),
    "max": lambda x, y: torch.maximum(x, y),
    "min": lambda x, y: torch.minimum(x, y),
}


def evaluate_tree(node: dict,
                  value: torch.Tensor,
                  missing: torch.Tensor,
                  confidence: torch.Tensor) -> torch.Tensor:
    """Evaluate a tree on 3 channels: value x, missing flag m, confidence c."""
    ntype = node["type"]

    if ntype == "input":
        mapping = {"value": value, "missing": missing, "confidence": confidence}
        return mapping[node["input_name"]]

    if ntype == "constant":
        return torch.full_like(value, node["value"])

    if ntype == "unary":
        child = evaluate_tree(node["children"][0], value, missing, confidence)
        op = UNARY_OPERATORS[node["operator"]]
        return clip_result(op(child))

    if ntype == "binary":
        left = evaluate_tree(node["children"][0], value, missing, confidence)
        right = evaluate_tree(node["children"][1], value, missing, confidence)
        op = BINARY_OPERATORS[node["operator"]]
        return clip_result(op(left, right))

    raise ValueError(f"Unknown node type: {ntype}")


def tree_to_string(node: dict, indent: int = 0) -> str:
    """Pretty-print the tree as nested text."""
    spacing = "  " * indent
    t = node["type"]

    if t == "input":
        return f"{spacing}{node['input_name']}"
    if t == "constant":
        return f"{spacing}{node['value']}"

    if t == "unary":
        child_str = tree_to_string(node["children"][0], indent + 1)
        return f"{spacing}{node['operator']}(\n{child_str}\n{spacing})"

    if t == "binary":
        left_str = tree_to_string(node["children"][0], indent + 1)
        right_str = tree_to_string(node["children"][1], indent + 1)
        return f"{spacing}{node['operator']}(\n{left_str},\n{right_str}\n{spacing})"

    return f"{spacing}<?>"


def tree_to_infix(node: dict) -> str:
    """Return a math-style string using x, m, c instead of 'value/missing/confidence'."""
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
            "identity": s,
            "negate": f"(-{s})",
            "abs": f"|{s}|",
            "square": f"({s})^2",
            "cube": f"({s})^3",
            "sqrt": f"sqrt({s})",
            "exp": f"exp({s})",
            "log": f"log(|{s}|)",
            "sin": f"sin({s})",
            "cos": f"cos({s})",
            "tanh": f"tanh({s})",
            "sigmoid": f"sigmoid({s})",
            "relu": f"relu({s})",
            "softplus": f"softplus({s})",
            "leaky_relu": f"leakyrelu({s})",
            "elu": f"elu({s})",
        }
        return mapping.get(op, f"{op}({s})")

    if t == "binary":
        op = node["operator"]
        a = tree_to_infix(node["children"][0])
        b = tree_to_infix(node["children"][1])
        mapping = {
            "add": f"({a} + {b})",
            "subtract": f"({a} - {b})",
            "multiply": f"({a} * {b})",
            "divide": f"({a} / {b})",
            "max": f"max({a}, {b})",
            "min": f"min({a}, {b})",
        }
        return mapping.get(op, f"{op}({a}, {b})")

    return "<?>"


def get_tree_depth(node: dict) -> int:
    """Tree depth in nodes, counting root as 1."""
    t = node["type"]
    if t in ["input", "constant"]:
        return 1
    if t == "unary":
        return 1 + get_tree_depth(node["children"][0])
    if t == "binary":
        return 1 + max(get_tree_depth(node["children"][0]),
                       get_tree_depth(node["children"][1]))
    raise ValueError(f"Invalid node type: {t}")


def count_nodes(node: dict) -> int:
    """Total number of nodes in the tree."""
    t = node["type"]
    if t in ["input", "constant"]:
        return 1
    if t == "unary":
        return 1 + count_nodes(node["children"][0])
    if t == "binary":
        return 1 + count_nodes(node["children"][0]) + count_nodes(node["children"][1])
    raise ValueError(f"Invalid node type: {t}")


def generate_terminal_node() -> dict:
    """Generate a random terminal node (input or constant)."""
    if random.random() < 0.7:
        input_name = random.choice(INPUTS)
        return create_input_node(input_name)
    constant_value = random.choice(CONSTANTS)
    return create_constant_node(constant_value)


def generate_non_terminal_node(max_depth: int, current_depth: int) -> dict:
    """Generate a random unary or binary node with children."""
    if random.random() < 0.4:
        op = random.choice(list(UNARY_OPERATORS.keys()))
        child = generate_random_tree(max_depth, current_depth + 1)
        return create_unary_node(op, child)
    op = random.choice(list(BINARY_OPERATORS.keys()))
    left = generate_random_tree(max_depth, current_depth + 1)
    right = generate_random_tree(max_depth, current_depth + 1)
    return create_binary_node(op, left, right)


def generate_random_tree(max_depth: int, current_depth: int = 0) -> dict:
    """Top-down recursive random tree generation with higher terminal prob near leaves."""
    if current_depth >= max_depth:
        return generate_terminal_node()

    terminal_prob = 0.3 + 0.4 * (current_depth / max_depth)
    if random.random() < terminal_prob:
        return generate_terminal_node()

    return generate_non_terminal_node(max_depth, current_depth)


def validate_tree(node: dict) -> None:
    """Consistency checks for tree structure."""
    t = node["type"]

    if t == "input":
        assert node["input_name"] in INPUTS
        assert len(node["children"]) == 0
        return

    if t == "constant":
        assert isinstance(node["value"], (int, float))
        assert len(node["children"]) == 0
        return

    if t == "unary":
        assert node["operator"] in UNARY_OPERATORS
        assert len(node["children"]) == 1
        validate_tree(node["children"][0])
        return

    if t == "binary":
        assert node["operator"] in BINARY_OPERATORS
        assert len(node["children"]) == 2
        validate_tree(node["children"][0])
        validate_tree(node["children"][1])
        return

    raise ValueError(f"Invalid node type: {t}")


def get_all_nodes(tree: dict) -> list:
    """Return a flat list of all nodes in the tree."""
    nodes = [tree]
    if tree["type"] in ["unary", "binary"]:
        for ch in tree["children"]:
            nodes.extend(get_all_nodes(ch))
    return nodes


def mutate_single_node(node: dict, max_depth: int, current_tree_depth: int) -> None:
    """In-place mutation of a single node depending on its type."""
    t = node["type"]

    if t == "input":
        current_input = node["input_name"]
        candidates = [inp for inp in INPUTS if inp != current_input]
        if candidates:
            node["input_name"] = random.choice(candidates)

    elif t == "constant":
        node["value"] = random.choice(CONSTANTS)

    elif t == "unary":
        if random.random() < 0.7:
            current_op = node["operator"]
            candidates = [op for op in UNARY_OPERATORS.keys() if op != current_op]
            if candidates:
                node["operator"] = random.choice(candidates)
        else:
            if current_tree_depth < max_depth:
                remaining = max_depth - current_tree_depth + get_tree_depth(node)
                new_subtree = generate_random_tree(min(remaining, 3))
                node.clear()
                node.update(new_subtree)

    elif t == "binary":
        if random.random() < 0.7:
            current_op = node["operator"]
            candidates = [op for op in BINARY_OPERATORS.keys() if op != current_op]
            if candidates:
                node["operator"] = random.choice(candidates)
        else:
            if current_tree_depth < max_depth:
                remaining = max_depth - current_tree_depth + get_tree_depth(node)
                new_subtree = generate_random_tree(min(remaining, 3))
                node.clear()
                node.update(new_subtree)


def mutate_tree(tree: dict,
                mutation_rate: float = 0.1,
                max_depth: int = 5) -> dict:
    """Return a mutated copy of the input tree."""
    mutated = copy.deepcopy(tree)
    all_nodes = get_all_nodes(mutated)

    for node in all_nodes:
        if random.random() < mutation_rate:
            mutate_single_node(node, max_depth, get_tree_depth(mutated))

    validate_tree(mutated)
    return mutated


def perform_crossover(node1: dict, node2: dict) -> None:
    """Swap contents of two nodes."""
    temp = copy.deepcopy(node1)
    node1.clear()
    node1.update(copy.deepcopy(node2))
    node2.clear()
    node2.update(temp)


def crossover_trees(parent1: dict, parent2: dict, max_depth: int = 5) -> tuple:
    """Random subtree crossover between two parents; returns two offspring."""
    offspring1 = copy.deepcopy(parent1)
    offspring2 = copy.deepcopy(parent2)

    nodes1 = get_all_nodes(offspring1)
    nodes2 = get_all_nodes(offspring2)

    crossover_node1 = random.choice(nodes1)
    crossover_node2 = random.choice(nodes2)
    perform_crossover(crossover_node1, crossover_node2)

    try:
        validate_tree(offspring1)
        validate_tree(offspring2)
    except Exception:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    return offspring1, offspring2


def create_initial_population(population_size: int, max_depth: int = 4) -> list:
    """Create initial population of random trees."""
    population = []
    for _ in range(population_size):
        tree = generate_random_tree(max_depth)
        population.append({"tree": tree, "fitness": None})
    return population


def weighted_random_choice(population: list, probabilities: list) -> dict:
    r = random.random()
    cumulative = 0.0
    for ind, p in zip(population, probabilities):
        cumulative += p
        if r <= cumulative:
            return ind
    return population[-1]


def fitness_proportional_selection(population: list, num_selected: int) -> list:
    """Softmax-based fitness proportional selection."""
    fitness_values = [ind["fitness"] for ind in population]
    min_f = min(fitness_values)
    if min_f < 0:
        shifted = [f - min_f + 1.0 for f in fitness_values]
    else:
        shifted = fitness_values

    max_f = max(shifted)
    exp_f = [math.exp(f - max_f) for f in shifted]
    total = sum(exp_f)
    probs = [e / total for e in exp_f]

    selected = []
    for _ in range(num_selected):
        selected.append(weighted_random_choice(population, probs))
    return selected


def evolve_population(population: list,
                      mutation_rate: float = 0.1,
                      crossover_rate: float = 0.7,
                      elite_size: int = 2) -> list:
    """Produce next generation via elitism + crossover + mutation."""
    population_size = len(population)
    sorted_pop = sorted(population, key=lambda x: x["fitness"], reverse=True)

    new_population = []

    # Elites
    for i in range(elite_size):
        new_population.append({"tree": copy.deepcopy(sorted_pop[i]["tree"]),
                               "fitness": None})

    # Rest
    while len(new_population) < population_size:
        if random.random() < crossover_rate and len(new_population) < population_size - 1:
            parent1 = fitness_proportional_selection(population, 1)[0]
            parent2 = fitness_proportional_selection(population, 1)[0]
            off1_tree, off2_tree = crossover_trees(parent1["tree"], parent2["tree"])
            new_population.append({"tree": off1_tree, "fitness": None})
            if len(new_population) < population_size:
                new_population.append({"tree": off2_tree, "fitness": None})
        else:
            parent = fitness_proportional_selection(population, 1)[0]
            mutated = mutate_tree(parent["tree"], mutation_rate)
            new_population.append({"tree": mutated, "fitness": None})

    return new_population


def count_input_diversity(node: dict) -> int:
    """How many of {value, missing, confidence} appear in the tree."""
    found = set()

    def walk(n):
        if n.get("type") == "input":
            found.add(n["input_name"])
        for ch in n.get("children", []):
            walk(ch)

    walk(node)
    return len(found)


# ============================================================================
# NEURAL NETWORK: THREE-CHANNEL MLP
# ============================================================================

class ChannelProp(nn.Module):
    """
    Propagate missing/confidence through a linear layer using |W|-based weights.

    Input: (x, m, c)
      - x: values
      - m: missing flags (1=missing, 0=observed)
      - c: confidence scores in [0,1]
    """

    def __init__(self, linear: nn.Linear, eps: float = 1e-8):
        super().__init__()
        self.linear = linear
        self.eps = eps

    def forward(self, x, m, c):
        y = self.linear(x)  # (batch, out_dim)

        W = self.linear.weight  # (out_dim, in_dim)
        W_abs = W.abs() + self.eps
        col_sum = W_abs.sum(dim=1, keepdim=True)  # (out_dim, 1)
        W_norm = W_abs / col_sum  # row-normalized
        W_norm_T = W_norm.t()     # (in_dim, out_dim)

        # propagate confidence
        c_out = c @ W_norm_T

        # propagate observed probability
        obs_in = 1.0 - m
        obs_out = obs_in @ W_norm_T
        m_out = 1.0 - obs_out

        c_out = c_out.clamp(0.0, 1.0)
        m_out = m_out.clamp(0.0, 1.0)
        return y, m_out, c_out


class AdaptiveLayer(nn.Module):
    """Apply evolved activation tree element-wise on (x, m, c)."""

    def __init__(self, activation_tree: dict):
        super().__init__()
        self.activation_tree = activation_tree

    def forward(self, x, m, c):
        return evaluate_tree(self.activation_tree, x, m, c)


class ThreeChannelBlock(nn.Module):
    """Linear layer + channel propagation + tree-based activation."""

    def __init__(self, in_dim: int, out_dim: int, activation_tree: dict):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.prop = ChannelProp(self.lin)
        self.act = AdaptiveLayer(activation_tree)

    def forward(self, x, m, c):
        x, m, c = self.prop(x, m, c)
        x = self.act(x, m, c)
        return x, m, c


class ThreeChannelMLP(nn.Module):
    """MLP using repeated ThreeChannelBlocks, then a standard linear output."""

    def __init__(self, input_size: int, hidden_sizes, output_size: int, activation_tree: dict):
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
