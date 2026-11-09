import math
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_breast_cancer, load_wine, load_iris


# ============================================================================
# TREE STRUCTURE COMPONENTS (Same as original)
# ============================================================================

def create_input_node(input_name):
    assert input_name in ['value', 'missing', 'confidence'], f"Invalid input name: {input_name}"
    return {
        'type': 'input',
        'input_name': input_name,
        'children': []
    }


def create_constant_node(value):
    """Create a leaf node with a constant value"""
    return {
        'type': 'constant',
        'value': float(value),
        'children': []
    }


def create_unary_node(operator, child):
    """Create a unary operation node with one child"""
    return {
        'type': 'unary',
        'operator': operator,
        'children': [child]
    }


def create_binary_node(operator, left_child, right_child):
    """Create a binary operation node with two children"""
    return {
        'type': 'binary',
        'operator': operator,
        'children': [left_child, right_child]
    }


def safe_divide(x, y):
    """Safe division that handles division by zero with larger epsilon"""
    denom = torch.where(torch.abs(y) < 1e-3,
                        torch.sign(y) * 1e-3 + (y == 0).float() * 1e-3,
                        y)
    return x / denom


def safe_sqrt(x):
    """Safe square root that handles negative values"""
    return torch.sqrt(torch.clamp(x, min=0.0))


def safe_log(x):
    """Safe logarithm that handles zero and negative values"""
    return torch.log(torch.clamp(torch.abs(x), min=1e-8))


def safe_pow(x, power):
    """Safe power operation with clamping"""
    return torch.pow(torch.clamp(torch.abs(x), min=1e-8, max=1000.0), power)


def clip_result(x, min_val=-1000.0, max_val=1000.0):
    """Clip results to prevent extreme values"""
    return torch.clamp(x, min=min_val, max=max_val)


# Unary operators registry
UNARY_OPERATORS = {
    'identity': lambda x: x,
    'negate': lambda x: -x,
    'abs': lambda x: torch.abs(x),
    'square': lambda x: clip_result(x * x),
    'cube': lambda x: clip_result(x * x * x),
    'sqrt': lambda x: safe_sqrt(x),
    'exp': lambda x: clip_result(torch.exp(torch.clamp(x, min=-10, max=10))),
    'log': lambda x: safe_log(x),
    'sin': lambda x: torch.sin(x),
    'cos': lambda x: torch.cos(x),
    'tanh': lambda x: torch.tanh(x),
    'sigmoid': lambda x: torch.sigmoid(x),
    'relu': lambda x: torch.clamp(x, min=0),
    'softplus': lambda x: torch.nn.functional.softplus(x)
}

# Binary operators registry
BINARY_OPERATORS = {
    'add': lambda x, y: x + y,
    'subtract': lambda x, y: x - y,
    'multiply': lambda x, y: clip_result(x * y),
    'divide': lambda x, y: safe_divide(x, y),
    'max': lambda x, y: torch.maximum(x, y),
    'min': lambda x, y: torch.minimum(x, y)
}

# Constants that can appear in trees
CONSTANTS = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1]

# Input names
INPUTS = ['value', 'missing', 'confidence']


def evaluate_tree(node, value, missing, confidence):
    if node['type'] == 'input':
        input_map = {
            'value': value,
            'missing': missing,
            'confidence': confidence
        }
        return input_map[node['input_name']]

    elif node['type'] == 'constant':
        return torch.full_like(value, node['value'])

    elif node['type'] == 'unary':
        child_result = evaluate_tree(node['children'][0], value, missing, confidence)
        operator_func = UNARY_OPERATORS[node['operator']]
        return clip_result(operator_func(child_result))

    elif node['type'] == 'binary':
        left_result = evaluate_tree(node['children'][0], value, missing, confidence)
        right_result = evaluate_tree(node['children'][1], value, missing, confidence)
        operator_func = BINARY_OPERATORS[node['operator']]
        return clip_result(operator_func(left_result, right_result))

    else:
        raise ValueError(f"Unknown node type: {node['type']}")


def tree_to_string(node, indent=0):
    """Convert tree to readable string representation"""
    spacing = "  " * indent

    if node['type'] == 'input':
        return f"{spacing}{node['input_name']}"
    elif node['type'] == 'constant':
        return f"{spacing}{node['value']}"
    elif node['type'] == 'unary':
        child_str = tree_to_string(node['children'][0], indent + 1)
        return f"{spacing}{node['operator']}(\n{child_str}\n{spacing})"
    elif node['type'] == 'binary':
        left_str = tree_to_string(node['children'][0], indent + 1)
        right_str = tree_to_string(node['children'][1], indent + 1)
        return f"{spacing}{node['operator']}(\n{left_str},\n{right_str}\n{spacing})"


def get_tree_depth(node):
    """Calculate the depth of a tree"""
    if node['type'] in ['input', 'constant']:
        return 1
    elif node['type'] == 'unary':
        return 1 + get_tree_depth(node['children'][0])
    elif node['type'] == 'binary':
        return 1 + max(get_tree_depth(node['children'][0]),
                       get_tree_depth(node['children'][1]))


def count_nodes(node):
    """Count total number of nodes in tree"""
    if node['type'] in ['input', 'constant']:
        return 1
    elif node['type'] == 'unary':
        return 1 + count_nodes(node['children'][0])
    elif node['type'] == 'binary':
        return 1 + count_nodes(node['children'][0]) + count_nodes(node['children'][1])


def generate_random_tree(max_depth, current_depth=0):
    if current_depth >= max_depth:
        return generate_terminal_node()

    terminal_prob = 0.3 + 0.4 * (current_depth / max_depth)

    if random.random() < terminal_prob:
        return generate_terminal_node()
    else:
        return generate_non_terminal_node(max_depth, current_depth)


def generate_terminal_node():
    """Generate a random terminal node (input or constant)"""
    if random.random() < 0.7:
        input_name = random.choice(INPUTS)
        return create_input_node(input_name)
    else:
        constant_value = random.choice(CONSTANTS)
        return create_constant_node(constant_value)


def generate_non_terminal_node(max_depth, current_depth):
    """Generate a random non-terminal node (unary or binary)"""
    if random.random() < 0.4:
        operator = random.choice(list(UNARY_OPERATORS.keys()))
        child = generate_random_tree(max_depth, current_depth + 1)
        return create_unary_node(operator, child)
    else:
        operator = random.choice(list(BINARY_OPERATORS.keys()))
        left_child = generate_random_tree(max_depth, current_depth + 1)
        right_child = generate_random_tree(max_depth, current_depth + 1)
        return create_binary_node(operator, left_child, right_child)


def validate_tree(node):
    if node['type'] == 'input':
        assert 'input_name' in node
        assert node['input_name'] in INPUTS
        assert len(node['children']) == 0

    elif node['type'] == 'constant':
        assert 'value' in node
        assert isinstance(node['value'], (int, float))
        assert len(node['children']) == 0

    elif node['type'] == 'unary':
        assert 'operator' in node
        assert node['operator'] in UNARY_OPERATORS
        assert len(node['children']) == 1
        validate_tree(node['children'][0])

    elif node['type'] == 'binary':
        assert 'operator' in node
        assert node['operator'] in BINARY_OPERATORS
        assert len(node['children']) == 2
        validate_tree(node['children'][0])
        validate_tree(node['children'][1])

    else:
        raise ValueError(f"Invalid node type: {node['type']}")


# ============================================================================
# GENETIC ALGORITHM OPERATIONS (Same as original)
# ============================================================================

def get_all_nodes(tree):
    """Get list of all nodes in tree (for mutation candidates)"""
    nodes = [tree]

    if tree['type'] in ['unary', 'binary']:
        for child in tree['children']:
            nodes.extend(get_all_nodes(child))

    return nodes


def mutate_single_node(node, max_depth, current_tree_depth):
    """Mutate a single node based on its type"""

    if node['type'] == 'input':
        current_input = node['input_name']
        available_inputs = [inp for inp in INPUTS if inp != current_input]
        if available_inputs:
            node['input_name'] = random.choice(available_inputs)

    elif node['type'] == 'constant':
        node['value'] = random.choice(CONSTANTS)

    elif node['type'] == 'unary':
        if random.random() < 0.7:
            current_op = node['operator']
            available_ops = [op for op in UNARY_OPERATORS.keys() if op != current_op]
            if available_ops:
                node['operator'] = random.choice(available_ops)
        else:
            if current_tree_depth < max_depth:
                remaining_depth = max_depth - current_tree_depth + get_tree_depth(node)
                new_subtree = generate_random_tree(min(remaining_depth, 3))
                node.update(new_subtree)

    elif node['type'] == 'binary':
        if random.random() < 0.7:
            current_op = node['operator']
            available_ops = [op for op in BINARY_OPERATORS.keys() if op != current_op]
            if available_ops:
                node['operator'] = random.choice(available_ops)
        else:
            if current_tree_depth < max_depth:
                remaining_depth = max_depth - current_tree_depth + get_tree_depth(node)
                new_subtree = generate_random_tree(min(remaining_depth, 3))
                node.update(new_subtree)


def mutate_tree(tree, mutation_rate=0.1, max_depth=5):
    mutated_tree = copy.deepcopy(tree)
    all_nodes = get_all_nodes(mutated_tree)

    for node in all_nodes:
        if random.random() < mutation_rate:
            mutate_single_node(node, max_depth, get_tree_depth(mutated_tree))

    validate_tree(mutated_tree)
    return mutated_tree


def find_node_in_tree(tree, target_node_id):
    """Find a specific node in tree by comparing content"""
    if tree == target_node_id:
        return tree

    if tree['type'] in ['unary', 'binary']:
        for child in tree['children']:
            result = find_node_in_tree(child, target_node_id)
            if result is not None:
                return result

    return None


def perform_crossover(node1, node2):
    """Exchange content between two nodes"""
    temp_content = copy.deepcopy(node1)

    node1.clear()
    node1.update(copy.deepcopy(node2))

    node2.clear()
    node2.update(temp_content)


def crossover_trees(parent1, parent2, max_depth=5):
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
    except:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    return offspring1, offspring2


def create_initial_population(population_size, max_depth=4):
    """Create initial population of random trees"""
    population = []
    for _ in range(population_size):
        tree = generate_random_tree(max_depth)
        population.append({
            'tree': tree,
            'fitness': None
        })
    return population


def fitness_proportional_selection(population, num_selected):
    """Select individuals based on fitness using softmax probabilities"""
    fitness_values = [individual['fitness'] for individual in population]

    min_fitness = min(fitness_values)
    if min_fitness < 0:
        shifted_fitness = [f - min_fitness + 1.0 for f in fitness_values]
    else:
        shifted_fitness = fitness_values

    max_fitness = max(shifted_fitness)
    exp_fitness = [math.exp(f - max_fitness) for f in shifted_fitness]
    total_exp = sum(exp_fitness)
    probabilities = [exp_f / total_exp for exp_f in exp_fitness]

    selected = []
    for _ in range(num_selected):
        selected_individual = weighted_random_choice(population, probabilities)
        selected.append(selected_individual)

    return selected


def weighted_random_choice(population, probabilities):
    """Select individual based on weighted probabilities"""
    r = random.random()
    cumulative_prob = 0.0

    for individual, prob in zip(population, probabilities):
        cumulative_prob += prob
        if r <= cumulative_prob:
            return individual

    return population[-1]


def evolve_population(population, mutation_rate=0.1, crossover_rate=0.7, elite_size=2):
    population_size = len(population)

    sorted_population = sorted(population, key=lambda x: x['fitness'], reverse=True)

    new_population = []
    for i in range(elite_size):
        new_population.append({
            'tree': copy.deepcopy(sorted_population[i]['tree']),
            'fitness': None
        })

    while len(new_population) < population_size:
        if random.random() < crossover_rate and len(new_population) < population_size - 1:
            parent1 = fitness_proportional_selection(population, 1)[0]
            parent2 = fitness_proportional_selection(population, 1)[0]
            offspring1, offspring2 = crossover_trees(parent1['tree'], parent2['tree'])

            new_population.append({'tree': offspring1, 'fitness': None})
            if len(new_population) < population_size:
                new_population.append({'tree': offspring2, 'fitness': None})
        else:
            parent = fitness_proportional_selection(population, 1)[0]
            mutated_tree = mutate_tree(parent['tree'], mutation_rate)
            new_population.append({'tree': mutated_tree, 'fitness': None})

    return new_population


def count_input_diversity(tree):
    """Count how many different inputs are used in tree"""
    inputs_used = set()

    def collect_inputs(node):
        if node['type'] == 'input':
            inputs_used.add(node['input_name'])
        elif node['type'] in ['unary', 'binary']:
            for child in node['children']:
                collect_inputs(child)

    collect_inputs(tree)
    return len(inputs_used)


# ============================================================================
# NEURAL NETWORK INTEGRATION (Same as original)
# ============================================================================

class ChannelProp(nn.Module):
    """Propagate missing/confidence through a linear layer using |W|-based weights."""

    def __init__(self, linear: nn.Linear, eps: float = 1e-8):
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
        c_out = c_out.clamp(0.0, 1.0)
        m_out = m_out.clamp(0.0, 1.0)
        return y, m_out, c_out


class AdaptiveLayer(nn.Module):
    """Applies evolved activation tree element-wise to (x, m, c)."""

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


def evaluate_fitness_neural_network(tree, X_train, y_train, X_val, y_val,
                                    missing_train, missing_val,
                                    conf_train, conf_val,
                                    num_epochs=30, device=None,
                                    hidden_sizes=[32, 16]):
    """Evaluate activation function by training a neural network"""
    if device is None:
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
            activation_tree=tree
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        patience = 2
        patience_counter = 0

        model.train()
        for epoch in range(num_epochs):
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
                _, predicted = torch.max(val_outputs, 1)
                val_acc = (predicted == y_val).float().mean().item()
            model.train()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        size_penalty = 0.001 * count_nodes(tree)
        depth_penalty = 0.002 * (get_tree_depth(tree) - 1)
        fitness = best_val_acc - size_penalty - depth_penalty

        del model, optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return fitness

    except Exception as e:
        print(f"Error evaluating tree: {e}")
        return 0.0


def run_evolution_with_nn(X_train, y_train, X_val, y_val,
                          missing_train, missing_val,
                          conf_train, conf_val,
                          population_size=10, generations=5, max_depth=4,
                          hidden_sizes=[32, 16]):
    """Evolution with neural network-based fitness evaluation"""
    print(f"Starting evolution: {population_size} individuals, {generations} generations")
    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
    print(f"Network architecture: {X_train.shape[1]} -> {hidden_sizes} -> {len(torch.unique(y_train))}")
    print(f"Max tree depth: {max_depth}")

    population = create_initial_population(population_size, max_depth)

    best_ever = None
    best_ever_fitness = -float('inf')

    for generation in range(generations):
        print(f"\n{'=' * 60}")
        print(f"Generation {generation + 1}/{generations}")
        print(f"{'=' * 60}")

        for idx, individual in enumerate(population):
            if individual['fitness'] is None:
                print(f"Evaluating individual {idx + 1}/{population_size}...", end=' ')

                fitness = evaluate_fitness_neural_network(
                    individual['tree'],
                    X_train, y_train, X_val, y_val,
                    missing_train, missing_val,
                    conf_train, conf_val,
                    num_epochs=30,
                    hidden_sizes=hidden_sizes
                )

                individual['fitness'] = fitness
                print(f"Fitness: {fitness:.4f}")

        population.sort(key=lambda x: x['fitness'], reverse=True)

        best_individual = population[0]
        if best_individual['fitness'] > best_ever_fitness:
            best_ever_fitness = best_individual['fitness']
            best_ever = copy.deepcopy(best_individual)
            print(f"\n🎯 NEW BEST! Fitness: {best_ever_fitness:.4f}")
            print(f"Tree complexity: {count_nodes(best_ever['tree'])} nodes, "
                  f"depth {get_tree_depth(best_ever['tree'])}, "
                  f"uses {count_input_diversity(best_ever['tree'])}/3 inputs")

        print(f"\nGeneration {generation + 1} Summary:")
        print(f"  Best fitness: {best_individual['fitness']:.4f}")
        print(f"  Mean fitness: {np.mean([ind['fitness'] for ind in population]):.4f}")
        print(f"  Worst fitness: {population[-1]['fitness']:.4f}")

        if generation < generations - 1:
            population = evolve_population(
                population,
                mutation_rate=0.1,
                crossover_rate=0.7,
                elite_size=2
            )

    print(f"\n{'=' * 60}")
    print(f"Evolution Complete!")
    print(f"Best fitness achieved: {best_ever_fitness:.4f}")
    print(f"{'=' * 60}")

    return best_ever


def train_final_model(activation_tree, X_train, y_train, X_val, y_val,
                      missing_train, missing_val, conf_train, conf_val,
                      epochs=100, patience=15, device=None):
    """Train final model with best activation function for more epochs"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print(f"Training Final Model on {device}")
    print(f"{'=' * 60}")
    print(f"Architecture: {X_train.shape[1]} -> [64, 32] -> {len(torch.unique(y_train))}")
    print(f"Training for up to {epochs} epochs with early stopping (patience={patience})")

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
        hidden_sizes=[64, 32],
        output_size=len(torch.unique(y_train)),
        activation_tree=activation_tree
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.5, patience=5)

    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None

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
            _, predicted = torch.max(val_outputs, 1)
            val_acc = (predicted == y_val).float().mean().item()

            train_outputs = model(X_train, missing_train, conf_train)
            _, train_predicted = torch.max(train_outputs, 1)
            train_acc = (train_predicted == y_train).float().mean().item()

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} - "
                  f"Loss: {loss.item():.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Val Acc: {val_acc:.4f}, "
                  f"Best Val: {best_val_acc:.4f}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nFinal Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    return model, best_val_acc


# ============================================================================
# MISSING DATA INJECTION FUNCTIONS - NEW!
# ============================================================================

def introduce_missing_data(df, missing_rate=0.10, mechanism='MCAR', seed=42):
    """
    Introduce missing data into the dataset with controllable missing rate.

    Parameters:
    -----------
    df : pandas.DataFrame
        The complete dataset (features only, no labels)
    missing_rate : float
        Proportion of values to make missing (0.0 to 1.0)
        Default: 0.10 (10%)
    mechanism : str
        Missing data mechanism: 'MCAR', 'MAR', or 'MNAR'
        - MCAR: Missing Completely At Random (uniform random)
        - MAR: Missing At Random (depends on other features)
        - MNAR: Missing Not At Random (depends on the value itself)
        Default: 'MCAR'
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    df_missing : pandas.DataFrame
        DataFrame with missing values introduced
    missing_mask : pandas.DataFrame
        Boolean mask indicating where values were made missing (True = originally complete, made missing)
    """
    np.random.seed(seed)
    df_missing = df.copy()
    n_rows, n_cols = df.shape

    # Create mask to track what we made missing
    missing_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    print(f"\n{'=' * 60}")
    print(f"INTRODUCING MISSING DATA")
    print(f"{'=' * 60}")
    print(f"Mechanism: {mechanism}")
    print(f"Target missing rate: {missing_rate * 100:.1f}%")
    print(f"Dataset shape: {n_rows} samples × {n_cols} features")

    if mechanism == 'MCAR':
        # Missing Completely At Random - uniform random selection
        print("Strategy: Uniform random selection across all values")

        total_values = n_rows * n_cols
        n_missing = int(total_values * missing_rate)

        # Generate random positions
        missing_positions = np.random.choice(total_values, size=n_missing, replace=False)

        for pos in missing_positions:
            row_idx = pos // n_cols
            col_idx = pos % n_cols
            df_missing.iloc[row_idx, col_idx] = np.nan
            missing_mask.iloc[row_idx, col_idx] = True

    elif mechanism == 'MAR':
        # Missing At Random - missingness depends on other observed features
        print("Strategy: Missingness depends on other features (first feature predicts missingness)")

        # Use first feature to predict missingness in others
        first_col_values = df.iloc[:, 0].values
        threshold = np.percentile(first_col_values, 50)  # median

        # Features with values in first column above median are more likely to have missing data
        for col_idx in range(1, n_cols):  # Skip first column
            col = df.columns[col_idx]

            # Probability of missing increases with first feature value
            probs = np.where(first_col_values > threshold,
                             missing_rate * 2,  # Higher probability
                             missing_rate * 0.5)  # Lower probability

            missing_in_col = np.random.random(n_rows) < probs
            df_missing.loc[missing_in_col, col] = np.nan
            missing_mask.loc[missing_in_col, col] = True

    elif mechanism == 'MNAR':
        # Missing Not At Random - missingness depends on the value itself
        print("Strategy: High/low values are more likely to be missing")

        for col in df.columns:
            col_values = df[col].values

            # Values in extreme quantiles are more likely to be missing
            lower_threshold = np.percentile(col_values, 25)
            upper_threshold = np.percentile(col_values, 75)

            # Higher probability for extreme values
            is_extreme = (col_values < lower_threshold) | (col_values > upper_threshold)

            probs = np.where(is_extreme,
                             missing_rate * 1.5,  # Higher for extreme
                             missing_rate * 0.5)  # Lower for middle range

            missing_in_col = np.random.random(n_rows) < probs
            df_missing.loc[missing_in_col, col] = np.nan
            missing_mask.loc[missing_in_col, col] = True

    else:
        raise ValueError(f"Unknown mechanism: {mechanism}. Choose from 'MCAR', 'MAR', 'MNAR'")

    # Calculate actual missing rate
    actual_missing_rate = df_missing.isnull().sum().sum() / (n_rows * n_cols)

    print(f"\nResults:")
    print(f"  Total values: {n_rows * n_cols}")
    print(f"  Missing values introduced: {df_missing.isnull().sum().sum()}")
    print(f"  Actual missing rate: {actual_missing_rate * 100:.2f}%")
    print(
        f"  Missing values per feature (range): {df_missing.isnull().sum().min()} to {df_missing.isnull().sum().max()}")
    print(f"{'=' * 60}\n")

    return df_missing, missing_mask


# ============================================================================
# SONAR DATASET EXPERIMENTS WITH MISSING DATA
# ============================================================================

def load_sonar_dataset():
    """Load and preprocess the Sonar dataset"""
    try:
        df = pd.read_csv('sonar_final_numeric.csv')
        print(f"Loaded Sonar dataset: {df.shape[0]} samples, {df.shape[1]} columns")

        # Check for existing missing values
        existing_missing = df.isnull().sum().sum()
        if existing_missing > 0:
            print(f"⚠️  Dataset already has {existing_missing} missing values")
        else:
            print("✅ Dataset is complete (no missing values)")

        return df

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def prepare_three_channel_data_sonar(df):
    """Convert Sonar data to three-channel format (value, missing, confidence)"""
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    column_means = X.mean()

    values = X.copy()
    missing_flags = X.isnull().astype(float)
    confidence_scores = (~X.isnull()).astype(float)

    values = values.fillna(column_means)

    for col in X.columns:
        missing_rate = missing_flags[col].mean()
        imputed_confidence = max(0.1, 1.0 - missing_rate)
        confidence_scores.loc[missing_flags[col] == 1, col] = imputed_confidence

    print("\nThree-channel data prepared:")
    print(f"  Values shape: {values.shape}")
    print(f"  Missing flags shape: {missing_flags.shape}")
    print(f"  Confidence scores shape: {confidence_scores.shape}")
    print(f"  Overall missing rate: {missing_flags.values.mean():.3f}")

    return values, missing_flags, confidence_scores, y


def test_evolved_functions_on_sonar(missing_rate=0.10, missing_mechanism='MCAR'):
    """
    Main function to test evolved activation functions on Sonar data with missing values

    Parameters:
    -----------
    missing_rate : float
        Proportion of values to make missing (0.05 to 0.50 recommended)
        Examples: 0.05 (5%), 0.10 (10%), 0.20 (20%), 0.30 (30%)
    missing_mechanism : str
        'MCAR', 'MAR', or 'MNAR'
    """
    print("=" * 60)
    print(f"EVOLVING ACTIVATION FUNCTIONS ON SONAR DATA")
    print(f"WITH {missing_rate * 100:.0f}% MISSING DATA ({missing_mechanism})")
    print("=" * 60)

    # Step 1: Load dataset
    print("\nStep 1: Loading Sonar dataset...")
    df = load_sonar_dataset()
    if df is None:
        return

    # Step 2: Introduce missing data BEFORE splitting
    print("\nStep 2: Introducing missing data to complete dataset...")

    # Separate features and labels
    X_complete = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Introduce missing data to features only
    X_with_missing, missing_mask = introduce_missing_data(
        X_complete,
        missing_rate=missing_rate,
        mechanism=missing_mechanism,
        seed=42
    )

    # Combine back with labels
    df_with_missing = pd.concat([X_with_missing, y], axis=1)

    print("\nStep 3: Converting to three-channel format...")
    values, missing_flags, confidence_scores, target = prepare_three_channel_data_sonar(df_with_missing)

    # Step 4: Split data (NO DATA LEAKAGE)
    print("\nStep 4: Splitting and preparing data (no leakage)...")

    X_train, X_temp, y_train, y_temp = train_test_split(
        values, target, test_size=0.4, random_state=42, stratify=target
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # Compute imputation statistics ONLY on train
    print("\nComputing imputation statistics from TRAIN set only...")

    train_column_means = X_train.mean()

    # Process TRAIN set
    values_train = X_train.copy()
    missing_flags_train = X_train.isnull().astype(float)
    confidence_scores_train = (~X_train.isnull()).astype(float)

    values_train = values_train.fillna(train_column_means)

    train_missing_rates = {}
    for col in X_train.columns:
        train_missing_rate = missing_flags_train[col].mean()
        train_missing_rates[col] = train_missing_rate
        imputed_confidence = max(0.1, 1.0 - train_missing_rate)
        confidence_scores_train.loc[missing_flags_train[col] == 1, col] = imputed_confidence

    # Process VAL set (using TRAIN statistics)
    values_val = X_val.copy()
    missing_flags_val = X_val.isnull().astype(float)
    confidence_scores_val = (~X_val.isnull()).astype(float)

    values_val = values_val.fillna(train_column_means)

    for col in X_val.columns:
        train_missing_rate = train_missing_rates[col]
        imputed_confidence = max(0.1, 1.0 - train_missing_rate)
        confidence_scores_val.loc[missing_flags_val[col] == 1, col] = imputed_confidence

    # Process TEST set (using TRAIN statistics)
    values_test = X_test.copy()
    missing_flags_test = X_test.isnull().astype(float)
    confidence_scores_test = (~X_test.isnull()).astype(float)

    values_test = values_test.fillna(train_column_means)

    for col in X_test.columns:
        train_missing_rate = train_missing_rates[col]
        imputed_confidence = max(0.1, 1.0 - train_missing_rate)
        confidence_scores_test.loc[missing_flags_test[col] == 1, col] = imputed_confidence

    # Encode labels (FIT on train only)
    print("Encoding labels (fit on train only)...")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    # Standardize features (FIT on train only)
    print("Standardizing features (fit on train only)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(values_train)
    X_val_scaled = scaler.transform(values_val)
    X_test_scaled = scaler.transform(values_test)

    # Convert to torch tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_enc, dtype=torch.long)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_enc, dtype=torch.long)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_enc, dtype=torch.long)

    missing_train_t = torch.tensor(missing_flags_train.values, dtype=torch.float32)
    missing_val_t = torch.tensor(missing_flags_val.values, dtype=torch.float32)
    missing_test_t = torch.tensor(missing_flags_test.values, dtype=torch.float32)

    conf_train_t = torch.tensor(confidence_scores_train.values, dtype=torch.float32)
    conf_val_t = torch.tensor(confidence_scores_val.values, dtype=torch.float32)
    conf_test_t = torch.tensor(confidence_scores_test.values, dtype=torch.float32)

    print(f"\n✅ Data prepared with NO LEAKAGE:")
    print(f"  Missing rate in train: {missing_flags_train.values.mean() * 100:.2f}%")
    print(f"  Missing rate in val:   {missing_flags_val.values.mean() * 100:.2f}%")
    print(f"  Missing rate in test:  {missing_flags_test.values.mean() * 100:.2f}%")
    print(f"  Features: {X_train_t.shape[1]}")
    print(f"  Classes: {len(torch.unique(y_train_t))}")

    # Step 5: Evolve activation functions
    print("\nStep 5: Evolving activation functions with neural network fitness...")
    best_activation = run_evolution_with_nn(
        X_train_t, y_train_t, X_val_t, y_val_t,
        missing_train_t, missing_val_t,
        conf_train_t, conf_val_t,
        population_size=30,
        generations=20,
        max_depth=3,
        hidden_sizes=[64, 32]
    )

    print("\n" + "=" * 60)
    print("BEST ACTIVATION FUNCTION DISCOVERED:")
    print("=" * 60)
    print(tree_to_string(best_activation['tree']))
    print(f"\nValidation fitness: {best_activation['fitness']:.4f}")
    print(f"Complexity: {count_nodes(best_activation['tree'])} nodes")
    print(f"Depth: {get_tree_depth(best_activation['tree'])}")
    print(f"Uses {count_input_diversity(best_activation['tree'])}/3 input channels")

    # Visualization
    print("\n" + "=" * 60)
    print("VISUALIZING THE EVOLVED FUNCTION:")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt

        x_range = torch.linspace(-5, 5, 200)
        x_range = torch.where(torch.abs(x_range) < 0.01,
                              torch.sign(x_range) * 0.01,
                              x_range)
        m_vals = torch.zeros_like(x_range)
        c_vals = torch.ones_like(x_range)

        with torch.no_grad():
            y_vals = evaluate_tree(best_activation['tree'], x_range, m_vals, c_vals)
            y_vals = torch.clamp(y_vals, -10, 10)

        plt.figure(figsize=(12, 4))

        # Plot 1: Evolved vs ReLU
        plt.subplot(1, 3, 1)
        plt.plot(x_range.cpu().numpy(), y_vals.cpu().numpy(),
                 label='Evolved Function', linewidth=2)
        plt.plot(x_range.cpu().numpy(), torch.relu(x_range).cpu().numpy(),
                 '--', label='ReLU', alpha=0.7, linewidth=2)
        plt.xlabel('x (value)', fontsize=10)
        plt.ylabel('activation(x)', fontsize=10)
        plt.title('Evolved vs ReLU\n(missing=0, confidence=1)', fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-5, 5)

        # Plot 2: Effect of confidence
        c_low = torch.ones_like(x_range) * 0.3
        c_high = torch.ones_like(x_range) * 1.0

        with torch.no_grad():
            y_vals_low = evaluate_tree(best_activation['tree'], x_range, m_vals, c_low)
            y_vals_high = evaluate_tree(best_activation['tree'], x_range, m_vals, c_high)
            y_vals_low = torch.clamp(y_vals_low, -10, 10)
            y_vals_high = torch.clamp(y_vals_high, -10, 10)

        plt.subplot(1, 3, 2)
        plt.plot(x_range.cpu().numpy(), y_vals_high.cpu().numpy(),
                 label='Confidence=1.0', linewidth=2)
        plt.plot(x_range.cpu().numpy(), y_vals_low.cpu().numpy(),
                 label='Confidence=0.3', alpha=0.7, linewidth=2)
        plt.xlabel('x (value)', fontsize=10)
        plt.ylabel('activation(x)', fontsize=10)
        plt.title('Effect of Confidence\n(missing=0)', fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-5, 5)

        # Plot 3: Effect of missing flag
        m_missing = torch.ones_like(x_range)
        m_observed = torch.zeros_like(x_range)

        with torch.no_grad():
            y_vals_missing = evaluate_tree(best_activation['tree'], x_range, m_missing, c_vals)
            y_vals_observed = evaluate_tree(best_activation['tree'], x_range, m_observed, c_vals)
            y_vals_missing = torch.clamp(y_vals_missing, -10, 10)
            y_vals_observed = torch.clamp(y_vals_observed, -10, 10)

        plt.subplot(1, 3, 3)
        plt.plot(x_range.cpu().numpy(), y_vals_observed.cpu().numpy(),
                 label='Observed (m=0)', linewidth=2)
        plt.plot(x_range.cpu().numpy(), y_vals_missing.cpu().numpy(),
                 label='Missing (m=1)', alpha=0.7, linewidth=2)
        plt.xlabel('x (value)', fontsize=10)
        plt.ylabel('activation(x)', fontsize=10)
        plt.title('Effect of Missing Flag\n(confidence=1)', fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-5, 5)

        plt.tight_layout()
        filename = f'/mnt/user-data/outputs/evolved_activation_sonar_{int(missing_rate * 100)}pct_{missing_mechanism}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to: {filename}")
        plt.close()

    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")

    # Train final model
    print("\nStep 6: Training final model with best activation...")
    final_model, final_val_acc = train_final_model(
        best_activation['tree'],
        X_train_t, y_train_t, X_val_t, y_val_t,
        missing_train_t, missing_val_t, conf_train_t, conf_val_t,
        epochs=100,
        patience=15
    )

    # Evaluate on test set
    print("\nStep 7: Evaluating on held-out test set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model.eval()
    with torch.no_grad():
        test_outputs = final_model(X_test_t.to(device), missing_test_t.to(device), conf_test_t.to(device))
        _, predicted = torch.max(test_outputs, 1)
        test_acc = (predicted == y_test_t.to(device)).float().mean().item()

    print(f"Test Accuracy: {test_acc:.4f}")

    # Baselines
    relu_tree = create_binary_node('max', create_input_node('value'), create_constant_node(0.0))
    swish_tree = create_binary_node(
        'multiply',
        create_input_node('value'),
        create_unary_node('sigmoid', create_input_node('value'))
    )

    # ReLU baseline
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON: Training with ReLU")
    print("=" * 60)

    relu_model, relu_val_acc = train_final_model(
        relu_tree,
        X_train_t, y_train_t, X_val_t, y_val_t,
        missing_train_t, missing_val_t, conf_train_t, conf_val_t,
        epochs=100,
        patience=15
    )

    relu_model.eval()
    with torch.no_grad():
        relu_test_outputs = relu_model(X_test_t.to(device), missing_test_t.to(device), conf_test_t.to(device))
        _, relu_predicted = torch.max(relu_test_outputs, 1)
        relu_test_acc = (relu_predicted == y_test_t.to(device)).float().mean().item()

    print(f"ReLU Test Accuracy: {relu_test_acc:.4f}")

    # Swish baseline
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON: Training with Swish")
    print("=" * 60)

    swish_model, swish_val_acc = train_final_model(
        swish_tree,
        X_train_t, y_train_t, X_val_t, y_val_t,
        missing_train_t, missing_val_t, conf_train_t, conf_val_t,
        epochs=100,
        patience=15
    )

    swish_model.eval()
    with torch.no_grad():
        swish_test_outputs = swish_model(X_test_t.to(device), missing_test_t.to(device), conf_test_t.to(device))
        _, swish_predicted = torch.max(swish_test_outputs, 1)
        swish_test_acc = (swish_predicted == y_test_t.to(device)).float().mean().item()

    # Final summary
    print("\n" + "=" * 60)
    print(f"FINAL RESULTS SUMMARY - SONAR ({missing_rate * 100:.0f}% Missing, {missing_mechanism})")
    print("=" * 60)
    print(f"{'Method':<25} {'Validation':<12} {'Test':<12}")
    print("-" * 60)
    print(f"{'Evolved Activation':<25} {final_val_acc:<12.4f} {test_acc:<12.4f}")
    print(f"{'ReLU Baseline':<25} {relu_val_acc:<12.4f} {relu_test_acc:<12.4f}")
    print(f"{'Swish Baseline':<25} {swish_val_acc:<12.4f} {swish_test_acc:<12.4f}")

    print("\nImprovement over ReLU:")
    print(
        f"  Evolved - Val: {(final_val_acc - relu_val_acc) * 100:+.2f}%, Test: {(test_acc - relu_test_acc) * 100:+.2f}%")
    print(
        f"  Swish   - Val: {(swish_val_acc - relu_val_acc) * 100:+.2f}%, Test: {(swish_test_acc - relu_test_acc) * 100:+.2f}%")

    best_test = max(test_acc, relu_test_acc, swish_test_acc)

    if test_acc == best_test:
        print("\n✅ Evolved activation is THE BEST! Beats both ReLU and Swish!")
    elif test_acc > relu_test_acc:
        print("\n✅ Evolved activation beats ReLU!")
        if swish_test_acc == best_test:
            print("⚠️  But Swish performed slightly better.")
    else:
        print("\n⚠️  Baselines performed better this time.")
        if swish_test_acc > relu_test_acc:
            print(f"   Best: Swish ({swish_test_acc:.4f})")
        else:
            print(f"   Best: ReLU ({relu_test_acc:.4f})")

    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)

    return {
        'missing_rate': missing_rate,
        'mechanism': missing_mechanism,
        'evolved_val': final_val_acc,
        'evolved_test': test_acc,
        'relu_val': relu_val_acc,
        'relu_test': relu_test_acc,
        'swish_val': swish_val_acc,
        'swish_test': swish_test_acc,
        'best_tree': best_activation['tree']
    }


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_multiple_missing_rates():
    """
    Run experiments with different missing rates for comparison
    """
    print("\n" + "🔬" * 30)
    print("RUNNING EXPERIMENTS WITH MULTIPLE MISSING RATES")
    print("🔬" * 30)

    # Test different missing rates
    missing_rates = [0.05, 0.10, 0.20, 0.30]
    mechanism = 'MCAR'  # You can change this to 'MAR' or 'MNAR'

    results = []

    for rate in missing_rates:
        set_seed(42)  # Reset seed for fair comparison
        print(f"\n{'#' * 60}")
        print(f"# EXPERIMENT: {rate * 100:.0f}% Missing Data ({mechanism})")
        print(f"{'#' * 60}\n")

        result = test_evolved_functions_on_sonar(
            missing_rate=rate,
            missing_mechanism=mechanism
        )
        results.append(result)

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY: COMPARISON ACROSS MISSING RATES")
    print("=" * 80)
    print(f"{'Missing %':<12} {'Evolved Val':<15} {'Evolved Test':<15} {'ReLU Test':<15} {'Swish Test':<15}")
    print("-" * 80)

    for res in results:
        print(f"{res['missing_rate'] * 100:<12.0f} "
              f"{res['evolved_val']:<15.4f} "
              f"{res['evolved_test']:<15.4f} "
              f"{res['relu_test']:<15.4f} "
              f"{res['swish_test']:<15.4f}")

    print("=" * 80)


if __name__ == "__main__":
    set_seed(42)

    # OPTION 1: Run single experiment with specified missing rate
    print("=== SINGLE EXPERIMENT ===\n")
    test_evolved_functions_on_sonar(
        missing_rate=0.3,  # Change this: 0.05, 0.10, 0.20, 0.30, etc.
        missing_mechanism='MCAR'  # Change this: 'MCAR', 'MAR', or 'MNAR'
    )

    # OPTION 2: Uncomment to run multiple experiments for comparison
    # print("\n\n=== MULTIPLE EXPERIMENTS ===\n")
    # run_multiple_missing_rates()

    print("\n=== All tests completed ===")