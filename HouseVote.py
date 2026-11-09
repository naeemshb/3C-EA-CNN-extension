import math
import random
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder  # ← ADD LabelEncoder here
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
# Node representation using dictionaries (as requested - not object-oriented)
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
    # Use larger epsilon (1e-3) to prevent spikes near zero
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

# Unary operators registry (from Paper 2)
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

# Binary operators registry (from Paper 2)
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
        # Return the appropriate input tensor
        input_map = {
            'value': value,
            'missing': missing,
            'confidence': confidence
        }
        return input_map[node['input_name']]

    elif node['type'] == 'constant':
        # Return constant value broadcast to same shape as inputs
        return torch.full_like(value, node['value'])

    elif node['type'] == 'unary':
        # Evaluate child and apply unary operator
        child_result = evaluate_tree(node['children'][0], value, missing, confidence)
        operator_func = UNARY_OPERATORS[node['operator']]
        return clip_result(operator_func(child_result))

    elif node['type'] == 'binary':
        # Evaluate both children and apply binary operator
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


    # Force terminal nodes at maximum depth
    if current_depth >= max_depth:
        return generate_terminal_node()

    # Probability of generating terminal vs non-terminal
    # Higher probability of terminals as we approach max depth
    terminal_prob = 0.3 + 0.4 * (current_depth / max_depth)

    if random.random() < terminal_prob:
        return generate_terminal_node()
    else:
        return generate_non_terminal_node(max_depth, current_depth)


def generate_terminal_node():
    """Generate a random terminal node (input or constant)"""
    if random.random() < 0.7:  # 70% chance of input, 30% chance of constant
        input_name = random.choice(INPUTS)
        return create_input_node(input_name)
    else:
        constant_value = random.choice(CONSTANTS)
        return create_constant_node(constant_value)


def generate_non_terminal_node(max_depth, current_depth):
    """Generate a random non-terminal node (unary or binary)"""
    if random.random() < 0.4:  # 40% chance of unary, 60% chance of binary
        # Generate unary node
        operator = random.choice(list(UNARY_OPERATORS.keys()))
        child = generate_random_tree(max_depth, current_depth + 1)
        return create_unary_node(operator, child)
    else:
        # Generate binary node
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


def test_known_functions():


    # Test 1: ReLU function - max(0, value)
    relu_tree = create_binary_node('max',
                                   create_constant_node(0.0),
                                   create_input_node('value'))

    # Test inputs
    test_values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    test_missing = torch.zeros_like(test_values)
    test_confidence = torch.ones_like(test_values)

    result = evaluate_tree(relu_tree, test_values, test_missing, test_confidence)
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])

    print("ReLU Test:")
    print(f"Input: {test_values}")
    print(f"Output: {result}")
    print(f"Expected: {expected}")
    print(f"Test passed: {torch.allclose(result, expected)}")
    print()

    # Test 2: Confidence scaling - value * confidence
    confidence_tree = create_binary_node('multiply',
                                         create_input_node('value'),
                                         create_input_node('confidence'))

    test_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    test_missing = torch.zeros_like(test_values)
    test_confidence = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2])

    result = evaluate_tree(confidence_tree, test_values, test_missing, test_confidence)
    expected = test_values * test_confidence

    print("Confidence Scaling Test:")
    print(f"Values: {test_values}")
    print(f"Confidence: {test_confidence}")
    print(f"Output: {result}")
    print(f"Expected: {expected}")
    print(f"Test passed: {torch.allclose(result, expected)}")
    print()


def test_random_tree_generation():

    print("Random Tree Generation Test:")

    for i in range(3):
        tree = generate_random_tree(max_depth=3)
        validate_tree(tree)

        print(f"Tree {i + 1}:")
        print(tree_to_string(tree))
        print(f"Depth: {get_tree_depth(tree)}")
        print(f"Nodes: {count_nodes(tree)}")

        # Test evaluation
        test_values = torch.tensor([1.5])
        test_missing = torch.tensor([0.0])
        test_confidence = torch.tensor([0.8])

        try:
            result = evaluate_tree(tree, test_values, test_missing, test_confidence)
            print(f"Evaluation result: {result}")
            print("✓ Tree evaluation successful")
        except Exception as e:
            print(f"✗ Tree evaluation failed: {e}")

        print("-" * 50)


# ============================================================================
# GENETIC ALGORITHM OPERATIONS
# ============================================================================

import copy
import math


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
        # Change to different input
        current_input = node['input_name']
        available_inputs = [inp for inp in INPUTS if inp != current_input]
        if available_inputs:
            node['input_name'] = random.choice(available_inputs)

    elif node['type'] == 'constant':
        # Change constant value
        node['value'] = random.choice(CONSTANTS)

    elif node['type'] == 'unary':
        if random.random() < 0.7:
            # Change operator (70% chance)
            current_op = node['operator']
            available_ops = [op for op in UNARY_OPERATORS.keys() if op != current_op]
            if available_ops:
                node['operator'] = random.choice(available_ops)
        else:
            # Replace with random subtree (30% chance)
            if current_tree_depth < max_depth:
                remaining_depth = max_depth - current_tree_depth + get_tree_depth(node)
                new_subtree = generate_random_tree(min(remaining_depth, 3))
                # Replace current node's content with new subtree's content
                node.update(new_subtree)

    elif node['type'] == 'binary':
        if random.random() < 0.7:
            # Change operator (70% chance)
            current_op = node['operator']
            available_ops = [op for op in BINARY_OPERATORS.keys() if op != current_op]
            if available_ops:
                node['operator'] = random.choice(available_ops)
        else:
            # Replace with random subtree (30% chance)
            if current_tree_depth < max_depth:
                remaining_depth = max_depth - current_tree_depth + get_tree_depth(node)
                new_subtree = generate_random_tree(min(remaining_depth, 3))
                # Replace current node's content with new subtree's content
                node.update(new_subtree)


def mutate_tree(tree, mutation_rate=0.1, max_depth=5):
    # Deep copy to avoid modifying original
    mutated_tree = copy.deepcopy(tree)

    # Get all nodes in tree for mutation candidates
    all_nodes = get_all_nodes(mutated_tree)

    # Mutate each node with given probability
    for node in all_nodes:
        if random.random() < mutation_rate:
            mutate_single_node(node, max_depth, get_tree_depth(mutated_tree))

    # Validate result
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
    # Store node1's content
    temp_content = copy.deepcopy(node1)

    # Replace node1 with node2's content
    node1.clear()
    node1.update(copy.deepcopy(node2))

    # Replace node2 with node1's original content
    node2.clear()
    node2.update(temp_content)


def crossover_trees(parent1, parent2, max_depth=5):

    # Deep copy parents to avoid modification
    offspring1 = copy.deepcopy(parent1)
    offspring2 = copy.deepcopy(parent2)

    # Get all possible crossover points (all nodes)
    nodes1 = get_all_nodes(offspring1)
    nodes2 = get_all_nodes(offspring2)

    # Select random crossover points
    crossover_node1 = random.choice(nodes1)
    crossover_node2 = random.choice(nodes2)

    # Perform crossover
    perform_crossover(crossover_node1, crossover_node2)

    # Validate offspring
    try:
        validate_tree(offspring1)
        validate_tree(offspring2)
    except:
        # If validation fails, return original parents
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    return offspring1, offspring2


def create_initial_population(population_size, max_depth=4):
    """Create initial population of random trees"""
    population = []
    for _ in range(population_size):
        tree = generate_random_tree(max_depth)
        population.append({
            'tree': tree,
            'fitness': None  # Will be calculated later
        })
    return population


def fitness_proportional_selection(population, num_selected):
    """
    Select individuals based on fitness using softmax probabilities
    (Following Paper 2's approach)
    """
    # Extract fitness values
    fitness_values = [individual['fitness'] for individual in population]

    # Handle negative fitness values by shifting
    min_fitness = min(fitness_values)
    if min_fitness < 0:
        shifted_fitness = [f - min_fitness + 1.0 for f in fitness_values]
    else:
        shifted_fitness = fitness_values

    # Apply softmax for selection probabilities
    max_fitness = max(shifted_fitness)
    exp_fitness = [math.exp(f - max_fitness) for f in shifted_fitness]  # Numerical stability
    total_exp = sum(exp_fitness)
    probabilities = [exp_f / total_exp for exp_f in exp_fitness]

    # Select individuals based on probabilities
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

    # Fallback (should not happen with proper probabilities)
    return population[-1]


def evolve_population(population, mutation_rate=0.1, crossover_rate=0.7, elite_size=2):

    population_size = len(population)

    # Sort by fitness (descending)
    sorted_population = sorted(population, key=lambda x: x['fitness'], reverse=True)

    # Keep elite individuals
    new_population = []
    for i in range(elite_size):
        new_population.append({
            'tree': copy.deepcopy(sorted_population[i]['tree']),
            'fitness': None
        })

    # Generate rest of population
    while len(new_population) < population_size:
        if random.random() < crossover_rate and len(new_population) < population_size - 1:
            # Crossover
            parent1 = fitness_proportional_selection(population, 1)[0]
            parent2 = fitness_proportional_selection(population, 1)[0]
            offspring1, offspring2 = crossover_trees(parent1['tree'], parent2['tree'])

            new_population.append({'tree': offspring1, 'fitness': None})
            if len(new_population) < population_size:
                new_population.append({'tree': offspring2, 'fitness': None})
        else:
            # Mutation only
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


def evaluate_fitness_placeholder(tree):
    """
    Placeholder fitness function for testing genetic operations
    Will be replaced with neural network training later
    """
    # Simple fitness: prefer smaller trees that use all three inputs
    tree_size_penalty = count_nodes(tree) * 0.01

    # Count input diversity (bonus for using all three inputs)
    input_diversity = count_input_diversity(tree)
    diversity_bonus = input_diversity * 0.1

    # Random component (simulating performance variation)
    random_performance = random.uniform(0.7, 1.0)

    return random_performance + diversity_bonus - tree_size_penalty


def run_evolution(population_size=10, generations=5, max_depth=4):
    """
    Main evolution loop (placeholder - fitness evaluation will be added later)
    """
    print(f"Starting evolution: {population_size} individuals, {generations} generations")

    # Create initial population
    population = create_initial_population(population_size, max_depth)

    # Evolution loop
    for generation in range(generations):
        print(f"\nGeneration {generation + 1}")

        # Evaluate fitness (placeholder - will connect to neural networks later)
        for individual in population:
            individual['fitness'] = evaluate_fitness_placeholder(individual['tree'])

        # Print best individual
        best_individual = max(population, key=lambda x: x['fitness'])
        print(f"Best fitness: {best_individual['fitness']:.4f}")
        print(f"Best tree nodes: {count_nodes(best_individual['tree'])}")
        print(f"Best tree inputs used: {count_input_diversity(best_individual['tree'])}")

        # Evolve to next generation (except last generation)
        if generation < generations - 1:
            population = evolve_population(population)

    # Return best individual
    best_individual = max(population, key=lambda x: x['fitness'])
    return best_individual


# ============================================================================
# GENETIC ALGORITHM TESTING FUNCTIONS
# ============================================================================

def test_mutation_operations():
    """Test mutation operations"""
    print("Testing Mutation Operations:")

    # Create a known tree
    original_tree = create_binary_node('max',
                                       create_input_node('value'),
                                       create_constant_node(0.0))

    print("Original tree:")
    print(tree_to_string(original_tree))
    print()

    # Test multiple mutations
    for i in range(3):
        mutated = mutate_tree(original_tree, mutation_rate=0.5)
        print(f"Mutation {i + 1}:")
        print(tree_to_string(mutated))

        # Validate mutation
        try:
            validate_tree(mutated)
            print("✓ Mutation valid")
        except Exception as e:
            print(f"✗ Mutation invalid: {e}")

        print("-" * 30)


def test_crossover_operations():
    """Test crossover operations"""
    print("Testing Crossover Operations:")

    # Create two parent trees
    parent1 = create_binary_node('max',
                                 create_input_node('value'),
                                 create_constant_node(0.0))

    parent2 = create_binary_node('multiply',
                                 create_input_node('confidence'),
                                 create_input_node('missing'))

    print("Parent 1:")
    print(tree_to_string(parent1))
    print("\nParent 2:")
    print(tree_to_string(parent2))
    print()

    # Perform crossover
    offspring1, offspring2 = crossover_trees(parent1, parent2)

    print("Offspring 1:")
    print(tree_to_string(offspring1))
    print("\nOffspring 2:")
    print(tree_to_string(offspring2))

    # Validate offspring
    try:
        validate_tree(offspring1)
        validate_tree(offspring2)
        print("✓ Crossover successful")
    except Exception as e:
        print(f"✗ Crossover failed: {e}")


def test_population_management():
    """Test population creation and selection"""
    print("Testing Population Management:")

    # Create small population
    population = create_initial_population(5, max_depth=3)

    # Assign dummy fitness values
    fitness_values = [0.8, 0.6, 0.9, 0.7, 0.5]
    for i, individual in enumerate(population):
        individual['fitness'] = fitness_values[i]

    print("Population fitness values:", [ind['fitness'] for ind in population])

    # Test selection
    selected = fitness_proportional_selection(population, 3)
    selected_fitness = [ind['fitness'] for ind in selected]

    print(f"Selected individuals fitness: {selected_fitness}")
    print("✓ Selection completed")


def test_evolution_loop():
    """Test complete evolution loop"""
    print("Testing Evolution Loop:")

    # Run short evolution
    best_individual = run_evolution(population_size=5, generations=3, max_depth=3)

    print(f"\nFinal best individual:")
    print(f"Fitness: {best_individual['fitness']:.4f}")
    print(f"Tree: {tree_to_string(best_individual['tree'])}")
    print("✓ Evolution completed")


# ============================================================================
# NEURAL NETWORK INTEGRATION
# ============================================================================

import torch
import torch.nn as nn


class ChannelProp(nn.Module):
    """Propagate missing/confidence through a linear layer using |W|-based weights."""

    def __init__(self, linear: nn.Linear, eps: float = 1e-8):
        super().__init__()
        self.linear = linear
        self.eps = eps

    def forward(self, x, m, c):
        # x: (B, Din); m,c: (B, Din) with m in [0,1], c in [0,1]
        y = self.linear(x)  # (B, Dout)
        W = self.linear.weight  # (Dout, Din)
        W_abs = W.abs() + self.eps  # (Dout, Din)
        col_sum = W_abs.sum(dim=1, keepdim=True)  # (Dout,1)
        W_norm = W_abs / col_sum  # rows sum to 1 (across Din)
        # propagate confidence and missing via W_norm
        # transpose to apply as (B, Din) @ (Din, Dout)
        W_norm_T = W_norm.t()  # (Din, Dout)
        c_out = c @ W_norm_T  # (B, Dout)
        obs_in = 1.0 - m  # observedness
        obs_out = obs_in @ W_norm_T  # (B, Dout)
        m_out = 1.0 - obs_out
        # clamp to [0,1]
        c_out = c_out.clamp(0.0, 1.0)
        m_out = m_out.clamp(0.0, 1.0)
        return y, m_out, c_out


class AdaptiveLayer(nn.Module):
    """Applies evolved activation tree element-wise to (x, m, c)."""

    def __init__(self, activation_tree):
        super().__init__()
        self.activation_tree = activation_tree  # dict; autograd flows through internal ops

    def forward(self, x, m, c):
        return evaluate_tree(self.activation_tree, x, m, c)  # same shape as x


class ThreeChannelBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation_tree):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.prop = ChannelProp(self.lin)
        self.act = AdaptiveLayer(activation_tree)

    def forward(self, x, m, c):
        x, m, c = self.prop(x, m, c)  # propagate channels
        x = self.act(x, m, c)  # apply evolved activation
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
    """
    Evaluate activation function by training a neural network
    Returns validation accuracy as fitness (with complexity penalty)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Move data to device
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        missing_train = missing_train.to(device)
        missing_val = missing_val.to(device)
        conf_train = conf_train.to(device)
        conf_val = conf_val.to(device)

        # Create model with this activation function
        model = ThreeChannelMLP(
            input_size=X_train.shape[1],
            hidden_sizes=hidden_sizes,
            output_size=len(torch.unique(y_train)),
            activation_tree=tree
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Early stopping for fitness evaluation
        best_val_acc = 0.0
        patience = 2
        patience_counter = 0

        # Training with early stopping
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X_train, missing_train, conf_train)
            loss = criterion(outputs, y_train)

            # Check for NaN/Inf
            if not torch.isfinite(loss):
                return 0.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Evaluate validation accuracy
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val, missing_val, conf_val)
                _, predicted = torch.max(val_outputs, 1)
                val_acc = (predicted == y_val).float().mean().item()
            model.train()

            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break  # Stop early if no improvement

        # Apply complexity penalty
        size_penalty = 0.001 * count_nodes(tree)
        depth_penalty = 0.002 * (get_tree_depth(tree) - 1)
        fitness = best_val_acc - size_penalty - depth_penalty

        # Clean up
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
    """
    Evolution with neural network-based fitness evaluation
    """
    print(f"Starting evolution: {population_size} individuals, {generations} generations")
    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
    print(f"Network architecture: {X_train.shape[1]} -> {hidden_sizes} -> {len(torch.unique(y_train))}")
    print(f"Max tree depth: {max_depth}")

    # Create initial population
    population = create_initial_population(population_size, max_depth)

    # Track best ever
    best_ever = None
    best_ever_fitness = -float('inf')  # Changed from 0.0 to allow negative fitness

    for generation in range(generations):
        print(f"\n{'=' * 60}")
        print(f"Generation {generation + 1}/{generations}")
        print(f"{'=' * 60}")

        # Evaluate fitness for all individuals
        for idx, individual in enumerate(population):
            if individual['fitness'] is None:
                print(f"Evaluating individual {idx + 1}/{population_size}...", end=' ')

                fitness = evaluate_fitness_neural_network(
                    individual['tree'],
                    X_train, y_train, X_val, y_val,
                    missing_train, missing_val,
                    conf_train, conf_val,
                    num_epochs=30,  # Increased from 10, but early stopping will save time
                    hidden_sizes=hidden_sizes
                )

                individual['fitness'] = fitness
                print(f"Fitness: {fitness:.4f}")

        # Sort by fitness
        population.sort(key=lambda x: x['fitness'], reverse=True)

        # Track best
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

        # Evolve to next generation (except last generation)
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
    """
    Train final model with best activation function for more epochs
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print(f"Training Final Model on {device}")
    print(f"{'=' * 60}")
    print(f"Architecture: {X_train.shape[1]} -> [64, 32] -> {len(torch.unique(y_train))}")
    print(f"Training for up to {epochs} epochs with early stopping (patience={patience})")

    # Move data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    missing_train = missing_train.to(device)
    missing_val = missing_val.to(device)
    conf_train = conf_train.to(device)
    conf_val = conf_val.to(device)

    # Larger network for final training
    model = ThreeChannelMLP(
        input_size=X_train.shape[1],
        hidden_sizes=[64, 32],  # Bigger network for final model
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
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train, missing_train, conf_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val, missing_val, conf_val)
            _, predicted = torch.max(val_outputs, 1)
            val_acc = (predicted == y_val).float().mean().item()

            train_outputs = model(X_train, missing_train, conf_train)
            _, train_predicted = torch.max(train_outputs, 1)
            train_acc = (train_predicted == y_train).float().mean().item()

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Early stopping
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

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nFinal Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    return model, best_val_acc
# ============================================================================
# HEPATITIS DATASET EXPERIMENTS (OPTION 1)
# ============================================================================
def load_hepatitis_dataset():
    """Load and preprocess the hepatitis dataset"""
    try:
        # Load the CSV file
        df = pd.read_csv('house-votes-84-processed.csv')
        print(f"Loaded hepatitis dataset: {df.shape[0]} samples, {df.shape[1]} columns")

        # Replace '?' with NaN for proper missing value handling
        df = df.replace('?', np.nan)

        # Convert all columns to numeric (except where impossible)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        print(f"Missing values per column:")
        missing_counts = df.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                percentage = (count / len(df)) * 100
                print(f"  Column {col}: {count}/{len(df)} ({percentage:.1f}%)")

        return df

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def prepare_three_channel_data(df):
    """Convert hepatitis data to our three-channel format (value, missing, confidence)"""
    # Separate features (columns 0-18) and target (column 19)
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]  # Target

    # Calculate column means for imputation (only from observed values)
    column_means = X.mean()

    # Create three channels
    values = X.copy()
    missing_flags = X.isnull().astype(float)  # 1 = missing, 0 = observed
    confidence_scores = (~X.isnull()).astype(float)  # 1 = observed, 0 = missing initially

    # Impute missing values with column means
    values = values.fillna(column_means)

    # Set confidence scores based on missingness patterns
    # High confidence for observed, lower for imputed
    for col in X.columns:
        missing_rate = missing_flags[col].mean()

        # Confidence for imputed values depends on column's overall missing rate
        # Higher missing rate → lower confidence in imputed values
        imputed_confidence = max(0.1, 1.0 - missing_rate)

        confidence_scores.loc[missing_flags[col] == 1, col] = imputed_confidence

    print("\nThree-channel data prepared:")
    print(f"  Values shape: {values.shape}")
    print(f"  Missing flags shape: {missing_flags.shape}")
    print(f"  Confidence scores shape: {confidence_scores.shape}")
    print(f"  Overall missing rate: {missing_flags.values.mean():.3f}")

    return values, missing_flags, confidence_scores, y


def create_test_cases_from_hepatitis(values, missing_flags, confidence_scores, num_cases=20):
    """Create test cases from real hepatitis data patterns"""
    test_cases = []

    # Sample different types of data patterns
    n_samples, n_features = values.shape

    for i in range(min(num_cases, n_samples)):
        for j in range(min(5, n_features)):  # Test first 5 features
            test_case = {
                'sample_id': i,
                'feature_id': j,
                'value': float(values.iloc[i, j]),
                'missing': float(missing_flags.iloc[i, j]),
                'confidence': float(confidence_scores.iloc[i, j]),
                'description': f"Sample {i}, Feature {j}"
            }

            # Add descriptive information
            if test_case['missing'] == 1.0:
                test_case['description'] += " (imputed)"
            else:
                test_case['description'] += " (observed)"

            test_cases.append(test_case)

    return test_cases


def test_evolved_functions_on_hepatitis():
    """Main function to test evolved activation functions on hepatitis data"""
    print("=" * 60)
    print("EVOLVING ACTIVATION FUNCTIONS ON HEPATITIS DATA")
    print("=" * 60)

    # Step 1: Load and prepare data
    print("\nStep 1: Loading hepatitis dataset...")
    df = load_hepatitis_dataset()
    if df is None:
        print("Failed to load dataset. Please check if hepatitis.csv is available.")
        return

    print("\nStep 2: Converting to three-channel format...")
    values, missing_flags, confidence_scores, target = prepare_three_channel_data(df)

    # Step 3: Split and prepare data (FIX DATA LEAKAGE)
    print("\nStep 3: Splitting and preparing data (no leakage)...")
    from sklearn.model_selection import train_test_split

    # First split: 60% train, 40% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        values, target, test_size=0.4, random_state=42, stratify=target
    )

    # Second split: 20% val, 20% test from the 40% temp
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nInitial split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # ===== CRITICAL: Compute imputation statistics ONLY on train =====
    print("\nComputing imputation statistics from TRAIN set only...")

    # Calculate column means for imputation (only from train observed values)
    train_column_means = X_train.mean()

    # === Process TRAIN set ===
    values_train = X_train.copy()
    missing_flags_train = X_train.isnull().astype(float)
    confidence_scores_train = (~X_train.isnull()).astype(float)

    # Impute missing values with TRAIN means
    values_train = values_train.fillna(train_column_means)

    # Set confidence scores based on TRAIN missing patterns
    train_missing_rates = {}  # Store for reuse on val/test
    for col in X_train.columns:
        train_missing_rate = missing_flags_train[col].mean()
        train_missing_rates[col] = train_missing_rate
        imputed_confidence = max(0.1, 1.0 - train_missing_rate)
        confidence_scores_train.loc[missing_flags_train[col] == 1, col] = imputed_confidence

    # === Process VAL set (using TRAIN statistics) ===
    values_val = X_val.copy()
    missing_flags_val = X_val.isnull().astype(float)
    confidence_scores_val = (~X_val.isnull()).astype(float)

    # Impute with TRAIN means (not val means!)
    values_val = values_val.fillna(train_column_means)

    # Set confidence using TRAIN missing rates (not val rates!)
    for col in X_val.columns:
        train_missing_rate = train_missing_rates[col]  # Use TRAIN rate
        imputed_confidence = max(0.1, 1.0 - train_missing_rate)
        confidence_scores_val.loc[missing_flags_val[col] == 1, col] = imputed_confidence

    # === Process TEST set (using TRAIN statistics) ===
    values_test = X_test.copy()
    missing_flags_test = X_test.isnull().astype(float)
    confidence_scores_test = (~X_test.isnull()).astype(float)

    # Impute with TRAIN means (not test means!)
    values_test = values_test.fillna(train_column_means)

    # Set confidence using TRAIN missing rates (not test rates!)
    for col in X_test.columns:
        train_missing_rate = train_missing_rates[col]  # Use TRAIN rate
        imputed_confidence = max(0.1, 1.0 - train_missing_rate)
        confidence_scores_test.loc[missing_flags_test[col] == 1, col] = imputed_confidence

    # ===== Encode labels to 0..C-1 (FIT on train only) =====
    print("Encoding labels (fit on train only)...")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)  # FIT on train
    y_val_enc = le.transform(y_val)  # TRANSFORM only
    y_test_enc = le.transform(y_test)  # TRANSFORM only

    # ===== Standardize features (FIT on train only) =====
    print("Standardizing features (fit on train only)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(values_train)  # FIT on train
    X_val_scaled = scaler.transform(values_val)  # TRANSFORM only
    X_test_scaled = scaler.transform(values_test)  # TRANSFORM only

    # ===== Convert to torch tensors =====
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
    print(f"  Train: {len(X_train)} samples ({len(X_train) / (len(X_train) + len(X_val) + len(X_test)) * 100:.1f}%)")
    print(f"  Val:   {len(X_val)} samples ({len(X_val) / (len(X_train) + len(X_val) + len(X_test)) * 100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({len(X_test) / (len(X_train) + len(X_val) + len(X_test)) * 100:.1f}%)")
    print(f"  Features: {X_train_t.shape[1]} (standardized on train only)")
    print(f"  Classes: {len(np.unique(y_train))} → {len(torch.unique(y_train_t))}")
    print(f"  Device available: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print(f"\n  ✅ Imputation means computed from TRAIN only")
    print(f"  ✅ Missing rates computed from TRAIN only")
    print(f"  ✅ Standardization fitted on TRAIN only")
    print(f"  ✅ Label encoding fitted on TRAIN only")

    # Step 4: Evolve activation functions
    print("\nStep 4: Evolving activation functions with neural network fitness...")
    best_activation = run_evolution_with_nn(
        X_train_t, y_train_t, X_val_t, y_val_t,
        missing_train_t, missing_val_t,
        conf_train_t, conf_val_t,
        population_size=100,  # Increased
        generations=20,  # Increased
        max_depth=5,  # Reduced to control bloat
        hidden_sizes=[64, 32]  # Match final model architecture
    )

    print("\n" + "=" * 60)
    print("BEST ACTIVATION FUNCTION DISCOVERED:")
    print("=" * 60)
    print(tree_to_string(best_activation['tree']))
    print(f"\nValidation accuracy during evolution: {best_activation['fitness']:.4f}")
    print(f"Complexity: {count_nodes(best_activation['tree'])} nodes")
    print(f"Depth: {get_tree_depth(best_activation['tree'])}")
    print(f"Uses {count_input_diversity(best_activation['tree'])}/3 input channels")
    print("\n" + "=" * 60)
    print("BEST ACTIVATION FUNCTION DISCOVERED:")
    print("=" * 60)
    print(tree_to_string(best_activation['tree']))
    print(f"\nValidation fitness during evolution: {best_activation['fitness']:.4f}")
    print(f"Complexity: {count_nodes(best_activation['tree'])} nodes")
    print(f"Depth: {get_tree_depth(best_activation['tree'])}")
    print(f"Uses {count_input_diversity(best_activation['tree'])}/3 input channels")

    # ============ ADD THIS VISUALIZATION BLOCK ============
    print("\n" + "=" * 60)
    print("VISUALIZING THE EVOLVED FUNCTION:")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt

        # Create input range (avoid exact zero for division safety)
        x_range = torch.linspace(-5, 5, 200)
        # Add small offset to avoid division by zero
        x_range = torch.where(torch.abs(x_range) < 0.01,
                              torch.sign(x_range) * 0.01,
                              x_range)
        m_vals = torch.zeros_like(x_range)
        c_vals = torch.ones_like(x_range)

        # Evaluate evolved function
        with torch.no_grad():
            y_vals = evaluate_tree(best_activation['tree'], x_range, m_vals, c_vals)
            # Clamp extreme values for visualization
            y_vals = torch.clamp(y_vals, -10, 10)

        # Create figure
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
        plt.savefig('evolved_activation.png', dpi=150, bbox_inches='tight')
        print("✅ Saved visualization to: evolved_activation.png")
        plt.close()

    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")
    # ============ END VISUALIZATION BLOCK ============
    # Step 5: Train final model with best activation
    print("\nStep 5: Training final model with best activation...")
    final_model, final_val_acc = train_final_model(
        best_activation['tree'],
        X_train_t, y_train_t, X_val_t, y_val_t,
        missing_train_t, missing_val_t, conf_train_t, conf_val_t,
        epochs=100,
        patience=15
    )

    # Step 6: Evaluate on test set
    print("\nStep 6: Evaluating on held-out test set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model.eval()
    with torch.no_grad():
        test_outputs = final_model(X_test_t.to(device), missing_test_t.to(device), conf_test_t.to(device))
        _, predicted = torch.max(test_outputs, 1)
        test_acc = (predicted == y_test_t.to(device)).float().mean().item()

    print(f"Test Accuracy: {test_acc:.4f}")
    # Build Swish tree (x * sigmoid(x))
    swish_tree = create_binary_node(
        'multiply',
        create_input_node('value'),
        create_unary_node('sigmoid', create_input_node('value'))
    )
    # Step 7: Compare with ReLU baseline
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON: Training with ReLU")
    print("=" * 60)
    relu_tree = create_binary_node('max', create_input_node('value'), create_constant_node(0.0))

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
    # Step 8: Swish baseline
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

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Method':<25} {'Validation':<12} {'Test':<12}")
    print("-" * 60)
    print(f"{'Evolved Activation':<25} {final_val_acc:<12.4f} {test_acc:<12.4f}")
    print(f"{'ReLU Baseline':<25} {relu_val_acc:<12.4f} {relu_test_acc:<12.4f}")
    print(f"{'Swish Baseline':<25} {swish_val_acc:<12.4f} {swish_test_acc:<12.4f}")

    print("\n" + "Improvement over ReLU:")
    print(
        f"  Evolved - Val: {(final_val_acc - relu_val_acc) * 100:+.2f}%, Test: {(test_acc - relu_test_acc) * 100:+.2f}%")
    print(
        f"  Swish   - Val: {(swish_val_acc - relu_val_acc) * 100:+.2f}%, Test: {(swish_test_acc - relu_test_acc) * 100:+.2f}%")

    # Determine best performer
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
        print("   Try: More generations or different operators")

    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)




def analyze_evolved_function_behavior(evolved_functions, test_cases):
    """Analyze the behavior patterns of evolved functions"""
    print("\nBehavioral Analysis of Evolved Functions:")

    # Test sensitivity to missing data
    print("\n1. MISSING DATA SENSITIVITY:")

    # Find cases with same value but different missing status
    values_seen = {}
    for case in test_cases:
        val = round(case['value'], 1)
        if val not in values_seen:
            values_seen[val] = []
        values_seen[val].append(case)

    # Find a value that appears in both missing and observed states
    for val, cases in values_seen.items():
        missing_cases = [c for c in cases if c['missing'] == 1.0]
        observed_cases = [c for c in cases if c['missing'] == 0.0]

        if missing_cases and observed_cases:
            print(f"  Value {val}: comparing missing vs observed")

            missing_case = missing_cases[0]
            observed_case = observed_cases[0]

            for func_idx, tree in enumerate(evolved_functions):
                try:
                    # Test missing case
                    result_missing = evaluate_tree(tree,
                                                   torch.tensor([missing_case['value']]),
                                                   torch.tensor([missing_case['missing']]),
                                                   torch.tensor([missing_case['confidence']]))

                    # Test observed case
                    result_observed = evaluate_tree(tree,
                                                    torch.tensor([observed_case['value']]),
                                                    torch.tensor([observed_case['missing']]),
                                                    torch.tensor([observed_case['confidence']]))

                    diff = abs(result_observed.item() - result_missing.item())
                    print(f"    Function {func_idx + 1}: {diff:.4f} difference")

                except:
                    print(f"    Function {func_idx + 1}: Error in evaluation")
            break

    print("\n2. CONFIDENCE SENSITIVITY:")
    # Test how functions respond to confidence changes
    test_value = 10.0
    confidence_levels = [0.2, 0.5, 0.8, 1.0]

    print(f"  Testing value {test_value} with different confidence levels:")

    for func_idx, tree in enumerate(evolved_functions):
        print(f"    Function {func_idx + 1}:")
        try:
            for conf in confidence_levels:
                result = evaluate_tree(tree,
                                       torch.tensor([test_value]),
                                       torch.tensor([0.0]),  # Not missing
                                       torch.tensor([conf]))
                print(f"      Confidence {conf}: {result.item():.4f}")
        except:
            print(f"      Error in evaluation")

    print("\n3. FUNCTION COMPLEXITY:")
    for func_idx, tree in enumerate(evolved_functions):
        depth = get_tree_depth(tree)
        nodes = count_nodes(tree)
        diversity = count_input_diversity(tree)

        print(f"  Function {func_idx + 1}: depth={depth}, nodes={nodes}, inputs_used={diversity}/3")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
def run_hepatitis_experiments():
    """Wrapper function to run all hepatitis experiments"""
    try:
        test_evolved_functions_on_hepatitis()
    except Exception as e:
        print(f"Error in hepatitis experiments: {e}")
        print("Make sure hepatitis.csv is in the same directory as this script")


if __name__ == "__main__":
    set_seed(42)

    print("=== Tree Structure Implementation Tests ===")
    print()

    # Original tests
    test_known_functions()
    test_random_tree_generation()

    print("=== Genetic Algorithm Tests ===")
    print()

    test_mutation_operations()
    print()
    test_crossover_operations()
    print()
    test_population_management()
    print()

    print("=== HEPATITIS DATASET EXPERIMENTS WITH NEURAL NETWORKS ===")
    print()

    # Run the full experiment
    run_hepatitis_experiments()

    print("=== All tests completed ===")