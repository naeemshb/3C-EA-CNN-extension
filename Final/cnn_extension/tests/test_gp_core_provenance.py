"""gp_core.py must be a verbatim copy of the original GP / ChannelProp code in Final/Pima/New.py."""
import ast
import os

EXPECTED = [
    "INPUTS", "CONSTANTS", "create_input_node", "create_constant_node", "create_unary_node", "create_binary_node",
    "safe_divide", "safe_sqrt", "safe_log", "safe_pow", "clip_result", "UNARY_OPERATORS", "BINARY_OPERATORS",
    "evaluate_tree", "tree_to_string", "tree_to_infix", "get_tree_depth", "count_nodes", "generate_terminal_node",
    "generate_non_terminal_node", "generate_random_tree", "validate_tree", "get_all_nodes", "mutate_single_node",
    "mutate_tree", "perform_crossover", "crossover_trees", "create_initial_population", "weighted_random_choice",
    "fitness_proportional_selection", "evolve_population", "count_input_diversity",
    "ChannelProp", "AdaptiveLayer", "ThreeChannelBlock", "ThreeChannelMLP",
]


def _definitions(path):
    with open(path) as f:
        tree = ast.parse(f.read())
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = ast.dump(node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = ast.dump(node)
    return out


def test_gp_core_matches_original_source(final_dir):
    src = _definitions(os.path.join(final_dir, "Pima", "New.py"))
    dst = _definitions(os.path.join(final_dir, "cnn_extension", "gp_core.py"))
    for name in EXPECTED:
        assert name in dst, f"{name} missing from gp_core"
        assert name in src, f"{name} missing from original"
        assert dst[name] == src[name], f"{name} differs from the original implementation"


def test_gp_core_contains_nothing_beyond_the_original(final_dir):
    src = _definitions(os.path.join(final_dir, "Pima", "New.py"))
    dst = _definitions(os.path.join(final_dir, "cnn_extension", "gp_core.py"))
    extra = set(dst) - set(src)
    assert not extra, f"unexpected additions in gp_core: {extra}"
    assert set(dst) == set(EXPECTED)
