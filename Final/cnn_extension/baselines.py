"""
Fixed-activation baseline trees.

In the original scripts the baselines are evaluated THROUGH the same
three-channel network as the evolved activation: the baseline is a GP tree
that references only the ``value`` terminal, so the missingness and
confidence streams are propagated but ignored by every activation.  For such
a tree the network is mathematically identical to a conventional network
with that activation (the metadata streams never touch the value path).

Two of the original baseline trees differ from the activation they are named
after (see README "Discrepancies"):

* ``leakyrelu`` in the original is ``max(x, 0.1*x)`` (slope 0.1, not the
  PyTorch default 0.01).  We keep 0.1 and expose the slope.
* ``elu`` in the original is ``max(x, exp(x) - 1)``.  Because
  ``exp(x) - 1 >= x`` for every real x, this equals ``exp(x) - 1`` everywhere
  and is NOT ELU.  The default here is the true ELU (alpha = 1) via the
  existing unary operator; the original tree is kept as ``elu_original``.
"""
from . import gp_core as g


def relu_tree() -> dict:
    # original: max(value, 0)
    return g.create_binary_node("max", g.create_input_node("value"), g.create_constant_node(0.0))


def swish_tree() -> dict:
    # original: value * sigmoid(value)  (== SiLU)
    return g.create_binary_node(
        "multiply",
        g.create_input_node("value"),
        g.create_unary_node("sigmoid", g.create_input_node("value")),
    )


def leakyrelu_tree(slope: float = 0.1) -> dict:
    # original: max(value, 0.1 * value)
    return g.create_binary_node(
        "max",
        g.create_input_node("value"),
        g.create_binary_node("multiply", g.create_constant_node(slope), g.create_input_node("value")),
    )


def elu_tree() -> dict:
    # true ELU(alpha=1) through the existing protected unary operator
    return g.create_unary_node("elu", g.create_input_node("value"))


def elu_original_tree() -> dict:
    # the tree the original scripts label "ELU": max(value, exp(value) - 1) == exp(value) - 1
    return g.create_binary_node(
        "max",
        g.create_input_node("value"),
        g.create_binary_node(
            "subtract",
            g.create_unary_node("exp", g.create_input_node("value")),
            g.create_constant_node(1.0),
        ),
    )


FIXED_ACTIVATIONS = {
    "relu": relu_tree,
    "swish": swish_tree,
    "leakyrelu": leakyrelu_tree,
    "elu": elu_tree,
    "elu_original": elu_original_tree,
}


def fixed_activation_tree(name: str) -> dict:
    if name not in FIXED_ACTIVATIONS:
        raise ValueError(f"Unknown fixed activation '{name}'. Choose from {sorted(FIXED_ACTIVATIONS)}")
    tree = FIXED_ACTIVATIONS[name]()
    g.validate_tree(tree)
    return tree
