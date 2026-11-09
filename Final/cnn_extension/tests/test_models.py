import pytest
import torch
import torch.nn.functional as F

from cnn_extension import gp_core as g
from cnn_extension.baselines import FIXED_ACTIVATIONS, fixed_activation_tree
from cnn_extension.conv_layers import ThreeStreamAvgPool, flatten_streams
from cnn_extension.models import (ConcatInputCNN, ThreeChannelCNN, ThreeChannelImageMLP, build_model,
                                  count_parameters, first_layer_parameter_count)


def full_tree():
    # f(z, m, c) = tanh(z) * c - 0.5 * m
    return g.create_binary_node(
        "subtract",
        g.create_binary_node("multiply", g.create_unary_node("tanh", g.create_input_node("value")),
                             g.create_input_node("confidence")),
        g.create_binary_node("multiply", g.create_constant_node(0.5), g.create_input_node("missing")),
    )


def inputs(n, c, h, w, seed=0):
    torch.manual_seed(seed)
    X = torch.rand(n, c, h, w)
    M = (torch.rand_like(X) < 0.2).float()
    C = torch.where(M > 0, torch.full_like(X, 0.7), torch.ones_like(X))
    y = torch.randint(0, 10, (n,))
    return X, M, C, y


@pytest.mark.parametrize("cin,size,flat", [(1, 28, 800), (3, 32, 1152)])
def test_flattened_size_inferred_from_input_dimensions(cin, size, flat):
    model = ThreeChannelCNN(cin, size, size, 10, full_tree())
    assert model.flat_dim == flat
    X, M, C, _ = inputs(3, cin, size, size)
    assert model(X, M, C).shape == (3, 10)
    concat = ConcatInputCNN(cin, size, size, 10, fixed_activation_tree("relu"))
    assert concat.flat_dim == flat and concat(X, M, C).shape == (3, 10)
    mlp = ThreeChannelImageMLP(cin, size, size, 10, full_tree())
    assert mlp(X, M, C).shape == (3, 10)


PROTECTED_TREES = {
    "divide_by_missing": g.create_binary_node("divide", g.create_input_node("value"), g.create_input_node("missing")),
    "log": g.create_unary_node("log", g.create_input_node("value")),
    "sqrt": g.create_unary_node("sqrt", g.create_input_node("value")),
    "exp": g.create_unary_node("exp", g.create_binary_node("multiply", g.create_constant_node(2.0),
                                                           g.create_input_node("value"))),
    "cube": g.create_unary_node("cube", g.create_input_node("value")),
    "max_min": g.create_binary_node("max", g.create_input_node("value"),
                                    g.create_binary_node("min", g.create_input_node("confidence"),
                                                         g.create_input_node("missing"))),
    "full": full_tree(),
}


@pytest.mark.parametrize("name", list(PROTECTED_TREES))
@pytest.mark.parametrize("backbone", ["cnn", "mlp", "concat_cnn"])
def test_forward_and_backward_with_protected_expressions(name, backbone):
    tree = PROTECTED_TREES[name]
    torch.manual_seed(0)
    model = build_model(backbone, 1, 28, 28, 10, tree)
    X, M, C, y = inputs(4, 1, 28, 28)
    logits = model(X, M, C)
    loss = F.cross_entropy(logits, y)
    assert torch.isfinite(loss)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


@pytest.mark.parametrize("name", list(FIXED_ACTIVATIONS))
def test_fixed_baseline_trees_match_their_activation(name):
    tree = fixed_activation_tree(name)
    z = torch.linspace(-4, 4, 81)
    m, c = torch.rand_like(z), torch.rand_like(z)       # must be ignored
    out = g.evaluate_tree(tree, z, m, c)
    expected = {"relu": F.relu(z), "swish": F.silu(z), "leakyrelu": F.leaky_relu(z, 0.1), "elu": F.elu(z),
                "elu_original": torch.exp(z) - 1}[name]
    assert torch.allclose(out, expected, atol=1e-5)
    assert g.count_input_diversity(tree) == 1


def test_elu_original_tree_is_not_elu():
    z = torch.linspace(0.5, 3, 6)
    assert not torch.allclose(g.evaluate_tree(fixed_activation_tree("elu_original"), z, z * 0, z * 0 + 1), F.elu(z))


def test_three_stream_pooling_and_flatten_preserve_alignment():
    pool = ThreeStreamAvgPool(2)
    H = torch.zeros(1, 2, 4, 4)
    M = torch.zeros_like(H)
    H[0, 1, 2, 3] = 1.0
    M[0, 1, 2, 3] = 1.0                      # the same (channel, row, col) is "hot" in both streams
    h, m, c = pool(H, M, torch.ones_like(H))
    assert torch.equal(h, F.avg_pool2d(H, 2)) and torch.equal(m, F.avg_pool2d(M, 2))
    assert torch.equal(c, torch.ones_like(c))
    hf, mf, cf = flatten_streams(h, m, c)
    assert hf.shape == mf.shape == cf.shape == (1, 8)
    assert hf.argmax() == mf.argmax()        # same flat index -> same ordering of the three streams


def test_first_only_mode_feeds_neutral_metadata_after_first_activation():
    torch.manual_seed(0)
    captured = {}

    def hook(module, args):
        captured["fc_m"], captured["fc_c"] = args[1].detach().clone(), args[2].detach().clone()

    for mode in ("full", "first_only"):
        model = ThreeChannelCNN(1, 28, 28, 10, full_tree(), metadata_mode=mode)
        handle = model.fc.register_forward_pre_hook(hook)
        X, M, C, _ = inputs(2, 1, 28, 28)
        model(X, M, C)
        handle.remove()
        if mode == "first_only":
            assert torch.allclose(captured["fc_m"], torch.zeros_like(captured["fc_m"]), atol=1e-6)
            assert torch.allclose(captured["fc_c"], torch.ones_like(captured["fc_c"]), atol=1e-6)
        else:
            assert captured["fc_m"].max() > 0 and captured["fc_c"].min() < 1


def test_fixed_activation_networks_ignore_metadata_streams():
    torch.manual_seed(0)
    model = ThreeChannelCNN(1, 28, 28, 10, fixed_activation_tree("relu"))
    X, M, C, _ = inputs(2, 1, 28, 28)
    out1 = model(X, M, C)
    out2 = model(X, torch.zeros_like(M), torch.ones_like(C))
    assert torch.allclose(out1, out2, atol=1e-6)


def test_concat_cnn_input_layer_parameter_count_is_three_times_larger():
    for cin in (1, 3):
        three = ThreeChannelCNN(cin, 28, 28, 10, fixed_activation_tree("relu"))
        concat = ConcatInputCNN(cin, 28, 28, 10, fixed_activation_tree("relu"))
        assert first_layer_parameter_count(three) == 16 * cin * 9 + 16
        assert first_layer_parameter_count(concat) == 16 * 3 * cin * 9 + 16
        assert count_parameters(concat) - count_parameters(three) == 16 * 2 * cin * 9


def test_invalid_backbone_and_mode_rejected():
    with pytest.raises(ValueError):
        build_model("resnet", 1, 28, 28, 10, full_tree())
    with pytest.raises(ValueError):
        ThreeChannelCNN(1, 28, 28, 10, full_tree(), metadata_mode="all")
