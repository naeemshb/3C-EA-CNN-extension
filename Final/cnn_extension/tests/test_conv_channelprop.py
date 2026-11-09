import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn_extension import gp_core as g
from cnn_extension.conv_layers import CHANNELPROP_EPS, ConvChannelProp, ConvThreeChannelBlock


def reference_propagation(t, W, eps, stride=1):
    """Explicit-loop reference for convolutional metadata propagation (no padding)."""
    A = W.abs() + eps
    N, Cin, H, Wd = t.shape
    Cout, _, kh, kw = W.shape
    Ho, Wo = (H - kh) // stride + 1, (Wd - kw) // stride + 1
    out = torch.zeros(N, Cout, Ho, Wo, dtype=t.dtype)
    for n in range(N):
        for o in range(Cout):
            a, denom = A[o], A[o].sum()
            for i in range(Ho):
                for j in range(Wo):
                    window = t[n, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
                    out[n, o, i, j] = (window * a).sum() / denom
    return out.clamp(0.0, 1.0)


def make_prop(cin=1, cout=1, k=2, stride=1, bias=False, seed=0):
    torch.manual_seed(seed)
    conv = nn.Conv2d(cin, cout, k, stride=stride, padding=0, bias=bias)
    return ConvChannelProp(conv)


def test_numerical_example_from_the_brief():
    prop = make_prop(bias=False)
    with torch.no_grad():
        prop.conv.weight.fill_(0.25)
    X = torch.tensor([[[[0.2, 0.4], [0.6, 0.8]]]])
    M = torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]])
    C = torch.tensor([[[[1.0, 0.8], [1.0, 1.0]]]])
    Z, M_out, C_out = prop(X, M, C)
    assert Z.shape == (1, 1, 1, 1)
    assert torch.allclose(Z, torch.tensor(0.50), atol=1e-6)
    assert torch.allclose(M_out, torch.tensor(0.25), atol=1e-6)
    assert torch.allclose(C_out, torch.tensor(0.95), atol=1e-6)
    # illustrative activation f(z, m, c) = z*c - 0.1*m through the ORIGINAL tree evaluator
    tree = g.create_binary_node(
        "subtract",
        g.create_binary_node("multiply", g.create_input_node("value"), g.create_input_node("confidence")),
        g.create_binary_node("multiply", g.create_constant_node(0.1), g.create_input_node("missing")),
    )
    H = g.evaluate_tree(tree, Z, M_out, C_out)
    assert torch.allclose(H, torch.tensor(0.45), atol=1e-6)


@pytest.mark.parametrize("cin,cout,k,stride,H", [(1, 4, 3, 1, 7), (3, 5, 3, 1, 8), (3, 2, 2, 2, 9), (1, 3, 5, 2, 11)])
def test_agrees_with_explicit_reference(cin, cout, k, stride, H):
    prop = make_prop(cin, cout, k, stride=stride, bias=True, seed=cin + cout + k)
    torch.manual_seed(1)
    M = (torch.rand(2, cin, H, H) < 0.3).float()
    C = torch.rand(2, cin, H, H)
    X = torch.rand(2, cin, H, H)
    Z, M_out, C_out = prop(X, M, C)
    W = prop.conv.weight.detach()
    assert torch.allclose(M_out, reference_propagation(M, W, prop.eps, stride), atol=1e-6)
    assert torch.allclose(C_out, reference_propagation(C, W, prop.eps, stride), atol=1e-6)
    assert torch.allclose(Z, F.conv2d(X, W, prop.conv.bias.detach(), stride=stride), atol=1e-6)


@pytest.mark.parametrize("cin,H,expected", [(1, 28, (4, 16, 26, 26)), (3, 32, (4, 16, 30, 30))])
def test_shapes_grayscale_and_rgb(cin, H, expected):
    prop = make_prop(cin, 16, 3, bias=True)
    X = torch.rand(4, cin, H, H)
    M = torch.zeros_like(X)
    C = torch.ones_like(X)
    Z, M_out, C_out = prop(X, M, C)
    assert Z.shape == M_out.shape == C_out.shape == expected


def test_values_use_signed_weights_but_metadata_uses_absolute_weights():
    prop = make_prop(2, 3, 3, bias=False, seed=5)
    W = prop.conv.weight.detach()
    assert (W < 0).any() and (W > 0).any()
    torch.manual_seed(2)
    X, M, C = torch.rand(2, 2, 6, 6), (torch.rand(2, 2, 6, 6) < 0.4).float(), torch.rand(2, 2, 6, 6)
    Z1, M1, C1 = prop(X, M, C)
    with torch.no_grad():
        prop.conv.weight.neg_()  # flip every sign
    Z2, M2, C2 = prop(X, M, C)
    assert torch.allclose(Z2, -Z1, atol=1e-6)       # value path is signed
    assert torch.allclose(M2, M1, atol=1e-6)        # metadata path only sees |W|
    assert torch.allclose(C2, C1, atol=1e-6)


def test_all_observed_inputs_stay_observed_and_all_missing_stay_missing():
    prop = make_prop(3, 4, 3, bias=True, seed=3)
    X = torch.rand(2, 3, 9, 9)
    _, M_out, C_out = prop(X, torch.zeros_like(X), torch.ones_like(X))
    assert torch.allclose(M_out, torch.zeros_like(M_out), atol=1e-6)
    assert torch.allclose(C_out, torch.ones_like(C_out), atol=1e-6)
    _, M_out, _ = prop(X, torch.ones_like(X), torch.full_like(X, 0.1))
    assert torch.allclose(M_out, torch.ones_like(M_out), atol=1e-6)


def test_zero_weights_give_finite_uniform_average_through_epsilon():
    prop = make_prop(1, 2, 2, bias=False)
    with torch.no_grad():
        prop.conv.weight.zero_()
    M = torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]])
    _, M_out, C_out = prop(M, M, 1.0 - 0.5 * M)
    assert torch.isfinite(M_out).all() and torch.isfinite(C_out).all()
    assert torch.allclose(M_out, torch.tensor(0.25), atol=1e-6)   # plain window average
    assert torch.allclose(C_out, torch.tensor(0.875), atol=1e-6)
    assert prop.eps == CHANNELPROP_EPS == 1e-8


def test_metadata_stays_in_unit_interval():
    prop = make_prop(3, 6, 3, bias=True, seed=11)
    torch.manual_seed(4)
    X = torch.rand(3, 3, 12, 12)
    M = (torch.rand_like(X) < 0.5).float()
    C = torch.rand_like(X)
    for _ in range(3):
        X, M, C = prop(X.abs() * 5, M, C)   # feed amplified values; metadata must stay bounded
        assert M.min() >= 0 and M.max() <= 1 and C.min() >= 0 and C.max() <= 1
        X, M, C = X[:, :3], M[:, :3], C[:, :3]


def test_direct_missingness_average_equals_observedness_complement():
    prop = make_prop(2, 3, 3, bias=True, seed=7)
    M = (torch.rand(2, 2, 7, 7) < 0.3).float()
    direct = prop.propagate(M)
    via_observed = 1.0 - prop.propagate(1.0 - M)
    assert torch.allclose(direct, via_observed, atol=1e-5)


def test_one_by_one_conv_on_one_pixel_matches_original_fc_channelprop():
    torch.manual_seed(9)
    lin = nn.Linear(5, 3)
    fc = g.ChannelProp(lin)
    conv = nn.Conv2d(5, 3, 1, bias=True)
    with torch.no_grad():
        conv.weight.copy_(lin.weight.view(3, 5, 1, 1))
        conv.bias.copy_(lin.bias)
    cp = ConvChannelProp(conv)
    x, m, c = torch.rand(4, 5), (torch.rand(4, 5) < 0.4).float(), torch.rand(4, 5)
    y1, m1, c1 = fc(x, m, c)
    y2, m2, c2 = cp(x.view(4, 5, 1, 1), m.view(4, 5, 1, 1), c.view(4, 5, 1, 1))
    assert torch.allclose(y1, y2.view(4, 3), atol=1e-5)
    assert torch.allclose(m1, m2.view(4, 3), atol=1e-5)
    assert torch.allclose(c1, c2.view(4, 3), atol=1e-5)


def test_routing_weights_are_not_detached_gradients_reach_W_through_metadata_path():
    prop = make_prop(1, 2, 3, bias=False, seed=12)
    C = torch.rand(2, 1, 6, 6)
    loss = (prop.propagate(C) ** 2).sum()
    loss.backward()
    assert prop.conv.weight.grad is not None and prop.conv.weight.grad.abs().sum() > 0


def test_unsupported_configurations_are_rejected():
    with pytest.raises(ValueError, match="padding"):
        ConvChannelProp(nn.Conv2d(1, 1, 3, padding=1))
    with pytest.raises(ValueError, match="padding"):
        ConvChannelProp(nn.Conv2d(1, 1, 3, padding="same"))
    with pytest.raises(ValueError, match="groups"):
        ConvChannelProp(nn.Conv2d(2, 2, 3, groups=2))
    with pytest.raises(ValueError, match="padding_mode"):
        ConvChannelProp(nn.Conv2d(1, 1, 3, padding=0, padding_mode="reflect"))
    with pytest.raises(ValueError, match="share a shape"):
        make_prop()(torch.rand(1, 1, 4, 4), torch.rand(1, 1, 3, 3), torch.rand(1, 1, 4, 4))


def test_conv_block_applies_tree_elementwise_on_propagated_streams():
    tree = g.create_binary_node("multiply", g.create_input_node("value"), g.create_input_node("confidence"))
    torch.manual_seed(0)
    block = ConvThreeChannelBlock(1, 2, 3, tree)
    X = torch.rand(2, 1, 6, 6)
    M = (torch.rand_like(X) < 0.3).float()
    C = 1.0 - 0.5 * M
    Z, M_out, C_out = block.prop(X, M, C)
    H, M2, C2 = block(X, M, C)
    assert torch.allclose(H, Z * C_out, atol=1e-6)
    assert torch.equal(M2, M_out) and torch.equal(C2, C_out)
