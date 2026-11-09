"""
Convolutional ChannelProp and three-stream pooling / flattening.

Every layer here accepts and returns the aligned triple ``(X, M, C)``:
``X`` values, ``M`` missingness (1 = missing, 0 = observed), ``C`` confidence,
all with identical shape ``[batch, channels, height, width]``.

Convolutional ChannelProp (this file) mirrors the fully connected ChannelProp
of the original method (``gp_core.ChannelProp``):

    Z      = conv2d(X, W, bias=b)                      value stream, signed W
    A      = |W| + eps                                 non-negative routing weights
    denom  = sum of A over (in_channels, kh, kw)       one scalar per output filter
    M_out  = conv2d(M, A, bias=None) / denom           clipped to [0, 1]
    C_out  = conv2d(C, A, bias=None) / denom           clipped to [0, 1]
    H      = f(Z, M_out, C_out)                        evolved activation, elementwise

Gradient behaviour: the original fully connected ChannelProp uses
``self.linear.weight`` directly (NOT ``.detach()``), so gradients from the
metadata path flow into W through |W|.  We preserve that: routing uses W.

Padding: only ``padding=0`` is supported.  With zero padding the metadata
normalisation would have to be computed over the valid input positions of
each output location; that is deliberately left for later and rejected here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gp_core import ChannelProp, AdaptiveLayer, ThreeChannelBlock  # noqa: F401  (re-exported for reuse)

# The original ChannelProp default epsilon (gp_core.ChannelProp.__init__).
CHANNELPROP_EPS = 1e-8


def _as_pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


class ConvChannelProp(nn.Module):
    """Propagate (M, C) through a Conv2d using |W|-based routing weights."""

    def __init__(self, conv: nn.Conv2d, eps: float = CHANNELPROP_EPS):
        super().__init__()
        if not isinstance(conv, nn.Conv2d):
            raise TypeError("ConvChannelProp expects an nn.Conv2d")
        pad = conv.padding
        if isinstance(pad, str) or any(int(p) != 0 for p in _as_pair(pad)):
            raise ValueError(
                f"ConvChannelProp supports padding=0 only (got padding={pad!r}); "
                "padded metadata propagation would need per-position normalisation over valid inputs."
            )
        if conv.groups != 1:
            raise ValueError("ConvChannelProp supports ungrouped convolutions only (groups=1)")
        if conv.padding_mode != "zeros":
            raise ValueError("ConvChannelProp supports padding_mode='zeros' only")
        self.conv = conv
        self.eps = float(eps)

    def routing_weights(self):
        """A = |W| + eps and its per-filter normaliser.  Uses W (not detached), as in the original."""
        W = self.conv.weight  # (out, in, kh, kw)
        A = W.abs() + self.eps
        denom = A.sum(dim=(1, 2, 3)).view(1, -1, 1, 1)
        return A, denom

    def propagate(self, t: torch.Tensor) -> torch.Tensor:
        A, denom = self.routing_weights()
        out = F.conv2d(
            t, A, bias=None,
            stride=self.conv.stride, padding=0, dilation=self.conv.dilation, groups=1,
        )
        return (out / denom).clamp(0.0, 1.0)

    def forward(self, x, m, c):
        if not (x.shape == m.shape == c.shape):
            raise ValueError(f"X, M, C must share a shape; got {tuple(x.shape)}, {tuple(m.shape)}, {tuple(c.shape)}")
        z = self.conv(x)          # signed weights + learned bias
        m_out = self.propagate(m)  # no bias, |W| routing
        c_out = self.propagate(c)
        return z, m_out, c_out


class ConvThreeChannelBlock(nn.Module):
    """Conv2d + convolutional ChannelProp + evolved activation (elementwise on the three streams)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, activation_tree: dict,
                 stride: int = 1, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, dilation=dilation, bias=True)
        self.prop = ConvChannelProp(self.conv)
        self.act = AdaptiveLayer(activation_tree)

    def forward(self, x, m, c):
        x, m, c = self.prop(x, m, c)
        x = self.act(x, m, c)
        return x, m, c


class ThreeStreamAvgPool(nn.Module):
    """Identical average pooling applied to H, M and C."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, h, m, c):
        k = self.kernel_size
        return F.avg_pool2d(h, k), F.avg_pool2d(m, k), F.avg_pool2d(c, k)


def flatten_streams(h, m, c):
    """Flatten the three streams with the same (channel, row, col) ordering."""
    return h.flatten(1), m.flatten(1), c.flatten(1)


def conv_output_size(size: int, kernel_size: int, stride: int = 1, dilation: int = 1) -> int:
    return (size - dilation * (kernel_size - 1) - 1) // stride + 1
