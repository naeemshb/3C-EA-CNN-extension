"""
Image backbones that all expose ``forward(X, M, C) -> logits``.

* ``ThreeChannelCNN``      - the starter architecture of the brief with
                             convolutional ChannelProp, three-stream average
                             pooling and the ORIGINAL fully connected
                             ChannelProp block before the classifier.
* ``ThreeChannelImageMLP`` - the original 3C-MLP (gp_core.ThreeChannelMLP)
                             applied to flattened images and metadata.
* ``ConcatInputCNN``       - a conventional CNN that receives ``cat([X, M, C])``
                             as its input channels (no propagation); the
                             activation tree is evaluated with neutral
                             metadata m = 0, c = 1, so fixed trees and
                             value-only evolved trees are both usable.

The final linear layer always produces raw class logits; no activation is
applied to them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gp_core import ThreeChannelBlock, ThreeChannelMLP, AdaptiveLayer
from .conv_layers import ConvThreeChannelBlock, ThreeStreamAvgPool, flatten_streams, conv_output_size

METADATA_MODES = ("full", "first_only")


class ThreeChannelCNN(nn.Module):
    def __init__(self, in_channels: int, height: int, width: int, num_classes: int, activation_tree: dict,
                 conv_channels=(16, 32), kernel_size: int = 3, pool_size: int = 2, fc_hidden: int = 64,
                 metadata_mode: str = "full"):
        super().__init__()
        if metadata_mode not in METADATA_MODES:
            raise ValueError(f"metadata_mode must be one of {METADATA_MODES}")
        self.metadata_mode = metadata_mode
        blocks = []
        ch, h, w = in_channels, height, width
        for out_ch in conv_channels:
            blocks.append(ConvThreeChannelBlock(ch, out_ch, kernel_size, activation_tree))
            h = conv_output_size(h, kernel_size) // pool_size
            w = conv_output_size(w, kernel_size) // pool_size
            ch = out_ch
            if h < 1 or w < 1:
                raise ValueError("input too small for this architecture")
        self.blocks = nn.ModuleList(blocks)
        self.pool = ThreeStreamAvgPool(pool_size)
        self.flat_dim = ch * h * w
        # original fully connected ChannelProp + evolved activation
        self.fc = ThreeChannelBlock(self.flat_dim, fc_hidden, activation_tree)
        self.out = nn.Linear(fc_hidden, num_classes)

    def forward(self, x, m, c):
        for i, block in enumerate(self.blocks):
            x, m, c = block(x, m, c)
            if self.metadata_mode == "first_only" and i == 0:
                # real metadata reached the first convolutional activation only
                m = torch.zeros_like(m)
                c = torch.ones_like(c)
            x, m, c = self.pool(x, m, c)
        x, m, c = flatten_streams(x, m, c)
        x, m, c = self.fc(x, m, c)
        return self.out(x)


class ThreeChannelImageMLP(nn.Module):
    """Original 3C-MLP on flattened images; hidden sizes default to the original [64, 32]."""

    def __init__(self, in_channels: int, height: int, width: int, num_classes: int, activation_tree: dict,
                 hidden_sizes=(64, 32)):
        super().__init__()
        self.mlp = ThreeChannelMLP(in_channels * height * width, list(hidden_sizes), num_classes, activation_tree)

    def forward(self, x, m, c):
        return self.mlp(x.flatten(1), m.flatten(1), c.flatten(1))


class ConcatInputCNN(nn.Module):
    """Conventional CNN with [X, M, C] concatenated along the channel axis (3x input channels)."""

    def __init__(self, in_channels: int, height: int, width: int, num_classes: int, activation_tree: dict,
                 conv_channels=(16, 32), kernel_size: int = 3, pool_size: int = 2, fc_hidden: int = 64):
        super().__init__()
        convs = []
        ch, h, w = 3 * in_channels, height, width
        for out_ch in conv_channels:
            convs.append(nn.Conv2d(ch, out_ch, kernel_size, padding=0))
            h = conv_output_size(h, kernel_size) // pool_size
            w = conv_output_size(w, kernel_size) // pool_size
            ch = out_ch
        self.convs = nn.ModuleList(convs)
        self.pool_size = pool_size
        self.flat_dim = ch * h * w
        self.fc = nn.Linear(self.flat_dim, fc_hidden)
        self.out = nn.Linear(fc_hidden, num_classes)
        self.act = AdaptiveLayer(activation_tree)

    def _act(self, z):
        return self.act(z, torch.zeros_like(z), torch.ones_like(z))

    def forward(self, x, m, c):
        h = torch.cat([x, m, c], dim=1)
        for conv in self.convs:
            h = F.avg_pool2d(self._act(conv(h)), self.pool_size)
        h = self._act(self.fc(h.flatten(1)))
        return self.out(h)

    def input_layer_parameter_count(self) -> int:
        conv = self.convs[0]
        return conv.weight.numel() + (conv.bias.numel() if conv.bias is not None else 0)


BACKBONES = ("cnn", "mlp", "concat_cnn")


def build_model(backbone: str, in_channels: int, height: int, width: int, num_classes: int,
                activation_tree: dict, architecture: dict | None = None, metadata_mode: str = "full") -> nn.Module:
    arch = dict(architecture or {})
    conv_channels = tuple(arch.get("conv_channels", (16, 32)))
    kernel_size = int(arch.get("kernel_size", 3))
    pool_size = int(arch.get("pool_size", 2))
    fc_hidden = int(arch.get("fc_hidden", 64))
    if backbone == "cnn":
        return ThreeChannelCNN(in_channels, height, width, num_classes, activation_tree,
                               conv_channels, kernel_size, pool_size, fc_hidden, metadata_mode)
    if backbone == "mlp":
        return ThreeChannelImageMLP(in_channels, height, width, num_classes, activation_tree,
                                    tuple(arch.get("mlp_hidden_sizes", (64, 32))))
    if backbone == "concat_cnn":
        return ConcatInputCNN(in_channels, height, width, num_classes, activation_tree,
                              conv_channels, kernel_size, pool_size, fc_hidden)
    raise ValueError(f"unknown backbone '{backbone}'; choose from {BACKBONES}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def first_layer_parameter_count(model: nn.Module) -> int:
    if isinstance(model, ConcatInputCNN):
        return model.input_layer_parameter_count()
    if isinstance(model, ThreeChannelCNN):
        conv = model.blocks[0].conv
        return conv.weight.numel() + conv.bias.numel()
    if isinstance(model, ThreeChannelImageMLP):
        lin = model.mlp.blocks[0].lin
        return lin.weight.numel() + lin.bias.numel()
    raise TypeError(type(model))
