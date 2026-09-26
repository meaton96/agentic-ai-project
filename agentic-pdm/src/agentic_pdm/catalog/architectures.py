"""Model architecture catalog. Every model maps (B, C, T) to logits (B, K)
and is sized for CPU training: the bounds keep a full cross-validation run
in minutes-to-hours on a few cores, not days."""
from __future__ import annotations

import torch
from torch import nn

from agentic_pdm.catalog.params import CatalogEntry, ParamSpec


def _dropout():
    return ParamSpec("float", 0.3, "dropout before the classifier head", min=0.0, max=0.7)


def conv_bn_relu(c_in, c_out, kernel, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=dilation * (kernel // 2),
                  dilation=dilation, bias=False),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
    )


class PooledHead(nn.Module):
    """Global mean + max pooling over time, dropout, linear."""

    def __init__(self, channels, n_classes, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(2 * channels, n_classes)

    def forward(self, h):
        return self.linear(self.dropout(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)))


# -- small_cnn -------------------------------------------------------------------


class SmallCNN(nn.Module):
    def __init__(self, n_channels, n_classes, width, depth, kernel, dropout):
        super().__init__()
        widths = [width * m for m in (1, 2, 2, 4, 4, 4, 4, 4)][:depth]
        layers, c_in = [], n_channels
        for i, c_out in enumerate(widths):
            layers.append(conv_bn_relu(c_in, c_out, kernel if i < 2 else max(3, kernel - 2),
                                       stride=2 if i < depth - 1 else 1))
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = PooledHead(c_in, n_classes, dropout)

    def forward(self, x):
        return self.head(self.features(x))


# -- inception_time ------------------------------------------------------------------


class InceptionModule(nn.Module):
    def __init__(self, c_in, filters, bottleneck, kernel):
        super().__init__()
        self.bottleneck = nn.Conv1d(c_in, bottleneck, 1, bias=False) if c_in > 1 else nn.Identity()
        c_b = bottleneck if c_in > 1 else c_in
        kernels = [max(3, (kernel // d) | 1) for d in (1, 2, 4)]
        self.branches = nn.ModuleList(nn.Conv1d(c_b, filters, k, padding=k // 2, bias=False) for k in kernels)
        self.pool_branch = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(c_in, filters, 1, bias=False))
        self.bn = nn.BatchNorm1d(4 * filters)

    def forward(self, x):
        b = self.bottleneck(x)
        out = torch.cat([branch(b) for branch in self.branches] + [self.pool_branch(x)], dim=1)
        return torch.relu(self.bn(out))


class InceptionTime(nn.Module):
    """One InceptionTime network (Fawaz et al. 2020) without the 5-model
    ensemble; residual connection every 3 modules. A strided stem first
    shortens long sequences so it stays affordable on CPU."""

    def __init__(self, n_channels, n_classes, filters, depth, kernel, bottleneck, stem_stride, dropout):
        super().__init__()
        self.stem = (conv_bn_relu(n_channels, 2 * filters, 5, stride=stem_stride)
                     if stem_stride > 1 else nn.Identity())
        c_in = 2 * filters if stem_stride > 1 else n_channels
        self.modules_ = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        res_in = c_in
        for i in range(depth):
            self.modules_.append(InceptionModule(c_in, filters, bottleneck, kernel))
            c_in = 4 * filters
            if i % 3 == 2:
                self.shortcuts.append(nn.Sequential(nn.Conv1d(res_in, c_in, 1, bias=False), nn.BatchNorm1d(c_in)))
                res_in = c_in
        self.head = PooledHead(c_in, n_classes, dropout)

    def forward(self, x):
        x = self.stem(x)
        residual = x
        for i, module in enumerate(self.modules_):
            x = module(x)
            if i % 3 == 2:
                x = torch.relu(x + self.shortcuts[i // 3](residual))
                residual = x
        return self.head(x)


# -- tcn -------------------------------------------------------------------------------


class TemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn_relu(c_in, c_out, kernel, dilation=dilation), nn.Dropout(dropout),
            conv_bn_relu(c_out, c_out, kernel, dilation=dilation), nn.Dropout(dropout),
        )
        self.shortcut = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return torch.relu(self.net(x) + self.shortcut(x))


class TCN(nn.Module):
    """Temporal convolutional network: stacked dilated residual blocks,
    dilation doubling per level, after a strided stem."""

    def __init__(self, n_channels, n_classes, channels, levels, kernel, stem_stride, dropout):
        super().__init__()
        self.stem = conv_bn_relu(n_channels, channels, 5, stride=stem_stride)
        self.blocks = nn.Sequential(*[TemporalBlock(channels, channels, kernel, 2 ** i, dropout) for i in range(levels)])
        self.head = PooledHead(channels, n_classes, dropout)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


# -- resnet1d ------------------------------------------------------------------------


class BasicBlock1d(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, 7, stride=stride, padding=3, bias=False), nn.BatchNorm1d(c_out), nn.ReLU(inplace=True),
            nn.Conv1d(c_out, c_out, 5, padding=2, bias=False), nn.BatchNorm1d(c_out),
        )
        self.shortcut = (nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out))
                         if stride != 1 or c_in != c_out else nn.Identity())

    def forward(self, x):
        return torch.relu(self.body(x) + self.shortcut(x))


class ResNet1D(nn.Module):
    def __init__(self, n_channels, n_classes, width, blocks_per_stage, dropout):
        super().__init__()
        self.stem = conv_bn_relu(n_channels, width, 7, stride=2)
        layers, c_in = [], width
        for stage, mult in enumerate((1, 2, 4)):
            for block in range(blocks_per_stage):
                c_out = width * mult
                layers.append(BasicBlock1d(c_in, c_out, stride=2 if block == 0 and stage > 0 else 1))
                c_in = c_out
        self.body = nn.Sequential(*layers)
        self.head = PooledHead(c_in, n_classes, dropout)

    def forward(self, x):
        return self.head(self.body(self.stem(x)))


# -- conv_mhsa ------------------------------------------------------------------------


class ConvMHSA(nn.Module):
    """A small version of the NGAFID-MC paper's Conv-MHSA: strided 1-D
    convolutions shorten the sequence to at most `max_tokens` positions,
    then a transformer encoder attends across the whole flight."""

    def __init__(self, n_channels, n_classes, seq_len, d_model, heads, layers, max_tokens, dropout):
        super().__init__()
        stem, c_in, length = [], n_channels, seq_len
        while length > max_tokens:
            stem.append(conv_bn_relu(c_in, d_model, 5, stride=2))
            c_in, length = d_model, (length + 1) // 2
        stem.append(nn.Conv1d(c_in, d_model, 1))
        self.stem = nn.Sequential(*stem)
        self.position = nn.Parameter(torch.zeros(1, length, d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2 * d_model, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, x):
        h = self.stem(x).transpose(1, 2)  # (B, L, D)
        h = self.encoder(h + self.position[:, : h.shape[1]])
        return self.linear(self.dropout(self.norm(h.mean(dim=1))))


# -- catalog --------------------------------------------------------------------------
# builder(n_channels, seq_len, n_classes, params) -> nn.Module


def _heads_divide(params):
    if params["d_model"] % params["heads"]:
        return f"d_model ({params['d_model']}) must be divisible by heads ({params['heads']})"
    return None


def _depth_multiple_of_3(params):
    return None if params["depth"] % 3 == 0 else f"depth must be a multiple of 3, got {params['depth']}"


ARCHITECTURES: dict[str, CatalogEntry] = {e.id: e for e in (
    CatalogEntry(
        "small_cnn", "architecture",
        "Plain strided 1-D CNN with global mean+max pooling. Fast and a strong baseline: the hand-written "
        "reference (width 32, depth 5, kernel 7, no schedule) reached ROC ~0.77 last-epoch / ~0.82 best-epoch "
        "on NGAFID C28 at pool 4.",
        {
            "width": ParamSpec("int", 32, "channels in the first layer (x2 then x4 deeper)", min=8, max=64),
            "depth": ParamSpec("int", 5, "conv layers; all but the last halve the sequence length", min=3, max=8),
            "kernel": ParamSpec("int", 7, "kernel size of the first two layers", min=3, max=15),
            "dropout": _dropout(),
        },
        builder=lambda c, t, k, p: SmallCNN(c, k, p["width"], p["depth"], p["kernel"], p["dropout"]),
    ),
    CatalogEntry(
        "inception_time", "architecture",
        "InceptionTime (single network, no 5x ensemble): multi-scale convolutions with residual links. "
        "Benchmarked in the NGAFID dataset paper at ~75% accuracy. Heavier than small_cnn; use stem_stride to "
        "shorten long inputs first.",
        {
            "filters": ParamSpec("int", 16, "filters per branch (module output = 4x this)", min=8, max=32),
            "depth": ParamSpec("int", 6, "inception modules (multiple of 3)", min=3, max=9),
            "kernel": ParamSpec("int", 21, "largest kernel; branches use k, k/2, k/4", min=9, max=41),
            "bottleneck": ParamSpec("int", 16, "1x1 bottleneck channels", min=8, max=32),
            "stem_stride": ParamSpec("choice", 4, "strided stem that shortens the sequence first", choices=(1, 2, 4, 8)),
            "dropout": _dropout(),
        },
        constraint=_depth_multiple_of_3,
        builder=lambda c, t, k, p: InceptionTime(c, k, p["filters"], p["depth"], p["kernel"], p["bottleneck"],
                                                 p["stem_stride"], p["dropout"]),
    ),
    CatalogEntry(
        "tcn", "architecture",
        "Temporal convolutional network: residual blocks of dilated convolutions (dilation doubles per level), "
        "so the receptive field grows exponentially with depth.",
        {
            "channels": ParamSpec("int", 48, "channels throughout", min=16, max=96),
            "levels": ParamSpec("int", 6, "residual levels (receptive field ~ kernel * 2^levels)", min=3, max=9),
            "kernel": ParamSpec("int", 5, "kernel size", min=3, max=9),
            "stem_stride": ParamSpec("choice", 4, "strided stem that shortens the sequence first", choices=(1, 2, 4, 8)),
            "dropout": ParamSpec("float", 0.2, "dropout inside blocks and before the head", min=0.0, max=0.6),
        },
        builder=lambda c, t, k, p: TCN(c, k, p["channels"], p["levels"], p["kernel"], p["stem_stride"], p["dropout"]),
    ),
    CatalogEntry(
        "resnet1d", "architecture",
        "1-D ResNet: three stages (width x1, x2, x4), stride 2 between stages, residual basic blocks.",
        {
            "width": ParamSpec("int", 32, "channels in the first stage", min=16, max=64),
            "blocks_per_stage": ParamSpec("int", 1, "residual blocks per stage", min=1, max=3),
            "dropout": _dropout(),
        },
        builder=lambda c, t, k, p: ResNet1D(c, k, p["width"], p["blocks_per_stage"], p["dropout"]),
    ),
    CatalogEntry(
        "conv_mhsa", "architecture",
        "Small convolutional transformer after the NGAFID-MC paper's Conv-MHSA (the paper's model: 7.9M params, "
        "4096 steps, TPU). Strided convs shorten the flight to <= max_tokens positions, then self-attention "
        "relates distant parts of the flight. The paper found it overfits without augmentation.",
        {
            "d_model": ParamSpec("int", 64, "embedding width", min=32, max=128),
            "heads": ParamSpec("int", 4, "attention heads (must divide d_model)", min=1, max=8),
            "layers": ParamSpec("int", 2, "transformer encoder layers", min=1, max=4),
            "max_tokens": ParamSpec("int", 128, "sequence positions left after the conv stem", min=32, max=512),
            "dropout": ParamSpec("float", 0.1, "dropout in attention/FF and before the head", min=0.0, max=0.5),
        },
        constraint=_heads_divide,
        builder=lambda c, t, k, p: ConvMHSA(c, k, t, p["d_model"], p["heads"], p["layers"], p["max_tokens"],
                                            p["dropout"]),
    ),
)}
