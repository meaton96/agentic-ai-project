"""
Hand-written reference plugin: a small 1D CNN plus the three temporal
augmentations from Yang, LaBella & Desell (2021), "Predictive Maintenance
for General Aviation Using Convolutional Transformers" (arXiv 2110.03757).

It serves two roles: the human baseline agent-written plugins are compared
against, and the known-good fixture the harness's contract tests (and the
reviewer agent's planted-bug evaluation) are built from.

Augmentations, as the paper specifies them (segment lengths are given for
4096-step flights, so they are expressed as fractions of the sequence length
here and scale with pooling):
  cutout  — random segment of 64-512 steps, each channel picked with p=0.3,
            set to 0.
  cutmix  — a random segment of that length from another random flight in
            the batch (any label) replaces the same-length segment, on
            channels picked with p=0.3.
  mixup   — channels picked with p=0.4 become m*x + (1-m)*x_other over all
            timesteps, m ~ U(0.6, 0.9).
Each is applied to a flight with probability 0.4. Labels are left unchanged:
the paper mixes signal, not targets.
"""
from __future__ import annotations

import torch
from torch import nn

PAPER_AUGMENTATIONS = ("cutout", "cutmix", "mixup")


class SmallCNN(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, width: int = 32, dropout: float = 0.3):
        super().__init__()

        def block(c_in, c_out, kernel, stride):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=kernel // 2, bias=False),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
            )

        w = width
        self.features = nn.Sequential(
            block(n_channels, w, 7, 2),
            block(w, 2 * w, 7, 2),
            block(2 * w, 2 * w, 5, 2),
            block(2 * w, 4 * w, 5, 2),
            block(4 * w, 4 * w, 3, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(8 * w, n_classes)

    def forward(self, x):
        h = self.features(x)
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(self.dropout(pooled))


def build_model(n_channels, seq_len, n_classes, config):
    return SmallCNN(n_channels, n_classes,
                    width=config.get("width", 32), dropout=config.get("dropout", 0.3))


def augment(x, y, generator, config):
    enabled = config.get("augmentations", PAPER_AUGMENTATIONS)
    if not enabled:
        return x, y
    unknown = set(enabled) - set(PAPER_AUGMENTATIONS)
    if unknown:
        raise ValueError(f"unknown augmentations: {sorted(unknown)}")

    p_apply = config.get("aug_p", 0.4)
    batch, n_channels, t = x.shape
    seg_min = max(1, round(t * config.get("seg_min_frac", 64 / 4096)))
    seg_max = min(t, max(seg_min, round(t * config.get("seg_max_frac", 512 / 4096))))

    def uniform():
        return torch.rand((), generator=generator).item()

    def randint(lo, hi):  # inclusive
        return int(torch.randint(lo, hi + 1, (), generator=generator).item())

    def pick_channels(p):
        return torch.rand(n_channels, generator=generator) < p

    source = x.clone()
    out = x
    for i in range(batch):
        if "cutout" in enabled and uniform() < p_apply:
            length = randint(seg_min, seg_max)
            start = randint(0, t - length)
            channels = pick_channels(0.3)
            out[i, channels, start:start + length] = 0.0
        if "cutmix" in enabled and uniform() < p_apply:
            other = randint(0, batch - 1)
            length = randint(seg_min, seg_max)
            dst, src = randint(0, t - length), randint(0, t - length)
            channels = pick_channels(0.3)
            out[i, channels, dst:dst + length] = source[other, channels, src:src + length]
        if "mixup" in enabled and uniform() < p_apply:
            other = randint(0, batch - 1)
            m = 0.6 + 0.3 * uniform()
            channels = pick_channels(0.4)
            out[i, channels, :] = m * out[i, channels, :] + (1 - m) * source[other, channels, :]
    return out, y
