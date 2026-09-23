"""Per-channel min/max scaling, fit on the training fold only.

The NGAFID-MC paper fit min/max over all of the data, which lets validation
flights shape the scaling. Here the statistics come from the training
indices alone, so a validation fold never influences its own inputs."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class ChannelMinMax:
    minimum: np.ndarray  # (C,)
    maximum: np.ndarray  # (C,)

    @classmethod
    def fit(cls, X: np.ndarray, batch: int = 512) -> "ChannelMinMax":
        c = X.shape[1]
        lo = np.full(c, np.inf, dtype=np.float64)
        hi = np.full(c, -np.inf, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN channels
            for start in range(0, X.shape[0], batch):
                block = X[start:start + batch]
                lo = np.fmin(lo, np.nanmin(block, axis=(0, 2)))
                hi = np.fmax(hi, np.nanmax(block, axis=(0, 2)))
        # A channel that was never observed in training scales to all zeros.
        lo = np.where(np.isfinite(lo), lo, 0.0)
        hi = np.where(np.isfinite(hi), hi, 0.0)
        return cls(minimum=lo.astype(np.float32), maximum=hi.astype(np.float32))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scales to roughly [0, 1] (validation values may fall outside) and
        replaces NaN (padding, sensor dropout) with 0."""
        span = self.maximum - self.minimum
        span = np.where(span > 0, span, 1.0).astype(np.float32)
        out = (np.asarray(X, dtype=np.float32) - self.minimum[None, :, None]) / span[None, :, None]
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
