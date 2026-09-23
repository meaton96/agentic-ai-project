import numpy as np
import pandas as pd
import pytest


def make_long_frame(n_seq=40, n_groups=8, min_len=40, max_len=120, seed=0, signal=1.5):
    """Long-format frame: channels a, b, c; label shifts channel a's level so
    the task is learnable; a `leak` column copies the label (to be excluded)."""
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(n_seq):
        length = int(rng.integers(min_len, max_len + 1))
        label = i % 2
        group = (i // 2) % n_groups  # both labels in every group
        t = np.arange(length)
        parts.append(pd.DataFrame({
            "a": rng.normal(label * signal, 1.0, length),
            "b": np.sin(t / 5.0) + rng.normal(0, 0.1, length),
            "c": rng.normal(0, 1.0, length),
            "seq": i,
            "unit": group,
            "fold": group % 4,
            "target": "bad" if label else "ok",
            "leak": label,
        }))
    df = pd.concat(parts, ignore_index=True)
    df.loc[df.sample(frac=0.02, random_state=seed).index, "c"] = np.nan
    return df


@pytest.fixture
def long_csv(tmp_path):
    path = tmp_path / "long.csv"
    make_long_frame().to_csv(path, index=False)
    return path
