"""The deterministic harness: everything a plugin is NOT allowed to own —
fold assignment, normalization statistics, the training/eval loop, and
metric computation. Plugins (agent-written or hand-written) only supply a
model, an augmentation function, and optionally an optimizer."""
