"""
src/utils/seed.py

Reproducible random state.

WHY THIS EXISTS
---------------
The inherited pipeline seeded only the Random Forest (`random_state=42`). The GNN, the
transformer, the stacking meta-learner and the ensemble were all unseeded, so every run
drew fresh weight initialisations, dropout masks and shuffle orders. Two consequences:

  1. The baseline's reported GNN / transformer numbers are a single, unreproducible draw.
     Re-running the same code gives different results, and nobody can tell how much of a
     reported difference is a real effect and how much is noise.
  2. Multi-seed evaluation -- running the same configuration under several seeds and
     reporting mean +/- confidence interval -- is impossible without controlling the seed.

`set_seed()` fixes Python's `random`, NumPy and PyTorch in one call. Each training script
calls it at the start of every dataset, so running one dataset on its own reproduces
exactly what running all of them produces (the seed does not drift with dataset order).

The seed can be overridden with the MPP_SEED environment variable, which is how the
multi-seed harness will sweep it:

    MPP_SEED=1 python -m src.train.train_gnn
"""

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 42


def get_seed(default=DEFAULT_SEED):
    """Read the seed from MPP_SEED, falling back to the default."""
    raw = os.environ.get("MPP_SEED")
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"MPP_SEED must be an integer, got {raw!r}")


def set_seed(seed=None, deterministic=True):
    """
    Seed Python, NumPy and PyTorch. Returns the seed actually used.

    `deterministic` additionally asks cuDNN for reproducible kernel choices. It costs a
    little speed and does nothing on CPU, but it matters once training moves to Colab.
    """
    if seed is None:
        seed = get_seed()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed
