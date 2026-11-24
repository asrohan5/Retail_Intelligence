"""Tiny, dependency-light transforms. Kept separate so the API can unpickle the model
without importing the training/plotting stack."""

import numpy as np


def signed_log1p(X):
    """log1p that preserves sign, so (rare) negative values do not produce NaN."""
    return np.sign(X) * np.log1p(np.abs(X))
