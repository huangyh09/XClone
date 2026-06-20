"""Utility functions for XClone-Torch."""

from .data_utils import (
    normalize,
    log_normalize,
    sparse_to_dense,
    ensure_tensor,
)
from .tensor_utils import (
    logsumexp_stable,
    log_softmax_stable,
    softmax_stable,
)

__all__ = [
    'normalize',
    'log_normalize',
    'sparse_to_dense',
    'ensure_tensor',
    'logsumexp_stable',
    'log_softmax_stable',
    'softmax_stable',
]
