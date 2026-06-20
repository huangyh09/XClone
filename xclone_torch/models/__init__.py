"""Variational inference models for XClone.

Models include:
    - BAFModel: Beta-Binomial mixture model for B-allele frequency
    - RDRModel: Negative Binomial mixture model for read depth ratio
    - CombinedModel: Joint BAF and RDR inference
"""

from .base import BaseVariationalModel
from .baf import BAFModel
from .rdr import RDRModel

__all__ = [
    'BaseVariationalModel',
    'BAFModel',
    'RDRModel',
]
