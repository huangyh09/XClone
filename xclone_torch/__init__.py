"""XClone-Torch: PyTorch-based reimplementation of XClone for efficient GPU-accelerated CNV inference.

A modern deep learning framework for detecting allele-specific subclonal copy number
alterations (CNAs) from single-cell RNA-seq data.

Modules:
    - models: Variational inference models (BAF, RDR, Combined)
    - losses: ELBO and KL divergence computations
    - distributions: Probability distributions (Beta, Gamma, BetaBinomial)
    - utils: Data handling and utility functions
    - data: Data preprocessing and loading utilities
"""

import torch

__version__ = "0.1.0"
__author__ = "XClone PyTorch Contributors"

# Check CUDA availability
if torch.cuda.is_available():
    print(f"[XClone-Torch] CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[XClone-Torch] Running on CPU. For large datasets, GPU acceleration is recommended.")

from . import models
from . import losses
from . import distributions
from . import utils
from . import data

__all__ = [
    'models',
    'losses', 
    'distributions',
    'utils',
    'data',
    '__version__',
]
