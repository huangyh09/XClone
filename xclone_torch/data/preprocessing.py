"""Data preprocessing utilities for XClone-Torch."""

import torch
import numpy as np
from typing import Tuple, Optional


def prepare_baf_data(AD: np.ndarray, DP: np.ndarray,
                    device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare B-allele frequency data for BAF module.
    
    Parameters
    ----------
    AD : np.ndarray
        Alternative depth (allele counts) - shape (n_features, n_samples)
    DP : np.ndarray
        Total depth - shape (n_features, n_samples)
    device : str
        Device to place tensors on
        
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (AD_tensor, DP_tensor) on specified device
    """
    # Convert to torch tensors
    AD_tensor = torch.from_numpy(AD).float().to(device=device)
    DP_tensor = torch.from_numpy(DP).float().to(device=device)
    
    # Ensure non-negative
    AD_tensor = torch.clamp(AD_tensor, min=0)
    DP_tensor = torch.clamp(DP_tensor, min=0)
    
    # Ensure DP >= AD
    DP_tensor = torch.maximum(DP_tensor, AD_tensor)
    
    return AD_tensor, DP_tensor


def prepare_rdr_data(counts: np.ndarray, library_sizes: Optional[np.ndarray] = None,
                    ref_counts: Optional[np.ndarray] = None,
                    device: str = 'cpu',
                    log_transform: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare read depth ratio data for RDR module.
    
    Parameters
    ----------
    counts : np.ndarray
        Gene expression count matrix - shape (n_features, n_samples)
    library_sizes : np.ndarray, optional
        Library size for each cell. If None, computed from counts.
    ref_counts : np.ndarray, optional
        Reference counts for each gene. If None, uses mean.
    device : str
        Device to place tensors on
    log_transform : bool
        Whether to log-transform counts
        
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (counts_tensor, library_ratio, ref_counts_tensor)
    """
    counts_tensor = torch.from_numpy(counts).float().to(device=device)
    counts_tensor = torch.clamp(counts_tensor, min=0)
    
    # Compute library sizes
    if library_sizes is None:
        library_sizes = counts.sum(axis=0)
    library_sizes = torch.from_numpy(library_sizes).float().to(device=device)
    library_ratio = library_sizes / library_sizes.mean()
    library_ratio = library_ratio.unsqueeze(-1).unsqueeze(-1)  # (n_samples, 1, 1)
    
    # Compute reference counts
    if ref_counts is None:
        ref_counts = counts.mean(axis=1)
    ref_counts_tensor = torch.from_numpy(ref_counts).float().to(device=device)
    ref_counts_tensor = ref_counts_tensor.unsqueeze(-1).unsqueeze(-1)  # (n_features, 1, 1)
    
    if log_transform:
        counts_tensor = torch.log(counts_tensor + 1)
    
    return counts_tensor, library_ratio, ref_counts_tensor
