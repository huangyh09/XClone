"""Data utility functions for XClone-Torch."""

import torch
import numpy as np
from typing import Union, Optional


def normalize(data: torch.Tensor, axis: int = -1, eps: float = 1e-10) -> torch.Tensor:
    """Normalize tensor along specified axis.
    
    Parameters
    ----------
    data : torch.Tensor
        Input tensor
    axis : int
        Axis along which to normalize
    eps : float
        Small epsilon for numerical stability
        
    Returns
    -------
    torch.Tensor
        Normalized tensor
    """
    sum_data = data.sum(dim=axis, keepdim=True)
    return data / (sum_data + eps)


def log_normalize(data: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Apply log-normalization to tensor.
    
    Parameters
    ----------
    data : torch.Tensor
        Input tensor (should be positive)
    axis : int
        Axis along which to normalize
        
    Returns
    -------
    torch.Tensor
        Log-normalized tensor
    """
    log_data = torch.log(data + 1e-10)
    return log_data - torch.logsumexp(log_data, dim=axis, keepdim=True)


def sparse_to_dense(sparse_matrix) -> torch.Tensor:
    """Convert sparse matrix to dense torch tensor.
    
    Parameters
    ----------
    sparse_matrix
        Scipy sparse matrix or PyTorch sparse tensor
        
    Returns
    -------
    torch.Tensor
        Dense tensor
    """
    if hasattr(sparse_matrix, 'toarray'):
        # Scipy sparse matrix
        return torch.from_numpy(sparse_matrix.toarray())
    elif isinstance(sparse_matrix, torch.sparse.FloatTensor):
        # PyTorch sparse tensor
        return sparse_matrix.to_dense()
    else:
        return sparse_matrix


def ensure_tensor(data: Union[torch.Tensor, np.ndarray, list],
                  device: Optional[torch.device] = None,
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Ensure input is a torch tensor.
    
    Parameters
    ----------
    data : Union[torch.Tensor, np.ndarray, list]
        Input data
    device : torch.device, optional
        Device to place tensor on
    dtype : torch.dtype
        Data type
        
    Returns
    -------
    torch.Tensor
        Tensor on specified device with specified dtype
    """
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        tensor = torch.tensor(data)
    
    tensor = tensor.to(dtype=dtype)
    
    if device is not None:
        tensor = tensor.to(device=device)
    
    return tensor
