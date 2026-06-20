"""Tensor utility functions with numerical stability."""

import torch


def logsumexp_stable(input_tensor: torch.Tensor, dim: int = -1,
                     keepdim: bool = False) -> torch.Tensor:
    """Numerically stable logsumexp.
    
    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor
    dim : int
        Dimension along which to compute logsumexp
    keepdim : bool
        Keep dimension
        
    Returns
    -------
    torch.Tensor
        Stable logsumexp
    """
    return torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim)


def log_softmax_stable(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable log-softmax.
    
    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor
    dim : int
        Dimension along which to apply log-softmax
        
    Returns
    -------
    torch.Tensor
        Log-softmax applied
    """
    return torch.log_softmax(input_tensor, dim=dim)


def softmax_stable(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax.
    
    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor
    dim : int
        Dimension along which to apply softmax
        
    Returns
    -------
    torch.Tensor
        Softmax applied
    """
    return torch.softmax(input_tensor, dim=dim)
