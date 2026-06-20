"""Compatibility utilities for different PyTorch versions."""

import torch


def get_betaln():
    """Get betaln function, with fallback for older PyTorch versions.
    
    torch.special.betaln was added in PyTorch 1.13.
    For older versions, we implement it using loggamma.
    
    Returns
    -------
    callable
        betaln function
    """
    try:
        from torch.special import betaln
        return betaln
    except ImportError:
        # Fallback for PyTorch < 1.13
        def betaln(alpha, beta):
            """Compute log of Beta function: log(B(a,b)) = log(Gamma(a)) + log(Gamma(b)) - log(Gamma(a+b))."""
            return torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
        return betaln


def get_loggamma():
    """Get loggamma function, with compatibility for torch.lgamma.
    
    Returns
    -------
    callable
        loggamma function
    """
    try:
        from torch.special import loggamma
        return loggamma
    except ImportError:
        # Fallback to torch.lgamma
        return torch.lgamma


def get_digamma():
    """Get digamma function, with compatibility.
    
    Returns
    -------
    callable
        digamma function
    """
    try:
        from torch.special import digamma
        return digamma
    except ImportError:
        # Fallback to torch.digamma
        return torch.digamma


def get_gammaln():
    """Get gammaln function (log gamma).
    
    Returns
    -------
    callable
        gammaln function
    """
    try:
        from torch.special import gammaln
        return gammaln
    except ImportError:
        # Fallback to torch.lgamma
        return torch.lgamma


# Create module-level functions
betaln = get_betaln()
loggamma = get_loggamma()
digamma = get_digamma()
gammaln = get_gammaln()

__all__ = ['betaln', 'loggamma', 'digamma', 'gammaln']
