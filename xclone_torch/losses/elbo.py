"""Evidence Lower Bound (ELBO) computations."""

import torch


def compute_elbo_baf(log_likelihood: torch.Tensor,
                     kl_divergences: dict) -> torch.Tensor:
    """Compute ELBO for BAF module.
    
    ELBO = E_q[log p(x|z)] - KL(q||p)
    
    Parameters
    ----------
    log_likelihood : torch.Tensor
        Expected log-likelihood term
    kl_divergences : dict
        Dictionary containing KL divergence terms:
        - 'id': KL(q(z)||p(z))
        - 'cnv': KL(q(theta)||p(theta))
        - 'params': KL divergences for distribution parameters
        
    Returns
    -------
    torch.Tensor
        ELBO value
    """
    kl_total = sum(kl_divergences.values())
    elbo = log_likelihood - kl_total
    return elbo


def compute_elbo_rdr(log_likelihood: torch.Tensor,
                     kl_divergences: dict) -> torch.Tensor:
    """Compute ELBO for RDR module.
    
    Parameters
    ----------
    log_likelihood : torch.Tensor
        Expected log-likelihood term
    kl_divergences : dict
        Dictionary containing KL divergence terms
        
    Returns
    -------
    torch.Tensor
        ELBO value
    """
    kl_total = sum(kl_divergences.values())
    elbo = log_likelihood - kl_total
    return elbo
