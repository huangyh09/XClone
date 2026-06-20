"""KL divergence computations."""

import torch


def kl_beta(q_alpha: torch.Tensor, q_beta: torch.Tensor,
           p_alpha: torch.Tensor, p_beta: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two Beta distributions.
    
    KL(Beta(q_a, q_b) || Beta(p_a, p_b))
    
    Parameters
    ----------
    q_alpha : torch.Tensor
        Alpha parameter of variational distribution
    q_beta : torch.Tensor
        Beta parameter of variational distribution
    p_alpha : torch.Tensor
        Alpha parameter of prior distribution
    p_beta : torch.Tensor
        Beta parameter of prior distribution
        
    Returns
    -------
    torch.Tensor
        KL divergence
    """
    from torch.special import betaln, digamma
    
    kl = (betaln(p_alpha, p_beta) - betaln(q_alpha, q_beta) +
          (q_alpha - p_alpha) * digamma(q_alpha) +
          (q_beta - p_beta) * digamma(q_beta) +
          (p_alpha + p_beta - q_alpha - q_beta) * digamma(q_alpha + q_beta))
    
    return kl.sum()


def kl_gamma(q_alpha: torch.Tensor, q_beta: torch.Tensor,
            p_alpha: torch.Tensor, p_beta: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two Gamma distributions.
    
    KL(Gamma(q_a, q_b) || Gamma(p_a, p_b))
    
    Parameters
    ----------
    q_alpha : torch.Tensor
        Alpha (shape) parameter of variational distribution
    q_beta : torch.Tensor
        Beta (rate) parameter of variational distribution
    p_alpha : torch.Tensor
        Alpha parameter of prior distribution
    p_beta : torch.Tensor
        Beta parameter of prior distribution
        
    Returns
    -------
    torch.Tensor
        KL divergence
    """
    from torch.special import gammaln, digamma
    
    kl = (p_alpha * torch.log(q_beta / p_beta) +
          gammaln(q_alpha) - gammaln(p_alpha) +
          (p_alpha - q_alpha) * digamma(q_alpha) +
          q_alpha * (p_beta - q_beta) / q_beta)
    
    return kl.sum()


def kl_categorical(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two Categorical distributions.
    
    KL(q || p) where q and p are categorical probability distributions
    
    Parameters
    ----------
    q : torch.Tensor
        Probabilities of variational distribution
    p : torch.Tensor
        Probabilities of prior distribution (or reference)
        
    Returns
    -------
    torch.Tensor
        KL divergence
    """
    eps = 1e-10
    q_safe = torch.clamp(q, min=eps)
    p_safe = torch.clamp(p, min=eps)
    
    kl = (q_safe * (torch.log(q_safe) - torch.log(p_safe))).sum()
    return kl
