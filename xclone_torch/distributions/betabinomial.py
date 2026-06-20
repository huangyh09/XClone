"""Beta-Binomial distribution for XClone BAF module."""

import torch
import torch.nn as nn
from .base import Distribution
from ..compat import betaln, loggamma


class BetaBinomialDistribution(Distribution):
    """Beta-Binomial distribution for modeling B-allele frequency data.
    
    The Beta-Binomial distribution is a compound distribution that models
    uncertainty in allele frequency. It is formed by integrating over
    a Beta distribution.
    
    Parameters
    ----------
    alpha : torch.Tensor
        Alpha parameter of Beta component
    beta : torch.Tensor
        Beta parameter of Beta component
    """
    
    def __init__(self, alpha, beta):
        """Initialize Beta-Binomial distribution.
        
        Parameters
        ----------
        alpha : torch.Tensor
            Alpha parameter of underlying Beta distribution
        beta : torch.Tensor
            Beta parameter of underlying Beta distribution
        """
        super().__init__()
        self.alpha = nn.Parameter(alpha.clamp(min=1e-6))
        self.beta = nn.Parameter(beta.clamp(min=1e-6))
    
    def log_prob(self, value, total_count):
        """Log probability of value under Beta-Binomial distribution.
        
        Parameters
        ----------
        value : torch.Tensor
            Number of successes (AD: alternative allele count)
        total_count : torch.Tensor
            Total number of trials (DP: total read depth)
            
        Returns
        -------
        torch.Tensor
            Log probability
        """
        # Beta-Binomial log pmf:
        # P(k|n,a,b) = C(n,k) * B(k+a, n-k+b) / B(a,b)
        # log P = log(C(n,k)) + log(B(k+a, n-k+b)) - log(B(a,b))
        
        k = value
        n = total_count
        
        # Binomial coefficient: log(C(n,k)) = log(n!) - log(k!) - log((n-k)!)
        log_binom_coeff = loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)
        
        # Beta function: log(B(a,b)) = log(Gamma(a)) + log(Gamma(b)) - log(Gamma(a+b))
        log_beta_ab = betaln(self.alpha, self.beta)
        log_beta_num = betaln(k + self.alpha, n - k + self.beta)
        
        return log_binom_coeff + log_beta_num - log_beta_ab
    
    def mean(self, total_count):
        """Mean of Beta-Binomial distribution.
        
        E[X] = n * alpha / (alpha + beta)
        
        Parameters
        ----------
        total_count : torch.Tensor
            Total number of trials
            
        Returns
        -------
        torch.Tensor
            Mean
        """
        return total_count * self.alpha / (self.alpha + self.beta)
    
    def variance(self, total_count):
        """Variance of Beta-Binomial distribution.
        
        Parameters
        ----------
        total_count : torch.Tensor
            Total number of trials
            
        Returns
        -------
        torch.Tensor
            Variance
        """
        p = self.alpha / (self.alpha + self.beta)
        n = total_count
        ab = self.alpha + self.beta
        return n * p * (1 - p) * (n + ab) / (ab + 1)
    
    def entropy(self):
        """Entropy of Beta-Binomial distribution (not typically used).
        
        Returns
        -------
        torch.Tensor
            Entropy (placeholder)
        """
        return torch.tensor(0.0, device=self.alpha.device)
    
    def sample(self, sample_shape=torch.Size()):
        """Not implemented for Beta-Binomial."""
        raise NotImplementedError("Sampling from Beta-Binomial requires both k and n parameters")
    
    def rsample(self, sample_shape=torch.Size()):
        """Not implemented for Beta-Binomial."""
        raise NotImplementedError("Sampling from Beta-Binomial requires both k and n parameters")
