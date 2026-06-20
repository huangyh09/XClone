"""Gamma distribution for RDR module variational inference."""

import torch
import torch.nn as nn
from torch.distributions import Gamma, constraints
from .base import Distribution


class GammaDistribution(Distribution):
    """Gamma distribution parameterized by concentration (alpha) and rate (beta).
    
    Parameters
    ----------
    concentration : torch.Tensor
        Alpha (shape) parameter of Gamma distribution
    rate : torch.Tensor
        Beta (rate) parameter of Gamma distribution
    """
    
    def __init__(self, concentration, rate, validate_args=False):
        """Initialize Gamma distribution.
        
        Parameters
        ----------
        concentration : torch.Tensor
            Alpha (concentration) parameter - shape parameter
        rate : torch.Tensor
            Beta (rate) parameter - rate parameter (inverse scale)
        validate_args : bool
            Whether to validate arguments
        """
        super().__init__()
        self.concentration = nn.Parameter(concentration.clamp(min=1e-6))
        self.rate = nn.Parameter(rate.clamp(min=1e-6))
        self._dist = Gamma(self.concentration, self.rate, validate_args=validate_args)
    
    @property
    def mean(self):
        """Mean of Gamma distribution: alpha / beta."""
        return self.concentration / self.rate
    
    @property
    def variance(self):
        """Variance of Gamma distribution: alpha / beta^2."""
        return self.concentration / (self.rate ** 2)
    
    def log_prob(self, value):
        """Log probability of value under Gamma distribution.
        
        Parameters
        ----------
        value : torch.Tensor
            Positive value
            
        Returns
        -------
        torch.Tensor
            Log probability
        """
        return self._dist.log_prob(value)
    
    def entropy(self):
        """Entropy of Gamma distribution.
        
        Returns
        -------
        torch.Tensor
            Entropy
        """
        return self._dist.entropy()
    
    def sample(self, sample_shape=torch.Size()):
        """Draw samples from Gamma distribution.
        
        Parameters
        ----------
        sample_shape : torch.Size
            Shape of samples
            
        Returns
        -------
        torch.Tensor
            Samples
        """
        return self._dist.sample(sample_shape)
    
    def rsample(self, sample_shape=torch.Size()):
        """Draw reparameterized samples from Gamma distribution.
        
        Parameters
        ----------
        sample_shape : torch.Size
            Shape of samples
            
        Returns
        -------
        torch.Tensor
            Reparameterized samples
        """
        return self._dist.rsample(sample_shape)
    
    def log_alpha(self):
        """Log of alpha parameter (concentration).
        
        Returns
        -------
        torch.Tensor
            log(alpha)
        """
        return torch.log(self.concentration)
    
    def log_beta(self):
        """Log of beta parameter (rate).
        
        Returns
        -------
        torch.Tensor
            log(beta)
        """
        return torch.log(self.rate)
    
    def digamma_alpha(self):
        """Digamma of alpha parameter.
        
        Returns
        -------
        torch.Tensor
            digamma(alpha)
        """
        return torch.digamma(self.concentration)
    
    def expected_log_alpha(self):
        """Expected value of log(X) where X ~ Gamma(alpha, beta).
        
        Returns log(alpha) - log(beta).
        
        Returns
        -------
        torch.Tensor
            E[log(X)]
        """
        return torch.digamma(self.concentration) - torch.log(self.rate)
