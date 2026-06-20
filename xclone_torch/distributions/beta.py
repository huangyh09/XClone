"""Beta distribution for BAF module variational inference."""

import torch
import torch.nn as nn
from torch.distributions import Beta, constraints
from .base import Distribution


class BetaDistribution(Distribution):
    """Beta distribution parameterized by mean and concentration.
    
    Parameters
    ----------
    concentration0 : torch.Tensor
        Alpha parameter of Beta distribution (shape parameter 1)
    concentration1 : torch.Tensor
        Beta parameter of Beta distribution (shape parameter 2)
    """
    
    def __init__(self, concentration0, concentration1, validate_args=False):
        """Initialize Beta distribution.
        
        Parameters
        ----------
        concentration0 : torch.Tensor
            Alpha (concentration0) parameter
        concentration1 : torch.Tensor
            Beta (concentration1) parameter
        validate_args : bool
            Whether to validate arguments
        """
        super().__init__()
        self.concentration0 = nn.Parameter(concentration0.clamp(min=1e-6))
        self.concentration1 = nn.Parameter(concentration1.clamp(min=1e-6))
        self._dist = Beta(self.concentration0, self.concentration1, validate_args=validate_args)
    
    @property
    def mean(self):
        """Mean of Beta distribution: alpha / (alpha + beta)."""
        return self.concentration0 / (self.concentration0 + self.concentration1)
    
    @property
    def variance(self):
        """Variance of Beta distribution."""
        return self._dist.variance
    
    def log_prob(self, value):
        """Log probability of value under Beta distribution.
        
        Parameters
        ----------
        value : torch.Tensor
            Value in [0, 1]
            
        Returns
        -------
        torch.Tensor
            Log probability
        """
        return self._dist.log_prob(value)
    
    def entropy(self):
        """Entropy of Beta distribution.
        
        Returns
        -------
        torch.Tensor
            Entropy
        """
        return self._dist.entropy()
    
    def sample(self, sample_shape=torch.Size()):
        """Draw samples from Beta distribution.
        
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
        """Draw reparameterized samples from Beta distribution.
        
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
        """Log of alpha parameter (used in variational inference).
        
        Returns
        -------
        torch.Tensor
            log(alpha)
        """
        return torch.log(self.concentration0)
    
    def log_beta(self):
        """Log of beta parameter (used in variational inference).
        
        Returns
        -------
        torch.Tensor
            log(beta)
        """
        return torch.log(self.concentration1)
    
    def digamma_sum(self):
        """Digamma of (alpha + beta) (used in coordinate ascent).
        
        Returns
        -------
        torch.Tensor
            digamma(alpha + beta)
        """
        return torch.digamma(self.concentration0 + self.concentration1)
    
    def digamma_alpha(self):
        """Digamma of alpha parameter.
        
        Returns
        -------
        torch.Tensor
            digamma(alpha)
        """
        return torch.digamma(self.concentration0)
    
    def digamma_beta(self):
        """Digamma of beta parameter.
        
        Returns
        -------
        torch.Tensor
            digamma(beta)
        """
        return torch.digamma(self.concentration1)
