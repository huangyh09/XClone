"""Base distribution class for XClone models."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Distribution(ABC, nn.Module):
    """Abstract base class for probability distributions.
    
    All distributions should inherit from this class and implement
    the required methods for variational inference.
    """
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def log_prob(self, value):
        """Compute log probability of value under this distribution.
        
        Parameters
        ----------
        value : torch.Tensor
            Value at which to compute log probability
            
        Returns
        -------
        torch.Tensor
            Log probability
        """
        pass
    
    @abstractmethod
    def entropy(self):
        """Compute entropy of the distribution.
        
        Returns
        -------
        torch.Tensor
            Entropy
        """
        pass
    
    @abstractmethod
    def mean(self):
        """Compute mean of the distribution.
        
        Returns
        -------
        torch.Tensor
            Mean
        """
        pass
    
    @abstractmethod
    def sample(self, sample_shape=torch.Size()):
        """Draw samples from the distribution.
        
        Parameters
        ----------
        sample_shape : torch.Size
            Shape of samples to draw
            
        Returns
        -------
        torch.Tensor
            Samples
        """
        pass
    
    @abstractmethod
    def rsample(self, sample_shape=torch.Size()):
        """Draw reparameterized samples (for gradient computation).
        
        Parameters
        ----------
        sample_shape : torch.Size
            Shape of samples to draw
            
        Returns
        -------
        torch.Tensor
            Reparameterized samples
        """
        pass
