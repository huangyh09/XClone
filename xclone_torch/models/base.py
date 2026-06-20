"""Base class for variational inference models."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict


class BaseVariationalModel(ABC, nn.Module):
    """Abstract base class for variational inference models.
    
    Implements common functionality for coordinate ascent variational
    inference (CAVI) and ELBO computation.
    """
    
    def __init__(self, n_features: int, n_samples: int, n_components: int,
                 device: Optional[str] = None, dtype: torch.dtype = torch.float32):
        """Initialize base variational model.
        
        Parameters
        ----------
        n_features : int
            Number of features (genes/blocks)
        n_samples : int
            Number of samples (cells)
        n_components : int
            Number of mixture components (clones)
        device : str, optional
            Device to use ('cpu' or 'cuda'). Auto-detects if None.
        dtype : torch.dtype
            Floating point dtype for computations
        """
        super().__init__()
        
        self.n_features = n_features
        self.n_samples = n_samples
        self.n_components = n_components
        self.dtype = dtype
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.to(self.device)
        self.to(dtype)
        
        # Initialize ELBO tracking
        self.elbo_history = []
        self.n_iterations = 0
    
    @abstractmethod
    def update_parameters(self, data: torch.Tensor) -> None:
        """Update variational parameters via coordinate ascent.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data
        """
        pass
    
    @abstractmethod
    def compute_elbo(self, data: torch.Tensor) -> torch.Tensor:
        """Compute Evidence Lower Bound (ELBO).
        
        Parameters
        ----------
        data : torch.Tensor
            Input data
            
        Returns
        -------
        torch.Tensor
            ELBO value
        """
        pass
    
    def fit(self, data: torch.Tensor, max_iter: int = 200, min_iter: int = 5,
            epsilon_conv: float = 1e-2, verbose: bool = True) -> Dict:
        """Fit model using coordinate ascent variational inference.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data
        max_iter : int
            Maximum number of iterations
        min_iter : int
            Minimum number of iterations
        epsilon_conv : float
            Convergence threshold for ELBO
        verbose : bool
            Whether to print progress
            
        Returns
        -------
        dict
            Training history including ELBO values
        """
        data = data.to(self.device).to(self.dtype)
        
        elbo_values = []
        
        for iteration in range(max_iter):
            # Update variational parameters
            self.update_parameters(data)
            
            # Compute ELBO
            elbo = self.compute_elbo(data)
            elbo_values.append(elbo.item())
            self.elbo_history.append(elbo.item())
            
            if verbose and (iteration % max(1, max_iter // 10) == 0):
                print(f"[Iteration {iteration}/{max_iter}] ELBO: {elbo.item():.4f}")
            
            # Check convergence
            if iteration > min_iter:
                if elbo_values[-1] < elbo_values[-2]:
                    if verbose:
                        print(f"Warning: ELBO decreased at iteration {iteration}")
                elif iteration == max_iter - 1:
                    if verbose:
                        print(f"Warning: Model did not converge after {max_iter} iterations")
                elif abs(elbo_values[-1] - elbo_values[-2]) < epsilon_conv:
                    if verbose:
                        print(f"Converged at iteration {iteration}")
                    break
            
            self.n_iterations += 1
        
        return {
            'elbo': elbo_values,
            'n_iterations': self.n_iterations,
            'converged': abs(elbo_values[-1] - elbo_values[-2]) < epsilon_conv if len(elbo_values) > 1 else False
        }
    
    def get_posterior_assignment(self) -> torch.Tensor:
        """Get posterior probability of each sample to each component.
        
        Returns
        -------
        torch.Tensor
            Assignment probabilities (n_samples, n_components)
        """
        raise NotImplementedError("Subclasses must implement get_posterior_assignment")
    
    def get_posterior_cnv_prob(self) -> torch.Tensor:
        """Get posterior CNV state probabilities.
        
        Returns
        -------
        torch.Tensor
            CNV state probabilities (n_features, n_components, n_states)
        """
        raise NotImplementedError("Subclasses must implement get_posterior_cnv_prob")
