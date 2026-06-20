"""RDR module: Negative Binomial mixture model for read depth ratio data."""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..distributions import GammaDistribution
from .base import BaseVariationalModel


class RDRModel(BaseVariationalModel):
    """Negative Binomial (or Poisson) mixture model for RDR module.
    
    Models read depth ratio (RDR) from expression data using a
    mixture of Negative Binomial distributions, enabling inference of
    copy number states from gene expression levels.
    
    Parameters
    ----------
    n_features : int
        Number of features (genes)
    n_samples : int
        Number of samples (cells)
    n_components : int
        Number of mixture components (clones)
    n_cnv_states : int
        Number of CNV states (default: 3)
    learn_theta : bool
        Whether to learn theta (rate) parameters
    learn_pi : bool
        Whether to learn mixture proportions
    fsp_mode : bool
        Feature-specific parameter mode
    """
    
    def __init__(self, n_features: int, n_samples: int, n_components: int,
                 n_cnv_states: int = 3, learn_theta: bool = True,
                 learn_pi: bool = True, fsp_mode: bool = True,
                 device: Optional[str] = None):
        """Initialize RDR model.
        
        Parameters
        ----------
        n_features : int
            Number of genes
        n_samples : int
            Number of cells
        n_components : int
            Number of clones
        n_cnv_states : int
            Number of CNV states
        learn_theta : bool
            Whether to update theta parameters
        learn_pi : bool
            Whether to update mixture proportions
        fsp_mode : bool
            Feature-specific parameter mode
        device : str, optional
            Device to use
        """
        super().__init__(n_features, n_samples, n_components, device=device)
        
        self.n_cnv_states = n_cnv_states
        self.learn_theta = learn_theta
        self.learn_pi = learn_pi
        self.fsp_mode = fsp_mode
        
        # Parameters for Gamma distribution (shape-rate parameterization)
        # alpha_s1: shape parameter (concentration)
        # beta_s2: rate parameter
        alpha_len = n_features if fsp_mode else 1
        
        # Initialize with mean [0.5, 1.0, 1.5] for [loss, neutral, gain]
        mu_init = torch.tensor([0.5, 1.0, 1.5])
        var_init = 0.01
        alpha_init = (mu_init ** 2) / var_init
        beta_init = mu_init / var_init
        
        self.register_buffer(
            'alpha_s1',
            torch.ones(alpha_len, n_cnv_states) * alpha_init
        )
        self.register_buffer(
            'beta_s2',
            torch.ones(alpha_len, n_cnv_states) * beta_init
        )
        
        # Library size ratios (cell-specific normalization)
        self.register_buffer('library_ratio', torch.ones(n_samples, 1, 1))
        self.register_buffer('ref_counts', torch.ones(n_features, 1, 1))
        
        # Cell cluster assignments
        self.register_buffer(
            'id_prob',
            torch.ones(n_samples, n_components) / n_components
        )
        
        # CNV state probabilities
        self.register_buffer(
            'cnv_prob',
            torch.ones(n_features, n_components, n_cnv_states) / n_cnv_states
        )
        
        # Priors
        self._set_priors()
    
    def _set_priors(self) -> None:
        """Set prior distributions."""
        alpha_len = self.n_features if self.fsp_mode else 1
        mu_init = torch.tensor([0.5, 1.0, 1.5])
        var_init = 0.01
        alpha_init = (mu_init ** 2) / var_init
        beta_init = mu_init / var_init
        
        self.register_buffer(
            'alpha_s1_prior',
            torch.ones(alpha_len, self.n_cnv_states) * alpha_init
        )
        self.register_buffer(
            'beta_s2_prior',
            torch.ones(alpha_len, self.n_cnv_states) * beta_init
        )
        
        self.register_buffer(
            'id_prior',
            torch.ones(self.n_samples, self.n_components) / self.n_components
        )
        
        self.register_buffer(
            'cnv_prior',
            torch.ones(self.n_features, self.n_components, self.n_cnv_states) / self.n_cnv_states
        )
    
    def update_parameters(self, data: torch.Tensor) -> None:
        """Update variational parameters via coordinate ascent.
        
        Parameters
        ----------
        data : torch.Tensor
            Gene expression count data (n_features, n_samples)
        """
        r_counts = data
        
        # Update ID_prob
        if self.learn_pi:
            self._update_id_prob(r_counts)
        
        # Update CNV_prob
        self._update_cnv_prob(r_counts)
        
        # Update theta (Gamma parameters)
        if self.learn_theta:
            self._update_theta(r_counts)
    
    def _update_id_prob(self, r_counts: torch.Tensor) -> None:
        """Update cell-to-cluster assignment probabilities.
        
        Parameters
        ----------
        r_counts : torch.Tensor
            Count data (n_features, n_samples)
        """
        log_lik_id = torch.zeros(r_counts.shape[1], self.n_components, device=self.device, dtype=self.dtype)
        
        for t in range(self.n_cnv_states):
            gamma_exp1 = torch.digamma(self.alpha_s1[:, t:t+1]) - torch.log(self.beta_s2[:, t:t+1])
            gamma_exp2 = self.alpha_s1[:, t:t+1] / self.beta_s2[:, t:t+1]
            
            s1 = (r_counts.T * torch.log(self.library_ratio.squeeze(-1) + 1e-10)) @ (self.cnv_prob[:, :, t] * gamma_exp2)
            s2 = r_counts.T @ (self.cnv_prob[:, :, t] * gamma_exp1)
            s3 = (self.library_ratio * self.ref_counts).squeeze(-1) @ (self.cnv_prob[:, :, t] * gamma_exp2)
            
            log_lik_id += s1 + s2 - s3
        
        # Update with numerical stability
        log_id_prob = log_lik_id + torch.log(self.id_prior + 1e-10)
        log_id_prob = log_id_prob - torch.logsumexp(log_id_prob, dim=1, keepdim=True)
        self.id_prob.copy_(torch.exp(log_id_prob))
    
    def _update_cnv_prob(self, r_counts: torch.Tensor) -> None:
        """Update CNV state assignment probabilities.
        
        Parameters
        ----------
        r_counts : torch.Tensor
            Count data
        """
        log_lik_cnv = torch.zeros(
            self.n_features, self.n_components, self.n_cnv_states,
            device=self.device, dtype=self.dtype
        )
        
        for t in range(self.n_cnv_states):
            gamma_exp1 = torch.digamma(self.alpha_s1[:, t:t+1]) - torch.log(self.beta_s2[:, t:t+1])
            gamma_exp2 = self.alpha_s1[:, t:t+1] / self.beta_s2[:, t:t+1]
            
            s1 = r_counts @ self.id_prob * gamma_exp1
            s2 = (self.library_ratio * self.ref_counts) @ self.id_prob * gamma_exp2
            s3 = (r_counts * torch.log(self.library_ratio + 1e-10)).sum(dim=1, keepdim=True) @ self.id_prob
            
            log_lik_cnv[:, :, t] = s1 + s2 - s3
        
        # Update with numerical stability
        log_cnv_prob = log_lik_cnv + torch.log(self.cnv_prior + 1e-10)
        log_cnv_prob = log_cnv_prob - torch.logsumexp(log_cnv_prob, dim=2, keepdim=True)
        self.cnv_prob.copy_(torch.exp(log_cnv_prob))
    
    def _update_theta(self, r_counts: torch.Tensor) -> None:
        """Update Gamma posterior parameters.
        
        Parameters
        ----------
        r_counts : torch.Tensor
            Count data
        """
        s1_ik = r_counts @ self.id_prob
        s2_ik = (self.library_ratio * self.ref_counts) @ self.id_prob
        
        new_alpha_s1 = self.alpha_s1_prior.clone()
        new_beta_s2 = self.beta_s2_prior.clone()
        
        for t in range(self.n_cnv_states):
            new_alpha_s1[:, t:t+1] += s1_ik * self.cnv_prob[:, :, t]
            new_beta_s2[:, t:t+1] += s2_ik * self.cnv_prob[:, :, t]
        
        self.alpha_s1.copy_(new_alpha_s1)
        self.beta_s2.copy_(new_beta_s2)
    
    def compute_elbo(self, data: torch.Tensor) -> torch.Tensor:
        """Compute Evidence Lower Bound (ELBO).
        
        Parameters
        ----------
        data : torch.Tensor
            Count data
            
        Returns
        -------
        torch.Tensor
            ELBO value
        """
        r_counts = data
        
        # Likelihood term
        ll = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for t in range(self.n_cnv_states):
            gamma_exp1 = torch.digamma(self.alpha_s1[:, t:t+1]) - torch.log(self.beta_s2[:, t:t+1])
            gamma_exp2 = self.alpha_s1[:, t:t+1] / self.beta_s2[:, t:t+1]
            
            s1 = ((r_counts * (torch.log(self.library_ratio + 1e-10) + torch.log(self.ref_counts + 1e-10))).T 
                  @ (self.cnv_prob[:, :, t] * gamma_exp1)).sum()
            s2 = (r_counts.T @ (self.cnv_prob[:, :, t] * gamma_exp1)).sum()
            s3 = ((self.library_ratio * self.ref_counts).squeeze(-1).unsqueeze(0) * r_counts 
                  @ (self.cnv_prob[:, :, t] * gamma_exp2)).sum()
            
            ll = ll + s1 + s2 - s3
        
        # KL divergences
        kl_id = -torch.sum(
            self.id_prob * (torch.log(self.id_prob + 1e-10) - torch.log(self.id_prior + 1e-10))
        )
        
        kl_cnv = -torch.sum(
            self.cnv_prob * (torch.log(self.cnv_prob + 1e-10) - torch.log(self.cnv_prior + 1e-10))
        )
        
        elbo = ll + kl_id + kl_cnv
        return elbo
    
    def get_posterior_assignment(self) -> torch.Tensor:
        """Get posterior probability of each cell to each clone.
        
        Returns
        -------
        torch.Tensor
            (n_samples, n_components)
        """
        return self.id_prob.detach().clone()
    
    def get_posterior_cnv_prob(self) -> torch.Tensor:
        """Get posterior CNV state probabilities.
        
        Returns
        -------
        torch.Tensor
            (n_features, n_components, n_cnv_states)
        """
        return self.cnv_prob.detach().clone()
