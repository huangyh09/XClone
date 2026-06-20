"""BAF module: Beta-Binomial mixture model for B-allele frequency data."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ..distributions import BetaDistribution
from .base import BaseVariationalModel


class BAFModel(BaseVariationalModel):
    """Beta-Binomial mixture model for BAF module.
    
    Models B-allele frequency (BAF) from phased SNP data using a
    mixture of Beta-Binomial distributions, enabling inference of
    allele-specific copy number states.
    
    Parameters
    ----------
    n_features : int
        Number of features (genes/blocks with SNPs)
    n_samples : int
        Number of samples (cells)
    n_components : int
        Number of mixture components (clones)
    n_cnv_states : int
        Number of CNV states (default: 3 or 5)
    learn_theta : bool
        Whether to learn theta parameters
    learn_pi : bool
        Whether to learn mixture proportions
    fsp_mode : bool
        Feature-specific parameter mode
    """
    
    def __init__(self, n_features: int, n_samples: int, n_components: int,
                 n_cnv_states: int = 3, learn_theta: bool = True,
                 learn_pi: bool = True, fsp_mode: bool = True,
                 device: Optional[str] = None):
        """Initialize BAF model.
        
        Parameters
        ----------
        n_features : int
            Number of features
        n_samples : int
            Number of cells
        n_components : int
            Number of clones
        n_cnv_states : int
            Number of CNV states (3 or 5)
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
        
        # Theta length depends on FSP mode
        theta_len = n_features if fsp_mode else 1
        
        # Initialize theta (Beta distribution parameters)
        # theta_mu: mean of theta ~ Beta distribution
        # theta_concentration: concentration of theta ~ Beta distribution
        theta_mu_init = torch.linspace(0.01, 0.99, n_cnv_states).reshape(1, -1).expand(theta_len, -1).clone()
        theta_conc_init = torch.ones(theta_len, n_cnv_states) * 50.0
        
        self.theta_mu = nn.Parameter(theta_mu_init)
        self.theta_concentration = nn.Parameter(theta_conc_init)
        
        # Mixture proportions (cell cluster assignments)
        # ID_prob: P(cluster_k | cell_j)
        self.register_buffer(
            'id_prob',
            torch.ones(n_samples, n_components) / n_components
        )
        
        # CNV state probabilities
        # CNV_prob: P(CNV_state_t | feature_i, cluster_k)
        self.register_buffer(
            'cnv_prob',
            torch.ones(n_features, n_components, n_cnv_states) / n_cnv_states
        )
        
        # Prior distributions
        self._set_priors()
    
    def _set_priors(self) -> None:
        """Set prior distributions for theta, ID_prob, and CNV_prob."""
        theta_len = self.n_features if self.fsp_mode else 1
        
        # Theta prior (Beta distribution)
        theta_mu_prior = torch.linspace(0.01, 0.99, self.n_cnv_states).reshape(1, -1).expand(theta_len, -1).clone()
        theta_conc_prior = torch.ones(theta_len, self.n_cnv_states) * 50.0
        
        self.register_buffer('theta_mu_prior', theta_mu_prior)
        self.register_buffer('theta_concentration_prior', theta_conc_prior)
        
        # ID_prob prior (uniform)
        self.register_buffer(
            'id_prior',
            torch.ones(self.n_samples, self.n_components) / self.n_components
        )
        
        # CNV_prob prior (uniform)
        self.register_buffer(
            'cnv_prior',
            torch.ones(self.n_features, self.n_components, self.n_cnv_states) / self.n_cnv_states
        )
    
    @property
    def theta_s1(self):
        """Alpha parameter of theta posterior (Beta): mu * concentration."""
        return self.theta_mu * self.theta_concentration
    
    @property
    def theta_s2(self):
        """Beta parameter of theta posterior: (1-mu) * concentration."""
        return (1 - self.theta_mu) * self.theta_concentration
    
    def update_parameters(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Update variational parameters via coordinate ascent.
        
        Parameters
        ----------
        data : tuple
            (AD, DP) - Alternative depth and total depth tensors
        """
        AD, DP = data[0], data[1]
        BD = DP - AD
        
        # Update ID_prob (cell-to-cluster assignment)
        if self.learn_pi:
            self._update_id_prob(AD, DP, BD)
        
        # Update CNV_prob (CNV state assignments)
        self._update_cnv_prob(AD, BD, DP)
        
        # Update theta parameters
        if self.learn_theta:
            self._update_theta(AD, BD)
    
    def _update_id_prob(self, AD: torch.Tensor, DP: torch.Tensor, BD: torch.Tensor) -> None:
        """Update cell-to-cluster assignment probabilities.
        
        Parameters
        ----------
        AD : torch.Tensor
            Alternative depth (n_features, n_samples)
        DP : torch.Tensor
            Total depth (n_features, n_samples)
        BD : torch.Tensor
            Reference depth = DP - AD
        """
        # Compute log-likelihood contributions from each CNV state
        log_lik_id = torch.zeros(AD.shape[1], self.n_components, device=self.device, dtype=self.dtype)
        
        for t in range(self.n_cnv_states):
            # Digamma of theta parameters
            digamma_s1 = torch.digamma(self.theta_s1[:, t:t+1])
            digamma_s2 = torch.digamma(self.theta_s2[:, t:t+1])
            digamma_sum = torch.digamma(self.theta_s1[:, t:t+1] + self.theta_s2[:, t:t+1])
            
            # Compute log-likelihood for this state
            s1 = AD.T @ (self.cnv_prob[:, :, t] * digamma_s1)
            s2 = BD.T @ (self.cnv_prob[:, :, t] * digamma_s2)
            ss = DP.T @ (self.cnv_prob[:, :, t] * digamma_sum)
            
            log_lik_id += s1 + s2 - ss
        
        # Update ID_prob with log-sum-exp trick for numerical stability
        log_id_prob = log_lik_id + torch.log(self.id_prior + 1e-10)
        log_id_prob = log_id_prob - torch.logsumexp(log_id_prob, dim=1, keepdim=True)
        self.id_prob.copy_(torch.exp(log_id_prob))
    
    def _update_cnv_prob(self, AD: torch.Tensor, BD: torch.Tensor, DP: torch.Tensor) -> None:
        """Update CNV state assignment probabilities.
        
        Parameters
        ----------
        AD : torch.Tensor
            Alternative depth
        BD : torch.Tensor
            Reference depth
        DP : torch.Tensor
            Total depth
        """
        log_lik_cnv = torch.zeros(
            self.n_features, self.n_components, self.n_cnv_states,
            device=self.device, dtype=self.dtype
        )
        
        for t in range(self.n_cnv_states):
            digamma_s1 = torch.digamma(self.theta_s1[:, t:t+1])
            digamma_s2 = torch.digamma(self.theta_s2[:, t:t+1])
            digamma_sum = torch.digamma(self.theta_s1[:, t:t+1] + self.theta_s2[:, t:t+1])
            
            s1_cnv = AD @ self.id_prob
            s2_cnv = BD @ self.id_prob
            ss_cnv = DP @ self.id_prob
            
            log_lik_cnv[:, :, t] = (
                s1_cnv * digamma_s1 +
                s2_cnv * digamma_s2 -
                ss_cnv * digamma_sum
            )
        
        # Update CNV_prob with numerical stability
        log_cnv_prob = log_lik_cnv + torch.log(self.cnv_prior + 1e-10)
        log_cnv_prob = log_cnv_prob - torch.logsumexp(log_cnv_prob, dim=2, keepdim=True)
        self.cnv_prob.copy_(torch.exp(log_cnv_prob))
    
    def _update_theta(self, AD: torch.Tensor, BD: torch.Tensor) -> None:
        """Update theta (allele ratio) posterior parameters.
        
        Parameters
        ----------
        AD : torch.Tensor
            Alternative depth
        BD : torch.Tensor
            Reference depth
        """
        s1_ik = AD @ self.id_prob  # (n_features, n_components)
        s2_ik = BD @ self.id_prob  # (n_features, n_components)
        
        # Update theta posterior parameters
        new_theta_s1 = self.theta_s1_prior.clone() if hasattr(self, 'theta_s1_prior') else torch.zeros_like(self.theta_mu)
        new_theta_s2 = self.theta_s2_prior.clone() if hasattr(self, 'theta_s2_prior') else torch.zeros_like(self.theta_mu)
        
        for t in range(self.n_cnv_states):
            axis = 1 if self.fsp_mode else None
            contribution_s1 = (s1_ik * self.cnv_prob[:, :, t]).sum(dim=axis, keepdim=True) if axis else (s1_ik * self.cnv_prob[:, :, t]).sum()
            contribution_s2 = (s2_ik * self.cnv_prob[:, :, t]).sum(dim=axis, keepdim=True) if axis else (s2_ik * self.cnv_prob[:, :, t]).sum()
            
            if axis is None:
                new_theta_s1[:, t] += contribution_s1
                new_theta_s2[:, t] += contribution_s2
            else:
                new_theta_s1[:, t:t+1] += contribution_s1
                new_theta_s2[:, t:t+1] += contribution_s2
        
        # Update parameters
        self.theta_mu.data = new_theta_s1 / (new_theta_s1 + new_theta_s2)
        self.theta_concentration.data = new_theta_s1 + new_theta_s2
    
    def compute_elbo(self, data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute Evidence Lower Bound (ELBO).
        
        Parameters
        ----------
        data : tuple
            (AD, DP) tensors
            
        Returns
        -------
        torch.Tensor
            ELBO value
        """
        AD, DP = data[0], data[1]
        BD = DP - AD
        
        # Likelihood term (approximated)
        ll = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for t in range(self.n_cnv_states):
            digamma_s1 = torch.digamma(self.theta_s1[:, t:t+1])
            digamma_s2 = torch.digamma(self.theta_s2[:, t:t+1])
            digamma_sum = torch.digamma(self.theta_s1[:, t:t+1] + self.theta_s2[:, t:t+1])
            
            s1 = (AD.T @ (self.cnv_prob[:, :, t] * digamma_s1)) * self.id_prob
            s2 = (BD.T @ (self.cnv_prob[:, :, t] * digamma_s2)) * self.id_prob
            ss = (DP.T @ (self.cnv_prob[:, :, t] * digamma_sum)) * self.id_prob
            
            ll = ll + (s1 + s2 - ss).sum()
        
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
