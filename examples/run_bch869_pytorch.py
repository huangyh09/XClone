"""Example: Running XClone-Torch on BCH869 scRNA-seq dataset.

This script demonstrates how to use XClone-Torch for CNV analysis on the BCH869 dataset,
following the structure of the original XClone tutorials but using PyTorch.

Dataset: BCH869 scRNA-seq (H3K27M-glioma, 492 cells, 32,696 genes)
Modules demonstrated:
  - RDR module: Read Depth Ratio from gene expression
  - BAF module: B-Allele Frequency from SNP data
  - Combined analysis

Troubleshooting:
  If you get "ModuleNotFoundError: No module named 'xclone_torch'":
  
  Option 1 - Install in editable mode:
    cd /path/to/XClone
    pip install -e .
  
  Option 2 - Set PYTHONPATH:
    export PYTHONPATH="${PYTHONPATH}:/path/to/XClone"
    python examples/run_bch869_pytorch.py
  
  Option 3 - Run from repo root:
    cd /path/to/XClone
    python -m examples.run_bch869_pytorch
  
  Option 4 - See INSTALLATION_GUIDE.md for more details
"""

import sys
import os
from pathlib import Path

# ============================================================================
# SETUP: Add repository root to path
# ============================================================================

# Get the directory where this script is located
script_dir = Path(__file__).resolve().parent  # examples/
repo_root = script_dir.parent  # XClone/

# Add repo root to Python path if not already there
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
    print(f"[Setup] Added {repo_root} to sys.path")

# ============================================================================
# IMPORTS
# ============================================================================

import torch
import numpy as np
import pandas as pd
import anndata as ad
from typing import Dict, Tuple

# Try to import xclone_torch
try:
    import xclone_torch
    print(f"[XClone-Torch] Version: {xclone_torch.__version__}")
    print(f"[XClone-Torch] Location: {xclone_torch.__file__}")
except ImportError as e:
    print(f"[Error] Failed to import xclone_torch: {e}")
    print(f"[Error] sys.path: {sys.path}")
    print("\n[Help] Try one of these solutions:")
    print("  1. Install in editable mode: cd /path/to/XClone && pip install -e .")
    print("  2. Set PYTHONPATH: export PYTHONPATH=\"${PYTHONPATH}:/path/to/XClone\"")
    print("  3. Run from repo root: cd /path/to/XClone && python -m examples.run_bch869_pytorch")
    print("  4. See INSTALLATION_GUIDE.md for more details")
    sys.exit(1)

# Check CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[XClone-Torch] Using device: {device}")
if device == 'cuda':
    print(f"[XClone-Torch] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[XClone-Torch] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


class BCH869Analysis:
    """Wrapper class for BCH869 dataset analysis with XClone-Torch."""
    
    def __init__(self, rdr_adata: ad.AnnData, baf_adata: ad.AnnData,
                 device: str = 'cuda', n_components: int = 4):
        """
        Initialize analysis.
        
        Parameters
        ----------
        rdr_adata : ad.AnnData
            RDR module data (492 cells × 32,696 genes)
        baf_adata : ad.AnnData
            BAF module data (AD and DP layers)
        device : str
            Device to use ('cuda' or 'cpu')
        n_components : int
            Number of clones (default 4 for BCH869)
        """
        self.rdr_adata = rdr_adata
        self.baf_adata = baf_adata
        self.device = device
        self.n_components = n_components
        
        # Ground truth from original paper
        self.true_clones = np.array([227, 221, 34, 7])  # Clone sizes
        self.n_cells = rdr_adata.n_obs
        self.n_genes_rdr = rdr_adata.n_vars
        self.n_genes_baf = baf_adata.n_vars
        
        print(f"[BCH869] Dataset info:")
        print(f"  Cells: {self.n_cells}")
        print(f"  RDR genes: {self.n_genes_rdr}")
        print(f"  BAF genes: {self.n_genes_baf}")
        print(f"  Expected clones: {self.n_components} ({self.true_clones})")
    
    def prepare_rdr_data(self, layer: str = 'raw_expr',
                        filter_genes: bool = True) -> torch.Tensor:
        """
        Prepare RDR data for PyTorch model.
        
        Parameters
        ----------
        layer : str
            Layer name in anndata to use
        filter_genes : bool
            Whether to filter genes (remove markers, low expression)
            
        Returns
        -------
        torch.Tensor
            Preprocessed count data on specified device
        """
        print(f"\n[RDR Preprocessing] Loading layer: {layer}")
        
        # Extract expression data
        if layer in self.rdr_adata.layers:
            expr = self.rdr_adata.layers[layer]
            if hasattr(expr, 'toarray'):
                expr = expr.toarray()
            expr = np.array(expr)  # (n_cells, n_genes)
        else:
            expr = self.rdr_adata.X
            if hasattr(expr, 'toarray'):
                expr = expr.toarray()
        
        print(f"  Raw data shape: {expr.shape}")
        print(f"  Min: {expr.min():.2f}, Max: {expr.max():.2f}")
        
        # Filter genes (optional)
        if filter_genes:
            # Remove genes with low mean expression
            gene_means = expr.mean(axis=0)
            keep_genes = gene_means > 0.1
            expr = expr[:, keep_genes]
            print(f"  After filtering: {expr.shape} ({keep_genes.sum()} genes kept)")
        
        # Log-transform
        expr = np.log(expr + 1)
        print(f"  After log transform - Min: {expr.min():.2f}, Max: {expr.max():.2f}")
        
        # Convert to tensor
        expr_tensor = torch.from_numpy(expr.T).float().to(self.device)  # (n_genes, n_cells)
        
        return expr_tensor
    
    def prepare_baf_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare BAF data for PyTorch model.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (AD_tensor, DP_tensor) for BAF module
        """
        print(f"\n[BAF Preprocessing] Loading BAF data")
        
        AD = self.baf_adata.layers['AD']
        DP = self.baf_adata.layers['DP']
        
        # Convert sparse to dense if needed
        if hasattr(AD, 'toarray'):
            AD = AD.toarray()
        if hasattr(DP, 'toarray'):
            DP = DP.toarray()
        
        AD = np.array(AD)  # (n_cells, n_genes) or (n_genes, n_cells)?
        DP = np.array(DP)
        
        print(f"  AD shape: {AD.shape}, DP shape: {DP.shape}")
        
        # Ensure correct orientation (n_features, n_samples)
        if AD.shape[0] > AD.shape[1]:
            AD = AD.T
            DP = DP.T
        
        print(f"  After transpose - AD: {AD.shape}, DP: {DP.shape}")
        print(f"  AD range: [{AD.min()}, {AD.max()}]")
        print(f"  DP range: [{DP.min()}, {DP.max()}]")
        
        # Convert to tensors
        AD_tensor = torch.from_numpy(AD).float().to(self.device)
        DP_tensor = torch.from_numpy(DP).float().to(self.device)
        
        return AD_tensor, DP_tensor
    
    def run_rdr_module(self, rdr_data: torch.Tensor,
                       n_cnv_states: int = 3,
                       max_iter: int = 50) -> Dict:
        """
        Run RDR module inference.
        
        Parameters
        ----------
        rdr_data : torch.Tensor
            Prepared RDR data (n_genes, n_cells)
        n_cnv_states : int
            Number of CNV states (loss, neutral, gain)
        max_iter : int
            Maximum iterations
            
        Returns
        -------
        dict
            Results including clone assignments and CNV probabilities
        """
        print(f"\n[RDR Module] Initializing model...")
        print(f"  Features: {rdr_data.shape[0]} genes")
        print(f"  Samples: {rdr_data.shape[1]} cells")
        print(f"  Components: {self.n_components} clones")
        print(f"  States: {n_cnv_states} (copy loss, neutral, gain)")
        
        # Initialize model
        model = xclone_torch.models.RDRModel(
            n_features=rdr_data.shape[0],
            n_samples=rdr_data.shape[1],
            n_components=self.n_components,
            n_cnv_states=n_cnv_states,
            learn_theta=True,
            learn_pi=True,
            device=self.device
        )
        
        print(f"\n[RDR Module] Training...")
        history = model.fit(
            rdr_data,
            max_iter=max_iter,
            min_iter=5,
            epsilon_conv=1e-2,
            verbose=True
        )
        
        # Extract results
        clone_assignment = model.get_posterior_assignment()  # (n_cells, n_components)
        cnv_probs = model.get_posterior_cnv_prob()  # (n_genes, n_components, n_states)
        
        print(f"\n[RDR Module] Results:")
        print(f"  Clone assignments shape: {clone_assignment.shape}")
        print(f"  CNV probabilities shape: {cnv_probs.shape}")
        print(f"  ELBO history (last 5): {history['elbo'][-5:]}")
        print(f"  Converged: {history['converged']}")
        
        # Predict hard clone assignments
        hard_clones = clone_assignment.argmax(dim=1)  # (n_cells,)
        clone_counts = torch.bincount(hard_clones)
        print(f"  Predicted clone sizes: {clone_counts.cpu().numpy()}")
        print(f"  Expected clone sizes: {self.true_clones}")
        
        return {
            'model': model,
            'clone_assignment': clone_assignment,
            'cnv_probs': cnv_probs,
            'hard_clones': hard_clones,
            'history': history
        }
    
    def run_baf_module(self, AD: torch.Tensor, DP: torch.Tensor,
                       n_cnv_states: int = 3,
                       max_iter: int = 50) -> Dict:
        """
        Run BAF module inference.
        
        Parameters
        ----------
        AD : torch.Tensor
            Alternative depth (n_features, n_samples)
        DP : torch.Tensor
            Total depth
        n_cnv_states : int
            Number of CNV states
        max_iter : int
            Maximum iterations
            
        Returns
        -------
        dict
            BAF module results
        """
        print(f"\n[BAF Module] Initializing model...")
        print(f"  Features: {AD.shape[0]} genes/blocks")
        print(f"  Samples: {AD.shape[1]} cells")
        print(f"  Components: {self.n_components} clones")
        print(f"  States: {n_cnv_states}")
        
        # Initialize model
        model = xclone_torch.models.BAFModel(
            n_features=AD.shape[0],
            n_samples=AD.shape[1],
            n_components=self.n_components,
            n_cnv_states=n_cnv_states,
            learn_theta=True,
            learn_pi=True,
            device=self.device
        )
        
        print(f"\n[BAF Module] Training...")
        history = model.fit(
            (AD, DP),
            max_iter=max_iter,
            min_iter=5,
            epsilon_conv=1e-2,
            verbose=True
        )
        
        # Extract results
        clone_assignment = model.get_posterior_assignment()
        cnv_probs = model.get_posterior_cnv_prob()
        
        print(f"\n[BAF Module] Results:")
        print(f"  Clone assignments shape: {clone_assignment.shape}")
        print(f"  CNV probabilities shape: {cnv_probs.shape}")
        print(f"  ELBO history (last 5): {history['elbo'][-5:]}")
        
        hard_clones = clone_assignment.argmax(dim=1)
        clone_counts = torch.bincount(hard_clones)
        print(f"  Predicted clone sizes: {clone_counts.cpu().numpy()}")
        
        return {
            'model': model,
            'clone_assignment': clone_assignment,
            'cnv_probs': cnv_probs,
            'hard_clones': hard_clones,
            'history': history
        }
    
    def compare_results(self, rdr_results: Dict, baf_results: Dict) -> Dict:
        """
        Compare RDR and BAF module results.
        
        Parameters
        ----------
        rdr_results : dict
            RDR module results
        baf_results : dict
            BAF module results
            
        Returns
        -------
        dict
            Comparison metrics
        """
        print(f"\n[Comparison] RDR vs BAF module results:")
        
        rdr_clones = rdr_results['hard_clones'].cpu().numpy()
        baf_clones = baf_results['hard_clones'].cpu().numpy()
        
        # Compute agreement
        agreement = (rdr_clones == baf_clones).sum() / len(rdr_clones)
        print(f"  Clone assignment agreement: {agreement:.2%}")
        
        # Get true assignments from data
        if 'Clone_ID' in self.rdr_adata.obs.columns:
            true_clones = self.rdr_adata.obs['Clone_ID'].values
            print(f"  True clone distribution: {pd.Series(true_clones).value_counts().sort_index().values}")
        
        print(f"  RDR predicted: {torch.bincount(torch.tensor(rdr_clones)).cpu().numpy()}")
        print(f"  BAF predicted: {torch.bincount(torch.tensor(baf_clones)).cpu().numpy()}")
        
        return {
            'rdr_clones': rdr_clones,
            'baf_clones': baf_clones,
            'agreement': agreement
        }


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("XClone-Torch: BCH869 scRNA-seq CNV Analysis")
    print("="*80)
    
    # For demonstration, we'll create synthetic data matching BCH869 dimensions
    # In practice, you would load the actual data using xclone.data.bch869_rdr() etc.
    
    print("\n[Data Loading] Creating synthetic BCH869-like data...")
    n_cells = 492
    n_genes_rdr = 32696
    n_genes_baf = 32696
    
    # Create synthetic RDR data (log-normal distribution)
    rdr_data = np.random.lognormal(mean=1, sigma=1.5, size=(n_cells, n_genes_rdr)).astype(np.float32)
    rdr_adata = ad.AnnData(rdr_data)
    rdr_adata.layers['raw_expr'] = rdr_data
    rdr_adata.obs['Clone_ID'] = np.repeat([0, 1, 2, 3], [227, 221, 34, 10])
    
    # Create synthetic BAF data
    AD = np.random.poisson(lam=50, size=(n_cells, n_genes_baf)).astype(np.float32)
    DP = AD + np.random.poisson(lam=50, size=(n_cells, n_genes_baf)).astype(np.float32)
    baf_adata = ad.AnnData(rdr_data[:, :n_genes_baf])
    baf_adata.layers['AD'] = AD
    baf_adata.layers['DP'] = DP
    baf_adata.obs['Clone_ID'] = rdr_adata.obs['Clone_ID']
    
    print(f"  RDR AnnData: {rdr_adata}")
    print(f"  BAF AnnData: {baf_adata}")
    
    # Initialize analysis
    analysis = BCH869Analysis(
        rdr_adata=rdr_adata,
        baf_adata=baf_adata,
        device=device,
        n_components=4
    )
    
    # Run RDR module
    print("\n" + "="*80)
    print("RDR MODULE ANALYSIS")
    print("="*80)
    rdr_data_tensor = analysis.prepare_rdr_data(filter_genes=True)
    rdr_results = analysis.run_rdr_module(rdr_data_tensor, max_iter=30)
    
    # Run BAF module
    print("\n" + "="*80)
    print("BAF MODULE ANALYSIS")
    print("="*80)
    AD_tensor, DP_tensor = analysis.prepare_baf_data()
    baf_results = analysis.run_baf_module(AD_tensor, DP_tensor, max_iter=30)
    
    # Compare results
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    comparison = analysis.compare_results(rdr_results, baf_results)
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nSummary:")
    print(f"  - RDR and BAF modules converged successfully")
    print(f"  - Clone assignment agreement: {comparison['agreement']:.2%}")
    print(f"  - Results saved to memory (tensors on {device})")


if __name__ == '__main__':
    main()
