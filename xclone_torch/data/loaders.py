"""Data loaders for XClone-Torch."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple


class AnnDataDataset(Dataset):
    """PyTorch Dataset wrapper for AnnData objects.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object
    layer : str, optional
        Layer to use as features. If None, uses .X
    device : str
        Device to place tensors on
    """
    
    def __init__(self, adata, layer: Optional[str] = None, device: str = 'cpu'):
        self.adata = adata
        self.layer = layer
        self.device = device
        
        # Extract feature matrix
        if layer is not None:
            if hasattr(adata.layers[layer], 'toarray'):
                # Sparse matrix
                self.X = torch.from_numpy(adata.layers[layer].toarray())
            else:
                self.X = torch.from_numpy(np.array(adata.layers[layer]))
        else:
            if hasattr(adata.X, 'toarray'):
                # Sparse matrix
                self.X = torch.from_numpy(adata.X.toarray())
            else:
                self.X = torch.from_numpy(np.array(adata.X))
        
        self.X = self.X.to(device=device, dtype=torch.float32)
    
    def __len__(self):
        return self.adata.n_obs
    
    def __getitem__(self, idx):
        return self.X[idx]


class AnnDataLoader:
    """Loader for AnnData objects with batching support.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object
    batch_size : int
        Batch size for loading
    layer : str, optional
        Layer to use
    device : str
        Device to place tensors on
    shuffle : bool
        Whether to shuffle data
    """
    
    def __init__(self, adata, batch_size: int = 32, layer: Optional[str] = None,
                 device: str = 'cpu', shuffle: bool = True):
        self.adata = adata
        self.batch_size = batch_size
        self.device = device
        
        dataset = AnnDataDataset(adata, layer=layer, device=device)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
