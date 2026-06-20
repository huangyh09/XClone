"""Quickstart guide for XClone-Torch installation and usage.

This guide helps troubleshoot common installation issues.
"""

# ============================================================================
# INSTALLATION TROUBLESHOOTING
# ============================================================================

# Problem: "ModuleNotFoundError: No module named 'xclone_torch'"
# Solution: Use one of these methods:

# Method 1: Install in development mode from the pytorch-reimplementation branch
# ============================================================================
# git clone https://github.com/single-cell-genetics/XClone.git
# cd XClone
# git checkout pytorch-reimplementation
# pip install -e .

# Method 2: Add the branch directory to PYTHONPATH
# ============================================================================
# export PYTHONPATH="${PYTHONPATH}:/path/to/XClone/pytorch-reimplementation"

# Method 3: Run from the repository root
# ============================================================================
# cd /path/to/XClone
# python -m examples.run_bch869_pytorch

# Method 4: Use Python path manipulation in script
# ============================================================================
# Add this at the top of your script:
import sys
import os

# Add parent directory to path if xclone_torch not found
if 'xclone_torch' not in sys.modules:
    # Try adding the current directory and parent directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)  # examples -> root
    root_dir = os.path.dirname(parent_dir)     # root -> parent
    
    for path in [current_dir, parent_dir, root_dir]:
        if path not in sys.path:
            sys.path.insert(0, path)
            print(f"[Setup] Added {path} to sys.path")

# Now import xclone_torch
import xclone_torch

print(f"[Setup] XClone-Torch version: {xclone_torch.__version__}")
print(f"[Setup] Location: {xclone_torch.__file__}")
