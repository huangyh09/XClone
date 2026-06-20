from setuptools import setup, find_packages

setup(
    name="xclone-torch",
    version="0.1.0",
    description="PyTorch reimplementation of XClone for GPU-accelerated CNV inference from scRNA-seq data",
    author="XClone PyTorch Contributors",
    author_email="rthuang@connect.hku.hk",
    url="https://github.com/single-cell-genetics/XClone",
    license="Apache-2.0",
    packages=find_packages(include=["xclone_torch", "xclone_torch.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.18",
        "pandas>=1.0",
        "scipy>=1.5",
        "anndata>=0.7",
        "scikit-learn>=0.23",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "nbsphinx>=0.8",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
