"""causalscale: One-Line Causal Discovery Engine.

causalscale wraps the LowRankGNN engine (ICLR 2027) into a single
pip-installable package that scales causal discovery from d=30 to d=100,000,000
on consumer GPUs.

Quick Start:
    pip install causalscale
    >>> import causalscale as cs
    >>> model = cs.CausalDiscovery(data)
    >>> model.fit()
    >>> network = model.get_network()
    >>> model.plot()

Author: Shuaidong Gao (ORCID: 0009-0004-5641-3581)
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="causalscale",
    version="3.2.0",
    description="Unified Causal Discovery Platform — 7 engines (DAGMA, ClusterAware, Causal Transformer, LowRankGNN, MultiScale, MultiModal, Ensemble), auto-selection, genome-scale",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Shuaidong Gao",
    author_email="sgao.academics@gmail.com",
    url="https://github.com/sgao-academics/causalscale",
    packages=find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "causalscale=causalscale.cli:main",
        ],
    },
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.2",
        "pandas>=1.5",
        "matplotlib>=3.7",
        "networkx>=3.0",
        "tqdm>=4.65",
        "dagma>=0.1",
    ],
    extras_require={
        "web": ["streamlit>=1.28", "plotly>=5.15"],
        "all": ["streamlit>=1.28", "plotly>=5.15", "huggingface_hub>=0.19"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords="causal-discovery dag low-rank gnn genomics drug-sensitivity",
)
