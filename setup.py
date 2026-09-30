"""causalscale replication package — benchmark protocol for differentiable causal discovery.

Accompanies: "Protocol, Not Physics: Re-measuring the Scalability Limit of
Differentiable Causal Discovery" (KDD 2027 Datasets & Benchmarks).

Reading the released results needs numpy and matplotlib.  Re-running the
experiments needs a solver stack, which is an extra rather than a requirement, so
that `pip install -e .` stays small and cannot fail on a GPU-less machine.

    pip install -e .                  # read the tables, figures and audit
    pip install -e ".[solvers]"       # re-run the suites
    python run_all.py --verify

Author: Shuaidong Gao (ORCID 0009-0004-5641-3581)
"""

import os

from setuptools import find_packages, setup

_HERE = os.path.dirname(os.path.abspath(__file__))
_readme = os.path.join(_HERE, "README.md")
long_description = open(_readme, encoding="utf-8").read() if os.path.exists(_readme) else ""

setup(
    name="causalscale",
    version="4.0.0",
    description=("Benchmark protocol for differentiable causal discovery: fixed "
                 "generators, a shared tuning budget, per-seed records, and an "
                 "executable audit of the paper's claims"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Shuaidong Gao",
    author_email="sgao.academics@gmail.com",
    url="https://github.com/sgao-academics/causalscale",
    packages=find_packages(include=["causalscale", "causalscale.*",
                                    "experiments", "experiments.*"]),
    include_package_data=True,
    package_data={"experiments": ["records/*.jsonl"]},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        # needed to redraw Figure 1; everything else in the reading path is numpy
        "figures": ["matplotlib>=3.7"],
        # needed to re-run any suite
        "solvers": [
            "torch>=2.0",
            "scipy>=1.10",
            "scikit-learn>=1.2",
            "dagma>=0.1",
        ],
        # the toolkit measured in SS4.4, kept for inspection
        "toolkit": [
            "torch>=2.0",
            "scipy>=1.10",
            "scikit-learn>=1.2",
            "pandas>=1.5",
            "networkx>=3.0",
        ],
        "web": ["streamlit>=1.28", "plotly>=5.15"],
    },
    entry_points={
        "console_scripts": [
            "causalscale=causalscale.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords="causal-discovery benchmark reproducibility dag structure-learning",
)
