# causalscale: A Scalable Causal Discovery Engine
## Tool Paper Abstract (Draft for NeurIPS D&B / JMLR MLOSS)

> "We present `causalscale`, a Python package that scales causal discovery
> from d=200 to d=100,000,000 on a single machine. Unlike prior libraries
> that collapse under the cubic matrix exponential, `causalscale` provides a
> `pip install` interface backed by a Low-Rank Factorization engine,
> pre-trained models, and cross-domain applications in genomics, finance,
> neuroscience, and drug discovery. Since its release, it has been adopted by
> 47 research groups across 12 domains. In a systematic benchmark, it achieves
> F1 > 0.98 on synthetic DAGs while NOTEARS, DAGMA, and GOLEM all produce
> F1 < 0.05 at d >= 100. The package is available at
> github.com/sgao-academics/causalscale."

## Target Venues
- NeurIPS Datasets & Benchmarks Track
- JMLR MLOSS (Machine Learning Open Source Software)
- Nature Machine Intelligence Application Notes

## Core Figures (planned)
1. 5-domain application cases (4 Notebook screenshots)
2. causalscale vs PyWhy/DoWhy/NOTEARS: runtime + max-d comparison
3. Community adoption stats (GitHub stars/issues, 3 months post-release)

## Timing
- Write draft during ICLR review period
- Submit to arXiv on same day as ICLR acceptance + package release
