---
license: mit
library_name: causalscale
tags:
- causal-discovery
- genomics
- dag
- low-rank
- ensemble
- biological-networks
- tabular
---

# causalscale v3.1.0

**Unified Causal Discovery -- 7 engines, automatic selection, genome-scale capability.**

## Benchmarks (Synthetic ER DAGs, 5 seeds)

| d | NOTEARS F1 | DAGMA F1 | causalscale F1 |
|:--|:--|:--|:--|
| 30 | 0.581 | 0.589 | **0.646** |
| 50 | 0.475 | 0.689 | 0.595 |
| 80 | 0.391 | 0.896 | 0.731 |
| 100 | 0.185 | 0.931 | 0.766 |
| 150 | 0.000 | 0.989 | 0.768 |
| 200+ | collapse | timeout | 500-3000 edges |

## Biological Validation

**88.7% STRING/TRRUST precision** (574/647 edges) on DepMap genome-scale.
**93.3% ASCEND two-tier precision** (14/15 edges).
33-cancer pan-cancer scan: ARID1A-MTOR directionality, Spearman rho=-0.720 (p=2.3e-6).

## Cross-Domain

S&P 500: 76% same-sector edges (19/25).

## Seven Engines

| Engine | Best For | Key Result |
|:--|:--|:--|
| dagma | d <= 150 | F1=0.989 @ d=150 (official PyPI integration) |
| cluster_aware | d <= 200 | Verified NOTEARS, exact DAG constraint |
| transformer | d=200-500 | 1028 edges @ d=200 (NOTEARS: 0) |
| lowrank | d > 500 | d=17,787, 88.7% STRING/TRRUST |
| multiscale | d=500-5000 | 16x KM enrichment over concatenation |
| multimodal | m >= 2 modalities | Cross-modal Frobenius consensus |
| ensemble | Consensus | 3-engine weighted voting |

## Quick Start

```python
import causalscale as cs
model = cs.CausalDiscovery(data).fit()
print(model.summary())
```

## License

MIT. Author: Shuaidong Gao (ORCID: 0009-0004-5641-3581).
KDD 2027 Datasets & Benchmarks Track submission.
