# causalscale v3.2.0

**Unified causal discovery — 7 engines under one API, from d=30 to genome-wide.** `pip install causalscale`.

```python
import causalscale as cs
model = cs.CausalDiscovery(data, method="auto")  # auto-selects best engine
model.fit()
print(model.summary())
```

## Seven Engines

| Engine | Best For | Method | Key Result |
|:--|:--|:--|:--|
| **dagma** | d <= 150 | DAGMA (Bello et al., NeurIPS 2022) | F1=0.989 @ d=150 |
| **cluster_aware** | d <= 200 | Verified NOTEARS with exact DAG constraint | Exceeds NOTEARS at all d, exceeds DAGMA at d=30 |
| **transformer** | d=200-500 | Causal Transformer (Gao 2026) | 1,028 edges @ d=200, NOTEARS = 0 |
| **lowrank** | d > 500 | LowRankGNN (Gao 2026) | d=17,787, 88.7% STRING/TRRUST precision |
| **multiscale** | d=500-5,000 | Multi-scale low-rank decomposition | 16x KM enrichment over concatenation |
| **multimodal** | m >= 2 | Cross-modal Frobenius consensus | Multi-omics causal discovery |
| **ensemble** | Any | 3-engine weighted voting | F1 exceeds best single engine by 22-35% |

Plus: **PCMCI** time-series engine (Runge et al., Sci. Adv. 2019) for lagged causal discovery.

## Benchmarks (Synthetic ER DAGs, d=30-150, 5 seeds)

| d | NOTEARS F1 | DAGMA F1 | causalscale F1 | Engine |
|:--|:--|:--|:--|:--|
| 30 | 0.581 | 0.589 | **0.646** | cluster_aware |
| 50 | 0.475 | 0.689 | 0.595 | cluster_aware |
| 80 | 0.391 | 0.896 | 0.731 | cluster_aware |
| 100 | 0.185 | 0.931 | 0.766 | cluster_aware |
| 150 | 0.000 | 0.989 | 0.768 | cluster_aware |
| 200+ | 0 (collapse) | timeout | 500-3,000 edges | transformer / lowrank |

NOTEARS death-line confirmed at d=150. Only causalscale survives beyond d=200.

## Biological Validation

**88.7% STRING/TRRUST precision** (574/647 edges) on DepMap genome-scale CRISPR data (1,208 cell lines).

ASCEND two-tier discovery: **93.3%** (14/15 edges, 95% CI [68.0%, 99.8%]).

33-cancer pan-cancer scan recovers tissue-specific ARID1A-MTOR directionality
(Spearman rho=-0.720, p=2.3e-6; 5.5 minutes on 8 GB GPU).

## Cross-Domain

S&P 500 equities (76% same-sector edges, 180 stocks), NOAA climate reanalysis
(correct west-to-east ENSO propagation, 19 Pacific stations). Same `fit()`
call across three independent domains.

## Install

```bash
pip install causalscale
```

GPU auto-detected; CPU fallback supported. Python >=3.10, PyTorch >=2.1.

Full replication package (pre-computed results, figures, one-command verify):
see `paper/causalscale_Replication_Package.zip` in this repository.

## Links

- **GitHub**: https://github.com/sgao-academics/causalscale
- **HuggingFace Hub** (pre-trained models): https://huggingface.co/sgao-academics/causalscale
- **KDD 2027 Datasets & Benchmarks Track**

## License

MIT. Author: Shuaidong Gao (ORCID: 0009-0004-5641-3581).
