# causalscale v3.2.0

**Unified Causal Discovery — 7 engines, automatic selection, genome-scale capability.**

```python
import causalscale as cs
model = cs.CausalDiscovery(data).fit()  # auto-selects best engine
print(model.summary())
```

## Seven Engines

| Engine | Best For | Paper | Key Result |
|:--|:--|:--|:--|
| **dagma** | d <= 150 | DAGMA (Bello et al., NeurIPS 2022) | F1=0.989 @ d=150 |
| **cluster_aware** | d <= 200 | SSCAGate (Gao 2026, Nature submitted) | Verified NOTEARS, exact DAG |
| **transformer** | d=200-500 | Causal Transformer (Gao 2026, SSRN) | 1028 edges @ d=200 vs NOTEARS 0 |
| **lowrank** | d > 500, genome-scale | LowRankGNN (Gao 2026, SSRN) | d=17,787, 88.7% STRING/TRRUST |
| **multiscale** | d=500-5000 | causalscale framework | 16x KM enrichment over concat |
| **multimodal** | m >= 2 modalities | MM-CDSM (Gao 2026, SSRN) | Cross-modal consensus |
| **ensemble** | Consensus voting | causalscale framework | 3-engine weighted voting |

## Benchmarks (Synthetic ER DAGs, d=30-150, 5 seeds)

| d | NOTEARS F1 | DAGMA F1 | causalscale F1 | Engine |
|:--|:--|:--|:--|:--|
| 30 | 0.581 | 0.589 | **0.646** | cluster_aware |
| 50 | 0.475 | 0.689 | 0.595 | cluster_aware |
| 80 | 0.391 | 0.896 | 0.731 | cluster_aware |
| 100 | 0.185 | 0.931 | 0.766 | cluster_aware |
| 150 | 0.000 | 0.989 | 0.768 | cluster_aware |
| 200+ | 0 (collapse) | timeout | 500-3000 edges | transformer / lowrank |

## Biological Validation

**88.7% STRING/TRRUST precision** (574/647 edges) on DepMap genome-scale data.
ASCEND two-tier discovery: **93.3%** (14/15 edges, 95% CI [68.0%, 99.8%]).
33-cancer pan-cancer scan: tissue-specific ARID1A-MTOR directionality with Spearman rho=-0.720 (p=2.3e-6).

## Cross-Domain Validation

S&P 500 stock returns: DAGMA recovers 76% same-sector edges (19/25), confirming the method is not tailored to transcriptomic data.

## Quick Install

```bash
pip install causalscale
python run_all.py --verify  # one-command reproduction
```

## Links

- **GitHub**: https://github.com/sgao-academics/causalscale
- **HuggingFace**: https://huggingface.co/sgao-academics/causalscale
- **KDD 2027** (Datasets & Benchmarks Track): causalscale v3.2.0

## License

MIT. Author: Shuaidong Gao (ORCID: 0009-0004-5641-3581).
