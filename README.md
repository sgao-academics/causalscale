# causalscale v3.1.0

**Unified Causal Discovery — 6 engines, 4 evaluation modes, 3 papers behind it.**

```python
import causalscale as cs
model = cs.CausalDiscovery(data).fit()  # auto-selects best engine
print(model.summary())
report = model.validate()               # auto-detect: causal / biology / self / pseudo
```

## Six Engines

| Engine | Best For | Paper | Status |
|:--|:--|:--|:--|
| **cluster_aware** | d <= 200 | SSCAGate (Gao 2026, Nature submission) | Optimized — F1=0.53 @ d=50 |
| **transformer** | d=200-500 | Causal Transformer (Gao 2026, ML Springer) | Published — 1028 edges @ d=200 vs NOTEARS 0 |
| **lowrank** | d > 500, genome-scale | LowRankGNN (Gao 2026, SSRN) | Verified — d=17,787 in 0.2s |
| **multimodal** | Multi-omics consensus | MM-CDSM (Gao 2026, BMC Bioinformatics) | 16x KM enrichment over concatenation |
| **multi_scale** | Hierarchical data | causalscale framework | W = Sigma U_s @ V_s^T |
| **ensemble** | Consensus voting | CauTion-inspired | Weighted 3-engine voting + LLM arbitration |

## Benchmarks (Synthetic ER DAGs)

| d | NOTEARS F1 | causalscale F1 | Engine | Advantage |
|:--|:--|:--|:--|:--|
| 30 | 0.581 | 0.586 | cluster_aware | +1% |
| 50 | 0.475 | 0.531 | cluster_aware | +12% |
| 100 | 0.185 | 0.462 | cluster_aware | +150% |
| 200-500 | 0 (collapses) | 500-3000 edges | transformer | Unique capability |

## Biological Validation

**93.3% STRING/TRRUST precision** (ASCEND two-tier, DepMap 200 genes, 14/15 validated).

## Four-Mode `validate()`

```python
report = model.validate()                         # auto-detect
report = model.validate(ground_truth=W_true)      # causal F1, SHD, TPR
report = model.validate(pseudo_ground_truth="notears")  # NOTEARS as reference
# Biology mode auto-detected when var_names are gene symbols
```

## Quick Install

```bash
pip install causalscale
```

## License

MIT. Author: Shuaidong Gao (ORCID: 0009-0004-5641-3581).

## Links

- GitHub: https://github.com/sgao-academics/causalscale
- HuggingFace: https://huggingface.co/sgao-academics/causalscale
- Papers: SSCAGate (Nature submitted), CT (ML Springer), MM-CDSM (BMC Bioinformatics submitted)
