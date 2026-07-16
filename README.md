# causalscale v3.1

**Unified Causal Discovery Platform — One Line, Any Scale. Honest Benchmarks. STRING/TRRUST Validated.**

`pip install causalscale` — auto engine selection, 4-mode validate(), ensemble, stability, ASCEND.

```python
import causalscale as cs

model = cs.CausalDiscovery(data)  # auto-everything
model.fit()
report = model.validate()          # auto: causal/bio/self/pseudo
```

## Benchmarks (v3.1, honest numbers)

| d | NOTEARS F1 | causalscale F1 | Advantage |
|:--|:--|:--|:--|
| 30 | 0.581 | 0.586 | +1% |
| 50 | 0.475 | 0.531 | +12% |
| 80 | 0.391 | 0.495 | +27% |
| 100 | 0.185 | 0.462 | +150% |

**Biology**: 93.3% STRING/TRRUST precision (ASCEND two-tier, DepMap d=200).

## Engines

| Mode | Best For | Status |
|:--|:--|:--|
| `cluster_aware` | d <= 500, best synthetic F1 | Verified |
| `lowrank` | d > 500, genome-scale | Verified (d=5000, 0.2s) |
| `ensemble` | Multi-engine consensus | CauTion-inspired |
| `ASCEND` | Two-tier biology | 93.3% precision |
| `multi_scale` | Routed to cluster_aware | Fallback |
| `transformer` | Routed to cluster_aware | Fallback |

## Paper

KDD 2027 Datasets & Benchmarks Track. [`paper/causalscale_kdd2027.pdf`](paper/causalscale_kdd2027.pdf)

## License

MIT. Author: Shuaidong Gao (ORCID: 0009-0004-5641-3581).
