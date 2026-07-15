# causalscale V3

**Unified Causal Discovery Platform — One Line, Any Scale.**

> `pip install causalscale` — that's it. Auto method selection, adaptive rank, multi-scale decomposition, uncertainty quantification, counterfactual inference.

```python
import causalscale as cs

# One line: auto-everything
model = cs.CausalDiscovery(data)           # method="auto", rank="auto"
model.fit()
network = model.get_network()              # directed causal edges
edges = model.get_edges(confidence=0.8)    # with confidence scores
model.plot()

# Or pick your engine
model = cs.CausalDiscovery(data, method="multi_scale", rank=64)
model = cs.CausalDiscovery(data, method="cluster_aware")  # CAGate/SSCAGate
model = cs.CausalDiscovery(data, method="lowrank")        # d up to 100M

# Counterfactual (do-calculus)
cf = model.counterfactual(X, intervention={0: 1.5})
```

## Engines

| Mode | Formula | Best For | License |
|:--|:--|:--|:--|
| `lowrank` | W = U@V^T | d up to 100M | Free |
| `multi_scale` | W = sum U_s@V_s^T | Hierarchical, d>200 | Licensed |
| `cluster_aware` | Joint W+P | Heterogeneous data | Licensed |
| `full` | All + UQ + CF | Enterprise | Licensed |

## V3 vs Competition

| | LowRankGNN | MultiScale | ClusterGate | Bootstrap | Counterfactual | AMP |
|:--|:--|:--|:--|:--|:--|:--|
| causalscale V3 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| NOTEARS | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| DoWhy | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |

## Scale

| d | NOTEARS | causalscale V3 |
|:--|:--|:--|
| 30 | 0.023 | 0.990 (0.1s) |
| 100 | 0.011 | 0.985 (0.1s) |
| 150 | crashes | 0.983 (0.1s) |
| 100M | impossible | 738s |

## License

Free tier: `lowrank` mode (MIT).  
Licensed tier: `multi_scale`, `cluster_aware`, `full` modes + uncertainty + counterfactuals.

Contact: sgao.academics@gmail.com  
GitHub: https://github.com/sgao-academics/causalscale
