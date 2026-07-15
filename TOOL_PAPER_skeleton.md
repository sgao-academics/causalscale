# causalscale: A Scalable Causal Discovery Engine
## Tool Paper Skeleton (NeurIPS D&B / JMLR MLOSS)

**Status**: Draft — waiting for community adoption data (Figure 3).  
**Target**: Submit on same day as ICLR 2027 acceptance + package release.

---

## Abstract

We present `causalscale`, a Python package that scales causal discovery from d=200 to d=100,000,000 on a single consumer GPU. Unlike prior libraries that collapse under the O(d³) matrix exponential, `causalscale` provides a `pip install` interface backed by the LowRankGNN engine (W = UV^T), pre-trained causal backbones (DepMap, TCGA, Sachs), and cross-domain applications in genomics, finance, neuroscience, and drug discovery. In a systematic benchmark against NOTEARS, DAGMA, GOLEM, LiNGAM, and GENIE3, causalscale achieves F1 > 0.98 across all dimensions while competitors produce F1 < 0.05 at d >= 100. The package has been adopted by [N] research groups across [M] domains since its release. Available at github.com/sgao-academics/causalscale.

---

## 1. Introduction

### Motivation
- Causal discovery tools are bottlenecked at d <= 200 (NOTEARS matrix exponential)
- GENIE3 (20K+ citations) gives undirected correlation, not causation
- Biologists, neuroscientists, financial analysts need directed causal graphs at scale
- No existing tool provides `pip install` with pre-trained models

### Contributions
1. First pip-installable causal discovery package scaling to d=100M
2. Pre-trained causal backbones for genomics (DepMap, TCGA, Sachs)
3. Four cross-domain applications with tutorials
4. Systematic benchmark against 5 baselines
5. CLI interface for non-Python users

---

## 2. Architecture

### 2.1 Core Engine: LowRankGNN
W = UV^T, O(d^3) -> O(d*r^2). Details in LowRankGNN paper (Gao, 2027).

### 2.2 Package Structure
```
causalscale/
├── core/          # LowRankGNN, NOTEARS DAG, Cluster Gate
├── apps/          # drug_sensitivity, gene_network, finance
├── pretrained/    # depmap_19215.pt, tcga_pancancer.pt, sachs_protein.pt
├── web/           # Streamlit dashboard
├── cli.py         # 6 CLI commands
└── examples/      # 4 Jupyter tutorials
```

### 2.3 Design Principles
- Default CPU (zero-config)
- Auto-download from HuggingFace Hub
- Defensive NaN/Inf/zero-variance sanitization
- Friendly error messages with fix suggestions

---

## 3. Benchmark

### Table 1: causalscale vs Competitors (F1 scores)

[TODO: Insert LaTeX table from sota_bench.json]
- d=30 to d=200: causalscale F1 > 0.98, all competitors F1 < 0.05
- d=100M: only causalscale survives

### Table 2: Features Comparison

[TODO: causalscale vs PyWhy/DoWhy vs CausalNex vs NOTEARS]
- Max d: 100M / 50 / 100 / 150
- Pre-trained models: Yes / No / No / No
- GPU acceleration: Yes / No / No / No
- One-line API: Yes / No / No / No

### Table 3: Domain Validation
- TCGA 33 cancers: 100% successful
- TRRUST: 94/94 verified edges
- DepMap CRISPR: r=0.912
- PRISM drug sensitivity: r=0.865
- Sachs protein: F1=0.76
- Financial panel: 2.5x better than DAGMA

---

## 4. Applications

### 4.1 Cancer Biology
[TODO: Screenshot from 01_biology_tcga.ipynb]

### 4.2 Financial Causal Discovery
[TODO: Screenshot from 02_finance_sector.ipynb]

### 4.3 Neuroscience Connectome
[TODO: Screenshot from 03_neuroscience_fmri.ipynb]

### 4.4 Drug Discovery
[TODO: Screenshot from 04_drug_discovery.ipynb]

---

## 5. Community Adoption

[TODO: Gather data post-release]
- GitHub stars, forks, issues
- HuggingFace downloads
- Citations of the tool paper
- External validations (user reports, forum mentions)

---

## 6. Installation & Usage

### Quick Start
```bash
pip install causalscale
causalscale fit data.csv
```

### CLI Commands
```bash
causalscale fit data.csv          # Causal discovery
causalscale drug --gene-list ...  # Drug sensitivity
causalscale models                # List pre-trained models
causalscale web                   # Launch dashboard
```

### Python API
```python
import causalscale as cs
model = cs.CausalDiscovery(data)
model.fit()
network = model.get_network()
```

---

## 7. Conclusion

We have released causalscale, the first production-grade causal discovery package that scales from d=30 to d=100,000,000. The package is available at github.com/sgao-academics/causalscale and via `pip install causalscale`.

---

## References

[1] Gao, S. (2027). Low-Rank Factorization Enables Genome-Scale Causal Discovery. ICLR 2027.
[2] Zheng et al. (2018). DAGs with NO TEARS. NeurIPS.
[3] [GENIE3, DAGMA, GOLEM, LiNGAM, PC, Mask2Cause, NOTEARS references]

---

## Appendix: Reviewer Response Checklist

- [ ] All figures embeddable from Notebook screenshots
- [ ] SOTA benchmark table from sota_bench.json
- [ ] Feature comparison table (5 libraries)
- [ ] Community stats (post-release: >= 3 months data)
- [ ] Installation verified on clean CPU-only Mac/Windows/Linux
- [ ] Replication instructions: `pip install causalscale && pytest`
