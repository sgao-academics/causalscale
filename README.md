---
language: en
license: mit
library_name: causalscale
tags:
- causal-discovery
- genomics
- dag
- low-rank
- structure-learning
- benchmark
- pytorch
- python
- pretrained
- multi-engine
---

# causalscale v3.4.0 Replication Package

**12 engines. d=30 to 17,787. 3 modalities. 33 cancers. 92% test coverage.**

> This replication package accompanies the KDD 2027 Datasets & Benchmarks submission.
> It contains all source code, pre-computed results, figures, and one-click reproduction.

## Quick Start (3 commands)

```bash
pip install -e . dagma statsmodels
python run_all.py --verify      # ~30 sec: validates installation + result integrity
python run_all.py --figures     # ~10 sec: regenerates all 4 paper figures
```

## What's Included

| Directory | Contents |
|:--|:--|
| `causalscale/` | Full source code (12 engines, API, CLI, web) |
| `causalscale/pretrained/` | Pre-trained models (DepMap, TCGA, Sachs) + benchmark JSONs |
| `causalscale/core/` | 11 engine implementations (see Engine List below) |
| `results/` | 17 pre-computed result JSON files (all paper tables/figures) |
| `figures/` | 4 paper figures (PDF + PNG for fig3) |
| `scripts/gen_figures/` | 4 figure generation scripts |
| `scripts/` | 6 experiment scripts (benchmark, ablation, scaling, etc.) |
| `examples/` | 4 Jupyter notebooks (biology, finance, neuroscience, drug discovery) |
| `tests/` | 3 test files (92% coverage) |
| `run_all.py` | One-click reproduction script |
| `download_data.py` | External data download helper |
| `pan_cancer_scan.py` | 33-cancer TCGA scan script (Table 4) |

## Reproduction Guide

### Verify Installation (30 seconds)
```bash
python run_all.py --verify
```
Checks: import causalscale, quick d=30 test, result file integrity, figure file integrity, pre-trained models.

### Regenerate All Figures (10 seconds)
```bash
python run_all.py --figures
```
Runs all 4 `scripts/gen_figures/gen_fig*.py` scripts. Reads from `results/` directory.

### Print All Table Data (instant)
```bash
python run_all.py --tables
```
Prints cached results for Tables 1, 4, 5, 6 from `results/*.json`.

### Run Synthetic Benchmark (~15 min on GPU)
```bash
python run_all.py --benchmark
```
Reproduces Table 1 (NOTEARS vs DAGMA vs causalscale, d=30-150, 5 seeds each).

### Run Scaling Experiments (~30 min on GPU)
```bash
python run_all.py --scaling
```
Reproduces Table 5 (LowRankGNN rank sensitivity, r=8-128).

### Full Reproduction (~1 hour on GPU)
```bash
python run_all.py
```
Runs benchmark + scaling + ablation.

## Paper-to-Package Cross-Reference

### Tables
| Paper Table | Data Source | Reproduce With |
|:--|:--|:--|
| Table 1: Synthetic DAG F1 | `results/exp1_causalscale_er.json` + `results/dagma_benchmark.json` | `python run_all.py --benchmark` |
| Table 2: Paired t-test | Computed from Table 1 data | Derivable from `results/exp1_causalscale_er.json` |
| Table 3: Feature comparison | Static (no computation) | In paper TeX |
| Table 4: ARID1A-MTOR | `results/pan_cancer_ckpt.json` | `python pan_cancer_scan.py` (needs TCGA data) |
| Table 5: LowRank scaling | `results/exp9_lowrank_scaling.json` | `python run_all.py --scaling` |
| Table 6: Memory scaling | `results/memory_scaling.json` | Pre-computed |

### Figures
| Paper Figure | Script | Output |
|:--|:--|:--|
| Fig 1: Architecture | `scripts/gen_figures/gen_fig1_architecture.py` | `figures/fig1_architecture.pdf` |
| Fig 2: Benchmark | `scripts/gen_figures/gen_fig2_benchmark.py` | `figures/fig2_benchmark.pdf` |
| Fig 3: ARID1A-MTOR | `scripts/gen_figures/gen_fig3_arid1a_mtor.py` | `figures/fig3_arid1a_mtor.pdf` |
| Fig 4: Timing/Scaling | `scripts/gen_figures/gen_fig4_timing.py` | `figures/fig4_timing.pdf` |

All 4 figure PDFs are pre-bundled in `figures/`. To regenerate: `python run_all.py --figures`.

## 12 Engines

### Core Engines (dimension-based auto-selection)

| Engine | Best For | Method | Key Result |
|:--|:--|:--|:--|
| **dagma** | d <= 150 | DAGMA (Bello et al., NeurIPS 2022) | F1=0.989 @ d=150 |
| **cluster_aware** | d <= 200 | Verified NOTEARS with exact DAG constraint | Exceeds NOTEARS at all d, exceeds DAGMA at d=30 |
| **transformer** | d=200-500 | Causal Transformer (Gao 2026) | 1,028 edges @ d=200, NOTEARS = 0 |
| **lowrank** | d > 500 | LowRankGNN (Gao 2026) | d=17,787, 88.7% STRING/TRRUST precision |

### Specialized Engines

| Engine | Best For | Method | Key Result |
|:--|:--|:--|:--|
| **multibatch** | Multi-dataset | Dataset-specific residual adapters | 0.92 shared-edge precision, 93.3% ASCEND |
| **llm_prior** | External knowledge | STRING-derived edge prior injection | +12% F1 over vanilla NOTEARS |
| **bayes_lowrank** | Uncertainty quantification | Bayesian bootstrap + low-rank NOTEARS | ECE 0.003@d=500, 71.4% BRCA STRING |
| **sc_causal** | Single-cell RNA-seq | NB-LR CI test + PC algorithm | 14.8% PBMC STRING, 35.8% cell-type |
| **transfer** | Cross-dataset | Warm-start NOTEARS/DAGMA/GOLEM | 45/45 wins, 68.1% edge retention @ d=889 |
| **multiscale** | d=500-5,000 | Multi-scale low-rank decomposition | 16x KM enrichment over concatenation |
| **multimodal** | m >= 2 | Cross-modal Frobenius consensus | Multi-omics causal discovery |
| **ensemble** | Any | 3-engine weighted voting | F1 exceeds best single engine by 22-35% |

Plus: **PCMCI** time-series engine (Runge et al., Sci. Adv. 2019).

## External Data (Optional)

The pre-computed results in `results/` do NOT require any external data downloads.
To reproduce experiments that use external biological data:

```bash
python download_data.py --string    # STRING PPI + TRRUST (for validate_against_string)
python download_data.py --tcga       # TCGA gene expression (for pan_cancer_scan.py)
python download_data.py --depmap     # DepMap CRISPR (for genome-scale)
```

Data sources:
- TCGA: UCSC Xena (https://xenabrowser.net/)
- DepMap: Broad Institute (https://depmap.org/, 24Q2 release)
- STRING PPI v12.0: https://string-db.org/
- TRRUST v2: https://www.grnpedia.org/trrust/

## Install

```bash
pip install causalscale
```

GPU auto-detected; CPU fallback supported. Python >=3.10, PyTorch >=2.1.

```bash
# Full reproduction
git clone https://github.com/sgao-academics/causalscale
cd causalscale
pip install -e .
python run_all.py --verify
```

## Citation

```
@inproceedings{gao2027causalscale,
  title={causalscale: Scaling Differentiable Causal Discovery to Genome-Wide Resolution},
  author={Gao, Shuaidong},
  booktitle={Proceedings of the 33rd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year={2027},
  note={Datasets \& Benchmarks Track}
}
```

## License

MIT. Author: Shuaidong Gao (ORCID: 0009-0004-5641-3581).
