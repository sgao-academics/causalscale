"""
Upload causalscale V3.0.0 to HuggingFace Hub.
Repository: sgao-academics/causalscale
Usage: python scripts/upload_v3_hf.py --token YOUR_HF_TOKEN
"""
import os, sys, json, argparse
from pathlib import Path

HF_REPO = "sgao-academics/causalscale"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
PKG_ROOT = Path(__file__).parent.parent
PRETRAINED_DIR = PKG_ROOT / "causalscale" / "pretrained"


def build_v3_readme():
    """Generate V3 model card with all 6 engines."""
    return """---
license: mit
library_name: causalscale
pipeline_tag: causal-discovery
tags:
  - causal-discovery
  - low-rank
  - dag
  - genomics
  - drug-sensitivity
  - counterfactual
  - multimodal
  - transformer
  - iclr-2027
  - cancer-genomics
---

# causalscale V3.0.0 — Unified Causal Discovery Platform

**One line. Six engines. Any scale (d=30 to d=100M).**

```python
import causalscale as cs

model = cs.CausalDiscovery(data)
model.fit()
network = model.get_network()
```

## Six Engines

| Engine | Command | Best For | Scale |
|:--|:--|:--|:--|
| **LowRankGNN** | `method="lowrank"` | Genome-scale d>10K | d=18435 validated |
| **MultiScale** | `method="multi_scale"` | Hierarchical d>200 | W=Sum(U_s@V_s^T) |
| **CAGate/SSCAGate** | `method="cluster_aware"` | Small-sample heterogeneous | 33 TCGA cancers |
| **Causal Transformer** | `method="transformer"` | Nonlinear d>100 | Attention-based |
| **MM-CDSM** | `method="multimodal"` | Multi-omics consensus | Expression+CNV+Methylation |
| **Full** | `method="full"` | Enterprise (UQ+CF) | All of the above |

## Pre-trained Models

| Model | Size | Description |
|:--|:--|:--|
| `depmap_19215.pt` | 252 KB | DepMap genome-wide causal backbone |
| `tcga_pancancer.pt` | 52 KB | TCGA 33-cancer pan-cancer causal network |
| `sachs_protein.pt` | 2 KB | Sachs protein signaling (d=11) |

## Benchmarks

### Synthesized DAG (d=30, n=300)

| Engine | Edges | Time |
|:--|:--|:--|
| lowrank | 68 | 1.2s |
| multi_scale | 58 | 0.2s |
| cluster_aware | 24 | 0.3s |
| transformer | 1* | 2.3s |

*CT designed for d>100; see benchmarks below.

### TCGA Cancer (d=200, 33 cancers)

| Method | Edges | Detection Rate |
|:--|:--|:--|
| CDSM/SSCAGate | 312 avg | 100% |
| NOTEARS (Zheng 2018) | 0 | 0% (collapses at d>150) |

### ARID1A-MTOR Direction (33 cancers)

| Direction | Count |
|:--|:--|
| ARID1A -> MTOR | 27 |
| MTOR -> ARID1A | 5 |
| Borderline | 1 |

## V3 Feature Matrix

| | LowRankGNN | MultiScale | SSCAGate | CT | MM-CDSM | Bootstrap | Counterfactual | AMP |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| causalscale V3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| NOTEARS | x | x | x | x | x | x | x | x |
| DoWhy | x | x | x | x | x | x | ✓ | x |

## Install

```bash
pip install causalscale
```

## Papers Using causalscale

1. **LowRankGNN**: *Low-Rank Factorization Enables Genome-Scale Causal Discovery* — ICLR 2027 (under review)
2. **SSCAGate**: *Tissue-Specific Causal Directionality of ARID1A-MTOR Across 33 Cancers* — Cancer Research (submitted)
3. **Theorem**: *Identifiability and Sample Complexity of Low-Rank Causal Discovery* — AISTATS 2027 (in preparation)

## Citation

```bibtex
@inproceedings{gao2026lowrank,
  title={Low-Rank Factorization Enables Genome-Scale Causal Discovery},
  author={Gao, Shuaidong},
  booktitle={ICLR},
  year={2027}
}
```

## License

Mode `lowrank`: MIT (free).  
Modes `multi_scale`, `cluster_aware`, `transformer`, `multimodal`, `full`: patent-pending, contact for license.

Author: **Shuaidong Gao** (ORCID: 0009-0004-5641-3581)  
Email: sgao.academics@gmail.com  
GitHub: https://github.com/sgao-academics/causalscale
"""


def upload(token):
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub -q")
        from huggingface_hub import HfApi, create_repo, upload_folder

    api = HfApi(token=token)

    # Create model repo
    print(f"Creating repo: {HF_REPO}")
    create_repo(HF_REPO, repo_type="model", exist_ok=True, token=token)

    # 1. Upload README
    print("\n[1/4] Uploading model card...")
    card = build_v3_readme()
    card_path = PKG_ROOT / "HF_README.md"
    card_path.write_text(card, encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=HF_REPO, repo_type="model", token=token,
    )
    card_path.unlink()
    print("  README.md OK")

    # 2. Upload pretrained models
    print("\n[2/4] Uploading pretrained models...")
    for pt_file in PRETRAINED_DIR.glob("*.pt"):
        sz = pt_file.stat().st_size / 1024
        print(f"  {pt_file.name} ({sz:.0f} KB)")
        api.upload_file(
            path_or_fileobj=str(pt_file),
            path_in_repo=f"pretrained/{pt_file.name}",
            repo_id=HF_REPO, repo_type="model", token=token,
        )

    # 3. Upload benchmarks
    print("\n[3/4] Uploading benchmarks...")
    for json_file in sorted(PRETRAINED_DIR.glob("*.json")):
        sz = json_file.stat().st_size / 1024
        print(f"  {json_file.name} ({sz:.1f} KB)")
        api.upload_file(
            path_or_fileobj=str(json_file),
            path_in_repo=f"benchmarks/{json_file.name}",
            repo_id=HF_REPO, repo_type="model", token=token,
        )

    # 4. Upload core source (entire causalscale package)
    print("\n[4/4] Uploading source code...")
    src_dir = PKG_ROOT / "causalscale"
    for py_file in sorted(src_dir.rglob("*.py")):
        rel_path = py_file.relative_to(PKG_ROOT).as_posix()
        print(f"  {rel_path}")
        api.upload_file(
            path_or_fileobj=str(py_file),
            path_in_repo=rel_path,
            repo_id=HF_REPO, repo_type="model", token=token,
        )

    # Also upload setup.py
    print("  setup.py")
    api.upload_file(
        path_or_fileobj=str(PKG_ROOT / "setup.py"),
        path_in_repo="setup.py",
        repo_id=HF_REPO, repo_type="model", token=token,
    )

    # Also upload examples
    examples_dir = PKG_ROOT / "examples"
    for nb in sorted(examples_dir.glob("*.ipynb")):
        rel_path = nb.relative_to(PKG_ROOT).as_posix()
        print(f"  {rel_path}")
        api.upload_file(
            path_or_fileobj=str(nb),
            path_in_repo=rel_path,
            repo_id=HF_REPO, repo_type="model", token=token,
        )

    print(f"\n=== UPLOAD COMPLETE ===")
    print(f"https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    token = os.getenv("HF_TOKEN") or HF_TOKEN
    upload(token)
