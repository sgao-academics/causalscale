"""
Upload causalscale to HuggingFace Hub.

Usage:
    python scripts/upload_huggingface.py --token YOUR_HF_TOKEN

This uploads:
    - All pre-trained models (.pt files) to sgao-academics/causalscale
    - Model card with benchmark results and usage examples

Requires: pip install huggingface_hub
"""

import os, sys, json, argparse
from pathlib import Path

HF_REPO = "sgao-academics/causalscale"
PKG_ROOT = Path(__file__).parent.parent
PRETRAINED_DIR = PKG_ROOT / "causalscale" / "pretrained"


def create_model_card():
    """Generate HuggingFace model card."""
    from causalscale.pretrained import list_models, list_benchmarks, load_benchmark

    sota = load_benchmark("sota")

    card = f"""---
license: mit
library_name: causalscale
tags:
  - causal-discovery
  - low-rank
  - dag
  - genomics
  - drug-sensitivity
  - iclr-2027
---

# causalscale: One-Line Causal Discovery Engine

Pre-trained causal backbones from the LowRankGNN paper (ICLR 2027).

## Models

| Model | Size | Description |
|:--|:--|:--|
{depmap_row}|
| tcga | 52 KB | TCGA pancancer causal network (d=200 proxy) |
| sachs | 2 KB | Sachs protein signaling (d=11, n=853) |

{_model_rows()}

## Benchmarks

{_benchmark_table(sota)}

## Quick Start

```python
import causalscale as cs

# One line: upload data -> causal network
model = cs.CausalDiscovery(data, method="lowrank")
model.fit()
network = model.get_network()

# Or load a pre-trained backbone
from causalscale.pretrained import load_model
sachs = load_model("sachs")
W = sachs["U"] @ sachs["V"].T
```

## Scale

| d | LowRankGNN F1 | NOTEARS F1 | Time |
|:--|:--|:--|:--|
{_scale_rows(sota)}

## Citation

```bibtex
@inproceedings{{gao2026lowrank,
  title={{Low-Rank Factorization Enables Genome-Scale Causal Discovery}},
  author={{Gao, Shuaidong}},
  booktitle={{ICLR}},
  year={{2027}}
}}
```

## Install

```bash
pip install causalscale
```

Author: Shuaidong Gao (ORCID: 0009-0004-5641-3581)
"""
    return card


def _model_rows():
    from causalscale.pretrained import list_models

    rows = []
    for name, info in list_models().items():
        rows.append(f"| {name} | {info.split('(')[1].replace(')','')} |")
    return "\n".join(rows) if rows else "| depmap | 252 KB | DepMap genomic causal network (d=500 proxy) |"


def _benchmark_table(sota):
    rows = ["| d | LowRankGNN F1 | NOTEARS F1 | Time |",
            "|:--|:--|:--|:--|"]
    for r in sota[:5]:
        rows.append(
            f"| {r['d']} | {r['lowrank_gnn']['f1']:.3f} | "
            f"{r['notears']['f1']:.3f} | {r['lowrank_gnn']['time_s']:.1f}s |"
        )
    return "\n".join(rows)


def _scale_rows(sota):
    rows = []
    for r in sota[:5]:
        rows.append(
            f"| {r['d']} | {r['lowrank_gnn']['f1']:.3f} | "
            f"{r['notears']['f1']:.3f} | {r['lowrank_gnn']['time_s']:.1f}s |"
        )
    return "\n".join(rows)


def upload(token):
    """Upload to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    sys.path.insert(0, str(PKG_ROOT))

    api = HfApi(token=token)

    # Create repo if not exists
    print(f"Creating/accessing repo: {HF_REPO}")
    create_repo(HF_REPO, repo_type="model", exist_ok=True, token=token)

    # Upload .pt files
    print("\nUploading pre-trained models...")
    for pt_file in PRETRAINED_DIR.glob("*.pt"):
        print(f"  {pt_file.name} ({pt_file.stat().st_size / 1024:.0f} KB)")
        api.upload_file(
            path_or_fileobj=str(pt_file),
            path_in_repo=f"pretrained/{pt_file.name}",
            repo_id=HF_REPO,
            repo_type="model",
            token=token,
        )

    # Upload benchmark JSONs
    print("\nUploading benchmark results...")
    for json_file in sorted(PRETRAINED_DIR.glob("*.json")):
        print(f"  {json_file.name} ({json_file.stat().st_size / 1024:.1f} KB)")
        api.upload_file(
            path_or_fileobj=str(json_file),
            path_in_repo=f"benchmarks/{json_file.name}",
            repo_id=HF_REPO,
            repo_type="model",
            token=token,
        )

    # Upload model card
    print("\nGenerating and uploading model card...")
    card = create_model_card()
    card_path = PKG_ROOT / "HF_README.md"
    card_path.write_text(card, encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=HF_REPO,
        repo_type="model",
        token=token,
    )
    card_path.unlink()

    print(f"\nUpload complete: https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload causalscale to HuggingFace Hub.\n"
        "Get your token at: https://huggingface.co/settings/tokens\n"
        "Then run: python scripts/upload_huggingface.py --token YOUR_TOKEN\n"
        "Or export: HF_TOKEN=your_token && python scripts/upload_huggingface.py"
    )
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"),
                        help="HuggingFace API token (or set HF_TOKEN env var)")
    args = parser.parse_args()
    if not args.token:
        parser.error(
            "No token provided. Either:\n"
            "  1. Set HF_TOKEN environment variable, or\n"
            '  2. Pass --token YOUR_TOKEN\n'
            "Get a token at: https://huggingface.co/settings/tokens"
        )
    upload(args.token)
