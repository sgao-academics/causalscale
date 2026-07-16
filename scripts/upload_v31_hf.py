"""
Upload causalscale v3.1.0 to HuggingFace Hub.
Usage: HF_TOKEN=your_token python scripts/upload_v31_hf.py
"""
import os, sys, json
from pathlib import Path
from huggingface_hub import HfApi, upload_file, create_repo

HF_REPO = "sgao-academics/causalscale"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
PKG_ROOT = Path(__file__).parent.parent
PRETRAINED_DIR = PKG_ROOT / "causalscale" / "pretrained"

if not HF_TOKEN:
    print("ERROR: Set HF_TOKEN environment variable.")
    print("  $env:HF_TOKEN='hf_...'; python scripts/upload_v31_hf.py")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

# Ensure repo exists
try:
    create_repo(HF_REPO, token=HF_TOKEN, exist_ok=True)
    print(f"Repo: {HF_REPO}")
except Exception as e:
    print(f"Repo check: {e}")

# Upload pretrained models
models = ["depmap_19215.pt", "tcga_pancancer.pt", "sachs_protein.pt"]
for m in models:
    path = PRETRAINED_DIR / m
    if path.exists():
        upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"pretrained/{m}",
            repo_id=HF_REPO,
            token=HF_TOKEN,
            commit_message=f"v3.1.0: {m}"
        )
        print(f"Uploaded: pretrained/{m}")

# Upload benchmark JSONs
benchmarks = ["sota_bench.json", "mega_33_full.json", "tcga_d200_10seed.json",
              "gap23_results.json", "extreme_scale.json"]
for b in benchmarks:
    path = PRETRAINED_DIR / b
    if path.exists():
        upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"pretrained/{b}",
            repo_id=HF_REPO,
            token=HF_TOKEN,
            commit_message=f"v3.1.0: {b}"
        )
        print(f"Uploaded: pretrained/{b}")

# Upload README
readme_path = PKG_ROOT / "README.md"
if readme_path.exists():
    upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=HF_REPO,
        token=HF_TOKEN,
        commit_message="v3.1.0: README with honest benchmarks"
    )
    print("Uploaded: README.md")

# Update model card
card = """---
license: mit
library_name: causalscale
pipeline_tag: causal-discovery
tags:
- causal-discovery
- genomics
- dag
- low-rank
- ensemble
- note-ars
- string-db
---

# causalscale v3.1.0

**Unified Causal Discovery Platform — 6 engines, honest benchmarks, STRING/TRRUST validated.**

## Pretrained Models

| Model | Description |
|:--|:--|
| `depmap_19215.pt` | Low-rank causal network on DepMap 24Q2 (CRISPR CERES, 17,787 genes) |
| `tcga_pancancer.pt` | Pan-cancer causal edges across 33 TCGA cancer types (d=100 genes) |
| `sachs_protein.pt` | Sachs protein signaling network (d=11) |

## Benchmarks (v3.1, honest numbers)

| d | NOTEARS F1 | causalscale F1 | Advantage |
|:--|:--|:--|:--|
| 30 | 0.581 | 0.586 | +1% |
| 50 | 0.475 | 0.531 | +12% |
| 100 | 0.185 | 0.462 | +150% |

**Biological validation**: 93.3% STRING/TRRUST precision (ASCEND two-tier, DepMap).

## Quick Start

```python
import causalscale as cs

# Load pretrained model
model = cs.load_model("depmap")
print(model.summary())  # 28 edges, GO/KEGG annotated

# Or run discovery from scratch
import numpy as np
data = np.random.randn(500, 50)
disc = cs.CausalDiscovery(data)
disc.fit()
report = disc.validate()  # auto-detect evaluation mode
```

## License

MIT. Author: Shuaidong Gao (ORCID: 0009-0004-5641-3581).
"""

card_path = PRETRAINED_DIR / "README.md"
card_path.write_text(card, encoding="utf-8")
upload_file(
    path_or_fileobj=str(card_path),
    path_in_repo="README.md",
    repo_id=HF_REPO,
    token=HF_TOKEN,
    commit_message="v3.1.0 model card"
)
print("Uploaded: model card")

print("\nDone! https://huggingface.co/sgao-academics/causalscale")
