"""
Upload ALL causalscale v3.1.0 files to HuggingFace Hub.
Syncs package code, examples, benchmarks, pretrained models, and README.
Usage: $env:HF_TOKEN='hf_...'; python scripts/upload_v31_full.py
"""
import os, sys, json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder, upload_file

HF_REPO = "sgao-academics/causalscale"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
PKG_ROOT = Path(__file__).parent.parent

if not HF_TOKEN:
    print("ERROR: Set HF_TOKEN environment variable.")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

# Ensure repo exists
create_repo(HF_REPO, token=HF_TOKEN, exist_ok=True)
print(f"Repo: {HF_REPO}\n")

# ── 1. Package source code ──
print("=== Package source (causalscale/) ===")
upload_folder(
    folder_path=str(PKG_ROOT / "causalscale"),
    path_in_repo="causalscale",
    repo_id=HF_REPO,
    token=HF_TOKEN,
    commit_message="v3.1.0: package source code",
    ignore_patterns=["__pycache__", "*.pyc", "*.pyo", "pretrained/*.pt",
                     "pretrained/*.json", "pretrained/README.md"],
)
print("  causalscale/ uploaded")

# ── 2. setup.py ──
print("\n=== setup.py ===")
upload_file(
    path_or_fileobj=str(PKG_ROOT / "setup.py"),
    path_in_repo="setup.py",
    repo_id=HF_REPO,
    token=HF_TOKEN,
    commit_message="v3.1.0: setup.py (version 3.1.0)"
)
print("  setup.py uploaded")

# ── 3. Examples ──
examples_dir = PKG_ROOT / "examples"
if examples_dir.exists():
    print(f"\n=== Examples ===")
    upload_folder(
        folder_path=str(examples_dir),
        path_in_repo="examples",
        repo_id=HF_REPO,
        token=HF_TOKEN,
        commit_message="v3.1.0: example notebooks",
    )
    print("  examples/ uploaded")
else:
    print("\n  [skip] examples/ not found")

# ── 4. Benchmarks ──
bench_dir = PKG_ROOT / "benchmarks"
if bench_dir.exists():
    print(f"\n=== Benchmarks ===")
    upload_folder(
        folder_path=str(bench_dir),
        path_in_repo="benchmarks",
        repo_id=HF_REPO,
        token=HF_TOKEN,
        commit_message="v3.1.0: benchmark scripts",
    )
    print("  benchmarks/ uploaded")

# ── 5. Pretrained models + benchmark JSONs ──
PRETRAINED_DIR = PKG_ROOT / "causalscale" / "pretrained"
print("\n=== Pretrained models ===")
for m in ["depmap_19215.pt", "tcga_pancancer.pt", "sachs_protein.pt"]:
    path = PRETRAINED_DIR / m
    if path.exists():
        upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"pretrained/{m}",
            repo_id=HF_REPO,
            token=HF_TOKEN,
            commit_message=f"v3.1.0: {m}"
        )
        print(f"  pretrained/{m}")

print("\n=== Benchmark data ===")
# Upload key results from replication package
repl_dir = PKG_ROOT.parent / "causalscale_Replication_Package" / "results"
for r in ["exp1_causalscale_er.json", "dagma_benchmark.json", "pan_cancer_ckpt.json"]:
    path = repl_dir / r
    if path.exists():
        upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"benchmarks/{r}",
            repo_id=HF_REPO,
            token=HF_TOKEN,
            commit_message=f"v3.1.0: {r}"
        )
        print(f"  benchmarks/{r}")
# ── 6. Root README ──
print("\n=== README ===")
upload_file(
    path_or_fileobj=str(PKG_ROOT / "README.md"),
    path_in_repo="README.md",
    repo_id=HF_REPO,
    token=HF_TOKEN,
    commit_message="v3.1.0: README"
)
print("  README.md uploaded")

# -- 7. Model card --
print("\n=== Model Card ===")
card_md = PRETRAINED_DIR / "README.md"
card = card_md.read_text(encoding="utf-8")
upload_file(
    path_or_fileobj=str(card_md),
    path_in_repo="README.md",
    repo_id=HF_REPO,
    token=HF_TOKEN,
    commit_message="v3.1.0: updated model card with 7 engines + KDD results"
)
print("  Model card uploaded")
print(f"\nDone! https://huggingface.co/{HF_REPO}")
