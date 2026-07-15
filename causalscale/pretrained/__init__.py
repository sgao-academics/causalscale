"""Pre-trained model loading and benchmark data.

Models:
    load_model("depmap")  -> DepMap genomic causal network (d=500 proxy)
    load_model("tcga")    -> TCGA pancancer causal network (d=200 proxy)
    load_model("sachs")   -> Sachs protein signaling (d=11)

Benchmark JSONs:
    load_benchmark("sota")      -> SOTA benchmark (d=30-200 vs NOTEARS)
    load_benchmark("mega_33")   -> 33 TCGA cancers head-to-head
    load_benchmark("tcga_d200") -> TCGA d=200 10-seed results
    load_benchmark("gap23")     -> Zero-shot transfer results
    load_benchmark("extreme")   -> Extreme scale (d=2K-19K)
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Optional

_PRETRAINED_DIR = os.path.dirname(__file__)

_MODEL_MAP = {
    "depmap": "depmap_19215.pt",
    "tcga": "tcga_pancancer.pt",
    "sachs": "sachs_protein.pt",
}

_BENCHMARK_MAP = {
    "sota": "sota_bench.json",
    "mega_33": "mega_33_full.json",
    "tcga_d200": "tcga_d200_10seed.json",
    "gap23": "gap23_results.json",
    "extreme": "extreme_scale.json",
}


def list_models() -> Dict[str, str]:
    """List available pre-trained models."""
    result = {}
    for name, fname in _MODEL_MAP.items():
        path = os.path.join(_PRETRAINED_DIR, fname)
        if os.path.exists(path):
            sz = os.path.getsize(path) / 1024
            result[name] = f"{fname} ({sz:.0f} KB)"
        else:
            result[name] = f"{fname} (NOT FOUND)"
    return result


_HF_REPO = "sgao-academics/causalscale"


def _download_from_hf(filename: str, dest_dir: str) -> bool:
    """Attempt to download a file from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=_HF_REPO,
            filename=f"pretrained/{filename}",
            local_dir=dest_dir,
            local_dir_use_symlinks=False,
        )
        return True
    except ImportError:
        raise RuntimeError(
            f"Model '{filename}' not found locally, and huggingface_hub is not installed. "
            f"Install it: pip install huggingface_hub"
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Model '{filename}' not found locally or on HuggingFace Hub. "
            f"Please download manually from https://huggingface.co/{_HF_REPO}\n"
            f"Error: {e}"
        )


def load_model(name: str) -> Dict:
    """Load a pre-trained causal backbone.

    If the model is not found locally, it will be automatically
    downloaded from HuggingFace Hub (sgao-academics/causalscale).

    Args:
        name: 'depmap', 'tcga', or 'sachs'

    Returns:
        dict with U, V, rank, d, model_name, description
    """
    filename = _MODEL_MAP.get(name)
    if filename is None:
        available = ", ".join(_MODEL_MAP.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    path = os.path.join(_PRETRAINED_DIR, filename)

    # Auto-download from HuggingFace if not found locally
    if not os.path.exists(path):
        print(f"Model '{name}' not found locally. Downloading from HuggingFace Hub...")
        _download_from_hf(filename, _PRETRAINED_DIR)
        # The file should be at path now; verify
        if not os.path.exists(path):
            # HF may have placed it in a subdirectory
            alt_path = os.path.join(_PRETRAINED_DIR, "pretrained", filename)
            if os.path.exists(alt_path):
                path = alt_path

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pre-trained model not found: {path}. "
            f"Download from: https://huggingface.co/{_HF_REPO}"
        )

    state = torch.load(path, map_location="cpu", weights_only=False)
    return state


def load_benchmark(name: str) -> Dict:
    """Load a benchmark results JSON.

    Args:
        name: 'sota', 'mega_33', 'tcga_d200', 'gap23', 'extreme'

    Returns:
        Parsed JSON dict
    """
    filename = _BENCHMARK_MAP.get(name)
    if filename is None:
        available = ", ".join(_BENCHMARK_MAP.keys())
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")

    path = os.path.join(_PRETRAINED_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Benchmark not found: {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_benchmarks() -> Dict[str, str]:
    """List available benchmark files."""
    result = {}
    for name, fname in _BENCHMARK_MAP.items():
        path = os.path.join(_PRETRAINED_DIR, fname)
        if os.path.exists(path):
            sz = os.path.getsize(path) / 1024
            result[name] = f"{fname} ({sz:.1f} KB)"
        else:
            result[name] = f"{fname} (NOT FOUND)"
    return result
