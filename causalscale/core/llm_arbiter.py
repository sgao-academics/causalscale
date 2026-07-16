"""
LLM Arbitration Module for causalscale ensemble.
Debates disagreement edges using DeepSeek/OpenAI API.
"""
import json, os, urllib.request
from typing import Optional, Dict, List, Tuple

_DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
_OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


def _call_llm(
    prompt: str,
    model: str = "deepseek-chat",
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 100,
) -> Optional[str]:
    """Call LLM API. Returns response text or None on failure."""
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        return None

    api_url = api_url or (_OPENAI_API_URL if "gpt" in model.lower() else _DEEPSEEK_API_URL)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        req = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return None


def _simulate_llm(
    gene_i: str, gene_j: str, string_pairs: set, trrust_pairs: set
) -> bool:
    """Simulate LLM response using STRING/TRRUST as ground truth.
    In production, this is replaced by real LLM API calls.
    Returns True if edge should be kept."""
    pair = (gene_i.upper(), gene_j.upper())
    return pair in string_pairs or pair in trrust_pairs


def arbitrate_disagreement_edges(
    disagreement_edges: List[Tuple[str, str, float]],
    var_names: List[str],
    string_data_dir: Optional[str] = None,
    use_llm: bool = False,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    model: str = "deepseek-chat",
    verbose: bool = True,
) -> Dict:
    """Arbitrate disagreement edges using LLM domain knowledge or STRING/TRRUST.

    Called after ensemble voting identifies edges where engines disagree.
    The LLM acts as a domain expert, voting "keep" or "drop" based on
    biological knowledge.

    Args:
        disagreement_edges: list of (source, target, mean_weight) tuples
        var_names: gene symbols or variable names
        string_data_dir: path to STRING/TRRUST for simulation mode
        use_llm: if True, call real LLM API. If False, use STRING simulation.
        api_key: LLM API key
        api_url: LLM API endpoint
        model: LLM model name
        verbose: print progress

    Returns:
        dict: kept_edges, dropped_edges, kept_count, dropped_count,
              total, keep_rate, mode ('llm' or 'simulated')
    """
    # ── Load STRING/TRRUST for simulated mode ──
    string_pairs = set()
    trrust_pairs = set()

    if not use_llm and string_data_dir and os.path.exists(string_data_dir):
        import gzip
        info_path = os.path.join(string_data_dir, "string_info.txt.gz")
        ppi_path = os.path.join(string_data_dir, "string_ppi_full.txt.gz")
        trrust_path = os.path.join(string_data_dir, "trrust_human.tsv")

        if os.path.exists(info_path) and os.path.exists(ppi_path):
            ensp2sym = {}
            with gzip.open(info_path, "rt", encoding="utf-8", errors="ignore") as f:
                next(f)
                for line in f:
                    p = line.strip().split("\t")
                    if len(p) >= 2:
                        eid = p[0]; sym = p[1].strip()
                        ensp2sym[eid] = sym
                        if eid.startswith("9606."):
                            ensp2sym[eid[5:]] = sym
            with gzip.open(ppi_path, "rt", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) >= 2:
                        s1 = ensp2sym.get(p[0]); s2 = ensp2sym.get(p[1])
                        if s1 and s2:
                            string_pairs.add((s1, s2))

        if os.path.exists(trrust_path):
            with open(trrust_path, encoding="utf-8") as f:
                for line in f:
                    p = line.strip().split("\t")
                    if len(p) >= 2:
                        trrust_pairs.add((p[0].upper(), p[1].upper()))

    kept = []
    dropped = []

    for src, tgt, w in disagreement_edges:
        if use_llm:
            prompt = (
                f"You are a molecular biologist evaluating causal gene relationships.\n"
                f"Based on known biology, does {src} causally regulate (upstream of) {tgt}?\n"
                f"Answer ONLY 'yes' or 'no'. Consider: transcription factor-target, "
                f"signaling pathway, protein-protein interaction.\n"
                f"Gene pair: {src} -> {tgt}"
            )
            response = _call_llm(prompt, model=model, api_key=api_key, api_url=api_url)
            keep = response and "yes" in response.lower()
        else:
            keep = _simulate_llm(src, tgt, string_pairs, trrust_pairs)

        if keep:
            kept.append((src, tgt, w))
        else:
            dropped.append((src, tgt, w))

    result = {
        "kept_edges": kept,
        "dropped_edges": dropped,
        "kept_count": len(kept),
        "dropped_count": len(dropped),
        "total": len(disagreement_edges),
        "keep_rate": round(len(kept) / max(len(disagreement_edges), 1), 4),
        "mode": "llm" if use_llm else "simulated",
    }

    if verbose and disagreement_edges:
        print(f"  LLM Arbitration ({result['mode']}): "
              f"{result['kept_count']} kept, {result['dropped_count']} dropped "
              f"({result['keep_rate']:.1%} keep rate)")
        if kept[:5]:
            print(f"  Top kept: {', '.join(f'{s}->{t}' for s,t,_ in kept[:5])}")

    return result
