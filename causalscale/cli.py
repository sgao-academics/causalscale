"""
causalscale CLI — Command-line causal discovery.

Usage:
    causalscale fit data.csv                    # discover causal network
    causalscale drug --gene-list TP53,ABCB1     # drug sensitivity prediction
    causalscale gene data.csv                   # gene regulatory network
    causalscale finance returns.csv             # financial causal graph
    causalscale models                          # list pre-trained models
    causalscale benchmarks                      # list benchmark results
    causalscale web                             # launch Streamlit web UI
"""

import argparse
import sys
import numpy as np
from pathlib import Path


def cmd_fit(args):
    """Run causal discovery on data file."""
    import causalscale as cs

    data = _load_data(args.data)
    print(f"Data: {data.shape[0]} samples x {data.shape[1]} variables")
    model = cs.CausalDiscovery(
        data, method=args.method, rank=args.rank, device=args.device
    )
    model.fit(verbose=True)
    net = model.get_network(top_k=args.top_k)

    print(f"\n{'='*50}")
    print(f"Discovered {net.edge_count} causal edges")
    if net.time_s:
        print(f"Time: {net.time_s:.1f}s")
    print(f"\nTop {min(args.top_k, len(net.edges))} edges:")
    for src, tgt, w in net.edges[: min(args.top_k, len(net.edges))]:
        arrow = "->" if w > 0 else "-|"
        print(f"  {src} {arrow} {tgt}: {w:+.4f}")

    if args.output:
        np.savetxt(args.output, net.adjacency, delimiter=",")
        print(f"\nAdjacency matrix saved to {args.output}")


def cmd_drug(args):
    """Predict drug sensitivity."""
    from causalscale.utils import make_synthetic_dag
    from causalscale.apps.drug_sensitivity import predict_drug_sensitivity

    genes = [g.strip() for g in args.gene_list.split(",")]
    X, _ = make_synthetic_dag(d=len(genes), n=300, edge_prob=0.02, seed=42)
    drug_data = np.random.randn(300, args.n_drugs).astype(np.float32) * 0.3

    result = predict_drug_sensitivity(
        expression=X,
        gene_list=genes,
        drug_data=drug_data,
        rank=min(32, len(genes) // 2),
        epochs=200,
        device=args.device,
        verbose=True,
    )

    print(f"\nCRISPR prediction r: {result['crispr_r']:.4f}")
    print(f"Mean drug r: {result.get('mean_drug_r', 'N/A'):.4f}")
    if "n_drugs_r_gt_05" in result:
        print(f"Drugs with r > 0.5: {result['n_drugs_r_gt_05']}/{args.n_drugs}")
    if "top_predictions" in result:
        print("\nTop drug predictions:")
        for p in result["top_predictions"][:10]:
            print(f"  Drug_{p['drug_index']}: r={p['r']:.4f}")


def cmd_gene(args):
    """Gene regulatory network analysis."""
    from causalscale.apps.gene_network import gene_causal_network

    data = _load_data(args.data)
    if args.genes is None:
        gene_names = [f"GENE_{i}" for i in range(data.shape[1])]
    else:
        gene_names = [g.strip() for g in args.genes.split(",")]

    net = gene_causal_network(
        data, gene_names, rank=args.rank, device=args.device, verbose=True
    )
    print(f"\nHub genes:")
    for h in net["hub_genes"][:10]:
        print(f"  {h['gene']}: {h['out_degree']} downstream targets")
    print(f"\nNetwork: {net['stats']['n_edges']} edges, "
          f"density={net['stats']['density']:.4f}")


def cmd_finance(args):
    """Financial causal discovery."""
    from causalscale.apps.finance import finance_causal_graph

    data = _load_data(args.data)
    tickers = [f"A{i}" for i in range(data.shape[1])]
    graph = finance_causal_graph(
        data, tickers=tickers, method=args.method, rank=args.rank, device=args.device, verbose=True
    )
    print(f"\nFound {graph['n_edges']} causal edges")
    print("Top influencers:")
    for inf in graph["top_influencers"][:5]:
        print(f"  {inf['ticker']}: influences {inf['influence']} others")


def cmd_models(args):
    """List pre-trained models."""
    from causalscale.pretrained import list_models, list_benchmarks

    print("Pre-trained Models:")
    for name, info in list_models().items():
        print(f"  {name}: {info}")
    print("\nBenchmark Results:")
    for name, info in list_benchmarks().items():
        print(f"  {name}: {info}")


def cmd_web(args):
    """Launch web UI."""
    try:
        import streamlit
    except ImportError:
        print("Streamlit not installed. Install: pip install streamlit")
        sys.exit(1)
    import subprocess

    app_path = Path(__file__).parent / "web" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


def _load_data(path):
    """Load data from CSV/TSV file."""
    import pandas as pd

    df = pd.read_csv(path, sep=None, engine="python", index_col=0)
    return df.values.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="causalscale: One-line causal discovery at scale",
        prog="causalscale",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # fit
    p_fit = subparsers.add_parser("fit", help="Discover causal network")
    p_fit.add_argument("data", help="CSV/TSV data file")
    p_fit.add_argument("--method", default="lowrank", choices=["lowrank", "notears", "cluster_aware"])
    p_fit.add_argument("--rank", type=int, default=64)
    p_fit.add_argument("--top-k", type=int, default=20)
    p_fit.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p_fit.add_argument("--output", "-o", help="Save adjacency matrix to CSV")

    # drug
    p_drug = subparsers.add_parser("drug", help="Drug sensitivity prediction")
    p_drug.add_argument("--gene-list", default="TP53,ABCB1,EGFR,MYC,BRCA1")
    p_drug.add_argument("--n-drugs", type=int, default=50)
    p_drug.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    # gene
    p_gene = subparsers.add_parser("gene", help="Gene regulatory network")
    p_gene.add_argument("data", help="Expression data CSV/TSV")
    p_gene.add_argument("--genes", help="Comma-separated gene names")
    p_gene.add_argument("--rank", type=int, default=64)
    p_gene.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    # finance
    p_fin = subparsers.add_parser("finance", help="Financial causal discovery")
    p_fin.add_argument("data", help="Returns data CSV/TSV")
    p_fin.add_argument("--method", default="lowrank", choices=["lowrank", "granger"])
    p_fin.add_argument("--rank", type=int, default=32)
    p_fin.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    # models
    subparsers.add_parser("models", help="List pre-trained models and benchmarks")

    # web
    subparsers.add_parser("web", help="Launch Streamlit web UI")

    args = parser.parse_args()

    commands = {
        "fit": cmd_fit,
        "drug": cmd_drug,
        "gene": cmd_gene,
        "finance": cmd_finance,
        "models": cmd_models,
        "web": cmd_web,
    }

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands[args.command](args)


if __name__ == "__main__":
    main()
