# Pan-cancer ARID1A-MTOR scan via causalscale cluster_aware
# Output: ./results/pan_cancer_ckpt.json
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# Add causalscale to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causalscale as cs

TCGA_DIR = os.environ.get("TCGA_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(OUTPUT_DIR, "pan_cancer_ckpt.json")

D = 100  # top D variable genes (matched to original analysis)
TARGET_GENES = ["ARID1A", "MTOR"]

# Load checkpoint
if os.path.exists(CKPT_PATH):
    with open(CKPT_PATH, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
    print(f"Resuming from checkpoint: {len(ckpt)} cancers done")
else:
    ckpt = {}

# Scan all TCGA files
tsv_files = sorted([
    f for f in os.listdir(TCGA_DIR)
    if f.startswith("TCGA_") and f.endswith("_HiSeqV2.tsv")
])
cancer_codes = [f.replace("TCGA_", "").replace("_HiSeqV2.tsv", "") for f in tsv_files]

total = len(cancer_codes)
for idx, (cancer, tsv) in enumerate(zip(cancer_codes, tsv_files)):
    if cancer in ckpt:
        print(f"[{idx+1}/{total}] {cancer}: SKIP (already done)")
        continue

    t0 = time.time()
    print(f"[{idx+1}/{total}] {cancer}: loading...", end=" ", flush=True)

    try:
        # Load and preprocess
        df = pd.read_csv(os.path.join(TCGA_DIR, tsv), sep="\t", index_col=0)
        # Transpose: genes=columns, samples=rows
        if df.shape[0] > df.shape[1]:
            df = df.T

        # Select top D variable genes + ensure ARID1A and MTOR present
        present_targets = [g for g in TARGET_GENES if g in df.columns]
        var_genes = df.var().nlargest(D).index.tolist()
        selected = list(dict.fromkeys(present_targets + [g for g in var_genes if g not in TARGET_GENES][:D]))

        X = df[selected].values.astype(np.float32)
        n, d = X.shape
        print(f"n={n}, d={d}, ARID1A_idx={selected.index('ARID1A') if 'ARID1A' in selected else 'MISSING'}, MTOR_idx={selected.index('MTOR') if 'MTOR' in selected else 'MISSING'}", end=" ", flush=True)

        # Run causalscale
        model = cs.CausalDiscovery(X, method="cluster_aware", device="cuda",
                                   var_names=selected)
        model.fit(verbose=False)
        net = model.get_network()

        # Extract ARID1A<->MTOR
        arid1a_to_mtor = 0.0
        mtor_to_arid1a = 0.0
        adj = net.adjacency
        for i in range(d):
            for j in range(d):
                if i != j and abs(adj[i, j]) > 0.3:
                    if selected[i] == "ARID1A" and selected[j] == "MTOR":
                        arid1a_to_mtor = float(adj[i, j])
                    if selected[i] == "MTOR" and selected[j] == "ARID1A":
                        mtor_to_arid1a = float(adj[i, j])

        result = {
            "n": n, "d": d,
            "edge_count": net.edge_count,
            "time_s": time.time() - t0,
            "arid1a_to_mtor": arid1a_to_mtor,
            "mtor_to_arid1a": mtor_to_arid1a,
            "net": arid1a_to_mtor - mtor_to_arid1a,
            "arid1a_idx": selected.index("ARID1A") if "ARID1A" in selected else -1,
            "mtor_idx": selected.index("MTOR") if "MTOR" in selected else -1,
        }
        print(f"-> A->M={arid1a_to_mtor:.4f}, M->A={mtor_to_arid1a:.4f}, edges={net.edge_count}, {result['time_s']:.1f}s")

    except Exception as e:
        result = {"error": str(e), "time_s": time.time() - t0}
        print(f"ERROR: {e}")

    ckpt[cancer] = result
    # Save checkpoint after every cancer
    with open(CKPT_PATH, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, indent=2, ensure_ascii=False)

print(f"\nDone. {len(ckpt)}/{total} cancers processed.")
print(f"Output: {CKPT_PATH}")
