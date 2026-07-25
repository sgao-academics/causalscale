#!/usr/bin/env python
"""Download external datasets needed for full reproduction.

Not all datasets can be bundled in this package due to size:
- STRING PPI: ~154 MB (string_ppi_full.txt.gz + string_info.txt.gz)
- TRRUST: ~1 MB (trrust_human.tsv)
- TCGA gene expression: ~500 MB per cancer type (33 types)
- DepMap CRISPR: ~200 MB

Usage:
    python download_data.py --string     # Download STRING + TRRUST
    python download_data.py --tcga        # Download TCGA (33 cancer types)
    python download_data.py --depmap      # Download DepMap CRISPR
    python download_data.py --all         # Download everything

Note: These downloads may take 10-60 minutes depending on network speed.
If you only want to verify the code works, run: python run_all.py --verify
The pre-computed results in results/ do NOT require downloading these datasets.
"""
import sys, os, urllib.request, zipfile, gzip

PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PKG_ROOT, 'causalscale', 'pretrained', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

STRING_BASE = "https://stringdb-downloads.org/download/protein.links.v12.0"
STRING_FULL = "https://stringdb-downloads.org/download/protein.links.full.v12.0/9606.protein.links.full.v12.0.txt.gz"
STRING_INFO = "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"
TRRUST_URL = "https://www.grnpedia.org/trrust/data/trrust_human.tsv"

TCGA_BASE = "https://toil.xenahubs.net/download/tcga_RSEM_gene_fpkm"

DEPMAP_URL = "https://depmap.org/portal/download/all/?download=CRISPR+Gene+Effect&release=Q2+2024"

def download(url, dest, desc=""):
    if os.path.exists(dest):
        sz = os.path.getsize(dest) / 1024 / 1024
        print(f"  SKIP (already exists): {dest} ({sz:.1f} MB)")
        return
    print(f"  Downloading {desc or url}...")
    try:
        urllib.request.urlretrieve(url, dest)
        sz = os.path.getsize(dest) / 1024 / 1024
        print(f"  OK: {dest} ({sz:.1f} MB)")
    except Exception as e:
        print(f"  FAILED: {e}")
        print(f"  Manual download: {url}")
        print(f"  Save to: {dest}")

def download_string():
    print("\n=== Downloading STRING + TRRUST ===")
    print("  These are needed for validate_against_string() (88.7% precision claim)")
    download(STRING_FULL, os.path.join(DATA_DIR, "string_ppi_full.txt.gz"), "STRING PPI (full)")
    download(STRING_INFO, os.path.join(DATA_DIR, "string_info.txt.gz"), "STRING gene info")
    download(TRRUST_URL, os.path.join(DATA_DIR, "trrust_human.tsv"), "TRRUST")
    print("  Done. validate_against_string() will now work.")

def download_tcga():
    print("\n=== Downloading TCGA Gene Expression ===")
    print("  33 cancer types, ~500 MB each. Needed for pan_cancer_scan.py (Table 4)")
    tcga_dir = os.path.join(PKG_ROOT, 'data', 'tcga')
    os.makedirs(tcga_dir, exist_ok=True)
    cancers = ["ACC","BLCA","BRCA","CESC","CHOL","COAD","DLBC","ESCA","GBM","HNSC",
               "KICH","KIRC","KIRP","LAML","LGG","LIHC","LUAD","LUSC","MESO","OV",
               "PAAD","PCPG","PRAD","READ","SARC","SKCM","STAD","TGCT","THCA","THYM",
               "UCEC","UCS","UCEC"]
    print(f"  {len(cancers)} cancer types to download")
    print(f"  Destination: {tcga_dir}")
    print(f"  Source: UCSC Xena (https://xenabrowser.net/)")
    print(f"  NOTE: TCGA data must be downloaded manually from UCSC Xena.")
    print(f"  Go to: https://xenabrowser.net/datapages/")
    print(f"  Search for: TCGA HiSeqV2 for each cancer type")
    print(f"  Save as: data/tcga/TCGA_XXX_HiSeqV2.tsv")
    print(f"  The pan-cancer results are pre-computed in results/pan_cancer_ckpt.json")

def download_depmap():
    print("\n=== Downloading DepMap CRISPR ===")
    print("  Needed for genome-scale validation (d=17,787)")
    print(f"  Source: {DEPMAP_URL}")
    print(f"  NOTE: DepMap data must be downloaded manually from Broad Institute.")
    print(f"  Go to: https://depmap.org/portal/download/all/")
    print(f"  Download: CRISPR Gene Effect (Q2 2024 release)")
    print(f"  The pre-trained model is bundled: causalscale/pretrained/depmap_19215.pt")

if '--string' in sys.argv or '--all' in sys.argv:
    download_string()
if '--tcga' in sys.argv or '--all' in sys.argv:
    download_tcga()
if '--depmap' in sys.argv or '--all' in sys.argv:
    download_depmap()

if not any(a in sys.argv for a in ['--string','--tcga','--depmap','--all']):
    print(__doc__)
    print("\nPre-computed results in results/ do NOT require any downloads.")
    print("Run 'python run_all.py --verify' to validate the installation.")
