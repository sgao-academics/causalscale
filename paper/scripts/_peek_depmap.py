"""Peek DepMap CRISPR data structure."""
import pandas as pd, numpy as np

DATA = r"D:\NO.1\cancer_application\data\depmap"

# Check CRISPRGeneEffect (likely the one we want - CERES scores)
print("=== CRISPRGeneEffect ===")
df = pd.read_csv(f"{DATA}/CRISPRGeneEffect.csv", nrows=5, index_col=0)
print(f"Shape (first 5 rows): {df.shape}")
print(f"Columns (first 10): {list(df.columns[:10])}")
print(f"Index (first 5): {list(df.index[:5])}")
print(f"dtypes sample: {df.dtypes.iloc[:3].to_dict()}")

# Full shape
print("\n=== Full shape estimate ===")
import csv
with open(f"{DATA}/CRISPRGeneEffect.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    n_cols = len(header)
    n_rows = sum(1 for _ in reader)
print(f"CRISPRGeneEffect: {n_rows} cell lines x {n_cols-1} genes")

# Check CRISPRGeneDependency
print("\n=== CRISPRGeneDependency ===")
with open(f"{DATA}/CRISPRGeneDependency.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    n_cols2 = len(header)
    n_rows2 = sum(1 for _ in reader)
print(f"CRISPRGeneDependency: {n_rows2} cell lines x {n_cols2-1} genes")
