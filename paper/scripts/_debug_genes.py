"""Quick debug: gene name matching between STRING-anchored genes and DepMap columns."""
import pandas as pd

DATA = r"D:\NO.1\cancer_application\data\depmap"
df = pd.read_csv(f"{DATA}/CRISPRGeneEffect.csv", index_col=0)

# Extract gene names from columns
col_genes = {}
for col in df.columns:
    gene = col.split(" (")[0] if " (" in col else col
    col_genes[gene] = col

print(f"Total DepMap columns: {len(df.columns)}")
print(f"Sample column names: {list(df.columns[:5])}")
print(f"Extracted gene names: {list(col_genes.keys())[:5]}")

# Now match with STRING
import gzip
STRING_PATH = r"D:\NO.1\cancer_application\data\validation\string_ppi_full.txt.gz"

all_gene_set = set(col_genes.keys())
print(f"Unique genes: {len(all_gene_set)}")

gene_degree = {}
count = 0
with gzip.open(STRING_PATH, "rt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            g1, g2 = parts[0], parts[1]
            if g1 in all_gene_set and g2 in all_gene_set:
                gene_degree[g1] = gene_degree.get(g1, 0) + 1
                gene_degree[g2] = gene_degree.get(g2, 0) + 1
        count += 1
        if count % 1000000 == 0:
            print(f"  Processed {count/1e6:.0f}M lines, found {len(gene_degree)} genes")

print(f"STRING-anchored genes in DepMap: {len(gene_degree)}")
print(f"Top 10 by degree:")
for g, dg in sorted(gene_degree.items(), key=lambda x: -x[1])[:10]:
    print(f"  {g}: degree={dg}")

# Check TRRUST overlap too
TRRUST_PATH = r"D:\NO.1\cancer_application\data\validation\trrust_human.tsv"
trrust_genes = set()
with open(TRRUST_PATH, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            g1, g2 = parts[0].upper(), parts[1].upper()
            if g1 in all_gene_set:
                trrust_genes.add(g1)
            if g2 in all_gene_set:
                trrust_genes.add(g2)

print(f"TRRUST-anchored genes: {len(trrust_genes)}")
print(f"STRING+TRRUST union: {len(set(gene_degree.keys()) | trrust_genes)}")
