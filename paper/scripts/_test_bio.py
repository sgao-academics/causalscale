import sys, numpy as np
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")
import causalscale as cs

X = np.random.randn(100, 5)
m = cs.CausalDiscovery(X, var_names=["TP53","MDM2","BRCA1","EGFR","KRAS"],
                        method="lowrank", rank=2, device="cpu", verbose=False)
m.fit(verbose=False)
r = m.validate(string_data_dir=r"D:\NO.1\cancer_application\data\validation", verbose=False)
print(f"Mode: {r['mode']}")
print(f"Edges: {r['total_edges']}")
print(f"Validated: {r['validated_edges']} / {r['total_edges']}")
print(f"Precision: {r['precision']:.4f}")
print("OK")
