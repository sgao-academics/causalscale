import sys
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")
import causalscale as cs
print(f"Version: {cs.__version__}")
print(f"validate_against_string: {hasattr(cs, 'validate_against_string')}")
print(f"LowRankGNN: {hasattr(cs, 'LowRankGNN')}")
print(f"train_lowrank_gnn: {hasattr(cs, 'train_lowrank_gnn')}")
print("OK")
