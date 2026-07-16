# Generate Fig 1: Architecture diagram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

OUT = r'C:\Users\高帅东\Desktop\causalscale\paper\figures'
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 7)
ax.axis('off')

# Colors
c_api = '#2C3E50'; c_eng = '#27AE60'; c_pretrain = '#2980B9'; c_cli = '#8E44AD'
c_bg = '#F8F9FA'

# Title
ax.text(5, 6.6, 'causalscale: Unified Causal Discovery Engine', ha='center', fontsize=13, fontweight='bold', color=c_api)

# API Box
rect = patches.FancyBboxPatch((1.5, 5.6), 7, 0.8, boxstyle='round,pad=0.1', facecolor=c_api, alpha=0.12, edgecolor=c_api, linewidth=1.5)
ax.add_patch(rect)
ax.text(5, 6.0, 'pip install causalscale', ha='center', fontsize=10, fontweight='bold', color=c_api)

# Three engine boxes
engines = [
    (1.3, 'ClusterAware', 'NOTEARS\n$d \\leq 500$\nAugmented Lagrangian', c_eng),
    (4.0, 'MultiScale', '$W=\\sum U_s V_s^\\top$\n$d=500$--$5{,}000$\nHierarchical', '#E67E22'),
    (6.7, 'LowRankGNN', '$W=UV^\\top$\n$d \\geq 5{,}000$\nGenome-scale', '#E74C3C'),
]
for x, name, desc, color in engines:
    box = patches.FancyBboxPatch((x-0.9, 3.8), 1.8, 1.5, boxstyle='round,pad=0.08', facecolor=color, alpha=0.1, edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x, 5.1, name, ha='center', fontsize=9, fontweight='bold', color=color)
    ax.text(x, 4.3, desc, ha='center', fontsize=7, color='#555555', va='center')

# Arrows from API to engines
for x, _, _, _ in engines:
    ax.annotate('', xy=(x, 5.4), xytext=(5, 5.4),
                arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1))

# Auto selector
rect2 = patches.FancyBboxPatch((3.5, 3.3), 3, 0.3, boxstyle='round,pad=0.05', facecolor='#ECF0F1', edgecolor='#95A5A6', linewidth=1)
ax.add_patch(rect2)
ax.text(5, 3.45, 'Automatic Engine Selection', ha='center', fontsize=8, color='#555555')

# Pre-trained box
rect3 = patches.FancyBboxPatch((0.5, 1.8), 9, 1.2, boxstyle='round,pad=0.1', facecolor=c_pretrain, alpha=0.08, edgecolor=c_pretrain, linewidth=1.5)
ax.add_patch(rect3)
ax.text(5, 2.8, 'Pre-trained Backbones', ha='center', fontsize=9, fontweight='bold', color=c_pretrain)
ax.text(5, 2.3, 'DepMap 19,215 genes (28 causal edges)    |    TCGA Pan-Cancer (7,960 edges, 33 types)', ha='center', fontsize=7, color='#555555')

# Bottom row
items = [('CLI', '6 commands', c_cli), ('Jupyter', '4 tutorials', '#16A085'), ('Tests', '21/22 PASS', '#C0392B'),
         ('HuggingFace', 'Models', '#F39C12'), ('PyPI', 'pip install', '#3498DB'), ('MIT License', 'Open Source', '#1ABC9C')]
for i, (name, desc, color) in enumerate(items):
    x = 1 + i * 1.4
    box = patches.FancyBboxPatch((x-0.5, 0.2), 1.0, 1.2, boxstyle='round,pad=0.05', facecolor=color, alpha=0.1, edgecolor=color, linewidth=1.2)
    ax.add_patch(box)
    ax.text(x, 1.2, name, ha='center', fontsize=7, fontweight='bold', color=color)
    ax.text(x, 0.6, desc, ha='center', fontsize=6, color='#777777')

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig1_architecture.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig1_architecture.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f'Fig 1 saved to {OUT}/')
print(f'  fig1_architecture.pdf: {os.path.getsize(os.path.join(OUT, "fig1_architecture.pdf"))/1024:.0f}KB')
print(f'  fig1_architecture.png: {os.path.getsize(os.path.join(OUT, "fig1_architecture.png"))/1024:.0f}KB')
