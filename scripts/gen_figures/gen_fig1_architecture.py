# =============================================================================
# Fig 1: Architecture Diagram (Page 3, Section 3 Architecture and Design)
# 11 engines: DAGMA, ClusterAware, Causal Transformer, LowRankGNN,
#             MultiBatch, LLMPrior, BayesLowRank, scCausal,
#             MultiScale, MultiModal, Ensemble
# Automatic dimension-based selection, pre-trained backbones, distribution.
# =============================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT = os.path.join(PKG_ROOT, 'figures')
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(10.0, 6.5))
ax.set_xlim(0, 11.5)
ax.set_ylim(0, 8.5)
ax.axis('off')

C = {
    'cluster':  '#27AE60',
    'dagma':    '#2980B9',
    'ct':       '#F39C12',
    'lowrank':  '#E74C3C',
    'multibatch': '#1ABC9C',
    'llmprior':  '#9B59B6',
    'bayes':     '#E91E63',
    'sccausal':  '#00BCD4',
    'mscale':   '#8E44AD',
    'mm':       '#16A085',
    'ensemble': '#2C3E50',
    'pretrain': '#1A6FB5',
}

ax.text(5.75, 8.10, 'causalscale: Unified Causal Discovery (v3.3.0, 11 Engines)',
        ha='center', fontsize=13, fontweight='bold', color='#2C3E50')

# API surface bar
r_api = patches.FancyBboxPatch(
    (0.8, 7.20), 9.9, 0.55, boxstyle='round,pad=0.08',
    facecolor='#2C3E50', alpha=0.10, edgecolor='#2C3E50', linewidth=1.5
)
ax.add_patch(r_api)
ax.text(5.75, 7.47,
        'pip install causalscale  |  cs.CausalDiscovery(data).fit()  |  model.summary()',
        ha='center', fontsize=8.5, fontweight='bold', color='#2C3E50')

# Row 1: 4 Core Engines (dimension-ordered)
core_engines = [
    (1.55, 'DAGMA',         'd <= 150\nLog-Det\nBest F1',          C['dagma']),
    (3.85, 'ClusterAware',  'd <= 200\nNOTEARS\nExact DAG',        C['cluster']),
    (6.15, 'Causal Trans.', '200 < d <= 500\nAttention\nDAG Penalty', C['ct']),
    (8.45, 'LowRankGNN',    'd > 500\nW = UV^T\nGenome-Scale',     C['lowrank']),
]

for x, name, desc, color in core_engines:
    box = patches.FancyBboxPatch(
        (x - 0.95, 5.20), 1.90, 1.50,
        boxstyle='round,pad=0.06',
        facecolor=color, alpha=0.10, edgecolor=color, linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(x, 6.45, name, ha='center', fontsize=7.5, fontweight='bold', color=color)
    ax.text(x, 5.75, desc, ha='center', fontsize=6.2, color='#444444',
            va='center', linespacing=1.25)

# Auto-selection label (row 1)
r_auto1 = patches.FancyBboxPatch(
    (1.6, 4.75), 8.3, 0.28, boxstyle='round,pad=0.04',
    facecolor='#ECF0F1', edgecolor='#BDC3C7', linewidth=1.0
)
ax.add_patch(r_auto1)
ax.text(5.75, 4.89, 'Automatic Dimension-Based Engine Selection',
        ha='center', fontsize=8, color='#555555', fontstyle='italic')

# Row 2: 7 Specialized Engines
spec_engines = [
    (1.10, 'MultiBatch',   'Multi-Dataset\nResidual Adapters',  C['multibatch']),
    (2.65, 'LLMPrior',     'STRING Edge\nPrior Injection',      C['llmprior']),
    (4.20, 'BayesLowRank', 'Bayesian Bootstrap\nUncertainty',   C['bayes']),
    (5.75, 'scCausal',     'Single-Cell\nNB-LR CI Test',        C['sccausal']),
    (7.30, 'MultiScale',   'd=500-5000\nHierarchical',           C['mscale']),
    (8.85, 'MultiModal',   'm >= 2 Modalities\nCross-Modal',     C['mm']),
    (10.40,'Ensemble',     'Voting\n3+ Engines',                 C['ensemble']),
]

for x, name, desc, color in spec_engines:
    box = patches.FancyBboxPatch(
        (x - 0.65, 3.15), 1.30, 1.30,
        boxstyle='round,pad=0.05',
        facecolor=color, alpha=0.10, edgecolor=color, linewidth=1.3
    )
    ax.add_patch(box)
    ax.text(x, 4.15, name, ha='center', fontsize=6.5, fontweight='bold', color=color)
    ax.text(x, 3.60, desc, ha='center', fontsize=5.5, color='#444444',
            va='center', linespacing=1.2)

# Specialized engines label
r_auto2 = patches.FancyBboxPatch(
    (1.6, 2.75), 8.3, 0.25, boxstyle='round,pad=0.04',
    facecolor='#ECF0F1', edgecolor='#BDC3C7', linewidth=1.0
)
ax.add_patch(r_auto2)
ax.text(5.75, 2.87, 'Specialized Engines (domain-specific selection)',
        ha='center', fontsize=7.5, color='#555555', fontstyle='italic')

# Pre-trained models section
r_pretrain = patches.FancyBboxPatch(
    (0.5, 1.45), 10.5, 1.05, boxstyle='round,pad=0.08',
    facecolor=C['pretrain'], alpha=0.06, edgecolor=C['pretrain'], linewidth=1.5
)
ax.add_patch(r_pretrain)
ax.text(5.75, 2.25, 'Pre-Trained Causal Graphs on HuggingFace Hub',
        ha='center', fontsize=9, fontweight='bold', color=C['pretrain'])
ax.text(5.75, 1.72,
        'DepMap 19,215 genes  -  TCGA Pan-Cancer (7,960 edges, 33 types)  -  cs.load_model("depmap")',
        ha='center', fontsize=7, color='#555555')

# Distribution channels
items = [
    (1.55, 'PyPI',        'pip install',    '#3498DB'),
    (3.85, 'CLI',         '6 commands',     '#8E44AD'),
    (6.15, 'Jupyter',     '4 tutorials',    '#16A085'),
    (8.45, 'HuggingFace', 'Pre-trained',    '#F39C12'),
    (10.75,'GitHub',      'MIT License',    '#2C3E50'),
]
for x, name, desc, color in items:
    box = patches.FancyBboxPatch(
        (x - 0.55, 0.20), 1.10, 1.00,
        boxstyle='round,pad=0.04',
        facecolor=color, alpha=0.10, edgecolor=color, linewidth=1.3
    )
    ax.add_patch(box)
    ax.text(x, 1.00, name, ha='center', fontsize=7, fontweight='bold', color=color)
    ax.text(x, 0.50, desc, ha='center', fontsize=6, color='#666666')

plt.tight_layout(pad=0.3)
for fmt, ext in [('pdf', '.pdf'), ('png', '.png')]:
    path = os.path.join(OUT, f'fig1_architecture{ext}')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    kb = os.path.getsize(path) / 1024
    print(f'Fig1 saved: {path} ({kb:.0f} KB)')
plt.close()
