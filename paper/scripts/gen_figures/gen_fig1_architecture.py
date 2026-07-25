# =============================================================================
# Fig 1: Architecture Diagram — causalscale v3.3.0, 11 engines
# Two-row layout: 4 core (dimension-ordered) + 7 specialized
# =============================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT = os.path.join(PKG_ROOT, 'figures')
if not os.path.exists(OUT):
    OUT = os.path.join(os.path.dirname(PKG_ROOT), 'paper', 'figures')
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(10.5, 9.0))
ax.set_xlim(0, 10.5)
ax.set_ylim(0, 9.0)
ax.axis('off')

# ── Color palette ──
C = {
    'dagma':    '#2980B9',  # blue
    'cluster':  '#27AE60',  # green
    'ct':       '#F39C12',  # orange
    'lowrank':  '#E74C3C',  # red
    'm_batch':  '#D35400',  # dark orange
    'llm':      '#1ABC9C',  # teal/turquoise
    'bayes':    '#9B59B6',  # purple
    'sc':       '#E91E63',  # pink
    'mscale':   '#8E44AD',  # purple
    'mm':       '#16A085',  # dark teal
    'ensemble': '#2C3E50',  # dark
    'pretrain': '#1A6FB5',
}

# ────────────────────────────────────────────────────────────────
# 1. TITLE
# ────────────────────────────────────────────────────────────────
ax.text(5.25, 8.70, 'causalscale v3.3.0 — 11 Engines under One API',
        ha='center', fontsize=14, fontweight='bold', color='#2C3E50')

# ────────────────────────────────────────────────────────────────
# 2. API SURFACE BAR
# ────────────────────────────────────────────────────────────────
r_api = patches.FancyBboxPatch(
    (0.5, 7.60), 9.5, 0.60, boxstyle='round,pad=0.08',
    facecolor='#2C3E50', alpha=0.10, edgecolor='#2C3E50', linewidth=1.5
)
ax.add_patch(r_api)
ax.text(5.25, 7.90,
        'pip install causalscale     |     cs.CausalDiscovery(data).fit()     |     model.summary()',
        ha='center', fontsize=9, fontweight='bold', color='#2C3E50')

# ────────────────────────────────────────────────────────────────
# 3. CORE ENGINES (Row 1 — dimension-ordered)
# ────────────────────────────────────────────────────────────────
CORE_Y = 6.10  # box top
CORE_H = 1.30  # box height
CORE_BASE = CORE_Y - CORE_H  # box bottom

core_engines = [
    (1.30,  'DAGMA',          'd \u2264 150\nLog-Det Acyclicity\nBest F\u2081 = 0.989', C['dagma']),
    (3.60,  'ClusterAware',   'd \u2264 200\nNOTEARS + Exact DAG\nSurvives d > 150',  C['cluster']),
    (5.90,  'Causal Trans.',  '200 < d \u2264 500\nSelf-Attention\n1028 edges @ d=200', C['ct']),
    (8.20,  'LowRankGNN',     'd > 500\nW = UV\u1d40\n88.7% STRING @ d=17,787',      C['lowrank']),
]

for x, name, desc, color in core_engines:
    box = patches.FancyBboxPatch(
        (x - 0.90, CORE_BASE), 1.80, CORE_H,
        boxstyle='round,pad=0.06',
        facecolor=color, alpha=0.10, edgecolor=color, linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(x, CORE_Y - 0.20, name, ha='center', fontsize=7.5, fontweight='bold', color=color)
    ax.text(x, CORE_Y - 0.80, desc, ha='center', fontsize=6.2, color='#444444',
            va='center', linespacing=1.20)

# Core engines label
ax.text(5.25, CORE_Y + 0.15, 'Core Engines — Automatic Dimension-Based Selection',
        ha='center', fontsize=8.5, color='#555555', fontstyle='italic')
# Arrow between core engines
for i in range(len(core_engines)-1):
    x1 = core_engines[i][0] + 0.90
    x2 = core_engines[i+1][0] - 0.90
    ax.annotate('', xy=(x2-0.05, CORE_Y - 0.65), xytext=(x1+0.05, CORE_Y - 0.65),
                arrowprops=dict(arrowstyle='->', color='#BDC3C7', lw=1.0))

# ────────────────────────────────────────────────────────────────
# 4. SPECIALIZED ENGINES (Row 2)
# ────────────────────────────────────────────────────────────────
SPEC_Y = 4.40
SPEC_H = 1.20
SPEC_BASE = SPEC_Y - SPEC_H

spec_engines = [
    (0.85,  'MultiBatch',   'Multi-Dataset\nW\u2080 + \u0394W\u2096\n93.3% ASCEND',    C['m_batch']),
    (2.20,  'LLMPrior',     'Edge Prior\nSTRING-Derived\n+12% F\u2081',              C['llm']),
    (3.55,  'BayesLowRank', 'Uncertainty\nBB Bootstrap\nECE 0.003',                C['bayes']),
    (4.90,  'scCausal',     'Single-Cell\nNB-LR CI Test\n35.8% Cell-Type',         C['sc']),
    (6.25,  'MultiScale',   'd = 500–5000\nHierarchical\nMulti-Res.',              C['mscale']),
    (7.60,  'MultiModal',   'm \u2265 2 Modalities\nCross-Modal\nConsensus',       C['mm']),
    (8.95,  'Ensemble',     'Voting\n3 Engines\n+22–35% F\u2081',                  C['ensemble']),
]

for x, name, desc, color in spec_engines:
    box = patches.FancyBboxPatch(
        (x - 0.55, SPEC_BASE), 1.10, SPEC_H,
        boxstyle='round,pad=0.06',
        facecolor=color, alpha=0.10, edgecolor=color, linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(x, SPEC_Y - 0.18, name, ha='center', fontsize=6.0, fontweight='bold', color=color)
    ax.text(x, SPEC_Y - 0.70, desc, ha='center', fontsize=5.5, color='#444444',
            va='center', linespacing=1.15)

ax.text(5.25, SPEC_Y + 0.15, 'Specialized Engines — Domain-Specific & Auxiliary',
        ha='center', fontsize=8.5, color='#555555', fontstyle='italic')

# ────────────────────────────────────────────────────────────────
# 5. PRE-TRAINED MODELS
# ────────────────────────────────────────────────────────────────
PRETRAIN_Y = 2.40
PRETRAIN_H = 0.80
r_pretrain = patches.FancyBboxPatch(
    (0.5, PRETRAIN_Y - PRETRAIN_H), 9.5, PRETRAIN_H,
    boxstyle='round,pad=0.08',
    facecolor=C['pretrain'], alpha=0.06, edgecolor=C['pretrain'], linewidth=1.5
)
ax.add_patch(r_pretrain)
ax.text(5.25, PRETRAIN_Y - 0.12, 'Pre-Trained Causal Graphs on HuggingFace Hub',
        ha='center', fontsize=9.5, fontweight='bold', color=C['pretrain'])
ax.text(5.25, PRETRAIN_Y - 0.52,
        'DepMap 19,215 genes  \u00b7  TCGA Pan-Cancer (7,960 edges, 33 types)  \u00b7  model = cs.load_pretrained("depmap")',
        ha='center', fontsize=7.2, color='#555555')

# ────────────────────────────────────────────────────────────────
# 6. DISTRIBUTION CHANNELS
# ────────────────────────────────────────────────────────────────
items = [
    (1.35,  'PyPI',        'pip install',     '#3498DB'),
    (3.20,  'CLI',         'causalscale fit', '#8E44AD'),
    (5.05,  'Jupyter',     '4 tutorials',     '#16A085'),
    (6.90,  'HuggingFace', 'Pre-trained',     '#F39C12'),
    (8.75,  'GitHub',      'MIT License',     '#2C3E50'),
]
for x, name, desc, color in items:
    box = patches.FancyBboxPatch(
        (x - 0.55, 0.15), 1.10, 1.08,
        boxstyle='round,pad=0.04',
        facecolor=color, alpha=0.10, edgecolor=color, linewidth=1.3
    )
    ax.add_patch(box)
    ax.text(x, 1.10, name, ha='center', fontsize=7.5, fontweight='bold', color=color)
    ax.text(x, 0.52, desc, ha='center', fontsize=6.5, color='#666666')

# ────────────────────────────────────────────────────────────────
# SAVE
# ────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.3)
for fmt, ext in [('pdf', '.pdf'), ('png', '.png')]:
    path = os.path.join(OUT, f'fig1_architecture{ext}')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    kb = os.path.getsize(path) / 1024
    print(f'Fig1 saved: {path} ({kb:.0f} KB)')
plt.close()
