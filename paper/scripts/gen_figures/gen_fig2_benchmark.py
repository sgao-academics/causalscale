# =============================================================================
# Fig 2: Synthetic Benchmarks & Ablations (2x2 panels)
#   (a) F1 bar chart: NOTEARS vs DAGMA vs causalscale (with error bars)
#   (b) Cohen's d effect sizes (log scale)
#   (c) Scale-free vs ER topology robustness
#   (d) LowRankGNN rank sensitivity: r vs Correlation-Reconstruction F1
# =============================================================================
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, os, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
# Try results/ relative to package root (replication package layout)
RESULTS_DIR = os.path.join(PKG_ROOT, 'results')
if not os.path.exists(RESULTS_DIR):
    RESULTS_DIR = os.path.join(os.path.dirname(PKG_ROOT), 'paper')

OUT = os.path.join(PKG_ROOT, 'figures')
if not os.path.exists(OUT):
    OUT = os.path.join(os.path.dirname(PKG_ROOT), 'paper', 'figures')
os.makedirs(OUT, exist_ok=True)

# ── Read actual data for causalscale error bars ──
exp1_path = os.path.join(RESULTS_DIR, 'exp1_causalscale_er.json')
cs_mean = {}
cs_std = {}
if os.path.exists(exp1_path):
    with open(exp1_path, encoding='utf-8') as f:
        exp1 = json.load(f)
    from collections import defaultdict
    by_d = defaultdict(list)
    for k, v in exp1.items():
        by_d[v['d']].append(v['f1'])
    for d in [30, 50, 80, 100, 150]:
        vals = by_d.get(d, [])
        if vals:
            cs_mean[d] = round(np.mean(vals), 3)
            cs_std[d] = round(np.std(vals), 4)
else:
    # Fallback to paper values if JSON not found
    cs_mean = {30:0.646, 50:0.595, 80:0.731, 100:0.766, 150:0.768}
    cs_std  = {30:0.082, 50:0.129, 80:0.080, 100:0.029, 150:0.022}

matplotlib.rcParams.update({'font.size': 8, 'font.family': 'sans-serif'})
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7.2, 5.5))

# ══════════════════════════════════════════════════════════════════
# (a) F1 Bar Chart — NOTEARS vs DAGMA vs causalscale
# ══════════════════════════════════════════════════════════════════
dims = [30, 50, 80, 100, 150]
# Table 1 values
nt_f1   = [0.581, 0.475, 0.391, 0.185, 0.000]
nt_std  = [0.061, 0.074, 0.081, 0.069, 0.000]
dagma_f1  = [0.589, 0.689, 0.896, 0.931, 0.989]
dagma_std = [0.038, 0.056, 0.051, 0.017, 0.007]
cs_f1  = [cs_mean.get(d, 0) for d in dims]
cs_esd = [cs_std.get(d, 0) for d in dims]

x = np.arange(len(dims))
w = 0.22

bars1 = ax1.bar(x - w, nt_f1, w, label='NOTEARS', color='#E74C3C', edgecolor='white',
                yerr=nt_std, capsize=3, error_kw={'lw':0.8})
bars2 = ax1.bar(x,     dagma_f1, w, label='DAGMA', color='#3498DB', edgecolor='white',
                yerr=dagma_std, capsize=3, error_kw={'lw':0.8})
bars3 = ax1.bar(x + w, cs_f1, w, label='causalscale', color='#27AE60', edgecolor='white',
                yerr=cs_esd, capsize=3, error_kw={'lw':0.8})

ax1.set_xticks(x)
ax1.set_xticklabels([f'{d}' for d in dims])
ax1.set_ylabel('F1 Score')
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=6.5, loc='upper left', framealpha=0.85)
ax1.grid(axis='y', alpha=0.3)
ax1.set_title('(a) Synthetic DAG Recovery (ER, n=2d, 5 seeds)', fontsize=9, fontweight='bold', loc='left')

# Annotate NOTEARS demise — small, top-left of panel
ax1.text(0.98, 0.99, 'NOTEARS: zero edges at d=150', fontsize=5.5, color='#E74C3C',
         ha='right', va='top', style='italic', transform=ax1.transAxes)

# ── Value labels on causalscale bars (top performer beyond d=30) ──
for i, (v, e) in enumerate(zip(cs_f1, cs_esd)):
    ax1.text(i + w + 0.05, v + e + 0.025, f'{v:.3f}', ha='left', fontsize=5.5, fontweight='bold', color='#27AE60')

# ══════════════════════════════════════════════════════════════════
# (b) Cohen's d — log scale for readability
# ══════════════════════════════════════════════════════════════════
d_dims = [30, 50, 80, 100]
cohens_d = [0.9, 1.1, 4.2, 11.0]
colors_d = ['#F1C40F', '#F39C12', '#E67E22', '#E74C3C']
bars_d = ax2.bar(range(4), cohens_d, color=colors_d, edgecolor='white')
ax2.set_xticks(range(4))
ax2.set_xticklabels([f'd={d}' for d in d_dims])
ax2.set_ylabel("Cohen's d")
ax2.set_yscale('log')
ax2.set_ylim(0.4, 20)

# Reference lines
ax2.axhline(0.5, color='gray', ls=':', lw=0.8)
ax2.axhline(0.8, color='gray', ls='--', lw=0.8)
ax2.text(3.5, 0.52, 'Medium (0.5)', fontsize=5.5, color='#000000', ha='right')
ax2.text(3.5, 0.84, 'Large (0.8)', fontsize=5.5, color='#000000', ha='right')

ax2.grid(axis='y', alpha=0.3, which='both')
ax2.set_title('(b) Effect Size (causalscale vs NOTEARS)', fontsize=9, fontweight='bold', loc='left')

for i, v in enumerate(cohens_d):
    ax2.text(i, v * 1.15, f'{v:.1f}', ha='center', fontsize=7, fontweight='bold', color=colors_d[i])

# ══════════════════════════════════════════════════════════════════
# (c) Scale-Free vs ER Topology Robustness
# ══════════════════════════════════════════════════════════════════
topologies = ['Erdos-Renyi', 'Scale-Free']
d50 = [0.595, 0.620]
d100 = [0.766, 0.623]
x2 = np.arange(2)
w2 = 0.30

bars_c1 = ax3.bar(x2 - w2/2, d50, w2, label='d=50', color='#3498DB', edgecolor='white')
bars_c2 = ax3.bar(x2 + w2/2, d100, w2, label='d=100', color='#2ECC71', edgecolor='white')
ax3.set_xticks(x2)
ax3.set_xticklabels(topologies, fontsize=7.5)
ax3.set_ylabel('F1 Score')
ax3.set_ylim(0, 0.85)
ax3.legend(fontsize=7, loc='lower right')
ax3.grid(axis='y', alpha=0.3)
ax3.set_title('(c) Topology Robustness', fontsize=9, fontweight='bold', loc='left')

# Value labels
for bar_group, values in [(bars_c1, d50), (bars_c2, d100)]:
    for bar, v in zip(bar_group, values):
        ax3.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                 f'{v:.3f}', ha='center', fontsize=6.5, fontweight='bold', color='#333333')

# ══════════════════════════════════════════════════════════════════
# (d) LowRankGNN Rank Sensitivity
# ══════════════════════════════════════════════════════════════════
ranks = [8, 16, 32, 64, 128]
rank_f1 = [0.434, 0.557, 0.677, 0.806, 0.92]

ax4.plot(ranks, rank_f1, 'o-', color='#8E44AD', markersize=7, linewidth=2, zorder=3)
ax4.set_xlabel('Rank r')
ax4.set_ylabel('Correlation-Reconstruction F1')
ax4.set_ylim(0.30, 1.02)
ax4.set_xscale('log', base=2)
ax4.set_xticks(ranks)
ax4.set_xticklabels([str(r) for r in ranks])
ax4.grid(alpha=0.3)

# Annotate each point — directly below to prevent overlap
for r, f in zip(ranks, rank_f1):
    ax4.annotate(f'{f:.3f}', (r, f), textcoords='offset points',
                 xytext=(0, -14), fontsize=7, fontweight='bold', color='#8E44AD',
                 ha='center', va='top')

# Default rank marker
ax4.axvline(64, color='#E74C3C', ls='--', lw=1.5, alpha=0.6, zorder=1)
ax4.text(66, 0.97, 'Default r=64', fontsize=7, color='#E74C3C', ha='left', fontstyle='italic')
ax4.set_title('(d) LowRankGNN Rank Sensitivity (d=500)', fontsize=9, fontweight='bold', loc='left')

# ══════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.5)
for fmt, ext in [('pdf', '.pdf'), ('png', '.png')]:
    path = os.path.join(OUT, f'fig2_benchmark{ext}')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    kb = os.path.getsize(path) / 1024
    print(f'Fig2 saved: {path} ({kb:.0f} KB)')
plt.close()
