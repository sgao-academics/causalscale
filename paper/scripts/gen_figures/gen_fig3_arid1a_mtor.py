# =============================================================================
# Fig 3: Application & Validation (2x2 panels)
#   (a) ARID1A-MTOR heatmap across 33 TCGA cancers
#   (b) Edge count vs sample size (Spearman from actual data)
#   (c) GO enrichment: conserved vs tissue-specific
#   (d) External validation: STRING/TRRUST + cross-cancer conservation
# =============================================================================
import os
import json, numpy as np, matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from numpy.polynomial.polynomial import polyfit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# SCRIPT_DIR = .../paper/scripts/gen_figures -> go up 3 levels to repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
OUT = os.path.join(REPO_ROOT, 'paper', 'figures')
os.makedirs(OUT, exist_ok=True)

# Find pan_cancer_ckpt.json (try multiple locations)
ckpt_path = None
for candidate in [
    os.path.join(REPO_ROOT, '..', 'causalscale_Replication_Package', 'results', 'pan_cancer_ckpt.json'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'pan_cancer_ckpt.json'),
]:
    if os.path.exists(candidate):
        ckpt_path = candidate
        break
if ckpt_path is None:
    raise FileNotFoundError('Cannot find pan_cancer_ckpt.json')

matplotlib.rcParams.update({'font.size': 8, 'font.family': 'sans-serif'})

# ── Load data ──
ckpt = json.load(open(ckpt_path, encoding='utf-8'))

a2m = sorted([c for c, d in ckpt.items() if abs(d.get('arid1a_to_mtor', 0)) > 0.3])
m2a = sorted([c for c, d in ckpt.items() if abs(d.get('mtor_to_arid1a', 0)) > 0.3])
noe = sorted([c for c, d in ckpt.items()
    if abs(d.get('arid1a_to_mtor', 0)) <= 0.3 and abs(d.get('mtor_to_arid1a', 0)) <= 0.3])
all_c = a2m + m2a + noe

# Compute Spearman rho from actual data
ns_list, ec_list = [], []
for c, d in ckpt.items():
    if d.get('n') and d.get('edge_count'):
        ns_list.append(d['n'])
        ec_list.append(d['edge_count'])
rho_sp, p_sp = spearmanr(ns_list, ec_list)
print(f'Spearman: rho={rho_sp:.4f}, p={p_sp:.2e}')

# ────────────────────────────────────────────────────────────────
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7.8, 5.8))

# ════════════════════════════════════════════════════════════════
# (a) ARID1A-MTOR Heatmap
# ════════════════════════════════════════════════════════════════
heatmap = np.zeros((2, len(all_c)))
for i, c in enumerate(all_c):
    heatmap[0, i] = ckpt[c].get('arid1a_to_mtor', 0)
    heatmap[1, i] = ckpt[c].get('mtor_to_arid1a', 0)

im = ax1.imshow(heatmap, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['ARID1A \u2192 MTOR', 'MTOR \u2192 ARID1A'], fontsize=6.5)
ax1.set_xticks(range(len(all_c)))

# Show every 2nd label to prevent overlap at 45 degrees
show_idx = list(range(0, len(all_c), 2))
ax1.set_xticks(show_idx)
ax1.set_xticklabels([all_c[i] for i in show_idx], fontsize=5, rotation=45, ha='right', va='top')
ax1.set_xlim(-0.5, len(all_c) - 0.5)

# Group separator lines
ax1.axvline(len(a2m) - 0.5, color='black', lw=1.2)
ax1.axvline(len(a2m) + len(m2a) - 0.5, color='black', lw=1.2)

# Group labels at bottom
group_positions = [
    (len(a2m)/2, f'A\u2192M ({len(a2m)})'),
    (len(a2m) + len(m2a)/2, f'M\u2192A ({len(m2a)})'),
    (len(a2m) + len(m2a) + len(noe)/2, f'No edge ({len(noe)})'),
]
for xpos, label in group_positions:
    ax1.text(xpos, -0.8, label, ha='center', fontsize=6, fontweight='bold', color='#333333',
             transform=ax1.get_xaxis_transform())

ax1.set_title('(a) ARID1A-MTOR Pan-Cancer (33 TCGA)', fontsize=9, fontweight='bold', loc='left')
cbar = plt.colorbar(im, ax=ax1, shrink=0.75)
cbar.set_label('Edge weight W', fontsize=6.5)

# ════════════════════════════════════════════════════════════════
# (b) Edges vs Sample Size — Spearman from real data
# ════════════════════════════════════════════════════════════════
ns = np.array([ckpt[c]['n'] for c, d in ckpt.items() if d.get('n') and d.get('edge_count')])
edges_ct = np.array([ckpt[c]['edge_count'] for c, d in ckpt.items() if d.get('n') and d.get('edge_count')])

ax2.scatter(ns, edges_ct, c='#3498DB', s=18, alpha=0.7, edgecolors='white', lw=0.4, zorder=3)

# Power-law fit
cfit = polyfit(np.log10(ns), np.log10(edges_ct), 1)
xfit = np.logspace(np.log10(min(ns)), np.log10(max(ns)), 50)
yfit = 10**cfit[0] * xfit**cfit[1]
ax2.plot(xfit, yfit, '--', color='#E74C3C', lw=1.5,
         label=f'Power-law fit (slope={cfit[1]:.2f})', zorder=2)

ax2.set_xlabel('Sample size n')
ax2.set_ylabel('Edges found')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend(fontsize=6.5, loc='upper right')
ax2.grid(alpha=0.3)
ax2.set_title(f'(b) Edges vs. $n$ ($\\rho$={rho_sp:+.3f}, $p$={p_sp:.1e})',
             fontsize=9, fontweight='bold', loc='left')

# ════════════════════════════════════════════════════════════════
# (c) GO Enrichment
# ════════════════════════════════════════════════════════════════
go_terms = ['Translation', 'Ribosome\nbiogenesis', 'Chromatin\norganization',
            'Keratinization\n(LUAD)', 'Neurotrans.\ntransport (GBM)', 'Antigen processing\n(BRCA)']
go_fdr = [15.7, 10.2, 8.9, 3.4, 3.0, 3.1]  # -log10(FDR)
colors_go = ['#2C3E50'] * 3 + ['#E74C3C', '#E67E22', '#3498DB']
bars = ax3.barh(range(len(go_terms)), go_fdr, color=colors_go, edgecolor='white', height=0.6)
ax3.set_yticks(range(len(go_terms)))
ax3.set_yticklabels(go_terms, fontsize=6.5)
ax3.set_xlabel('-log10(FDR)')
ax3.axvline(1.3, color='#000000', ls='--', lw=0.8)
ax3.text(1.5, 5.6, 'p = 0.05', fontsize=6, color='#000000', va='bottom')
ax3.grid(axis='x', alpha=0.3)
ax3.set_title('(c) GO Enrichment', fontsize=9, fontweight='bold', loc='left')

# Value labels
for bar, v in zip(bars, go_fdr):
    ax3.text(v + 0.3, bar.get_y() + bar.get_height()/2, f'{v:.1f}',
             va='center', fontsize=6.5, fontweight='bold', color='#333333')

# ════════════════════════════════════════════════════════════════
# (d) External Validation
# ════════════════════════════════════════════════════════════════
categories = ['STRING/TRRUST\nValidation', 'Cross-Cancer\nConservation']
values = [88.7, 40.4]
colors_d = ['#27AE60', '#2980B9']
bars_d = ax4.bar(categories, values, color=colors_d, edgecolor='white', width=0.45)
ax4.set_ylabel('Percentage (%)')
ax4.set_ylim(0, 105)
for bar, v in zip(bars_d, values):
    ax4.text(bar.get_x() + bar.get_width()/2, v + 3, f'{v:.1f}%',
             ha='center', fontsize=11, fontweight='bold', color=bar.get_facecolor())
ax4.grid(axis='y', alpha=0.3)
ax4.set_title('(d) External Validation', fontsize=9, fontweight='bold', loc='left')

# ════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.2)
fig.subplots_adjust(hspace=0.25, wspace=0.30)
for fmt, ext in [('pdf', '.pdf'), ('png', '.png')]:
    path = os.path.join(OUT, f'fig3_arid1a_mtor{ext}')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    kb = os.path.getsize(path) / 1024
    print(f'Fig3 saved: {path} ({kb:.0f} KB)')
plt.close()
