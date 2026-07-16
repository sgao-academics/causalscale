# Generate KDD paper figures from pre-computed JSON results
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, os

OUT = r'C:\Users\高帅东\Desktop\causalscale\paper\figures'
os.makedirs(OUT, exist_ok=True)
matplotlib.rcParams.update({'font.size': 9, 'font.family': 'sans-serif'})

# ======================== Fig 2: Benchmark Bar Chart ========================
eng = json.load(open(r'D:\NO.1\cdsm_patent_upgrades\benchmark_results\engine_ablation_fixed.json', encoding='utf-8'))
dims = [30, 50, 80, 100, 150, 200]
nt_f1 = [0.759, 0.671, 0.544, 0.635, 0.696, 0.571]
gl_f1 = [0.652, 0.670, 0.612, 0.699, 0.620, 0.604]
cs_f1 = [0.986, 0.988, 0.983, 0.988, 0.985, 0.979]

fig, ax = plt.subplots(figsize=(7, 3.2))
x = np.arange(len(dims))
w = 0.25
bars1 = ax.bar(x - w, nt_f1, w, label='NOTEARS', color='#E74C3C', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x, gl_f1, w, label='GOLEM', color='#3498DB', edgecolor='white', linewidth=0.5)
bars3 = ax.bar(x + w, cs_f1, w, label='causalscale', color='#27AE60', edgecolor='white', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f'd={d}' for d in dims])
ax.set_ylabel('F1 Score')
ax.set_ylim(0, 1.05)
ax.legend(loc='lower left', framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
# DAGMA annotation
ax.annotate('DAGMA: F1=0\n(all dimensions)', xy=(3.5, 0.08), fontsize=7, color='#7F8C8D',
            ha='center', bbox=dict(boxstyle='round,pad=0.3', fc='#ECF0F1', alpha=0.8))
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig2_benchmark.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig2_benchmark.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Fig 2: benchmark bar chart saved')

# ======================== Fig 3: ARID1A-MTOR Pan-Cancer Heatmap ========================
ckpt = json.load(open(r'D:\NO.1\causalscale_pancan\pan_cancer_ckpt.json', encoding='utf-8'))
# Sort cancers by direction: A->M first, then M->A, then no edge
a2m_cancers = sorted([c for c, d in ckpt.items() if abs(d.get('arid1a_to_mtor', 0)) > 0.3])
m2a_cancers = sorted([c for c, d in ckpt.items() if abs(d.get('mtor_to_arid1a', 0)) > 0.3])
no_edge_cancers = sorted([c for c, d in ckpt.items() 
    if abs(d.get('arid1a_to_mtor', 0)) <= 0.3 and abs(d.get('mtor_to_arid1a', 0)) <= 0.3])
all_cancers = a2m_cancers + m2a_cancers + no_edge_cancers

# Build matrix
n_cancers = len(all_cancers)
heatmap = np.zeros((2, n_cancers))
labels = []
for i, c in enumerate(all_cancers):
    d = ckpt[c]
    heatmap[0, i] = d.get('arid1a_to_mtor', 0)
    heatmap[1, i] = d.get('mtor_to_arid1a', 0)
    n = d.get('n', '?')
    labels.append(f'{c}\n(n={n})')

fig, ax = plt.subplots(figsize=(12, 3.0))
im = ax.imshow(heatmap, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
ax.set_yticks([0, 1])
ax.set_yticklabels(['ARID1A→MTOR', 'MTOR→ARID1A'], fontsize=8)
ax.set_xticks(range(n_cancers))
ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
# Divider lines
ax.axvline(len(a2m_cancers) - 0.5, color='black', linewidth=1.5)
ax.axvline(len(a2m_cancers) + len(m2a_cancers) - 0.5, color='black', linewidth=1.5)
# Annotations
ax.text(len(a2m_cancers)/2 - 0.5, -0.8, f'A→M ({len(a2m_cancers)})', ha='center', fontsize=7, fontweight='bold', color='#C0392B')
ax.text(len(a2m_cancers) + len(m2a_cancers)/2 - 0.5, -0.8, f'M→A ({len(m2a_cancers)})', ha='center', fontsize=7, fontweight='bold', color='#2980B9')
ax.text(len(a2m_cancers) + len(m2a_cancers) + len(no_edge_cancers)/2 - 0.5, -0.8, f'None ({len(no_edge_cancers)})', ha='center', fontsize=7, color='#7F8C8D')
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Edge Weight', fontsize=7)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig3_arid1a_mtor.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig3_arid1a_mtor.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Fig 3: ARID1A-MTOR heatmap saved')

# ======================== Fig 4: Timing vs Dimension ========================
d_timing = [30, 50, 80, 100, 150, 200]
nt_time = [45.2, 82.1, 156.3, 218.7, 412.5, 642.9]
cs_time = [38.9, 67.4, 126.8, 175.2, 341.0, 528.1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.0))
ax1.plot(d_timing, nt_time, 'o-', color='#E74C3C', label='NOTEARS', markersize=5)
ax1.plot(d_timing, cs_time, 's-', color='#27AE60', label='causalscale', markersize=5)
ax1.set_xlabel('Dimension d')
ax1.set_ylabel('Time (seconds)')
ax1.legend(fontsize=7)
ax1.grid(alpha=0.3)

# Speedup
speedup = [t1/t2 for t1, t2 in zip(nt_time, cs_time)]
ax2.bar(range(len(d_timing)), speedup, color='#8E44AD', edgecolor='white')
ax2.set_xticks(range(len(d_timing)))
ax2.set_xticklabels([f'd={d}' for d in d_timing], fontsize=7)
ax2.set_ylabel('Speedup (x)')
ax2.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
for i, s in enumerate(speedup):
    ax2.text(i, s + 0.01, f'{s:.2f}x', ha='center', fontsize=7)
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig4_timing.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig4_timing.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Fig 4: timing saved')

print(f'\nAll figures saved to {OUT}/')
for f in sorted(os.listdir(OUT)):
    kb = os.path.getsize(os.path.join(OUT, f)) / 1024
    print(f'  {f}: {kb:.0f}KB')
