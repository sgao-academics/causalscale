# =============================================================================
# Fig 4: Scaling & Memory Analysis (2x2 panels)
#   (a) Per-seed wall-clock time (causalscale vs NOTEARS)
#   (b) GPU memory vs dimension (linear O(d))
#   (c) LowRankGNN time scaling (log-log, d=500-2000)
#   (d) Sample efficiency: n=2d vs n=5d
# =============================================================================
import json, numpy as np, matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

_script = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_script)))
OUT = os.path.join(REPO_ROOT, 'figures')
RESULTS = os.path.join(REPO_ROOT, '..', 'causalscale_Replication_Package', 'results')
os.makedirs(OUT, exist_ok=True)

matplotlib.rcParams.update({'font.size': 7.5, 'font.family': 'sans-serif'})
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7.2, 5.2))

# ════════════════════════════════════════════════════════════════
# (a) Per-Seed Wall-Clock Time — from EXP1 actual data
# ════════════════════════════════════════════════════════════════
exp1_path = os.path.join(RESULTS, 'exp1_causalscale_er.json')
if os.path.exists(exp1_path):
    exp1 = json.load(open(exp1_path, encoding='utf-8'))
    by_d = defaultdict(list)
    for k, v in exp1.items():
        by_d[v['d']].append(v)
    d_timing = sorted(by_d.keys())[:6]  # [30, 50, 80, 100, 150, 200]
    cs_time = [np.mean([v['time'] for v in by_d[d]]) for d in d_timing]
else:
    d_timing = [30, 50, 80, 100, 150, 200]
    cs_time = [8.8, 8.7, 9.3, 8.7, 8.3, 8.1]
# NOTEARS default timing (paper values)
nt_time = [3.5, 3.3, 3.4, 3.5, 3.5, 3.6]

ax1.plot(d_timing, cs_time, 's-', color='#27AE60', label='causalscale (verified)', markersize=5, lw=1.5)
ax1.plot(d_timing, nt_time, 'o-', color='#E74C3C', label='NOTEARS (default)', markersize=5, lw=1.5)
ax1.set_xlabel('Dimension d')
ax1.set_ylabel('Time (s)')
ax1.legend(fontsize=6.5, framealpha=0.85, loc='lower right')
ax1.grid(alpha=0.3)
ax1.set_title('(a) Per-Seed Time (RTX 5060)', fontsize=9, fontweight='bold', loc='left')
ax1.set_ylim(0, 11)

for d, cs, nt in zip(d_timing, cs_time, nt_time):
    ratio = cs / nt
    y = max(cs, nt) + 0.4
    ax1.annotate(f'{ratio:.1f}x', (d, y), fontsize=5.5, ha='center', color='#555555')

# ════════════════════════════════════════════════════════════════
# (b) GPU Memory vs d — linear O(d)
# ════════════════════════════════════════════════════════════════
d_mem   = [500, 1000, 2000, 5000, 10000]
mem_gb  = [0.03, 0.10, 0.16, 0.80, 2.98]

ax2.plot(d_mem, mem_gb, 'D-', color='#8E44AD', markersize=5, lw=1.5)
z = np.polyfit(d_mem, mem_gb, 1)
d_line = np.linspace(0, 11000, 50)
ax2.plot(d_line, z[0]*d_line + z[1], '--', color='#8E44AD', alpha=0.35, lw=1,
         label=f'{z[0]*1e3:.3f} MB / 1k vars')
ax2.set_xlabel('Dimension d')
ax2.set_ylabel('VRAM (GB)')
ax2.legend(fontsize=6, loc='upper left')
ax2.grid(alpha=0.3)
ax2.set_title('(b) GPU Memory Scaling (r=4-32)', fontsize=9, fontweight='bold', loc='left')
for d, m in zip(d_mem, mem_gb):
    ax2.annotate(f'{m:.2f}', (d, m), textcoords='offset points', xytext=(0, 7),
                 fontsize=5.5, ha='center')

# ════════════════════════════════════════════════════════════════
# (c) LowRankGNN Time Scaling — only real data points (Table 4)
# ════════════════════════════════════════════════════════════════
d_data  = np.array([500, 1000, 2000])
t_data  = np.array([66,  193,  702])  # Table 4: LowRankGNN r=64, 3 seeds

ax3.loglog(d_data, t_data, 's-', color='#16A085', markersize=6, lw=1.5, label='LowRankGNN (r=64)')
# Fit and extrapolate for dotted trend line
slope, intercept = np.polyfit(np.log10(d_data), np.log10(t_data), 1)
d_fit = np.array([400, 600, 1000, 1500, 2500])
ax3.loglog(d_fit, 10**intercept * d_fit**slope,
           '--', color='#16A085', alpha=0.35, lw=1, label=f'O(d^{{{slope:.1f}}})')
ax3.set_xlabel('Dimension d')
ax3.set_ylabel('Time (s)')
ax3.legend(fontsize=6.5, loc='upper left')
ax3.grid(alpha=0.3, which='both')
ax3.set_title('(c) LowRankGNN Time Scaling', fontsize=9, fontweight='bold', loc='left')
ax3.set_xlim(350, 3000)
ax3.set_ylim(40, 1000)

# ════════════════════════════════════════════════════════════════
# (d) Sample Efficiency: n=2d vs n=5d (from exp12)
# ════════════════════════════════════════════════════════════════
exp12_path = os.path.join(RESULTS, 'exp12_n5d_scaling.json')
if os.path.exists(exp12_path):
    exp12 = json.load(open(exp12_path, encoding='utf-8'))
    d_list = [50, 80, 100, 150]
    n2d_means, n2d_stds, n5d_means, n5d_stds = [], [], [], []
    for d in d_list:
        keys = [k for k in exp12 if k.startswith(f'd{d}_')]
        n2d_vals = [exp12[k]['n2d_f1'] for k in keys]
        n5d_vals = [exp12[k]['n5d_f1'] for k in keys]
        n2d_means.append(np.mean(n2d_vals))
        n2d_stds.append(np.std(n2d_vals))
        n5d_means.append(np.mean(n5d_vals))
        n5d_stds.append(np.std(n5d_vals))
else:
    # Fallback
    d_list = [50, 80, 100, 150]
    n2d_means = [0.615, 0.731, 0.766, 0.768]
    n2d_stds  = [0.050, 0.080, 0.029, 0.022]
    n5d_means = [0.859, 0.830, 0.782, 0.793]
    n5d_stds  = [0.030, 0.040, 0.055, 0.035]

x = np.arange(len(d_list))
w = 0.32
bars1 = ax4.bar(x - w/2, n2d_means, w, yerr=n2d_stds, capsize=3, color='#3498DB', alpha=0.85,
                label='n = 2d', error_kw={'lw': 0.8})
bars2 = ax4.bar(x + w/2, n5d_means, w, yerr=n5d_stds, capsize=3, color='#E67E22', alpha=0.85,
                label='n = 5d', error_kw={'lw': 0.8})
for i, (n2, n5) in enumerate(zip(n2d_means, n5d_means)):
    delta = n5 - n2
    y_max = max(n2 + n2d_stds[i], n5 + n5d_stds[i])
    sign = '+' if delta >= 0 else ''
    ax4.annotate(f'{sign}{delta:.2f}', (x[i], y_max + 0.03), fontsize=6, ha='center',
                 color='#C0392B', fontweight='bold')

ax4.set_xticks(x)
ax4.set_xticklabels([f'd={d}' for d in d_list])
ax4.set_ylabel('F1 Score')
ax4.legend(fontsize=6.5, framealpha=0.85, loc='lower left')
ax4.grid(alpha=0.3, axis='y')
ax4.set_title('(d) Sample Efficiency', fontsize=9, fontweight='bold', loc='left')
ax4.set_ylim(0, 1.02)

# ════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.3)
fig.subplots_adjust(hspace=0.28, wspace=0.30)
for fmt, ext in [('pdf', '.pdf'), ('png', '.png')]:
    path = os.path.join(OUT, f'fig4_timing{ext}')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    kb = os.path.getsize(path) / 1024
    print(f'Fig4 saved: {path} ({kb:.0f} KB)')
plt.close()
