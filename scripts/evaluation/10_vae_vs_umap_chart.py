"""
Generate a VAE-vs-UMAP-only comparison figure for slide 11.

Uses the numbers already computed in 09_three_way_comparison.py.
Just plots VAE vs UMAP-only (omits PCA).
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ = '/Users/zimozhang/Desktop/GSE163577_RAW/microglia_pipeline'
OUT  = f'{PROJ}/results/positive_control_visuals'
os.makedirs(OUT, exist_ok=True)

# Numbers from 09_three_way_comparison.py output
data = {
    'VAE\n(32 dims)':       {'purity': 0.305, 'accuracy': 0.413, 'silhouette': -0.036},
    'UMAP-only\n(2 dims)':  {'purity': 0.227, 'accuracy': 0.298, 'silhouette': -0.078},
}
methods = list(data.keys())
colors = ['#1A3A5C', '#A06CD5']
random_baseline_purity = 0.143
naive_baseline_acc = 0.311

fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))

# ---- Panel 1: kNN purity ------------------------------------
ax = axes[0]
purity_vals = [data[m]['purity'] for m in methods]
bars = ax.bar(methods, purity_vals, color=colors, width=0.55)
ax.axhline(random_baseline_purity, color='darkred', linestyle='--', linewidth=1.2,
           label=f'random ({random_baseline_purity:.3f})')
for bar, val in zip(bars, purity_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('kNN purity (k=15)', fontsize=11)
ax.set_title('kNN purity', fontsize=12, color='#1A3A5C', fontweight='bold')
ax.set_ylim(0, max(purity_vals) * 1.30)
ax.legend(fontsize=9, loc='upper right')

# ---- Panel 2: label transfer accuracy -----------------------
ax = axes[1]
acc_vals = [data[m]['accuracy'] for m in methods]
bars = ax.bar(methods, acc_vals, color=colors, width=0.55)
ax.axhline(naive_baseline_acc, color='darkred', linestyle='--', linewidth=1.2,
           label=f'naive ({naive_baseline_acc:.3f})')
for bar, val in zip(bars, acc_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('label transfer accuracy', fontsize=11)
ax.set_title('Label transfer accuracy', fontsize=12, color='#1A3A5C', fontweight='bold')
ax.set_ylim(0, max(acc_vals) * 1.25)
ax.legend(fontsize=9, loc='upper right')

# ---- Panel 3: silhouette ------------------------------------
ax = axes[2]
sil_vals = [data[m]['silhouette'] for m in methods]
bars = ax.bar(methods, sil_vals, color=colors, width=0.55)
ax.axhline(0, color='darkred', linestyle='--', linewidth=1.2, label='no structure')
for bar, val in zip(bars, sil_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (-0.005 if val < 0 else 0.005),
            f'{val:.3f}', ha='center',
            va='top' if val < 0 else 'bottom',
            fontsize=11, fontweight='bold')
ax.set_ylabel('silhouette score', fontsize=11)
ax.set_title('Silhouette by ground truth', fontsize=12, color='#1A3A5C', fontweight='bold')
ax.set_ylim(min(sil_vals) * 1.4, 0.01)
ax.legend(fontsize=9, loc='lower right')

fig.suptitle('VAE 32-dim latent vs UMAP 2-dim — head to head on the same evaluation',
             fontsize=14, color='#1A3A5C', fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(f'{OUT}/vae_vs_umap_only.png', dpi=180, bbox_inches='tight')
plt.close(fig)

print(f'Saved {OUT}/vae_vs_umap_only.png')
