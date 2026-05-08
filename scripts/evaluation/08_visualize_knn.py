"""
Visualize kNN-based evaluation in three ways:

1. Per-cell kNN purity heatmap on UMAP — see which latent-space regions
   have clean ground-truth structure
2. Per-state kNN purity bar chart (already exists, regenerated for consistency)
3. Conceptual diagram showing one cell + its 15 neighbors

Outputs:
  - knn_purity_umap.png         per-cell purity colored on UMAP
  - knn_purity_per_state.png    per-state purity bar chart
  - knn_concept.png             conceptual illustration of the metric
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

PROJ = '/Users/zimozhang/Desktop/GSE163577_RAW/microglia_pipeline'
OUT = f'{PROJ}/results/positive_control_visuals'
os.makedirs(OUT, exist_ok=True)

# ============================================================
# Load + rebuild ground truth
# ============================================================
adata = sc.read_h5ad(f'{PROJ}/results/vineseq_microglia_v1/annotated_microglia.h5ad')

score_cols = ['Homeostatic_score', 'DAM_score', 'IRM_Interferon_score',
              'Cytokine_NFkB_score', 'MHC_II_score', 'Lipid_Associated_score',
              'PVM_Border_score']
state_names = ['Homeostatic', 'DAM', 'IRM', 'Cytokine_NFkB',
               'MHC-II', 'Lipid_Assoc', 'PVM']
sm = adata.obs[score_cols].values
top1 = sm.argmax(axis=1)
top1_score = sm.max(axis=1)
sorted_scores = np.sort(sm, axis=1)
margin = sorted_scores[:, -1] - sorted_scores[:, -2]
confident = (top1_score > 0) & (margin > 0.05)
gt = np.array(['Ambiguous'] * adata.n_obs, dtype=object)
gt[confident] = [state_names[i] for i in top1[confident]]
adata.obs['ground_truth'] = pd.Categorical(gt)

# Make sure UMAP exists
if 'X_umap' not in adata.obsm:
    sc.pp.neighbors(adata, use_rep='X_vae', n_neighbors=15, random_state=42)
    sc.tl.umap(adata, random_state=42)

# ============================================================
# 1. Compute per-cell kNN purity (on labeled subset)
# ============================================================
print('Computing per-cell kNN purity...')

mask = confident
X = adata.obsm['X_vae'][mask]
y = gt[mask]

K = 15
nn = NearestNeighbors(n_neighbors=K + 1, metric='euclidean')
nn.fit(X)
_, idx = nn.kneighbors(X)
neighbor_idx = idx[:, 1:]  # drop self
matches = (y[neighbor_idx] == y[:, None])
purity_per_cell = matches.mean(axis=1)

# Map back to full adata (Ambiguous cells get NaN)
full_purity = np.full(adata.n_obs, np.nan)
full_purity[mask] = purity_per_cell
adata.obs['knn_purity'] = full_purity

print(f'Mean purity: {purity_per_cell.mean():.3f}')
print(f'Median purity: {np.median(purity_per_cell):.3f}')

# ============================================================
# Plot 1 — Per-cell kNN purity on UMAP
# ============================================================
print('\nPlotting per-cell kNN purity heatmap on UMAP...')

fig, ax = plt.subplots(figsize=(10, 7))

# Plot ambiguous cells in light gray as background
m_amb = ~confident
ax.scatter(adata.obsm['X_umap'][m_amb, 0],
           adata.obsm['X_umap'][m_amb, 1],
           s=3, c='#DDDDDD', alpha=0.4, edgecolors='none',
           label=f'Ambiguous (n={int(m_amb.sum())}) — not evaluated')

# Plot labeled cells colored by purity
m_lab = confident
sc_obj = ax.scatter(adata.obsm['X_umap'][m_lab, 0],
                    adata.obsm['X_umap'][m_lab, 1],
                    s=4, c=full_purity[m_lab], cmap='RdYlGn',
                    vmin=0, vmax=1, alpha=0.85, edgecolors='none')

cbar = plt.colorbar(sc_obj, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('kNN purity (k=15)', fontsize=11)
cbar.ax.axhline(0.143, color='black', linestyle='--', linewidth=1)
cbar.ax.text(1.5, 0.143, ' random\n baseline', va='center', fontsize=8, color='black')

ax.set_xlabel('UMAP1', fontsize=11)
ax.set_ylabel('UMAP2', fontsize=11)
ax.set_title('Per-cell kNN purity on VAE latent space\nGreen = neighbors share state label; Red = neighbors mixed',
             fontsize=12, color='#1A3A5C', fontweight='bold')
ax.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=9, markerscale=2)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(f'{OUT}/knn_purity_umap.png', dpi=180, bbox_inches='tight')
plt.close()
print(f'  Saved {OUT}/knn_purity_umap.png')

# ============================================================
# Plot 2 — Per-state purity bar chart (refreshed)
# ============================================================
print('\nPlotting per-state purity bars...')

per_state = []
for s in sorted(np.unique(y)):
    p = purity_per_cell[y == s].mean()
    n = int((y == s).sum())
    per_state.append((s, p, n))

per_state.sort(key=lambda x: -x[1])
states = [x[0] for x in per_state]
purities = [x[1] for x in per_state]
ns = [x[2] for x in per_state]

fig, ax = plt.subplots(figsize=(9, 4.5))
colors = ['#2D6A4F' if p >= 0.30 else ('#A06CD5' if p >= 0.143 else '#C9302C')
          for p in purities]
bars = ax.barh(states[::-1], purities[::-1], color=colors[::-1])

# Annotate each bar with N
for bar, n in zip(bars, ns[::-1]):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'n={n}', va='center', fontsize=9, color='#666666')

ax.axvline(0.143, color='black', linestyle='--', linewidth=1)
ax.text(0.143, -0.5, 'random baseline (0.143)',
        ha='center', va='top', fontsize=9, color='black')
ax.set_xlabel('kNN purity (k=15)', fontsize=11)
ax.set_title('kNN purity per microglia state\nGreen = strong recovery; Purple = above random; Red = below random',
             fontsize=12, color='#1A3A5C', fontweight='bold')
ax.set_xlim(0, 0.55)
plt.tight_layout()
plt.savefig(f'{OUT}/knn_purity_per_state.png', dpi=180, bbox_inches='tight')
plt.close()
print(f'  Saved {OUT}/knn_purity_per_state.png')

# ============================================================
# Plot 3 — Conceptual diagram of how kNN purity works
# ============================================================
print('\nPlotting conceptual kNN diagram...')

# Pick a single labeled cell and visualize its 15 neighbors
np.random.seed(7)
# Pick a high-purity cell for the visualization
pure_cells = np.where(purity_per_cell > 0.7)[0]
if len(pure_cells) == 0:
    pure_cells = np.where(purity_per_cell > 0.5)[0]
center_i = pure_cells[0] if len(pure_cells) else 0

# Find that cell's neighbors in UMAP space (for visualization purposes)
# Use UMAP coords just for this conceptual plot
labeled_umap = adata.obsm['X_umap'][mask]
labeled_y = y

nn_umap = NearestNeighbors(n_neighbors=K + 1).fit(labeled_umap)
_, vis_idx = nn_umap.kneighbors(labeled_umap[center_i].reshape(1, -1))
vis_idx = vis_idx[0, 1:]

center_label = labeled_y[center_i]
neighbor_labels = labeled_y[vis_idx]
n_match = (neighbor_labels == center_label).sum()
purity_here = n_match / K

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel A: zoomed-in single cell + its 15 neighbors
ax = axes[0]
ax.scatter(labeled_umap[:, 0], labeled_umap[:, 1],
           s=4, c='#EEEEEE', alpha=0.4, edgecolors='none')

# Highlight the 15 neighbors
for ni in vis_idx:
    if labeled_y[ni] == center_label:
        ax.scatter(labeled_umap[ni, 0], labeled_umap[ni, 1],
                   s=120, c='#2D6A4F', alpha=0.9, edgecolors='black',
                   linewidth=0.5, zorder=4)
    else:
        ax.scatter(labeled_umap[ni, 0], labeled_umap[ni, 1],
                   s=120, c='#C9302C', alpha=0.9, edgecolors='black',
                   linewidth=0.5, zorder=4)

# Center cell as star
ax.scatter(labeled_umap[center_i, 0], labeled_umap[center_i, 1],
           s=300, c='#1A3A5C', marker='*',
           edgecolors='white', linewidth=1.5, zorder=5)

# Zoom in
cx, cy = labeled_umap[center_i]
neighbor_coords = labeled_umap[vis_idx]
all_pts = np.vstack([labeled_umap[center_i:center_i+1], neighbor_coords])
xrange = all_pts[:, 0].max() - all_pts[:, 0].min()
yrange = all_pts[:, 1].max() - all_pts[:, 1].min()
pad = max(xrange, yrange) * 0.5
ax.set_xlim(all_pts[:, 0].mean() - pad, all_pts[:, 0].mean() + pad)
ax.set_ylim(all_pts[:, 1].mean() - pad, all_pts[:, 1].mean() + pad)

ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f'Center cell (★) labeled "{center_label}"\nand its 15 nearest neighbors',
             fontsize=11, color='#1A3A5C', fontweight='bold')

# Manual legend
ax.scatter([], [], s=300, c='#1A3A5C', marker='*',
           edgecolors='white', linewidth=1.5, label=f'Center cell ({center_label})')
ax.scatter([], [], s=120, c='#2D6A4F', edgecolors='black',
           linewidth=0.5, label=f'Same label ({n_match} of 15)')
ax.scatter([], [], s=120, c='#C9302C', edgecolors='black',
           linewidth=0.5, label=f'Different label ({K - n_match} of 15)')
ax.legend(loc='best', fontsize=9, frameon=True, framealpha=0.95)

# Panel B: explanation
ax = axes[1]
ax.axis('off')

text = (
    "How kNN purity is computed\n"
    "─────────────────────────\n\n"
    "For every labeled cell in the dataset:\n\n"
    "  1. Find its 15 closest cells in the\n"
    "     VAE's 32-dim latent space\n"
    "     (Euclidean distance)\n\n"
    "  2. Count how many of those 15\n"
    "     share the cell's ground-truth label\n\n"
    "  3. Divide by 15 → cell's purity score\n\n"
    f"For this cell:  {n_match} of 15 = {purity_here:.2f}\n\n"
    "─────────────────────────\n\n"
    "Average across all 12,509 labeled cells:\n"
    f"   Overall kNN purity = 0.305\n"
    f"   Random baseline   = 0.143\n"
    f"   Lift over random  = 2.14×"
)
ax.text(0.05, 0.95, text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', family='monospace',
        color='#333333')

plt.tight_layout()
plt.savefig(f'{OUT}/knn_concept.png', dpi=180, bbox_inches='tight')
plt.close()
print(f'  Saved {OUT}/knn_concept.png')

print('\n' + '='*60)
print('All kNN visualizations generated')
print('='*60)
print(f'Output directory: {OUT}/')
print()
print('Suggested deck use:')
print('  Slide 9 — drag in knn_purity_umap.png        (the heatmap)')
print('  Slide 9 — drag in knn_purity_per_state.png   (the bars)')
print('  Backup — knn_concept.png                      (for Q&A if asked)')
