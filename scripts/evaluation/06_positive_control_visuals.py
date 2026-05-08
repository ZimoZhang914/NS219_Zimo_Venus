"""
Visualize both positive controls (synthetic + real marker-derived) for the deck.

Outputs:
  - synthetic_umap_truth.png      UMAP of synthetic data colored by true state
  - synthetic_umap_leiden.png     UMAP of synthetic data colored by Leiden cluster
  - real_umap_groundtruth.png     UMAP of real microglia colored by ground-truth label
  - synthetic_vs_real_panel.png   Side-by-side comparison panel
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PROJ = '/Users/zimozhang/Desktop/GSE163577_RAW/microglia_pipeline'
sys.path.insert(0, f'{PROJ}/src')

OUT = f'{PROJ}/results/positive_control_visuals'
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 1. Load data + rebuild ground-truth labels for real data
# ============================================================
print('Loading real-data AnnData...')
adata_real = sc.read_h5ad(f'{PROJ}/results/vineseq_microglia_v1/annotated_microglia.h5ad')

score_cols = ['Homeostatic_score', 'DAM_score', 'IRM_Interferon_score',
              'Cytokine_NFkB_score', 'MHC_II_score', 'Lipid_Associated_score',
              'PVM_Border_score']
state_names = ['Homeostatic', 'DAM', 'IRM', 'Cytokine_NFkB',
               'MHC-II', 'Lipid_Assoc', 'PVM']
score_matrix = adata_real.obs[score_cols].values
top1_idx = score_matrix.argmax(axis=1)
top1_score = score_matrix.max(axis=1)
sorted_scores = np.sort(score_matrix, axis=1)
margin = sorted_scores[:, -1] - sorted_scores[:, -2]
confident = (top1_score > 0) & (margin > 0.05)
gt = np.array(['Ambiguous'] * adata_real.n_obs, dtype=object)
gt[confident] = [state_names[i] for i in top1_idx[confident]]
adata_real.obs['ground_truth'] = pd.Categorical(gt)

print('Loading synthetic AnnData...')
adata_syn = sc.read_h5ad(f'{PROJ}/results/synthetic_positive_control/synthetic_microglia_annotated.h5ad')

# ============================================================
# 2. Synthetic UMAP — by ground truth and by Leiden
# ============================================================
print('\nBuilding synthetic UMAPs...')

# UMAP by ground truth
fig, ax = plt.subplots(figsize=(6, 5))
colors_truth = {'Homeostatic': '#2D6A4F', 'DAM': '#C9302C', 'Proliferative': '#1A3A5C'}
for ct in ['Homeostatic', 'DAM', 'Proliferative']:
    m = adata_syn.obs['cell_type'].values == ct
    ax.scatter(adata_syn.obsm['X_umap'][m, 0],
               adata_syn.obsm['X_umap'][m, 1],
               s=12, c=colors_truth[ct], label=ct, alpha=0.85, edgecolors='none')
ax.set_xlabel('UMAP1', fontsize=11)
ax.set_ylabel('UMAP2', fontsize=11)
ax.set_title('Synthetic data — colored by ground-truth cell type\n(known by construction)',
             fontsize=12, color='#1A3A5C')
ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=10)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(f'{OUT}/synthetic_umap_truth.png', dpi=180, bbox_inches='tight')
plt.close()

# UMAP by Leiden cluster
fig, ax = plt.subplots(figsize=(6, 5))
leiden_vals = sorted(adata_syn.obs['leiden'].unique())
cmap = plt.cm.get_cmap('Set2', len(leiden_vals))
for i, lv in enumerate(leiden_vals):
    m = adata_syn.obs['leiden'].values == lv
    ax.scatter(adata_syn.obsm['X_umap'][m, 0],
               adata_syn.obsm['X_umap'][m, 1],
               s=12, c=[cmap(i)], label=f'Cluster {lv}', alpha=0.85, edgecolors='none')
ax.set_xlabel('UMAP1', fontsize=11)
ax.set_ylabel('UMAP2', fontsize=11)
ax.set_title('Synthetic data — colored by VAE+Leiden cluster\n(every cluster 100% pure for one cell type)',
             fontsize=12, color='#1A3A5C')
ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=9)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(f'{OUT}/synthetic_umap_leiden.png', dpi=180, bbox_inches='tight')
plt.close()

print(f'  Saved {OUT}/synthetic_umap_truth.png')
print(f'  Saved {OUT}/synthetic_umap_leiden.png')

# ============================================================
# 3. Real UMAP — colored by ground-truth label
# ============================================================
print('\nBuilding real-data UMAP by ground truth...')

# Compute UMAP on real data if not already present
if 'X_umap' not in adata_real.obsm:
    sc.pp.neighbors(adata_real, use_rep='X_vae', n_neighbors=15, random_state=42)
    sc.tl.umap(adata_real, random_state=42)

state_palette = {
    'Homeostatic':   '#2D6A4F',
    'Lipid_Assoc':   '#8B6F47',
    'PVM':           '#1A3A5C',
    'IRM':           '#5E548E',
    'MHC-II':        '#E07A5F',
    'DAM':           '#C9302C',
    'Cytokine_NFkB': '#D4A017',
    'Ambiguous':     '#CCCCCC',
}

fig, ax = plt.subplots(figsize=(8, 6))
# Plot ambiguous first (background)
m_amb = adata_real.obs['ground_truth'].values == 'Ambiguous'
ax.scatter(adata_real.obsm['X_umap'][m_amb, 0],
           adata_real.obsm['X_umap'][m_amb, 1],
           s=4, c=state_palette['Ambiguous'], alpha=0.4, edgecolors='none',
           label=f"Ambiguous (n={int(m_amb.sum())})")

# Plot each labeled state on top, ordered by size (smallest last so they're visible)
labeled_states = [s for s in state_palette if s != 'Ambiguous']
state_sizes = [(s, int((adata_real.obs['ground_truth'].values == s).sum())) for s in labeled_states]
state_sizes.sort(key=lambda x: -x[1])  # largest first

for state, n in state_sizes:
    m = adata_real.obs['ground_truth'].values == state
    if m.sum() == 0:
        continue
    ax.scatter(adata_real.obsm['X_umap'][m, 0],
               adata_real.obsm['X_umap'][m, 1],
               s=5, c=state_palette[state], alpha=0.75, edgecolors='none',
               label=f"{state} (n={n})")

ax.set_xlabel('UMAP1', fontsize=11)
ax.set_ylabel('UMAP2', fontsize=11)
ax.set_title(f'Real data — colored by marker-derived ground-truth label\n12,509 confidently labeled / 19,001 total cells',
             fontsize=12, color='#1A3A5C')
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
          frameon=True, framealpha=0.9, fontsize=9, markerscale=2)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(f'{OUT}/real_umap_groundtruth.png', dpi=180, bbox_inches='tight')
plt.close()
print(f'  Saved {OUT}/real_umap_groundtruth.png')

# ============================================================
# 4. Side-by-side comparison panel for the deck
# ============================================================
print('\nBuilding side-by-side comparison panel...')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: synthetic by ground truth
ax = axes[0]
for ct in ['Homeostatic', 'DAM', 'Proliferative']:
    m = adata_syn.obs['cell_type'].values == ct
    ax.scatter(adata_syn.obsm['X_umap'][m, 0],
               adata_syn.obsm['X_umap'][m, 1],
               s=14, c=colors_truth[ct], label=ct, alpha=0.85, edgecolors='none')
ax.set_xlabel('UMAP1', fontsize=11)
ax.set_ylabel('UMAP2', fontsize=11)
ax.set_title('Synthetic — known ground truth\nkNN purity = 1.000 (random 0.333)',
             fontsize=13, color='#2D6A4F', fontweight='bold')
ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=10)
ax.set_xticks([]); ax.set_yticks([])

# Right: real by ground truth
ax = axes[1]
m_amb = adata_real.obs['ground_truth'].values == 'Ambiguous'
ax.scatter(adata_real.obsm['X_umap'][m_amb, 0],
           adata_real.obsm['X_umap'][m_amb, 1],
           s=3, c=state_palette['Ambiguous'], alpha=0.35, edgecolors='none',
           label='Ambiguous')
for state, n in state_sizes:
    m = adata_real.obs['ground_truth'].values == state
    if m.sum() == 0:
        continue
    ax.scatter(adata_real.obsm['X_umap'][m, 0],
               adata_real.obsm['X_umap'][m, 1],
               s=4, c=state_palette[state], alpha=0.7, edgecolors='none',
               label=state)
ax.set_xlabel('UMAP1', fontsize=11)
ax.set_ylabel('UMAP2', fontsize=11)
ax.set_title('Real data — marker-derived ground truth\nkNN purity = 0.305 (random 0.143; 2.14x lift)',
             fontsize=13, color='#1A3A5C', fontweight='bold')
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
          frameon=True, framealpha=0.9, fontsize=8, markerscale=2)
ax.set_xticks([]); ax.set_yticks([])

fig.suptitle('Positive control evaluation: pipeline recovers known structure',
             fontsize=15, color='#1A3A5C', fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/synthetic_vs_real_panel.png', dpi=180, bbox_inches='tight')
plt.close()
print(f'  Saved {OUT}/synthetic_vs_real_panel.png')

print()
print('='*60)
print('All positive control visualizations generated')
print('='*60)
print(f'Output directory: {OUT}/')
print()
print('For your slides:')
print(f'  Slide 9 — drag in synthetic_vs_real_panel.png')
print(f'  Backup figures (in case you want a different layout):')
print(f'    - synthetic_umap_truth.png')
print(f'    - synthetic_umap_leiden.png')
print(f'    - real_umap_groundtruth.png')
