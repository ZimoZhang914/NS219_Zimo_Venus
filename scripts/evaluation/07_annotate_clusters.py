"""
Annotate Leiden clusters by their dominant microglia state signature.

Naming rules (no "Mixed" hedges):
  * top signature > 0.05 AND margin > 0.02:  "{State}"
  * top signature > 0      (positive but weak): "{State}-leaning"
  * top signature <= 0     (no positive signature): "Activated (low-signature)"
  * cluster size < MIN_CLUSTER_SIZE:           "Outlier"

If multiple clusters share the same name, the strongest gets the clean
name and the others are tagged with their second-strongest signature.

Outputs:
  - cluster_state_summary.csv          per-cluster signature score table
  - umap_annotated_state.png           UMAP with biological state names
  - umap_annotated_state_donors.png    same UMAP colored by AD/Control
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

# Tunables
RESOLUTION = 0.1
MIN_CLUSTER_SIZE = 100
STRONG_TOP_THRESHOLD = 0.05  # top signature must clear this for clean label
STRONG_MARGIN = 0.02         # AND margin to runner-up must exceed this

# ============================================================
# Load and re-cluster
# ============================================================
print('='*70)
print(f'Re-clustering at Leiden resolution {RESOLUTION}')
print('='*70)

adata = sc.read_h5ad(f'{PROJ}/results/vineseq_microglia_v1/annotated_microglia.h5ad')

if 'neighbors' not in adata.uns:
    sc.pp.neighbors(adata, use_rep='X_vae', n_neighbors=15, random_state=42)

sc.tl.leiden(adata, resolution=RESOLUTION, random_state=42,
             key_added=f'leiden_r{RESOLUTION}')
if 'X_umap' not in adata.obsm:
    sc.tl.umap(adata, random_state=42)

cluster_col = f'leiden_r{RESOLUTION}'
adata.obs['fine_cluster'] = adata.obs[cluster_col].copy()
n_clusters = adata.obs['fine_cluster'].nunique()
print(f'Found {n_clusters} clusters at resolution {RESOLUTION}')
print()

# ============================================================
# Score signatures per cluster
# ============================================================
score_cols = ['Homeostatic_score', 'DAM_score', 'IRM_Interferon_score',
              'Cytokine_NFkB_score', 'MHC_II_score', 'Lipid_Associated_score',
              'PVM_Border_score']

state_display = {
    'Homeostatic_score':     'Homeostatic',
    'DAM_score':             'DAM',
    'IRM_Interferon_score':  'IRM',
    'Cytokine_NFkB_score':   'Cytokine',
    'MHC_II_score':          'MHC-II',
    'Lipid_Associated_score':'Lipid-Assoc',
    'PVM_Border_score':      'PVM',
}

cluster_means = adata.obs.groupby('fine_cluster', observed=True)[score_cols].mean()
cluster_sizes = adata.obs['fine_cluster'].value_counts()

# ============================================================
# Decide label for each cluster
# ============================================================
print('='*70)
print('Per-cluster identity')
print('='*70)

# First pass — preliminary call for each cluster
prelim = {}
for cluster in cluster_means.index:
    means = cluster_means.loc[cluster]
    n = int(cluster_sizes[cluster])

    if n < MIN_CLUSTER_SIZE:
        prelim[cluster] = {
            'kind': 'outlier', 'n': n,
            'top': None, 'top_val': None,
            'second': None, 'second_val': None, 'margin': None,
        }
        continue

    sorted_states = means.sort_values(ascending=False)
    top_col = sorted_states.index[0]
    top_val = float(sorted_states.iloc[0])
    second_col = sorted_states.index[1]
    second_val = float(sorted_states.iloc[1])
    margin = top_val - second_val

    if top_val <= 0:
        kind = 'activated_low'
    elif top_val >= STRONG_TOP_THRESHOLD and margin >= STRONG_MARGIN:
        kind = 'strong'
    else:
        kind = 'leaning'

    prelim[cluster] = {
        'kind': kind, 'n': n,
        'top': state_display[top_col], 'top_val': top_val,
        'second': state_display[second_col], 'second_val': second_val,
        'margin': margin,
    }

# Second pass — build labels, ensuring uniqueness when clusters share top state
state_to_clusters = {}
for cluster, info in prelim.items():
    if info['kind'] == 'outlier':
        continue
    if info['kind'] == 'activated_low':
        continue
    state_to_clusters.setdefault(info['top'], []).append(
        (cluster, info['top_val'], info['margin'], info['n'], info['kind'])
    )

final_labels = {}
for cluster, info in prelim.items():
    if info['kind'] == 'outlier':
        final_labels[cluster] = f"Outlier (n={info['n']})"
        continue
    if info['kind'] == 'activated_low':
        final_labels[cluster] = 'Activated (low-signature)'
        continue
    # 'strong' or 'leaning' — handled together
    state = info['top']
    members = state_to_clusters[state]
    members_sorted = sorted(members, key=lambda x: -x[1])  # by top_val desc

    if len(members_sorted) == 1:
        # only one cluster has this top state
        final_labels[cluster] = state if info['kind'] == 'strong' else f'{state}-leaning'
    else:
        # multiple clusters — strongest gets the clean name, others get tag
        for rank, (c, tv, mg, n, knd) in enumerate(members_sorted):
            if c == cluster:
                if rank == 0:
                    final_labels[cluster] = state if info['kind'] == 'strong' else f'{state}-leaning'
                else:
                    second = info['second']
                    base = state if info['kind'] == 'strong' else f'{state}-leaning'
                    final_labels[cluster] = f'{base} ({second}+)'
                break

# Disambiguate any duplicates that snuck through
seen = {}
for c in sorted(final_labels.keys(), key=lambda x: int(x)):
    lbl = final_labels[c]
    if lbl in seen.values():
        # add a small numeric suffix
        n_dupes = sum(1 for v in seen.values() if v.startswith(lbl))
        final_labels[c] = f'{lbl} #{n_dupes + 1}'
    seen[c] = final_labels[c]

# ============================================================
# Print breakdown
# ============================================================
for cluster in sorted(prelim.keys(), key=lambda x: int(x)):
    info = prelim[cluster]
    label = final_labels[cluster]
    if info['kind'] == 'outlier':
        print(f'  cluster {cluster:>3} | n={info["n"]:>5} | tiny → "{label}"')
        continue
    print(f'  cluster {cluster:>3} | n={info["n"]:>5} | '
          f'top={info["top"]:<12} ({info["top_val"]:+.3f})  '
          f'2nd={info["second"]:<12} ({info["second_val"]:+.3f})  '
          f'margin={info["margin"]:+.3f}  →  "{label}"')

# ============================================================
# Apply labels and save
# ============================================================
adata.obs['state_annotation'] = adata.obs['fine_cluster'].astype(str).map(final_labels)
adata.obs['state_annotation'] = pd.Categorical(adata.obs['state_annotation'])

summary = cluster_means.copy()
summary.columns = [state_display[c] for c in summary.columns]
summary.insert(0, 'n_cells', cluster_sizes)
summary.insert(1, 'annotation', [final_labels[c] for c in summary.index])
summary.to_csv(f'{OUT}/cluster_state_summary.csv')
print(f'\nSaved per-cluster breakdown to {OUT}/cluster_state_summary.csv')

# ============================================================
# Plot annotated UMAP
# ============================================================
print('\nGenerating annotated UMAP...')

state_colors_base = {
    'Homeostatic': '#2D6A4F',
    'DAM':         '#C9302C',
    'Lipid-Assoc': '#8B6F47',
    'IRM':         '#5E548E',
    'MHC-II':      '#E07A5F',
    'PVM':         '#1A3A5C',
    'Cytokine':    '#D4A017',
}
extra_colors = ['#7A9CC6', '#A06CD5', '#F4A261', '#6B9080', '#B5838D',
                '#52B788', '#F08080', '#9C6644', '#E8A87C', '#85B79D']

def lighten(hex_color, amount=0.45):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f'#{r:02X}{g:02X}{b:02X}'

unique_labels = sorted(adata.obs['state_annotation'].cat.categories)
palette = {}
extra_i = 0
state_seen = set()

for lbl in unique_labels:
    if lbl.startswith('Outlier'):
        palette[lbl] = '#999999'
        continue
    if lbl == 'Activated (low-signature)':
        palette[lbl] = '#666666'
        continue
    base_state = lbl.split(' (')[0].replace('-leaning', '')
    if base_state in state_colors_base:
        if base_state in state_seen:
            palette[lbl] = lighten(state_colors_base[base_state], 0.45)
        else:
            palette[lbl] = state_colors_base[base_state]
            state_seen.add(base_state)
    else:
        palette[lbl] = extra_colors[extra_i % len(extra_colors)]
        extra_i += 1

fig, ax = plt.subplots(figsize=(10.5, 7))
for lbl in unique_labels:
    m = adata.obs['state_annotation'].values == lbl
    n = int(m.sum())
    ax.scatter(adata.obsm['X_umap'][m, 0],
               adata.obsm['X_umap'][m, 1],
               s=4, c=palette[lbl], alpha=0.75, edgecolors='none',
               label=f'{lbl} (n={n})')

ax.set_xlabel('UMAP1', fontsize=11)
ax.set_ylabel('UMAP2', fontsize=11)
ax.set_title(f'Annotated microglia subpopulations\nLeiden resolution {RESOLUTION} → {n_clusters} clusters, labeled by dominant signature',
             fontsize=13, color='#1A3A5C', fontweight='bold')
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
          frameon=True, framealpha=0.95, fontsize=8.5, markerscale=2.5)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(f'{OUT}/umap_annotated_state.png', dpi=180, bbox_inches='tight')
plt.close()
print(f'  Saved {OUT}/umap_annotated_state.png')

# AD vs Control overlay
fig, ax = plt.subplots(figsize=(10.5, 7))
for cond, color in [('Control', '#1A78A0'), ('AD', '#C9302C')]:
    m = adata.obs['condition'].values == cond
    n = int(m.sum())
    ax.scatter(adata.obsm['X_umap'][m, 0],
               adata.obsm['X_umap'][m, 1],
               s=3, c=color, alpha=0.55, edgecolors='none', label=f'{cond} (n={n})')
ax.set_xlabel('UMAP1', fontsize=11)
ax.set_ylabel('UMAP2', fontsize=11)
ax.set_title('Same UMAP — colored by AD vs Control donor',
             fontsize=13, color='#1A3A5C', fontweight='bold')
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
          frameon=True, framealpha=0.95, fontsize=10, markerscale=2.5)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(f'{OUT}/umap_annotated_state_donors.png', dpi=180, bbox_inches='tight')
plt.close()
print(f'  Saved {OUT}/umap_annotated_state_donors.png')

adata.write_h5ad(f'{PROJ}/results/vineseq_microglia_v1/annotated_microglia.h5ad')

# Final summary
print('\n' + '='*70)
print('FINAL CLUSTER → STATE MAPPING')
print('='*70)
mapping_df = pd.DataFrame([
    {
        'cluster': c,
        'n_cells': int(cluster_sizes[c]),
        'top_signature': prelim[c]['top'] or '—',
        'top_score': round(prelim[c]['top_val'], 3) if prelim[c]['top_val'] is not None else None,
        'annotation': final_labels[c],
    }
    for c in sorted(cluster_means.index, key=lambda x: int(x))
])
print(mapping_df.to_string(index=False))
