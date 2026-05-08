"""
Three-way comparison: VAE vs PCA vs UMAP-only as the basis for downstream
evaluation.

Question: is the VAE's 32-dim representation actually earning its keep,
or would a simpler approach work just as well?

For each method we use the SAME input (log-normalized HVG counts),
the SAME ground-truth labels (12,509 marker-derived), and the SAME
evaluation metrics (kNN purity, label transfer accuracy, silhouette).

Methods compared:
  1. VAE      — neural-network 32-dim encoding (what your pipeline uses)
  2. PCA      — linear 32-dim projection
  3. UMAP-only — non-linear 2-dim projection (the visualization tool)
                 used as if it were the primary representation
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

PROJ = '/Users/zimozhang/Desktop/GSE163577_RAW/microglia_pipeline'
sys.path.insert(0, f'{PROJ}/src')

from microglia_annotator.config import load_config
from microglia_annotator import preprocessing as pp_mod

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import silhouette_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import umap
from collections import Counter

OUT = f'{PROJ}/results/vineseq_microglia_v1/positive_control'
os.makedirs(OUT, exist_ok=True)

# ============================================================
# Step 1: Load data + ground-truth labels
# ============================================================
print('Loading data...')
cfg = load_config(f'{PROJ}/configs/default.yaml')
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

# Prepare matched input for PCA and UMAP-only
print('Preparing log-normalized input matrix for PCA / UMAP-only...')
X_counts = pp_mod.get_vae_input(adata, cfg)
X_log = np.log1p(X_counts / X_counts.sum(axis=1, keepdims=True).clip(min=1) * 1e4)
scaler = StandardScaler()
X_log_scaled = scaler.fit_transform(X_log)

# ============================================================
# Step 2: Get the three representations
# ============================================================
print('Getting VAE latent (already trained, 32 dim)...')
latent_vae = adata.obsm['X_vae']

print('Computing PCA latent (32 dim)...')
pca = PCA(n_components=32, random_state=42)
latent_pca = pca.fit_transform(X_log_scaled)
print(f'  PCA explains {pca.explained_variance_ratio_.sum()*100:.1f}% of variance in 32 components')

print('Computing UMAP-only embedding (2 dim) on log-normalized counts...')
print('  This may take 1-3 minutes...')
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
latent_umap_only = reducer.fit_transform(X_log_scaled)
print(f'  UMAP-only shape: {latent_umap_only.shape}')

# ============================================================
# Step 3: Run identical evaluation on all three
# ============================================================
def evaluate(X, y, name):
    """Run kNN purity, silhouette, label transfer."""
    mask = y != 'Ambiguous'
    Xm = X[mask]
    ym = y[mask]

    # kNN purity
    K = 15
    nn = NearestNeighbors(n_neighbors=K + 1, metric='euclidean')
    nn.fit(Xm)
    _, idx = nn.kneighbors(Xm)
    matches = (ym[idx[:, 1:]] == ym[:, None])
    purity = float(matches.mean(axis=1).mean())
    purity_per_state = {
        s: float(matches[ym == s].mean(axis=1).mean())
        for s in sorted(np.unique(ym))
    }

    # Silhouette (subsampled for speed)
    rng = np.random.default_rng(42)
    sub = rng.choice(len(Xm), size=min(5000, len(Xm)), replace=False)
    try:
        sil = float(silhouette_score(Xm[sub], ym[sub]))
    except Exception:
        sil = float('nan')

    # Label transfer
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xm, ym, test_size=0.2, random_state=42, stratify=ym
    )
    clf = KNeighborsClassifier(n_neighbors=15, weights='distance')
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = float(accuracy_score(y_te, y_pred))
    bacc = float(balanced_accuracy_score(y_te, y_pred))
    most_common = Counter(y_tr).most_common(1)[0][0]
    naive = float((y_te == most_common).mean())

    return {
        'method': name, 'n_dims': X.shape[1],
        'kNN_purity': purity, 'silhouette': sil,
        'accuracy': acc, 'balanced_acc': bacc, 'naive_baseline': naive,
        'per_state': purity_per_state,
    }

print('\nEvaluating all three methods...')
res_vae = evaluate(latent_vae, gt, 'VAE')
res_pca = evaluate(latent_pca, gt, 'PCA')
res_umap = evaluate(latent_umap_only, gt, 'UMAP-only')

# ============================================================
# Step 4: Print comparison
# ============================================================
print()
print('=' * 75)
print('THREE-WAY COMPARISON: VAE vs PCA vs UMAP-only')
print('=' * 75)
print(f'(Same 12,509 labeled microglia, same metrics, same ground truth)')
print()

table = pd.DataFrame([
    {
        'Metric': 'Dimensions',
        'VAE': str(res_vae['n_dims']),
        'PCA': str(res_pca['n_dims']),
        'UMAP-only': str(res_umap['n_dims']),
    },
    {
        'Metric': 'kNN purity (k=15)',
        'VAE': f"{res_vae['kNN_purity']:.3f}",
        'PCA': f"{res_pca['kNN_purity']:.3f}",
        'UMAP-only': f"{res_umap['kNN_purity']:.3f}",
    },
    {
        'Metric': 'Silhouette',
        'VAE': f"{res_vae['silhouette']:.3f}",
        'PCA': f"{res_pca['silhouette']:.3f}",
        'UMAP-only': f"{res_umap['silhouette']:.3f}",
    },
    {
        'Metric': 'Label transfer accuracy',
        'VAE': f"{res_vae['accuracy']:.3f}",
        'PCA': f"{res_pca['accuracy']:.3f}",
        'UMAP-only': f"{res_umap['accuracy']:.3f}",
    },
    {
        'Metric': 'Balanced accuracy',
        'VAE': f"{res_vae['balanced_acc']:.3f}",
        'PCA': f"{res_pca['balanced_acc']:.3f}",
        'UMAP-only': f"{res_umap['balanced_acc']:.3f}",
    },
])
print(table.to_string(index=False))

# Per-state purity
print()
print('Per-state kNN purity:')
states = sorted(res_vae['per_state'].keys())
print(f"{'State':<20} {'VAE':<10} {'PCA':<10} {'UMAP-only':<10}")
print('-' * 52)
for s in states:
    v = res_vae['per_state'][s]
    p = res_pca['per_state'][s]
    u = res_umap['per_state'][s]
    print(f'{s:<20} {v:<10.3f} {p:<10.3f} {u:<10.3f}')

# Save
out_df = pd.DataFrame([
    {
        'method': 'VAE',
        'n_dims': res_vae['n_dims'],
        'kNN_purity': res_vae['kNN_purity'],
        'silhouette': res_vae['silhouette'],
        'accuracy': res_vae['accuracy'],
        'balanced_acc': res_vae['balanced_acc'],
    },
    {
        'method': 'PCA',
        'n_dims': res_pca['n_dims'],
        'kNN_purity': res_pca['kNN_purity'],
        'silhouette': res_pca['silhouette'],
        'accuracy': res_pca['accuracy'],
        'balanced_acc': res_pca['balanced_acc'],
    },
    {
        'method': 'UMAP-only',
        'n_dims': res_umap['n_dims'],
        'kNN_purity': res_umap['kNN_purity'],
        'silhouette': res_umap['silhouette'],
        'accuracy': res_umap['accuracy'],
        'balanced_acc': res_umap['balanced_acc'],
    },
])
out_df.to_csv(f'{OUT}/three_way_comparison.csv', index=False)

# ============================================================
# Step 5: Visualization
# ============================================================
print('\nBuilding comparison bar chart...')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

methods = ['VAE\n(32 dims)', 'PCA\n(32 dims)', 'UMAP-only\n(2 dims)']
purity = [res_vae['kNN_purity'], res_pca['kNN_purity'], res_umap['kNN_purity']]
accuracy = [res_vae['accuracy'], res_pca['accuracy'], res_umap['accuracy']]
sil = [res_vae['silhouette'], res_pca['silhouette'], res_umap['silhouette']]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

# kNN purity
ax = axes[0]
colors = ['#1A3A5C', '#4A6FA5', '#A06CD5']
bars = ax.bar(methods, purity, color=colors)
ax.axhline(0.143, color='darkred', linestyle='--', linewidth=1.2,
           label='random baseline')
for bar, val in zip(bars, purity):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('kNN purity (k=15)')
ax.set_title('kNN purity', fontsize=12, color='#1A3A5C', fontweight='bold')
ax.set_ylim(0, max(purity) * 1.25)
ax.legend(fontsize=9)

# Label transfer accuracy
ax = axes[1]
naive = res_vae['naive_baseline']
bars = ax.bar(methods, accuracy, color=colors)
ax.axhline(naive, color='darkred', linestyle='--', linewidth=1.2,
           label=f'naive baseline ({naive:.3f})')
for bar, val in zip(bars, accuracy):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('label transfer accuracy')
ax.set_title('Label transfer accuracy', fontsize=12, color='#1A3A5C', fontweight='bold')
ax.set_ylim(0, max(accuracy) * 1.25)
ax.legend(fontsize=9)

# Silhouette
ax = axes[2]
bars = ax.bar(methods, sil, color=colors)
ax.axhline(0, color='darkred', linestyle='--', linewidth=1.2,
           label='no structure')
for bar, val in zip(bars, sil):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.005 if val >= 0 else -0.015),
            f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('silhouette score')
ax.set_title('Silhouette by ground truth', fontsize=12, color='#1A3A5C', fontweight='bold')
ax.legend(fontsize=9)

fig.suptitle('Three-way comparison: VAE vs PCA vs UMAP-only as primary representation',
             fontsize=14, color='#1A3A5C', fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(f'{OUT}/three_way_comparison.png', dpi=180, bbox_inches='tight')
plt.close(fig)

print()
print('=' * 75)
print('Saved comparison to:')
print(f'  {OUT}/three_way_comparison.csv')
print(f'  {OUT}/three_way_comparison.png')
