"""
Head-to-head: VAE latent vs PCA on identical evaluation.

Both methods get:
  - The same input matrix (3,000 HVGs from the same 19,001 microglia)
  - The same number of dimensions (32)
  - The same evaluation metrics (kNN purity, label transfer, silhouette)
  - The same ground-truth labels (12,509 marker-derived state labels)

If the VAE outperforms PCA, that's evidence the non-linear encoder
captures microglia biology better than a linear projection.
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/zimozhang/Desktop/GSE163577_RAW/microglia_pipeline/src')

from microglia_annotator.config import load_config
from microglia_annotator import preprocessing as pp_mod

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import silhouette_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

PROJ = '/Users/zimozhang/Desktop/GSE163577_RAW/microglia_pipeline'
OUT  = f'{PROJ}/results/vineseq_microglia_v1/positive_control'

# ============================================================
# Step 1: Load data and rebuild ground truth
# ============================================================
print('Loading data...')
cfg = load_config(f'{PROJ}/configs/default.yaml')
adata = sc.read_h5ad(f'{PROJ}/results/vineseq_microglia_v1/annotated_microglia.h5ad')

# Rebuild ground truth (same as Step 1)
score_cols = ['Homeostatic_score', 'DAM_score', 'IRM_Interferon_score',
              'Cytokine_NFkB_score', 'MHC_II_score', 'Lipid_Associated_score',
              'PVM_Border_score']
state_names = ['Homeostatic', 'DAM', 'IRM', 'Cytokine_NFkB',
               'MHC-II', 'Lipid_Assoc', 'PVM']
score_matrix = adata.obs[score_cols].values
top1_idx = score_matrix.argmax(axis=1)
top1_score = score_matrix.max(axis=1)
sorted_scores = np.sort(score_matrix, axis=1)
margin = sorted_scores[:, -1] - sorted_scores[:, -2]
confident = (top1_score > 0) & (margin > 0.05)
gt = np.array(['Ambiguous'] * adata.n_obs, dtype=object)
gt[confident] = [state_names[i] for i in top1_idx[confident]]

# Get the SAME input matrix the VAE saw (raw counts on HVGs)
print('Preparing matched input matrices...')
X_counts = pp_mod.get_vae_input(adata, cfg)
print(f'Input shape: {X_counts.shape}')

# For PCA we'll use the standard preprocessing pipeline:
# log1p-normalize the counts, then standardize per gene, then PCA
# This is the most charitable comparison for PCA.
X_log = np.log1p(X_counts / X_counts.sum(axis=1, keepdims=True).clip(min=1) * 1e4)
scaler = StandardScaler()
X_log_scaled = scaler.fit_transform(X_log)

# ============================================================
# Step 2: Compute PCA latent (32 components, same as VAE)
# ============================================================
print('Running PCA (32 components)...')
pca = PCA(n_components=32, random_state=42)
latent_pca = pca.fit_transform(X_log_scaled)
var_explained = pca.explained_variance_ratio_.sum() * 100
print(f'PCA explains {var_explained:.1f}% of variance in 32 components')

# ============================================================
# Step 3: Get VAE latent (already computed)
# ============================================================
latent_vae = adata.obsm['X_vae']
print(f'VAE latent shape: {latent_vae.shape}')
print(f'PCA latent shape: {latent_pca.shape}')

# ============================================================
# Step 4: Run identical evaluation on both
# ============================================================
def evaluate(X, y, name):
    """Run kNN purity, silhouette, label transfer on a latent embedding."""
    mask = y != 'Ambiguous'
    Xm = X[mask]
    ym = y[mask]

    # METRIC 1: kNN purity
    K = 15
    nn = NearestNeighbors(n_neighbors=K + 1, metric='euclidean')
    nn.fit(Xm)
    _, idx = nn.kneighbors(Xm)
    neighbor_idx = idx[:, 1:]
    neighbor_labels = ym[neighbor_idx]
    matches = (neighbor_labels == ym[:, None])
    purity = matches.mean(axis=1).mean()

    # METRIC 2: silhouette (subsampled)
    rng = np.random.default_rng(42)
    sub = rng.choice(len(Xm), size=min(5000, len(Xm)), replace=False)
    try:
        sil = silhouette_score(Xm[sub], ym[sub])
    except Exception:
        sil = float('nan')

    # METRIC 3: label transfer
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xm, ym, test_size=0.2, random_state=42, stratify=ym
    )
    clf = KNeighborsClassifier(n_neighbors=15, weights='distance')
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    bacc = balanced_accuracy_score(y_te, y_pred)
    most_common = Counter(y_tr).most_common(1)[0][0]
    naive = (y_te == most_common).mean()

    # Per-state purity
    per_state = {}
    for s in sorted(np.unique(ym)):
        per_state[s] = float(matches[ym == s].mean(axis=1).mean())

    return {
        'method': name,
        'kNN_purity': float(purity),
        'silhouette': float(sil),
        'accuracy': float(acc),
        'balanced_acc': float(bacc),
        'naive_baseline': float(naive),
        'lift_over_naive': float(acc - naive),
        'per_state': per_state,
    }

print()
print('Evaluating VAE latent...')
res_vae = evaluate(latent_vae, gt, 'VAE')
print('Evaluating PCA latent...')
res_pca = evaluate(latent_pca, gt, 'PCA')

# ============================================================
# Step 5: Print and save head-to-head comparison
# ============================================================
print()
print('=' * 70)
print('HEAD-TO-HEAD COMPARISON: VAE vs PCA')
print('=' * 70)
print(f'(Both use 32 dimensions on the same 12,509 labeled microglia)')
print()

table = pd.DataFrame([
    {
        'Metric': 'kNN purity (k=15)',
        'VAE': f"{res_vae['kNN_purity']:.3f}",
        'PCA': f"{res_pca['kNN_purity']:.3f}",
        'Random/Naive baseline': '0.143',
        'Winner': 'VAE' if res_vae['kNN_purity'] > res_pca['kNN_purity'] else 'PCA',
        'Margin': f"{abs(res_vae['kNN_purity'] - res_pca['kNN_purity']):.3f}",
    },
    {
        'Metric': 'Silhouette',
        'VAE': f"{res_vae['silhouette']:.3f}",
        'PCA': f"{res_pca['silhouette']:.3f}",
        'Random/Naive baseline': '0.000',
        'Winner': 'VAE' if res_vae['silhouette'] > res_pca['silhouette'] else 'PCA',
        'Margin': f"{abs(res_vae['silhouette'] - res_pca['silhouette']):.3f}",
    },
    {
        'Metric': 'Label transfer accuracy',
        'VAE': f"{res_vae['accuracy']:.3f}",
        'PCA': f"{res_pca['accuracy']:.3f}",
        'Random/Naive baseline': f"{res_vae['naive_baseline']:.3f}",
        'Winner': 'VAE' if res_vae['accuracy'] > res_pca['accuracy'] else 'PCA',
        'Margin': f"{abs(res_vae['accuracy'] - res_pca['accuracy']):.3f}",
    },
    {
        'Metric': 'Balanced accuracy',
        'VAE': f"{res_vae['balanced_acc']:.3f}",
        'PCA': f"{res_pca['balanced_acc']:.3f}",
        'Random/Naive baseline': '0.143',
        'Winner': 'VAE' if res_vae['balanced_acc'] > res_pca['balanced_acc'] else 'PCA',
        'Margin': f"{abs(res_vae['balanced_acc'] - res_pca['balanced_acc']):.3f}",
    },
])
print(table.to_string(index=False))

print()
print('Per-state kNN purity:')
print(f"{'State':<20} {'VAE':<10} {'PCA':<10} {'Δ (VAE-PCA)':<12}")
print('-' * 55)
for state in sorted(res_vae['per_state'].keys()):
    v = res_vae['per_state'][state]
    p = res_pca['per_state'][state]
    print(f'{state:<20} {v:<10.3f} {p:<10.3f} {v-p:+.3f}')

# Save comparison CSV
os.makedirs(OUT, exist_ok=True)
comparison_df = pd.DataFrame([
    {'method': 'VAE',
     'kNN_purity': res_vae['kNN_purity'],
     'silhouette': res_vae['silhouette'],
     'accuracy': res_vae['accuracy'],
     'balanced_acc': res_vae['balanced_acc'],
     'naive_baseline': res_vae['naive_baseline']},
    {'method': 'PCA',
     'kNN_purity': res_pca['kNN_purity'],
     'silhouette': res_pca['silhouette'],
     'accuracy': res_pca['accuracy'],
     'balanced_acc': res_pca['balanced_acc'],
     'naive_baseline': res_pca['naive_baseline']},
])
comparison_df.to_csv(f'{OUT}/vae_vs_pca_comparison.csv', index=False)

per_state_df = pd.DataFrame({
    'state': sorted(res_vae['per_state'].keys()),
    'VAE_purity': [res_vae['per_state'][s] for s in sorted(res_vae['per_state'].keys())],
    'PCA_purity': [res_pca['per_state'][s] for s in sorted(res_vae['per_state'].keys())],
})
per_state_df['delta_VAE_minus_PCA'] = per_state_df['VAE_purity'] - per_state_df['PCA_purity']
per_state_df.to_csv(f'{OUT}/vae_vs_pca_per_state.csv', index=False)

print()
print(f'Saved comparison CSVs to {OUT}/')
print(f'  - vae_vs_pca_comparison.csv  (overall metrics)')
print(f'  - vae_vs_pca_per_state.csv   (per-state purity)')
