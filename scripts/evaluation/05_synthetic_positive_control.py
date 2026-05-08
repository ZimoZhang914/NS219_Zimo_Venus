"""
Synthetic positive control for the microglia VAE pipeline.

Generates a synthetic dataset with KNOWN ground-truth cell states using
biologically realistic gene expression rules:

  * Homeostatic cells:    high P2RY12/TMEM119/CX3CR1 family, low DAM, low prolif
  * DAM cells:            high TREM2/APOE/CST7 family, downregulated homeostatic
  * Proliferative cells:  high MKI67/TOP2A family, partial homeostatic, low DAM
  * Shared microglia identity genes: expressed across all states

We then run the same VAE + clustering + evaluation pipeline used on the
real data, and report whether the pipeline recovers the 3 known states.

If the pipeline works, kNN purity and label-transfer accuracy on synthetic
data should be substantially higher than on real data — because the
synthetic ground truth is exact, while the real ground truth is a
marker-based proxy.
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
import warnings
from scipy.sparse import csr_matrix

warnings.filterwarnings('ignore')

PROJ = '/Users/zimozhang/Desktop/GSE163577_RAW/microglia_pipeline'
sys.path.insert(0, f'{PROJ}/src')

from microglia_annotator.config import load_config
from microglia_annotator.vae import VAE
from microglia_annotator.train import train_vae

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score, accuracy_score, balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split
from collections import Counter

OUT = f'{PROJ}/results/synthetic_positive_control'
os.makedirs(OUT, exist_ok=True)
np.random.seed(42)

# ============================================================
# Step 1: Define gene sets per state (your biology)
# ============================================================
HOMEOSTATIC_GENES = [
    "P2RY12", "P2RY13", "TMEM119", "CX3CR1", "SALL1", "FCRLS",
    "OLFML3", "HEXB", "TGFBR1", "SLC2A5", "GPR34", "SPARC",
    "CSF1R", "SELPLG", "GLUL", "GAS6", "PROS1", "AXL",
    "SERINC3", "SIGLECH", "CD33", "ITGB5",
]
DAM_GENES = [
    "TREM2", "APOE", "CST7", "LPL", "SPP1", "CTSD", "CTSL",
    "CD9", "CLEC7A", "ITGAX", "CSF1", "LILRB4", "CCL3", "CCL4",
    "CTSB", "LGALS3", "GPNMB", "FTH1", "FTL", "CD63", "LAMP1",
    "LIPA", "GRN", "TYROBP", "FCER1G",
]
PROLIFERATIVE_GENES = [
    "MKI67", "TOP2A", "PCNA", "STMN1", "HMGB2", "BIRC5",
    "CCNB1", "CDK1", "CCNA2", "UBE2C", "AURKB", "TPX2",
    "CENPF", "PLK1", "BUB1", "MCM2", "MCM5", "MCM6",
    "TYMS", "TK1", "RRM2", "HIST1H4C",
]
SHARED_GENES = [
    "IBA1", "AIF1", "CX3CR1", "CD68", "ITGAM", "PTPRC",
    "RUNX1", "IRF8", "PU1", "SALL1", "HEXB",
    "B2M", "GAPDH", "ACTB", "MALAT1", "NEAT1",
]
all_genes = list(dict.fromkeys(
    HOMEOSTATIC_GENES + DAM_GENES + PROLIFERATIVE_GENES + SHARED_GENES
))
n_genes = len(all_genes)
gene_idx = {g: i for i, g in enumerate(all_genes)}

print(f'Total genes in synthetic dataset: {n_genes}')
print(f'  Homeostatic markers : {len(HOMEOSTATIC_GENES)}')
print(f'  DAM markers         : {len(DAM_GENES)}')
print(f'  Proliferative markers: {len(PROLIFERATIVE_GENES)}')
print(f'  Shared identity     : {len(SHARED_GENES)}')

# ============================================================
# Step 2: Helpers — biologically realistic count generation
# ============================================================
def nb_counts(n_cells, mean, dispersion=2.0):
    """Sample negative binomial counts (overdispersed Poisson)."""
    p = dispersion / (dispersion + mean + 1e-8)
    counts = np.random.negative_binomial(dispersion, p, size=n_cells)
    return counts.astype(np.float32)


def make_cells(n_cells, gene_means, dispersion=2.0, noise_scale=0.3):
    """Generate a count matrix for n_cells cells."""
    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    for gene, base_mean in gene_means.items():
        if gene not in gene_idx:
            continue
        j = gene_idx[gene]
        cell_means = base_mean * np.random.lognormal(
            mean=0, sigma=noise_scale, size=n_cells
        )
        X[:, j] = nb_counts(n_cells, cell_means.mean(), dispersion)
    return X

# ============================================================
# Step 3: Generate cells per state
# ============================================================
print('\nGenerating synthetic cells with known ground truth...')

# --- Homeostatic (n=300) -------------------------------------
N_HOMEO = 300
homeo_means = {g: np.random.uniform(8, 20)  for g in HOMEOSTATIC_GENES}
homeo_means.update({g: np.random.uniform(5, 15)  for g in SHARED_GENES})
homeo_means.update({g: np.random.uniform(0.0, 0.3) for g in DAM_GENES})
homeo_means.update({g: np.random.uniform(0.0, 0.2) for g in PROLIFERATIVE_GENES})
X_homeo = make_cells(N_HOMEO, homeo_means, dispersion=3.0, noise_scale=0.25)
print(f'  Homeostatic cells:    {X_homeo.shape}')

# --- DAM (n=300) ---------------------------------------------
N_DAM = 300
dam_means = {g: np.random.uniform(10, 25) for g in DAM_GENES}
dam_means.update({g: np.random.uniform(5, 15)  for g in SHARED_GENES})
dam_means.update({g: np.random.uniform(0.2, 1.5) for g in HOMEOSTATIC_GENES})
dam_means.update({g: np.random.uniform(0.0, 0.3) for g in PROLIFERATIVE_GENES})
X_dam = make_cells(N_DAM, dam_means, dispersion=2.5, noise_scale=0.3)
print(f'  DAM cells:            {X_dam.shape}')

# --- Proliferative (n=150) -----------------------------------
N_PROLIF = 150
prolif_means = {g: np.random.uniform(12, 30) for g in PROLIFERATIVE_GENES}
prolif_means.update({g: np.random.uniform(5, 15)  for g in SHARED_GENES})
prolif_means.update({g: np.random.uniform(2, 8)   for g in HOMEOSTATIC_GENES})
prolif_means.update({g: np.random.uniform(0.0, 0.5) for g in DAM_GENES})
X_prolif = make_cells(N_PROLIF, prolif_means, dispersion=2.0, noise_scale=0.35)
print(f'  Proliferative cells:  {X_prolif.shape}')

# --- Assemble AnnData ----------------------------------------
X_all = np.vstack([X_homeo, X_dam, X_prolif])
cell_types = (
    ['Homeostatic']   * N_HOMEO  +
    ['DAM']           * N_DAM    +
    ['Proliferative'] * N_PROLIF
)
obs = pd.DataFrame({
    'cell_type': cell_types,
    'dataset':   'synthetic',
    'sample_id': (
        ['ctrl_sample_1'] * (N_HOMEO // 2) +
        ['ctrl_sample_2'] * (N_HOMEO - N_HOMEO // 2) +
        ['ctrl_sample_1'] * (N_DAM // 2) +
        ['ctrl_sample_2'] * (N_DAM - N_DAM // 2) +
        ['ctrl_sample_1'] * N_PROLIF
    ),
}, index=[f'syn_cell_{i}' for i in range(len(cell_types))])
var = pd.DataFrame(index=all_genes)
var.index.name = 'gene_name'

adata_syn = ad.AnnData(X=csr_matrix(X_all), obs=obs, var=var)
adata_syn.layers['counts'] = adata_syn.X.copy()
print(f'\nSynthetic dataset: {adata_syn.shape[0]} cells x {adata_syn.shape[1]} genes')
print(adata_syn.obs['cell_type'].value_counts().to_string())

# Save
adata_syn.write_h5ad(f'{OUT}/synthetic_microglia.h5ad')

# ============================================================
# Step 4: Run the same VAE pipeline on synthetic data
# ============================================================
print('\n' + '='*60)
print('Training VAE on synthetic data')
print('='*60)

# Use same config but adjust for the smaller dataset / fewer genes
cfg = load_config(f'{PROJ}/configs/default.yaml')
cfg.vae.epochs = 60
cfg.vae.patience = 10
cfg.vae.hidden_dims = [128, 64]   # smaller, matches the smaller input
cfg.vae.latent_dim = 16
cfg.vae.batch_size = 64

# Prepare counts as float32
X_in = adata_syn.layers['counts']
X_in = X_in.toarray() if hasattr(X_in, 'toarray') else np.asarray(X_in)
X_in = X_in.astype(np.float32)
print(f'VAE input shape: {X_in.shape}')

model, latent, history = train_vae(X_in, cfg, device='cpu')
print(f'Trained, best epoch: {history.best_epoch}')
print(f'Latent shape: {latent.shape}')

adata_syn.obsm['X_vae'] = latent

# ============================================================
# Step 5: Cluster + evaluate on synthetic data
# ============================================================
print('\n' + '='*60)
print('Evaluating recovery of known ground truth')
print('='*60)

sc.pp.neighbors(adata_syn, use_rep='X_vae', n_neighbors=15, random_state=42)
sc.tl.leiden(adata_syn, resolution=0.5, random_state=42)
sc.tl.umap(adata_syn, random_state=42)

n_clusters = adata_syn.obs['leiden'].nunique()
print(f'Leiden found {n_clusters} clusters (ground truth has 3 states)')

# Cross-tab
print('\nCluster composition (ground truth x Leiden):')
ct = pd.crosstab(adata_syn.obs['cell_type'], adata_syn.obs['leiden'])
print(ct.to_string())

y_true = adata_syn.obs['cell_type'].values
y_clust = adata_syn.obs['leiden'].values
ari = adjusted_rand_score(y_true, y_clust)
nmi = normalized_mutual_info_score(y_true, y_clust)
print(f'\nARI (ground truth vs Leiden):  {ari:.3f}')
print(f'NMI (ground truth vs Leiden):  {nmi:.3f}')

# kNN purity in latent space
print('\n--- kNN purity (k=15) on VAE latent ---')
nn = NearestNeighbors(n_neighbors=16, metric='euclidean')
nn.fit(latent)
_, idx = nn.kneighbors(latent)
neighbor_idx = idx[:, 1:]
matches = (y_true[neighbor_idx] == y_true[:, None])
purity = matches.mean(axis=1).mean()
print(f'Overall kNN purity:  {purity:.3f}')
print(f'Random baseline:     {1/3:.3f}  (3 states)')
print('Per-state:')
for s in sorted(np.unique(y_true)):
    p = matches[y_true == s].mean(axis=1).mean()
    print(f'  {s:<15} {p:.3f}  (n={int((y_true == s).sum())})')

# Silhouette
sil = silhouette_score(latent, y_true)
print(f'\nSilhouette (vs ground truth): {sil:.3f}')

# Label transfer
X_tr, X_te, y_tr, y_te = train_test_split(
    latent, y_true, test_size=0.2, random_state=42, stratify=y_true
)
clf = KNeighborsClassifier(n_neighbors=15, weights='distance')
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
acc = accuracy_score(y_te, y_pred)
bacc = balanced_accuracy_score(y_te, y_pred)
naive = (y_te == Counter(y_tr).most_common(1)[0][0]).mean()
print(f'\nLabel transfer accuracy:   {acc:.3f}')
print(f'Balanced accuracy:         {bacc:.3f}')
print(f'Naive baseline:            {naive:.3f}')

# ============================================================
# Step 6: Save metrics + plots
# ============================================================
metrics = pd.DataFrame([
    {'metric': 'ARI_leiden_vs_truth',         'value': float(ari)},
    {'metric': 'NMI_leiden_vs_truth',         'value': float(nmi)},
    {'metric': 'kNN_purity_k15',              'value': float(purity)},
    {'metric': 'silhouette_groundtruth',      'value': float(sil)},
    {'metric': 'label_transfer_accuracy',     'value': float(acc)},
    {'metric': 'label_transfer_balanced',     'value': float(bacc)},
    {'metric': 'naive_baseline',              'value': float(naive)},
    {'metric': 'random_baseline_purity',      'value': 1.0/3.0},
])
metrics.to_csv(f'{OUT}/synthetic_metrics.csv', index=False)
ct.to_csv(f'{OUT}/cluster_vs_truth_crosstab.csv')

# UMAP plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sc.pl.umap(adata_syn, color='cell_type',
           title='Synthetic — ground-truth cell type',
           show=False, save=False)
plt.gcf().savefig(f'{OUT}/umap_synthetic_truth.png', dpi=150, bbox_inches='tight')
plt.close('all')

sc.pl.umap(adata_syn, color='leiden',
           title='Synthetic — VAE+Leiden clusters',
           show=False, save=False)
plt.gcf().savefig(f'{OUT}/umap_synthetic_leiden.png', dpi=150, bbox_inches='tight')
plt.close('all')

adata_syn.write_h5ad(f'{OUT}/synthetic_microglia_annotated.h5ad')

print('\n' + '='*60)
print(f'SYNTHETIC POSITIVE CONTROL — SUMMARY')
print('='*60)
print(f'Saved outputs to {OUT}/')
print(f'  synthetic_microglia.h5ad           (input data)')
print(f'  synthetic_microglia_annotated.h5ad (with VAE latent + clusters)')
print(f'  synthetic_metrics.csv              (all metrics)')
print(f'  cluster_vs_truth_crosstab.csv      (cluster composition)')
print(f'  umap_synthetic_truth.png           (UMAP colored by ground truth)')
print(f'  umap_synthetic_leiden.png          (UMAP colored by Leiden)')
print()
print('Headline numbers:')
print(f'  kNN purity:               {purity:.3f}  (random {1/3:.3f}; lift {purity/(1/3):.2f}x)')
print(f'  ARI (Leiden vs truth):    {ari:.3f}')
print(f'  Label transfer accuracy:  {acc:.3f}  (naive {naive:.3f})')
