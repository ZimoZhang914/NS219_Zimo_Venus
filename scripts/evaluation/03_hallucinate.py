"""
Use the trained VAE decoder to hallucinate synthetic microglia.

Pipeline:
  1. Load the trained VAE
  2. Sample 1000 random latent vectors from N(0, I)
  3. Push them through the decoder to get fake gene expression
  4. Compare fake cells to real cells on:
     - Distribution of total counts
     - Expression of canonical microglia markers
     - Marker-state assignment (do fakes look like real microglia states?)
  5. Save fake cells + plots
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/zimozhang/Desktop/GSE163577_RAW/microglia_pipeline/src')

from microglia_annotator.vae import VAE
from microglia_annotator.config import load_config
from microglia_annotator import preprocessing as pp_mod

PROJ = '/Users/zimozhang/Desktop/GSE163577_RAW/microglia_pipeline'
OUT  = f'{PROJ}/results/vineseq_microglia_v1/hallucination'
os.makedirs(OUT, exist_ok=True)

# ============================================================
# Step 1: Load real data and trained VAE
# ============================================================
print('Loading real data and trained VAE...')
cfg = load_config(f'{PROJ}/configs/default.yaml')
adata = sc.read_h5ad(f'{PROJ}/results/vineseq_microglia_v1/annotated_microglia.h5ad')

# We need to know which 3,000 HVGs were used during training
# These are stored in adata.var (the saved file is post-HVG)
hvg_genes = adata.var_names.tolist()
n_genes = len(hvg_genes)
print(f'Number of HVGs: {n_genes}')

# Reconstruct the VAE architecture and load weights
vae = VAE(
    n_input=n_genes,
    hidden_dims=cfg.vae.hidden_dims,
    latent_dim=cfg.vae.latent_dim,
    dropout=cfg.vae.dropout,
    likelihood=cfg.vae.likelihood,
)
state = torch.load(f'{PROJ}/results/vineseq_microglia_v1/vae.pt', map_location='cpu')
vae.load_state_dict(state)
vae.eval()
print('VAE loaded.')

# ============================================================
# Step 2: Hallucinate 1000 fake cells
# ============================================================
print()
print('Hallucinating 1000 synthetic microglia from random latent vectors...')
N_FAKE = 1000
torch.manual_seed(42)

with torch.no_grad():
    z = torch.randn(N_FAKE, cfg.vae.latent_dim)  # sample from N(0, I)
    fake_log_rate = vae.decoder(z)  # decoder outputs log-rate for Poisson
    fake_rate = torch.exp(fake_log_rate)  # convert to expression rate
    # Sample actual counts from the Poisson with these rates
    fake_counts = torch.poisson(fake_rate).numpy().astype(np.int32)

print(f'Fake count matrix shape: {fake_counts.shape}')
print(f'Total counts per fake cell: mean={fake_counts.sum(1).mean():.0f}, '
      f'median={np.median(fake_counts.sum(1)):.0f}')

# ============================================================
# Step 3: Compare fake vs real on key statistics
# ============================================================
print()
print('='*60)
print('REALISM CHECK — fake cells vs real cells')
print('='*60)

# Real raw counts on the same HVG set
real_counts = adata.layers['counts']
real_counts = real_counts.toarray() if hasattr(real_counts, 'toarray') else np.asarray(real_counts)

# 3a: Total counts per cell
fake_totals = fake_counts.sum(axis=1)
real_totals = real_counts.sum(axis=1)
print()
print('Total counts per cell (sequencing depth):')
print(f'  Real:  mean={real_totals.mean():.0f}  median={np.median(real_totals):.0f}')
print(f'  Fake:  mean={fake_totals.mean():.0f}  median={np.median(fake_totals):.0f}')

# 3b: Genes detected per cell
fake_ngenes = (fake_counts > 0).sum(axis=1)
real_ngenes = (real_counts > 0).sum(axis=1)
print()
print('Genes detected per cell:')
print(f'  Real:  mean={real_ngenes.mean():.0f}  median={np.median(real_ngenes):.0f}')
print(f'  Fake:  mean={fake_ngenes.mean():.0f}  median={np.median(fake_ngenes):.0f}')

# 3c: Microglia markers — do fakes express them?
print()
print('Microglia marker expression (% of cells expressing > 0):')
markers = ['P2RY12', 'TMEM119', 'CX3CR1', 'CSF1R', 'AIF1', 'C1QA', 'C1QB', 'TYROBP',
           'APOE', 'TREM2', 'CST7']
for g in markers:
    if g in hvg_genes:
        gi = hvg_genes.index(g)
        real_pct = 100 * (real_counts[:, gi] > 0).mean()
        fake_pct = 100 * (fake_counts[:, gi] > 0).mean()
        print(f'  {g:<10}  real: {real_pct:5.1f}%   fake: {fake_pct:5.1f}%')
    else:
        print(f'  {g:<10}  NOT in HVG set')

# 3d: Do fake cells get plausible state assignments?
print()
print('State signature scoring on fake cells...')

# Build a small AnnData of fake cells so we can use scanpy
adata_fake = sc.AnnData(
    X=fake_counts.astype(np.float32),
    var=pd.DataFrame(index=hvg_genes),
)
adata_fake.layers['counts'] = adata_fake.X.copy()
sc.pp.normalize_total(adata_fake, target_sum=1e4)
sc.pp.log1p(adata_fake)

# Same signature definitions as the pipeline
SIGS = {
    'Homeostatic': ['P2RY12', 'CX3CR1', 'TMEM119', 'CSF1R', 'SELPLG', 'MARCKS', 'MEF2C'],
    'DAM':         ['TREM2', 'APOE', 'CST7', 'LPL', 'TYROBP', 'AXL', 'ITGAX', 'CLEC7A',
                    'SPP1', 'CD9', 'GPNMB'],
    'IRM':         ['IFIT1', 'IFIT3', 'ISG15', 'MX1', 'OAS1', 'OAS2', 'STAT1', 'IRF7'],
    'MHC-II':      ['HLA-DRA', 'HLA-DRB1', 'HLA-DPA1', 'HLA-DPB1', 'CD74'],
    'Lipid_Assoc': ['APOE', 'LPL', 'TREM2', 'PLIN2', 'FABP5', 'GPNMB', 'SPP1'],
    'PVM':         ['MRC1', 'LYVE1', 'CD163', 'F13A1', 'STAB1'],
}

# Score fake cells
fake_scores = {}
for name, genes in SIGS.items():
    present = [g for g in genes if g in hvg_genes]
    if len(present) >= 3:
        sc.tl.score_genes(adata_fake, present, score_name=f'{name}_score',
                          random_state=42, use_raw=False)
        fake_scores[name] = adata_fake.obs[f'{name}_score'].values

# Assign each fake cell to its top state
fake_score_matrix = np.column_stack([fake_scores[k] for k in fake_scores])
fake_state_idx = fake_score_matrix.argmax(axis=1)
fake_state_labels = np.array(list(fake_scores.keys()))[fake_state_idx]

print()
print('Fake-cell state assignments (top scoring signature):')
fake_state_counts = pd.Series(fake_state_labels).value_counts()
print(fake_state_counts.to_string())

# Compare to real cells' ground-truth distribution
real_score_matrix = adata.obs[['Homeostatic_score', 'DAM_score', 'IRM_Interferon_score',
                                'MHC_II_score', 'Lipid_Associated_score',
                                'PVM_Border_score']].values
real_state_idx = real_score_matrix.argmax(axis=1)
real_state_labels_full = np.array(list(SIGS.keys()))[real_state_idx]
real_state_counts = pd.Series(real_state_labels_full).value_counts()
print()
print('Real-cell state distribution (for comparison):')
print(real_state_counts.to_string())

# ============================================================
# Step 4: Save plots
# ============================================================
print()
print('Generating comparison plots...')

# Plot 1: total counts distribution
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(np.log10(real_totals + 1), bins=50, alpha=0.6, label='Real', color='steelblue')
axes[0].hist(np.log10(fake_totals + 1), bins=50, alpha=0.6, label='Fake', color='darkorange')
axes[0].set_xlabel('log10(total counts + 1)'); axes[0].set_ylabel('# cells')
axes[0].set_title('Sequencing depth: real vs fake'); axes[0].legend()

axes[1].hist(real_ngenes, bins=50, alpha=0.6, label='Real', color='steelblue')
axes[1].hist(fake_ngenes, bins=50, alpha=0.6, label='Fake', color='darkorange')
axes[1].set_xlabel('# genes detected'); axes[1].set_ylabel('# cells')
axes[1].set_title('Gene complexity: real vs fake'); axes[1].legend()
fig.tight_layout()
fig.savefig(f'{OUT}/realism_check.png', dpi=150)
plt.close(fig)

# Plot 2: state distribution comparison (bar chart)
all_states = sorted(set(list(fake_state_counts.index) + list(real_state_counts.index)))
real_pct = [100 * real_state_counts.get(s, 0) / len(real_state_labels_full) for s in all_states]
fake_pct = [100 * fake_state_counts.get(s, 0) / len(fake_state_labels) for s in all_states]
x = np.arange(len(all_states))
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - 0.2, real_pct, width=0.4, label='Real', color='steelblue')
ax.bar(x + 0.2, fake_pct, width=0.4, label='Fake', color='darkorange')
ax.set_xticks(x); ax.set_xticklabels(all_states, rotation=30, ha='right')
ax.set_ylabel('% of cells')
ax.set_title('State distribution: real vs hallucinated cells')
ax.legend()
fig.tight_layout()
fig.savefig(f'{OUT}/state_distribution_comparison.png', dpi=150)
plt.close(fig)

# Save the fake count matrix
np.save(f'{OUT}/fake_counts.npy', fake_counts)
pd.DataFrame({
    'fake_cell_id': range(N_FAKE),
    'predicted_state': fake_state_labels,
    'total_counts': fake_totals,
    'n_genes': fake_ngenes,
}).to_csv(f'{OUT}/fake_cell_metadata.csv', index=False)

print()
print(f'Saved 1000 fake cells to {OUT}/')
print(f'  - fake_counts.npy           (the fake count matrix)')
print(f'  - fake_cell_metadata.csv    (per-cell predicted state)')
print(f'  - realism_check.png         (sequencing depth and gene complexity)')
print(f'  - state_distribution_comparison.png  (state distribution comparison)')
