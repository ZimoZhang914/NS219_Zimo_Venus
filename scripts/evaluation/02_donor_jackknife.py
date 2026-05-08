"""
Donor jackknife evaluation for the microglia VAE pipeline.

For each of 6 folds:
  1. Hold out 4 donors (2 AD + 2 Control) as the test set
  2. Train a fresh VAE on the remaining 20 donors
  3. Encode the held-out cells with the trained VAE
  4. For each held-out cell, find its 15 nearest neighbors among TRAINING cells
  5. Predict its state from those neighbors' marker-derived ground-truth labels
  6. Compute kNN purity and label-transfer accuracy on held-out cells

If the pipeline generalizes, held-out kNN purity should be > random baseline (0.143)
across all 6 folds.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import warnings
warnings.filterwarnings('ignore')

# Make project source importable
sys.path.insert(0, '/Users/zimozhang/Desktop/GSE163577_RAW/microglia_pipeline/src')

from microglia_annotator.vae import VAE
from microglia_annotator.train import train_vae
from microglia_annotator.config import load_config
from microglia_annotator import preprocessing as pp_mod
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

# ============================================================
# Fold definitions
# ============================================================
FOLDS = [
    {'fold': 1, 'ad': ['H2004_AD', '04_15'],  'ctrl': ['01_06', '03_03']},
    {'fold': 2, 'ad': ['04_11AD', '03_02'],   'ctrl': ['02_11', '03_09']},
    {'fold': 3, 'ad': ['01_05',   '05_10'],   'ctrl': ['O4_O9C', '04_09']},
    {'fold': 4, 'ad': ['O4_15AD', '04_11'],   'ctrl': ['02_01', '01_09']},
    {'fold': 5, 'ad': ['O4_13AD', '04_13'],   'ctrl': ['O3_O9C', '01_10']},
    {'fold': 6, 'ad': ['O5_1OAD', '05_04'],   'ctrl': ['O1_1OC', 'O3_O3C']},
]

# ============================================================
# Load and rebuild ground-truth labels (same logic as Step 1)
# ============================================================
print('='*70)
print('Loading data and rebuilding ground-truth labels...')
print('='*70)

cfg = load_config('/Users/zimozhang/Desktop/GSE163577_RAW/microglia_pipeline/configs/default.yaml')
adata = sc.read_h5ad('/Users/zimozhang/Desktop/GSE163577_RAW/microglia_pipeline/results/vineseq_microglia_v1/annotated_microglia.h5ad')

# Rebuild ground truth
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
adata.obs['ground_truth'] = gt

print(f'Total cells: {adata.n_obs}')
print(f'Confidently labeled cells: {confident.sum()}')

# Get the count matrix the VAE will use (raw counts on HVGs)
print('Preparing count matrix for VAE...')
X_full = pp_mod.get_vae_input(adata, cfg)
print(f'Count matrix shape: {X_full.shape}')

# ============================================================
# Run jackknife
# ============================================================
results = []
t_start = time.time()

for fold_def in FOLDS:
    fold_id = fold_def['fold']
    held_out_donors = fold_def['ad'] + fold_def['ctrl']

    print()
    print('='*70)
    print(f'FOLD {fold_id} / 6')
    print(f'Held-out donors: {held_out_donors}')
    print('='*70)
    fold_t = time.time()

    # Build train/test masks
    test_mask = adata.obs['donor_id'].isin(held_out_donors).values
    train_mask = ~test_mask

    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())
    print(f'Train cells: {n_train}  Test cells: {n_test}')

    # Train fresh VAE on training cells only
    X_train = X_full[train_mask]
    print(f'Training fresh VAE on {n_train} cells, {X_full.shape[1]} genes...')

    # Use fewer epochs for jackknife to keep runtime manageable
    cfg.vae.epochs = 60
    cfg.vae.patience = 10
    model, latent_train, history = train_vae(X_train, cfg, device='cpu')
    print(f'  Trained in {time.time() - fold_t:.0f}s, best epoch: {history.best_epoch}')

    # Encode held-out test cells
    print('Encoding held-out cells...')
    X_test = X_full[test_mask]
    model.eval()
    with torch.no_grad():
        latent_test = model.encode(torch.from_numpy(X_test.astype(np.float32))).cpu().numpy()

    # Get ground-truth labels for train and test
    y_train = adata.obs['ground_truth'].values[train_mask]
    y_test = adata.obs['ground_truth'].values[test_mask]

    # Restrict to confidently-labeled cells in BOTH sets for the metric
    train_labeled = y_train != 'Ambiguous'
    test_labeled = y_test != 'Ambiguous'

    Xtr_lab = latent_train[train_labeled]
    ytr_lab = y_train[train_labeled]
    Xte_lab = latent_test[test_labeled]
    yte_lab = y_test[test_labeled]

    print(f'Labeled train cells: {len(ytr_lab)}  Labeled test cells: {len(yte_lab)}')

    # METRIC 1: kNN-purity for held-out cells (find their 15 neighbors AMONG TRAINING SET)
    clf = KNeighborsClassifier(n_neighbors=15, weights='distance', metric='euclidean')
    clf.fit(Xtr_lab, ytr_lab)

    # purity: for each test cell, fraction of its 15 NN in training that share its label
    distances, neighbor_idx = clf.kneighbors(Xte_lab)
    neighbor_labels = ytr_lab[neighbor_idx]
    matches = (neighbor_labels == yte_lab[:, None])
    purity_per_cell = matches.mean(axis=1)
    held_out_purity = purity_per_cell.mean()

    # METRIC 2: label transfer accuracy
    yte_pred = clf.predict(Xte_lab)
    accuracy = (yte_pred == yte_lab).mean()
    most_common = Counter(ytr_lab).most_common(1)[0][0]
    naive_acc = (yte_lab == most_common).mean()

    # Per-state breakdown
    per_state = {}
    for s in np.unique(yte_lab):
        per_state[s] = float(purity_per_cell[yte_lab == s].mean())

    fold_result = {
        'fold': fold_id,
        'ad_donors': ', '.join(fold_def['ad']),
        'ctrl_donors': ', '.join(fold_def['ctrl']),
        'n_train': n_train,
        'n_test': n_test,
        'n_test_labeled': len(yte_lab),
        'held_out_kNN_purity': float(held_out_purity),
        'held_out_accuracy': float(accuracy),
        'naive_baseline': float(naive_acc),
        'lift_over_naive': float(accuracy - naive_acc),
        'fold_runtime_sec': time.time() - fold_t,
        'best_epoch': history.best_epoch,
    }
    fold_result.update({f'purity_{s}': v for s, v in per_state.items()})
    results.append(fold_result)

    print(f'Held-out kNN purity:     {held_out_purity:.3f}')
    print(f'Held-out accuracy:       {accuracy:.3f}  (naive baseline: {naive_acc:.3f})')
    print(f'Total elapsed: {(time.time() - t_start)/60:.1f} min')

# ============================================================
# Save and summarize
# ============================================================
print()
print('='*70)
print('JACKKNIFE SUMMARY')
print('='*70)

df = pd.DataFrame(results)
out_dir = '/Users/zimozhang/Desktop/GSE163577_RAW/microglia_pipeline/results/vineseq_microglia_v1/positive_control'
os.makedirs(out_dir, exist_ok=True)
df.to_csv(f'{out_dir}/jackknife_results.csv', index=False)

# Summary stats
print()
print(f"Mean held-out kNN purity:    {df['held_out_kNN_purity'].mean():.3f} ± {df['held_out_kNN_purity'].std():.3f}")
print(f"Random baseline:             0.143")
print(f"Mean held-out accuracy:      {df['held_out_accuracy'].mean():.3f} ± {df['held_out_accuracy'].std():.3f}")
print(f"Mean naive baseline:         {df['naive_baseline'].mean():.3f}")
print(f"Mean lift over naive:        {df['lift_over_naive'].mean():.3f}")
print()
print('Per-fold results:')
print(df[['fold', 'n_test', 'held_out_kNN_purity', 'held_out_accuracy', 'naive_baseline']].to_string(index=False))
print()
print(f'Total runtime: {(time.time() - t_start)/60:.1f} minutes')
print(f'Results saved to: {out_dir}/jackknife_results.csv')
