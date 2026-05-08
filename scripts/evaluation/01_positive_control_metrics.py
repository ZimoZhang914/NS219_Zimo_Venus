import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import silhouette_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter
import os

print('Loading annotated AnnData...')
adata = sc.read_h5ad('results/vineseq_microglia_v1/annotated_microglia.h5ad')

# Rebuild ground truth labels (same logic as before)
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

mask = confident
X = adata.obsm['X_vae'][mask]
y = gt[mask]
print(f'Evaluating on {mask.sum()} confidently-labeled cells')
print(f'State distribution: {dict(pd.Series(y).value_counts())}')
print()

# METRIC 1 - kNN purity
print('='*60)
print('METRIC 1: kNN purity (k=15) on VAE latent space')
print('='*60)
print('Question: do nearest neighbors share my state label?')
print()

K = 15
nn = NearestNeighbors(n_neighbors=K + 1, metric='euclidean')
nn.fit(X)
_, idx = nn.kneighbors(X)
neighbor_idx = idx[:, 1:]

neighbor_labels = y[neighbor_idx]
matches = (neighbor_labels == y[:, None])
purity_per_cell = matches.mean(axis=1)
overall_purity = purity_per_cell.mean()

n_states = len(np.unique(y))
random_baseline = 1.0 / n_states

print(f'Overall kNN purity:     {overall_purity:.3f}')
print(f'Random baseline:        {random_baseline:.3f}  (1 / n_states)')
print(f'Lift over random:       {overall_purity / random_baseline:.2f}x')
print()
print('Per-state purity:')
for s in sorted(np.unique(y)):
    s_purity = purity_per_cell[y == s].mean()
    print(f'  {s:<18} {s_purity:.3f}  (n={int((y==s).sum())})')
print()

# METRIC 2 - Silhouette
print('='*60)
print('METRIC 2: Silhouette score by ground-truth state')
print('='*60)
print('Question: are state groups tight and well-separated?')
print()

rng = np.random.default_rng(42)
sub = rng.choice(len(X), size=min(5000, len(X)), replace=False)
sil = silhouette_score(X[sub], y[sub])
print(f'Silhouette (subsampled to 5000 cells): {sil:.3f}')
print(f'  -1.0 = wrong grouping')
print(f'   0.0 = no structure')
print(f'  +1.0 = perfect tight clusters')
print()

# METRIC 3 - Label transfer
print('='*60)
print('METRIC 3: Label transfer accuracy (k=15, 80/20 split)')
print('='*60)
print('Question: can I predict state from latent neighbors?')
print()

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
clf = KNeighborsClassifier(n_neighbors=15, weights='distance')
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
acc = accuracy_score(y_te, y_pred)
bacc = balanced_accuracy_score(y_te, y_pred)

most_common = Counter(y_tr).most_common(1)[0][0]
naive_acc = (y_te == most_common).mean()

print(f'Test accuracy:          {acc:.3f}')
print(f'Balanced accuracy:      {bacc:.3f}')
print(f'Naive baseline:         {naive_acc:.3f}  (always predict {repr(most_common)})')
print(f'Lift over naive:        {(acc - naive_acc):.3f} absolute')
print()

# Save summary
out = 'results/vineseq_microglia_v1/positive_control'
os.makedirs(out, exist_ok=True)
summary = pd.DataFrame([
    {'metric': 'kNN_purity_k15',           'value': overall_purity, 'baseline': random_baseline,           'lift': overall_purity / random_baseline},
    {'metric': 'silhouette_groundtruth',   'value': sil,            'baseline': 0.0,                       'lift': float('nan')},
    {'metric': 'label_transfer_accuracy',  'value': acc,            'baseline': naive_acc,                 'lift': acc - naive_acc},
    {'metric': 'label_transfer_balanced',  'value': bacc,           'baseline': 1.0/n_states,              'lift': bacc - 1.0/n_states},
])
summary.to_csv(f'{out}/positive_control_metrics.csv', index=False)
print(f'Saved metrics to {out}/positive_control_metrics.csv')

per_state = pd.DataFrame({'state': sorted(np.unique(y))})
per_state['n_cells'] = [int((y == s).sum()) for s in per_state['state']]
per_state['knn_purity'] = [purity_per_cell[y == s].mean() for s in per_state['state']]
per_state.to_csv(f'{out}/per_state_purity.csv', index=False)
print(f'Saved per-state purity to {out}/per_state_purity.csv')
