"""
Leiden clustering on the VAE latent space, with a resolution sweep
scored by silhouette (and ARI vs reference labels when available).
"""

from __future__ import annotations

import logging
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .config import Config

log = logging.getLogger(__name__)


def _attach_latent(adata: ad.AnnData, latent: np.ndarray) -> None:
    """Put the VAE latent in adata.obsm and build a kNN graph on it."""
    adata.obsm["X_vae"] = latent


def cluster_latent(
    adata: ad.AnnData,
    latent: np.ndarray,
    cfg: Config,
    reference_label_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Sweep Leiden resolutions and pick the one with the best score.

    Score is silhouette on the latent space (subsampled for speed).
    If reference_label_col is provided and present in adata.obs, ARI is also
    reported — the final pick uses 0.5*silhouette + 0.5*ARI when available.

    Returns a dataframe with one row per resolution and writes the chosen
    cluster assignment into adata.obs['leiden'].
    """
    cc = cfg.clustering
    _attach_latent(adata, latent)

    sc.pp.neighbors(
        adata,
        n_neighbors=cc.n_neighbors,
        use_rep="X_vae",
        random_state=cfg.seed,
    )

    # subsample for silhouette scoring (it's O(n^2))
    rng = np.random.default_rng(cfg.seed)
    n_sub = min(5000, latent.shape[0])
    sub_idx = rng.choice(latent.shape[0], size=n_sub, replace=False)
    sub_latent = latent[sub_idx]

    rows = []
    for res in cc.resolutions:
        sc.tl.leiden(adata, resolution=res, random_state=cfg.seed, key_added=f"leiden_{res}")
        labels = adata.obs[f"leiden_{res}"].astype(str).to_numpy()
        n_clusters = len(set(labels))

        if n_clusters < 2:
            sil = float("nan")
        else:
            sil = float(silhouette_score(sub_latent, labels[sub_idx]))

        ari = float("nan")
        if reference_label_col and reference_label_col in adata.obs:
            ref = adata.obs[reference_label_col].astype(str).to_numpy()
            ari = float(adjusted_rand_score(ref, labels))

        score = sil if np.isnan(ari) else 0.5 * sil + 0.5 * ari
        rows.append({"resolution": res, "n_clusters": n_clusters,
                     "silhouette": sil, "ari": ari, "score": score})
        log.info("res=%.2f  k=%d  sil=%.3f  ari=%.3f  score=%.3f",
                 res, n_clusters, sil, ari, score)

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    best_res = df.iloc[0]["resolution"]
    log.info("Best resolution: %s", best_res)

    adata.obs["leiden"] = adata.obs[f"leiden_{best_res}"]
    adata.uns["leiden_sweep"] = df.to_dict(orient="list")
    adata.uns["leiden_best_resolution"] = float(best_res)
    return df
