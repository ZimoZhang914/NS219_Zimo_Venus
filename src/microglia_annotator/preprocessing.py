"""
QC, normalization, HVG selection, and batch correction.

The VAE will see two flavours of input:
  * raw counts (``adata.layers['counts']``) when ``vae.likelihood == 'poisson'``
  * log-normalized HVG matrix (``adata.X``) when ``vae.likelihood == 'mse'``
"""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import scanpy as sc

from .config import Config

log = logging.getLogger(__name__)


def quality_control(adata: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Standard scRNA-seq QC."""
    qc = cfg.qc

    # Mitochondrial / ribosomal flags. VINE-seq is human (gene symbols MT-/RPS/RPL).
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.upper().str.startswith(("RPS", "RPL"))

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo"], percent_top=None, log1p=False, inplace=True
    )

    n0 = adata.n_obs
    keep = (
        (adata.obs["n_genes_by_counts"] >= qc.min_genes_per_cell)
        & (adata.obs["pct_counts_mt"] <= qc.max_mt_percent)
        & (adata.obs["total_counts"] <= qc.max_counts)
    )
    adata = adata[keep].copy()
    sc.pp.filter_genes(adata, min_cells=qc.min_cells_per_gene)
    log.info("QC: kept %d / %d cells, %d genes", adata.n_obs, n0, adata.n_vars)
    return adata


def normalize_and_hvg(adata: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Normalize total, log1p, select HVGs."""
    pp = cfg.preprocess

    # keep raw counts
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=pp.target_sum)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=pp.n_top_genes,
        flavor="seurat",
        batch_key=pp.batch_key if pp.batch_key in adata.obs else None,
    )
    log.info(
        "HVG: %d genes flagged (top %d requested)",
        int(adata.var["highly_variable"].sum()),
        pp.n_top_genes,
    )

    # subset to HVGs for downstream modeling
    adata = adata[:, adata.var["highly_variable"]].copy()

    if pp.scale:
        sc.pp.scale(adata, max_value=10)

    return adata


def batch_correct(adata: ad.AnnData, cfg: Config) -> ad.AnnData:
    """
    Optional Harmony correction on PCA. We compute PCA *before* the VAE so the
    VAE sees a corrected signal; alternatively, you can let the VAE learn the
    correction itself by passing batch as a covariate (not implemented here).
    """
    pp = cfg.preprocess
    sc.tl.pca(adata, n_comps=50, random_state=cfg.seed)

    if pp.batch_correction == "harmony" and pp.batch_key in adata.obs:
        try:
            import scanpy.external as sce

            log.info("Running Harmony on batch_key=%s", pp.batch_key)
            sce.pp.harmony_integrate(adata, key=pp.batch_key)
            adata.obsm["X_pca_corrected"] = adata.obsm["X_pca_harmony"]
        except Exception as e:
            log.warning("Harmony failed (%s) — falling back to uncorrected PCA", e)
            adata.obsm["X_pca_corrected"] = adata.obsm["X_pca"]
    else:
        adata.obsm["X_pca_corrected"] = adata.obsm["X_pca"]

    return adata


def get_vae_input(adata: ad.AnnData, cfg: Config) -> np.ndarray:
    """
    Return the matrix that the VAE will see.

    Poisson likelihood -> raw counts on HVGs (must be non-negative integers).
    MSE likelihood     -> log-normalized HVG matrix.
    """
    likelihood = cfg.vae.likelihood
    if likelihood == "poisson":
        x = adata.layers["counts"]
        x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        x = np.asarray(x, dtype=np.float32)
        x[x < 0] = 0
        return x
    if likelihood == "mse":
        x = adata.X
        x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        return np.asarray(x, dtype=np.float32)
    raise ValueError(f"Unknown likelihood: {likelihood}")
