"""
Data loading for VINE-seq.

VINE-seq (Yang et al., Nature 2022) is a vasculature-enriched snRNA-seq dataset.
It is usually published as an AnnData (.h5ad) with mixed cell types — endothelial,
mural, fibroblast, microglia/macrophage, etc. We extract the microglial compartment.

Supports two inputs:
  * a pre-built .h5ad           (preferred)
  * a 10x Cell Ranger directory (mtx.gz / barcodes / features)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import anndata as ad
import scanpy as sc

from .config import Config

log = logging.getLogger(__name__)


def load_anndata(cfg: Config) -> ad.AnnData:
    """Load AnnData from h5ad or 10x mtx according to ``cfg.data``."""
    d = cfg.data

    if d.h5ad_path:
        path = Path(d.h5ad_path)
        if not path.exists():
            raise FileNotFoundError(f"h5ad not found: {path}")
        log.info("Loading h5ad: %s", path)
        adata = sc.read_h5ad(path)
    elif d.tenx_dir:
        path = Path(d.tenx_dir)
        if not path.exists():
            raise FileNotFoundError(f"10x dir not found: {path}")
        log.info("Loading 10x mtx: %s", path)
        adata = sc.read_10x_mtx(path, var_names="gene_symbols", cache=False)
    else:
        raise ValueError("Either data.h5ad_path or data.tenx_dir must be set")

    # Make var/obs names unique (frequent footgun with VINE-seq merged data)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    # Stash a copy of raw counts so the VAE can use Poisson likelihood later.
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    log.info("Loaded AnnData: %d cells x %d genes", adata.n_obs, adata.n_vars)
    return adata


def subset_microglia(adata: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Keep only the microglial / macrophage compartment.

    Looks at ``adata.obs[cfg.data.celltype_col]``. Match is case-insensitive
    and tolerant of whitespace.

    If celltype_col is None, assume the data is already pre-filtered.
    """
    col = cfg.data.celltype_col
    if col is None:
        log.info("celltype_col is None — assuming data is already filtered to microglia.")
        return adata
    if col not in adata.obs:
        log.warning(
            "celltype column '%s' not found — skipping subset. "
            "Found columns: %s",
            col,
            list(adata.obs.columns),
        )
        return adata

    wanted = {x.strip().lower() for x in cfg.data.microglia_labels}
    obs_lower = adata.obs[col].astype(str).str.strip().str.lower()
    mask = obs_lower.isin(wanted)

    n = int(mask.sum())
    if n == 0:
        raise ValueError(
            f"No cells matched microglia labels {cfg.data.microglia_labels} "
            f"in column '{col}'. Unique values: "
            f"{adata.obs[col].astype(str).unique().tolist()[:20]}"
        )

    log.info("Subset to microglia: %d / %d cells", n, adata.n_obs)
    return adata[mask].copy()


def attach_condition(adata: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Standardize the condition column into adata.obs['condition']."""
    col = cfg.data.condition_col
    if col in adata.obs:
        adata.obs["condition"] = adata.obs[col].astype(str)
    else:
        log.warning(
            "condition column '%s' not found — DAM enrichment will be skipped.",
            col,
        )
    return adata
