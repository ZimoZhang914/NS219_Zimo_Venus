"""
Disease-associated microglia (DAM) signature scoring.

Built-in signatures are drawn from canonical microglia state references:

  * Homeostatic         — Keren-Shaul et al. 2017 (Cell), Sun et al. 2023 (Cell)
  * DAM (stage 1 + 2)   — Keren-Shaul et al. 2017 (Cell)
  * IRM (interferon)    — Ellwanger et al. 2021 (PNAS), Sun et al. 2023
  * Cytokine / NF-kB    — Sun et al. 2023, Olah et al. 2020 (Nat Commun)
  * MHC-II / antigen    — Olah et al. 2020
  * Lipid / lipid-assoc — Marschallinger et al. 2020 (Nat Neurosci)

These are HUMAN ortholog gene symbols. Adjust as needed for your data
and append extra signatures via cfg.dam.extra_signatures.

After scoring with scanpy.tl.score_genes we test whether each signature
is enriched in AD vs control donors.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

from .config import Config

log = logging.getLogger(__name__)


# Built-in microglia signatures (human gene symbols).
DEFAULT_SIGNATURES: Dict[str, List[str]] = {
    "Homeostatic": [
        "P2RY12", "CX3CR1", "TMEM119", "CSF1R", "SELPLG", "MARCKS",
        "SIGLEC8", "MEF2C", "SALL1", "FCRLS",
    ],
    "DAM": [
        "TREM2", "APOE", "CST7", "LPL", "TYROBP", "AXL", "ITGAX",
        "CLEC7A", "SPP1", "CD9", "GPNMB", "CTSD", "CTSB", "B2M",
    ],
    "IRM_Interferon": [
        "IFIT1", "IFIT3", "ISG15", "MX1", "OAS1", "OAS2", "STAT1",
        "RSAD2", "IFI44L", "IRF7",
    ],
    "Cytokine_NFkB": [
        "IL1B", "TNF", "CCL2", "CCL3", "CCL4", "CXCL8", "NFKB1",
        "NFKB2", "RELB", "NLRP3",
    ],
    "MHC_II": [
        "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1",
        "HLA-DQB1", "CD74",
    ],
    "Lipid_Associated": [
        "APOE", "LPL", "TREM2", "ABCA1", "ABCA7", "PLIN2", "FABP5",
        "GPNMB", "SPP1",
    ],
    "Cycling": [
        "MKI67", "TOP2A", "CENPF", "BIRC5", "UBE2C", "STMN1", "CCNB1",
    ],
    "PVM_Border": [
        "MRC1", "LYVE1", "CD163", "F13A1", "STAB1", "SIGLEC1",
    ],
}


def get_signatures(cfg: Config) -> Dict[str, List[str]]:
    """Merge built-in and user-supplied signatures."""
    sigs = {k: list(v) for k, v in DEFAULT_SIGNATURES.items()}
    for k, v in (cfg.dam.extra_signatures or {}).items():
        sigs[k] = list(v)
    return sigs


def score_signatures(adata: ad.AnnData, cfg: Config) -> List[str]:
    """
    Add one column per signature into ``adata.obs`` using ``sc.tl.score_genes``.
    Returns the list of column names that were created.
    """
    sigs = get_signatures(cfg)
    added: List[str] = []

    for name, genes in sigs.items():
        present = [g for g in genes if g in adata.var_names]
        if len(present) < 3:
            log.warning("signature '%s' has only %d/%d genes present — skipping",
                        name, len(present), len(genes))
            continue
        col = f"{name}_score"
        sc.tl.score_genes(
            adata,
            gene_list=present,
            score_name=col,
            random_state=cfg.seed,
            use_raw=False,
        )
        added.append(col)
        log.info("scored signature %s on %d/%d genes", name, len(present), len(genes))

    # Convenience: an overall DAM_score = DAM - Homeostatic.
    if "DAM_score" in adata.obs and "Homeostatic_score" in adata.obs:
        adata.obs["DAM_minus_Homeostatic"] = (
            adata.obs["DAM_score"] - adata.obs["Homeostatic_score"]
        )
        added.append("DAM_minus_Homeostatic")

    return added


def test_ad_vs_control(
    adata: ad.AnnData,
    score_cols: List[str],
    cfg: Config,
) -> pd.DataFrame:
    """
    Per-signature AD vs control test (Wilcoxon rank-sum on per-cell scores
    by default; you can swap to per-donor means for a more conservative test).
    Returns a tidy dataframe.
    """
    if "condition" not in adata.obs:
        log.warning("no 'condition' column — skipping AD vs control")
        return pd.DataFrame()

    ad_label = cfg.data.ad_label
    ctrl_label = cfg.data.control_label
    cond = adata.obs["condition"].astype(str)

    if ad_label not in cond.unique() or ctrl_label not in cond.unique():
        log.warning("AD/Control labels not both present (found %s) — skipping",
                    cond.unique().tolist())
        return pd.DataFrame()

    rows = []
    for col in score_cols:
        x = adata.obs.loc[cond == ad_label, col].dropna().to_numpy()
        y = adata.obs.loc[cond == ctrl_label, col].dropna().to_numpy()
        if len(x) < 5 or len(y) < 5:
            continue
        if cfg.dam.test == "wilcoxon":
            stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        else:
            stat, p = stats.ttest_ind(x, y, equal_var=False)
        rows.append({
            "signature": col,
            "n_AD": len(x),
            "n_Control": len(y),
            "mean_AD": float(np.mean(x)),
            "mean_Control": float(np.mean(y)),
            "delta": float(np.mean(x) - np.mean(y)),
            "stat": float(stat),
            "pvalue": float(p),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # BH FDR
        from statsmodels.stats.multitest import multipletests  # noqa: WPS433
        try:
            df["padj_bh"] = multipletests(df["pvalue"], method="fdr_bh")[1]
        except Exception:
            df["padj_bh"] = df["pvalue"]
        df = df.sort_values("padj_bh").reset_index(drop=True)
    return df
