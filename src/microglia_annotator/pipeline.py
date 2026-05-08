"""
End-to-end pipeline orchestrator.

Stages:
  1. Load + subset to microglia
  2. QC + normalize + HVG + (optional) Harmony
  3. Train VAE encoder-decoder
  4. Cluster on VAE latent (Leiden sweep)
  5. Train + calibrate classifier on a labeled reference column
  6. Predict cell states for all microglia
  7. Score microglia signatures (DAM / Homeostatic / IRM ...) + AD vs control test
  8. Plots, metrics, save annotated AnnData
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data as data_mod
from . import preprocessing as pp_mod
from . import train as train_mod
from . import cluster as cluster_mod
from . import classifier as clf_mod
from . import dam_score as dam_mod
from . import evaluate as eval_mod
from .config import Config

log = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline(cfg: Config) -> dict:
    """Execute the pipeline. Returns a dict of outputs / paths."""
    _set_seeds(cfg.seed)
    out = cfg.output_path()
    log.info("=== Run: %s -> %s ===", cfg.run_name, out)

    # save the resolved config alongside the outputs for provenance
    cfg.save(out / "config_used.yaml")

    # -------- 1. load + subset ----------------------------------------
    adata = data_mod.load_anndata(cfg)
    adata = data_mod.subset_microglia(adata, cfg)
    adata = data_mod.attach_condition(adata, cfg)

    # -------- 2. QC + preprocess --------------------------------------
    adata = pp_mod.quality_control(adata, cfg)
    adata = pp_mod.normalize_and_hvg(adata, cfg)
    adata = pp_mod.batch_correct(adata, cfg)

    # -------- 3. VAE --------------------------------------------------
    x = pp_mod.get_vae_input(adata, cfg)
    model, latent, history = train_mod.train_vae(x, cfg)
    train_mod.save_checkpoint(model, out / "vae.pt")
    eval_mod.plot_loss_history(history, out)
    pd.DataFrame({
        "epoch": list(range(len(history.train_loss))),
        "train_loss": history.train_loss,
        "val_loss": history.val_loss,
        "recon": history.recon,
        "kl": history.kl,
    }).to_csv(out / "vae_loss_history.csv", index=False)

    # -------- 4. clustering -------------------------------------------
    cc = cfg.classifier
    ref_col = cc.reference_label_col if cc.reference_label_col in adata.obs else None
    sweep_df = cluster_mod.cluster_latent(adata, latent, cfg, reference_label_col=ref_col)
    sweep_df.to_csv(out / "leiden_sweep.csv", index=False)

    # -------- 5. classifier (only if a reference label column exists) -
    classifier_metrics = None
    pred_df = None
    if ref_col is not None:
        artifacts, labeled_mask = clf_mod.train_classifier(
            latent=latent,
            labels=adata.obs[ref_col],
            cfg=cfg,
        )
        classifier_metrics = artifacts.metrics

        # save metrics + confusion matrix (computed on the held-out test split
        # internally; here we re-derive a confusion matrix for visualization
        # using predictions on labeled cells)
        eval_mod.save_metrics(classifier_metrics, out / "classifier_metrics.json")
        eval_mod.save_classification_report(
            classifier_metrics, out / "classification_report.csv"
        )

        # 6. predict for all microglia
        pred_df = clf_mod.predict(
            artifacts, latent, confidence_threshold=cc.confidence_threshold
        )
        pred_df.index = adata.obs_names
        for col in pred_df.columns:
            adata.obs[col] = pred_df[col].values

        # confusion matrix on the labeled subset (predicted vs reference)
        ref_labels = adata.obs[ref_col].astype(str).to_numpy()[labeled_mask]
        pred_labels = pred_df["predicted_state_raw"].to_numpy()[labeled_mask]
        classes = sorted(set(ref_labels) | set(pred_labels))
        eval_mod.plot_confusion(
            ref_labels, pred_labels, classes, out / "confusion_matrix.png"
        )

        # confidence panel
        eval_mod.plot_confidence(
            pred_df["confidence"].to_numpy(),
            pred_df["predicted_state"].to_numpy(),
            out / "confidence_analysis.png",
        )
    else:
        log.warning(
            "classifier.reference_label_col='%s' not found in adata.obs — "
            "skipping supervised annotation. Cells will be labeled only by Leiden cluster.",
            cc.reference_label_col,
        )
        adata.obs["predicted_state"] = "cluster_" + adata.obs["leiden"].astype(str)

    # -------- 7. DAM / state signature scoring -----------------------
    score_cols = dam_mod.score_signatures(adata, cfg)
    enrichment = dam_mod.test_ad_vs_control(adata, score_cols, cfg)
    if not enrichment.empty:
        enrichment.to_csv(out / "dam_ad_vs_control.csv", index=False)
        log.info("AD vs control enrichment:\n%s", enrichment.to_string(index=False))

    # -------- 8. UMAP + save ----------------------------------------
    if cfg.eval.make_umap:
        eval_mod.make_umap(adata, cfg, out)

    if cfg.eval.save_h5ad:
        adata.write_h5ad(out / "annotated_microglia.h5ad")

    log.info("=== Done. Outputs in %s ===", out)

    return {
        "output_dir": str(out),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "vae_best_epoch": history.best_epoch,
        "leiden_best_resolution": adata.uns.get("leiden_best_resolution"),
        "classifier_metrics": classifier_metrics,
        "score_columns": score_cols,
        "enrichment_table": enrichment if not enrichment.empty else None,
    }
