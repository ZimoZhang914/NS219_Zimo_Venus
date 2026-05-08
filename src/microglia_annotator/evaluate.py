"""
Evaluation, plotting, and reporting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .config import Config

log = logging.getLogger(__name__)


def make_umap(adata: ad.AnnData, cfg: Config, out_dir: Path) -> None:
    """Compute UMAP on the VAE latent and save plots colored by each requested key."""
    if "X_vae" not in adata.obsm:
        log.warning("X_vae missing — skipping UMAP")
        return
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, use_rep="X_vae", n_neighbors=cfg.clustering.n_neighbors,
                        random_state=cfg.seed)
    sc.tl.umap(adata, random_state=cfg.seed)

    keys = [k for k in cfg.eval.umap_color_by if k in adata.obs]
    if not keys:
        keys = ["leiden"] if "leiden" in adata.obs else []
    for k in keys:
        try:
            sc.pl.umap(adata, color=k, show=False, save=False)
            plt.gcf().tight_layout()
            plt.gcf().savefig(out_dir / f"umap_{k}.png", dpi=200, bbox_inches="tight")
            plt.close("all")
            log.info("saved UMAP colored by %s", k)
        except Exception as e:
            log.warning("UMAP plot for %s failed: %s", k, e)


def plot_loss_history(history, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.train_loss, label="train")
    axes[0].plot(history.val_loss, label="val")
    axes[0].axvline(history.best_epoch, color="red", linestyle="--", label=f"best={history.best_epoch}")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("ELBO loss"); axes[0].legend()
    axes[0].set_title("VAE total loss")

    axes[1].plot(history.recon, label="recon")
    axes[1].plot(history.kl, label="kl")
    axes[1].set_xlabel("epoch"); axes[1].legend()
    axes[1].set_title("VAE recon vs KL")

    fig.tight_layout()
    fig.savefig(out_dir / "vae_loss_history.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], out_path: Path
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.6),
                                    max(5, len(classes) * 0.6)))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax, cbar=True)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized colors)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confidence(
    confidences: np.ndarray, predicted: np.ndarray, out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(confidences, bins=40, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("max-class probability"); axes[0].set_ylabel("cells")
    axes[0].set_title("Confidence distribution")

    df = pd.DataFrame({"state": predicted, "confidence": confidences})
    sns.boxplot(data=df, x="state", y="confidence", ax=axes[1])
    axes[1].set_xlabel(""); axes[1].set_ylabel("confidence")
    axes[1].set_title("Confidence by predicted state")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metrics(metrics: Dict, path: Path) -> None:
    path.write_text(json.dumps(metrics, indent=2, default=str))


def save_classification_report(metrics: Dict, path: Path) -> None:
    pc = metrics.get("per_class", {})
    rows = []
    for cls, vals in pc.items():
        if isinstance(vals, dict):
            row = {"class": cls, **vals}
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
