"""
Training loop for the VAE with KL warm-up and early stopping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .vae import VAE

log = logging.getLogger(__name__)


@dataclass
class TrainHistory:
    train_loss: List[float]
    val_loss: List[float]
    recon: List[float]
    kl: List[float]
    best_epoch: int


def train_vae(
    x: np.ndarray,
    cfg: Config,
    device: str | None = None,
) -> tuple[VAE, np.ndarray, TrainHistory]:
    """
    Train a VAE on ``x`` (cells x genes) and return:
      * the trained model
      * the latent embedding for *all* cells (mu)
      * the training history
    """
    vc = cfg.vae
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training VAE on %s | input shape %s", device, x.shape)

    # train/val split
    rng = np.random.default_rng(cfg.seed)
    n = x.shape[0]
    idx = rng.permutation(n)
    n_val = max(1, int(n * vc.val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    x_t = torch.from_numpy(x.astype(np.float32))
    train_ds = TensorDataset(x_t[train_idx])
    val_ds = TensorDataset(x_t[val_idx])
    train_dl = DataLoader(train_ds, batch_size=vc.batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=vc.batch_size, shuffle=False)

    model = VAE(
        n_input=x.shape[1],
        hidden_dims=vc.hidden_dims,
        latent_dim=vc.latent_dim,
        dropout=vc.dropout,
        likelihood=vc.likelihood,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=vc.lr, weight_decay=vc.weight_decay)

    history = TrainHistory(train_loss=[], val_loss=[], recon=[], kl=[], best_epoch=-1)
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(vc.epochs):
        # --- KL annealing -------------------------------------------------
        if vc.warmup_epochs > 0:
            beta = vc.beta * min(1.0, (epoch + 1) / vc.warmup_epochs)
        else:
            beta = vc.beta

        # --- train --------------------------------------------------------
        model.train()
        train_losses, recon_losses, kl_losses = [], [], []
        for (batch,) in train_dl:
            batch = batch.to(device, non_blocking=True)
            opt.zero_grad()
            recon, mu, logvar, _ = model(batch)
            loss, rl, kl = model.loss(batch, recon, mu, logvar, beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(loss.item())
            recon_losses.append(rl.item())
            kl_losses.append(kl.item())

        # --- validate -----------------------------------------------------
        model.eval()
        with torch.no_grad():
            v_losses = []
            for (batch,) in val_dl:
                batch = batch.to(device, non_blocking=True)
                recon, mu, logvar, _ = model(batch)
                v, _, _ = model.loss(batch, recon, mu, logvar, beta=beta)
                v_losses.append(v.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(v_losses)) if v_losses else float("nan")
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.recon.append(float(np.mean(recon_losses)))
        history.kl.append(float(np.mean(kl_losses)))

        log.info(
            "epoch %3d  train=%.3f  val=%.3f  recon=%.3f  kl=%.3f  beta=%.2f",
            epoch, train_loss, val_loss,
            history.recon[-1], history.kl[-1], beta,
        )

        # --- early stop ---------------------------------------------------
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= vc.patience:
                log.info("Early stopping at epoch %d (best=%d).", epoch, history.best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- compute latent for all cells ------------------------------------
    model.eval()
    with torch.no_grad():
        all_dl = DataLoader(TensorDataset(x_t), batch_size=vc.batch_size, shuffle=False)
        latents = []
        for (batch,) in all_dl:
            batch = batch.to(device, non_blocking=True)
            mu = model.encode(batch).cpu().numpy()
            latents.append(mu)
        latent = np.concatenate(latents, axis=0)
    log.info("Latent shape: %s", latent.shape)

    return model, latent, history


def save_checkpoint(model: VAE, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
