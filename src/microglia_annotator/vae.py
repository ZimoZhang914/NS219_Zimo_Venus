"""
Variational Autoencoder for scRNA-seq.

Encoder-decoder architecture with two heads on the encoder (mu, logvar) and a
choice of two reconstruction likelihoods on the decoder:

  * Poisson  — recommended for raw UMI counts (scVI-style; Lopez et al. 2018)
  * Gaussian (MSE) — for log-normalized data

Reference:
  Grønbech et al. 2020 (scVAE), Lopez et al. 2018 (scVI), Lotfollahi et al.
  2022 (scArches), and the scVAE-Annotator implementation.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(dims: List[int], dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers += [
            nn.Linear(dims[i], dims[i + 1]),
            nn.LayerNorm(dims[i + 1]),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, n_input: int, hidden_dims: List[int], latent_dim: int, dropout: float):
        super().__init__()
        self.trunk = _mlp([n_input, *hidden_dims], dropout=dropout)
        last = hidden_dims[-1]
        self.fc_mu = nn.Linear(last, latent_dim)
        self.fc_logvar = nn.Linear(last, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-10.0, 10.0)  # numerical stability
        return mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        n_output: int,
        dropout: float,
        likelihood: str,
    ):
        super().__init__()
        self.likelihood = likelihood
        rev = list(reversed(hidden_dims))
        self.trunk = _mlp([latent_dim, *rev], dropout=dropout)
        last = rev[-1]
        self.fc_out = nn.Linear(last, n_output)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.trunk(z)
        out = self.fc_out(h)
        if self.likelihood == "poisson":
            # interpret as log-rate, clamp to keep exp() in float range
            out = out.clamp(-15.0, 15.0)
        return out


class VAE(nn.Module):
    """Encoder-decoder VAE with Poisson or Gaussian likelihood."""

    def __init__(
        self,
        n_input: int,
        hidden_dims: List[int] = (512, 256, 128),
        latent_dim: int = 32,
        dropout: float = 0.1,
        likelihood: str = "poisson",
    ):
        super().__init__()
        if likelihood not in ("poisson", "mse"):
            raise ValueError(f"likelihood must be 'poisson' or 'mse', got {likelihood}")
        self.likelihood = likelihood
        self.latent_dim = latent_dim
        self.encoder = Encoder(n_input, list(hidden_dims), latent_dim, dropout)
        self.decoder = Decoder(latent_dim, list(hidden_dims), n_input, dropout, likelihood)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

    # ----- losses ----------------------------------------------------------

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    def reconstruction_loss(self, recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.likelihood == "poisson":
            # recon is log-rate. -log p(x|rate) = rate - x*log(rate) + log(x!)
            # the log(x!) term is constant w.r.t. params so we can drop it.
            rate = torch.exp(recon)
            return torch.sum(rate - x * recon, dim=1)
        # MSE on log-normalized data
        return torch.sum((recon - x).pow(2), dim=1)

    def loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ):
        recon_l = self.reconstruction_loss(recon, x)
        kl = self.kl_divergence(mu, logvar)
        total = (recon_l + beta * kl).mean()
        return total, recon_l.mean(), kl.mean()

    # ----- inference -------------------------------------------------------

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return mu (deterministic embedding) for downstream use."""
        self.eval()
        mu, _ = self.encoder(x)
        return mu
