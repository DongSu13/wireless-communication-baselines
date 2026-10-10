from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        groups = min(32, cin)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(min(32, cout), cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1),
        )
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.skip(x) + self.block(x)


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.norm(x).flatten(2).transpose(1, 2)
        q = q + self.attn(q, q, q, need_weights=False)[0]
        return x + q.transpose(1, 2).reshape(b, c, h, w)


class AutoencoderKL(nn.Module):
    """f4 KL autoencoder: RGB [-1,1] <-> quarter-resolution latent map."""
    def __init__(self, base: int = 64, latent_channels: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1), ResBlock(base, base),
            nn.Conv2d(base, base * 2, 4, 2, 1), ResBlock(base * 2, base * 2),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1), ResBlock(base * 4, base * 4),
            SpatialAttention(base * 4), nn.GroupNorm(32, base * 4), nn.SiLU(),
            nn.Conv2d(base * 4, 2 * latent_channels, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, base * 4, 3, padding=1), ResBlock(base * 4, base * 4),
            SpatialAttention(base * 4), nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1),
            ResBlock(base * 2, base * 2), nn.ConvTranspose2d(base * 2, base, 4, 2, 1),
            ResBlock(base, base), nn.GroupNorm(min(32, base), base), nn.SiLU(), nn.Conv2d(base, 3, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x, sample: bool = True):
        mean, logvar = self.encoder(x).chunk(2, dim=1)
        logvar = logvar.clamp(-20, 10)
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar) if sample else mean
        return z, mean, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        return self.decode(z), mean, logvar


def vae_loss(recon, target, mean, logvar, kl_weight: float = 1e-6):
    reconstruction = F.l1_loss(recon, target)
    kl = -0.5 * (1 + logvar - mean.square() - logvar.exp()).mean()
    return reconstruction + kl_weight * kl, {"recon": reconstruction.detach(), "kl": kl.detach()}
