from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_time(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    x = t.float()[:, None] * f[None]
    return torch.cat((x.sin(), x.cos()), dim=-1)


class CSIStatEncoder(nn.Module):
    def __init__(self, d_model: int, patch=(4, 4)):
        super().__init__()
        self.patch = nn.Conv2d(2, d_model, patch, stride=patch)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * .02)
        self.cross = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        q = self.query.expand(len(x), -1, -1)
        return self.norm(q + self.cross(q, tokens, tokens, need_weights=False)[0]).squeeze(1)


class CoarseWorldModel(nn.Module):
    """Causal multimodal state model; coarse CSI output is a statistical token."""
    def __init__(self, latent_channels=4, d_model=256, depth=6, heads=8, max_time=64):
        super().__init__()
        self.d_model = d_model
        self.rgb_patch = nn.Conv2d(latent_channels, d_model, 2, 2)
        self.rgb_query = nn.Parameter(torch.randn(1, 1, d_model) * .02)
        self.rgb_cross = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.csi_encoder = CSIStatEncoder(d_model)
        self.gps_encoder = nn.Sequential(nn.Linear(2, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.modality = nn.Parameter(torch.randn(3, d_model) * .02)
        self.time_embed = nn.Parameter(torch.randn(max_time, d_model) * .02)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4*d_model, .1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.csi_predictor = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.gps_predictor = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 2))
        self.csi_location = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))
        self.rgb_condition = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

    def encode_rgb(self, z):
        p = self.rgb_patch(z).flatten(2).transpose(1, 2)
        q = self.rgb_query.expand(len(z), -1, -1)
        return q.add(self.rgb_cross(q, p, p, need_weights=False)[0]).squeeze(1)

    def encode_step(self, z, csi, gps):
        return torch.stack((self.encode_rgb(z), self.csi_encoder(csi), self.gps_encoder(gps)), dim=2)

    def forward(self, z_seq, csi_seq, gps_seq):
        b, t = gps_seq.shape[:2]
        rgb_tokens = torch.stack([self.encode_rgb(z_seq[:, i]) for i in range(t)], 1)
        csi_tokens = torch.stack([self.csi_encoder(csi_seq[:, i]) for i in range(t)], 1)
        return self.forward_tokens(rgb_tokens, csi_tokens, gps_seq)

    def forward_rgb(self, z_seq, gps_seq):
        """RGB baseline: no CSI tensor is read or replaced by a learned proxy."""
        rgb_tokens = torch.stack([self.encode_rgb(z_seq[:, i]) for i in range(z_seq.shape[1])], 1)
        return self.forward_tokens(rgb_tokens, None, gps_seq)

    def forward_tokens(self, rgb_tokens, csi_tokens, gps_seq):
        b, t = gps_seq.shape[:2]
        gps_tokens = self.gps_encoder(gps_seq)
        if csi_tokens is None:
            x = torch.stack((rgb_tokens, gps_tokens), dim=2)
            x = x + self.modality[[0, 2]][None, None] + self.time_embed[None, :t, None]
            modalities = 2
        else:
            x = torch.stack((rgb_tokens, csi_tokens, gps_tokens), dim=2)
            x = x + self.modality[None, None] + self.time_embed[None, :t, None]
            modalities = 3
        x = x.reshape(b, t * modalities, self.d_model)
        time_id = torch.arange(t, device=x.device).repeat_interleave(modalities)
        causal = time_id[:, None] < time_id[None, :]
        state = self.norm(self.transformer(x, mask=causal).reshape(b, t, modalities, -1)[:, -1].mean(1))
        return state, self.rgb_condition(state), self.csi_predictor(state), self.gps_predictor(state)


class CondResBlock(nn.Module):
    def __init__(self, cin, cout, cond_dim):
        super().__init__()
        self.n1 = nn.GroupNorm(min(32, cin), cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(min(32, cout), cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.cond = nn.Linear(cond_dim, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cond):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.cond(cond).chunk(2, -1)
        h = self.n2(h) * (1 + scale[..., None, None]) + shift[..., None, None]
        return self.skip(x) + self.c2(F.silu(h))


class LatentDiffusion(nn.Module):
    """Multi-scale latent U-Net conditioned by the predicted world state."""
    def __init__(self, channels=4, base=128, cond_dim=256, diffusion_steps=1000):
        super().__init__()
        self.steps = diffusion_steps
        self.time_mlp = nn.Sequential(nn.Linear(cond_dim, cond_dim*4), nn.SiLU(), nn.Linear(cond_dim*4, cond_dim))
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim*2, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.inp = nn.Conv2d(channels, base, 3, padding=1)
        self.down1 = CondResBlock(base, base, cond_dim)
        self.downsample1 = nn.Conv2d(base, base*2, 4, 2, 1)
        self.down2 = CondResBlock(base*2, base*2, cond_dim)
        self.downsample2 = nn.Conv2d(base*2, base*4, 4, 2, 1)
        self.mid1 = CondResBlock(base*4, base*4, cond_dim)
        self.attn = nn.MultiheadAttention(base*4, 8, batch_first=True)
        self.mid2 = CondResBlock(base*4, base*4, cond_dim)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 4, 2, 1)
        self.dec2 = CondResBlock(base*4, base*2, cond_dim)
        self.up1 = nn.ConvTranspose2d(base*2, base, 4, 2, 1)
        self.dec1 = CondResBlock(base*2, base, cond_dim)
        self.out = nn.Sequential(nn.GroupNorm(min(32, base), base), nn.SiLU(), nn.Conv2d(base, channels, 3, padding=1))
        betas = torch.linspace(1e-4, .02, diffusion_steps)
        self.register_buffer("alpha_bar", torch.cumprod(1 - betas, 0))

    def forward(self, noisy, t, world_cond):
        te = self.time_mlp(sinusoidal_time(t, world_cond.shape[-1]))
        cond = self.cond_mlp(torch.cat((world_cond, te), -1))
        x1 = self.down1(self.inp(noisy), cond)
        x2 = self.down2(self.downsample1(x1), cond)
        h = self.mid1(self.downsample2(x2), cond)
        b, c, hh, ww = h.shape
        q = h.flatten(2).transpose(1, 2)
        h = h + self.attn(q, q, q, need_weights=False)[0].transpose(1, 2).reshape(b, c, hh, ww)
        h = self.mid2(h, cond)
        h = self.dec2(torch.cat((self.up2(h), x2), 1), cond)
        h = self.dec1(torch.cat((self.up1(h), x1), 1), cond)
        return self.out(h)

    def loss(self, clean, cond):
        t = torch.randint(0, self.steps, (len(clean),), device=clean.device)
        noise = torch.randn_like(clean)
        a = self.alpha_bar[t, None, None, None]
        noisy = a.sqrt() * clean + (1-a).sqrt() * noise
        return F.mse_loss(self(noisy, t, cond), noise)

    @torch.no_grad()
    def sample(self, shape, cond, sample_steps=50):
        x = torch.randn(shape, device=cond.device)
        timeline = torch.linspace(self.steps-1, 0, sample_steps, device=cond.device).long().unique_consecutive()
        for i, t0 in enumerate(timeline):
            t = torch.full((shape[0],), int(t0), device=cond.device, dtype=torch.long)
            eps = self(x, t, cond)
            a = self.alpha_bar[t0]
            x0 = (x - (1-a).sqrt()*eps) / a.sqrt()
            if i == len(timeline)-1:
                x = x0
            else:
                an = self.alpha_bar[timeline[i+1]]
                x = an.sqrt()*x0 + (1-an).sqrt()*eps
        return x


class FineCSIHead(nn.Module):
    """Eight learned future queries cross-attend to 8x5ms CSI patches and CSI-stat state."""
    def __init__(self, csi_shape, d_model=256, heads=8, history=8, future=8, patch=(4,4)):
        super().__init__()
        self.csi_shape, self.future = tuple(csi_shape), future
        self.patch = nn.Conv2d(2, d_model, patch, stride=patch)
        self.time = nn.Parameter(torch.randn(history, d_model)*.02)
        self.queries = nn.Parameter(torch.randn(future, d_model)*.02)
        layer = nn.TransformerDecoderLayer(d_model, heads, 4*d_model, .1, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, 4)
        self.out = nn.Linear(d_model, int(torch.tensor(csi_shape).prod()))

    def forward(self, history, csi_stat):
        b, h = history.shape[:2]
        memory = torch.cat([self.patch(history[:, i]).flatten(2).transpose(1,2) + self.time[i] for i in range(h)], 1)
        memory = torch.cat((csi_stat[:, None], memory), 1)
        q = self.queries[None].expand(b, -1, -1)
        return self.out(self.decoder(q, memory)).reshape(b, self.future, *self.csi_shape)


class BeamHead(nn.Module):
    def __init__(self, d_model=256, num_beams=64):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_beams))

    def forward(self, csi_stat):
        return self.net(csi_stat)


def complex_nmse(pred, target):
    dims = tuple(range(1, pred.ndim))
    return ((pred-target).square().sum(dims) / target.square().sum(dims).clamp_min(1e-8)).mean()
