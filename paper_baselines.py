"""Paper-protocol baseline families with explicit task-specific outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class BaselineOutput:
    rgb: Optional[torch.Tensor] = None
    gps: Optional[torch.Tensor] = None
    coarse_csi: Optional[torch.Tensor] = None
    fine_csi: Optional[torch.Tensor] = None
    beam_logits: Optional[torch.Tensor] = None


class _RGBPredictor(nn.Module):
    def __init__(self, shapes: dict, depth: int):
        super().__init__()
        channels, height, width = shapes["image"]
        self.history = int(shapes["history"])
        self.pred_steps = int(shapes["pred_steps"])
        hidden = 32
        layers: list[nn.Module] = [nn.Conv2d(channels * self.history, hidden, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU()])
        layers.append(nn.Conv2d(hidden, channels * self.pred_steps, 3, padding=1))
        self.net = nn.Sequential(*layers)
        self.channels, self.height, self.width = channels, height, width

    def forward(self, batch: dict[str, torch.Tensor]) -> BaselineOutput:
        rgb = batch["rgb_history"]
        prediction = self.net(rgb.flatten(1, 2)).view(len(rgb), self.pred_steps, self.channels, self.height, self.width)
        return BaselineOutput(rgb=prediction)


class _TrajectoryPredictor(nn.Module):
    def __init__(self, shapes: dict, family: str):
        super().__init__()
        self.pred_steps = int(shapes["pred_steps"])
        history = int(shapes["history"])
        if family in {"gru", "lstm"}:
            recurrent = nn.GRU if family == "gru" else nn.LSTM
            self.encoder: nn.Module = recurrent(2, 64, batch_first=True)
            self.encoder_kind = "recurrent"
        elif family == "wayformer_style":
            self.input_proj = nn.Linear(2, 64)
            layer = nn.TransformerEncoderLayer(64, nhead=4, dim_feedforward=128, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.time = nn.Parameter(torch.zeros(1, history, 64))
            self.encoder_kind = "transformer"
        else:
            self.encoder = nn.Sequential(
                nn.Conv1d(2, 64, 3, padding=1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU()
            )
            self.encoder_kind = "convolution"
        self.head = nn.Linear(64, self.pred_steps * 2)
        self.history = history

    def forward(self, batch: dict[str, torch.Tensor]) -> BaselineOutput:
        gps = batch["gps_history"]
        if self.encoder_kind == "convolution":
            features = self.encoder(gps.transpose(1, 2)).mean(-1)
        elif self.encoder_kind == "transformer":
            features = self.encoder(self.input_proj(gps) + self.time[:, : gps.shape[1]])[:, -1]
        else:
            features = self.encoder(gps)[0][:, -1]
        return BaselineOutput(gps=self.head(features).view(len(gps), self.pred_steps, 2))


class _FineCSIPredictor(nn.Module):
    def __init__(self, shapes: dict, depth: int):
        super().__init__()
        csi_shape = tuple(shapes["csi"])
        self.fine_history = int(shapes.get("fine_history", 8))
        self.fine_future = int(shapes.get("fine_future", 8))
        elements = int(torch.tensor(csi_shape).prod())
        hidden = 256
        layers: list[nn.Module] = [nn.Linear(self.fine_history * elements, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        layers.append(nn.Linear(hidden, self.fine_future * elements))
        self.net = nn.Sequential(*layers)
        self.csi_shape = csi_shape

    def forward(self, batch: dict[str, torch.Tensor]) -> BaselineOutput:
        context = batch["fine_csi_context"]
        if context.ndim == 6:
            context = context[:, 0]
        prediction = self.net(context.flatten(1)).view(len(context), self.fine_future, *self.csi_shape)
        return BaselineOutput(fine_csi=prediction)


class _ChannelTransformerPredictor(nn.Module):
    """Sequence transformer adapter for the historical ChannelTransformer10 task line."""
    def __init__(self, shapes: dict):
        super().__init__()
        self.csi_shape = tuple(shapes["csi"])
        self.fine_history = int(shapes.get("fine_history", 8))
        self.fine_future = int(shapes.get("fine_future", 8))
        elements = int(torch.tensor(self.csi_shape).prod())
        hidden = 128
        self.input_proj = nn.Linear(elements, hidden)
        self.time = nn.Parameter(torch.zeros(1, self.fine_history, hidden))
        layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=256, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(hidden, self.fine_future * elements)

    def forward(self, batch: dict[str, torch.Tensor]) -> BaselineOutput:
        context = batch["fine_csi_context"]
        if context.ndim == 6:
            context = context[:, 0]
        tokens = self.input_proj(context.flatten(2)) + self.time[:, : context.shape[1]]
        prediction = self.head(self.encoder(tokens)[:, -1]).view(len(context), self.fine_future, *self.csi_shape)
        return BaselineOutput(fine_csi=prediction)


class _BeamPredictor(nn.Module):
    def __init__(self, shapes: dict, family: str, num_beams: int):
        super().__init__()
        self.family = family
        self.pred_steps = int(shapes["pred_steps"])
        channels, height, width = shapes["image"]
        csi_elements = int(torch.tensor(shapes["csi"]).prod())
        self.rgb_encoder = nn.Sequential(nn.Flatten(), nn.Linear(int(shapes["history"]) * channels * height * width, 64), nn.ReLU())
        self.csi_encoder = nn.Sequential(nn.Flatten(), nn.Linear(int(shapes["history"]) * csi_elements, 64), nn.ReLU())
        self.gps_encoder = nn.Sequential(nn.Flatten(), nn.Linear(int(shapes["history"]) * 2, 64), nn.ReLU())
        if family == "crossattn":
            self.cross_attention = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
            self.fuse = nn.Linear(64, 64)
        elif family == "gmu":
            self.gate = nn.Linear(192, 3)
            self.fuse = nn.Linear(64, 64)
        else:
            self.rgb_head = nn.Linear(64, num_beams)
            self.csi_head = nn.Linear(64, num_beams)
            self.gps_head = nn.Linear(64, num_beams)
        self.head = nn.Linear(64, self.pred_steps * num_beams) if family != "decisionfusion" else None
        self.num_beams = int(num_beams)

    def forward(self, batch: dict[str, torch.Tensor]) -> BaselineOutput:
        rgb = self.rgb_encoder(batch["rgb_history"])
        csi = self.csi_encoder(batch["csi_history"])
        gps = self.gps_encoder(batch["gps_history"])
        if self.family == "crossattn":
            query = gps[:, None]
            keys = torch.stack((rgb, csi), 1)
            features = self.fuse(self.cross_attention(query, keys, keys, need_weights=False)[0][:, 0])
            logits = self.head(features).view(len(rgb), self.pred_steps, self.num_beams)
        elif self.family == "gmu":
            modalities = torch.stack((rgb, csi, gps), 1)
            weights = self.gate(torch.cat((rgb, csi, gps), -1)).softmax(-1)
            features = self.fuse((modalities * weights[..., None]).sum(1))
            logits = self.head(features).view(len(rgb), self.pred_steps, self.num_beams)
        else:
            logits = (self.rgb_head(rgb) + self.csi_head(csi) + self.gps_head(gps)) / 3
            logits = logits[:, None].expand(-1, self.pred_steps, -1)
        return BaselineOutput(beam_logits=logits)


class _RGBCSIMMBPPredictor(nn.Module):
    """RGB+CSI-only late-fusion adapter for the historical MM-BP-style task line."""
    def __init__(self, shapes: dict, num_beams: int):
        super().__init__()
        self.pred_steps = int(shapes["pred_steps"])
        channels, height, width = shapes["image"]
        csi_elements = int(torch.tensor(shapes["csi"]).prod())
        history = int(shapes["history"])
        self.rgb_encoder = nn.Sequential(nn.Flatten(), nn.Linear(history * channels * height * width, 64), nn.ReLU())
        self.csi_encoder = nn.Sequential(nn.Flatten(), nn.Linear(history * csi_elements, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, self.pred_steps * num_beams))
        self.num_beams = int(num_beams)

    def forward(self, batch: dict[str, torch.Tensor]) -> BaselineOutput:
        features = torch.cat((self.rgb_encoder(batch["rgb_history"]), self.csi_encoder(batch["csi_history"])), -1)
        logits = self.head(features).view(len(features), self.pred_steps, self.num_beams)
        return BaselineOutput(beam_logits=logits)


def build_baseline(name: str, shapes: dict, num_beams: int) -> nn.Module:
    """Build exactly one task-specific baseline under the frozen data protocol."""
    if name == "convlstm":
        return _RGBPredictor(shapes, depth=1)
    if name == "predrnn_style":
        return _RGBPredictor(shapes, depth=2)
    if name == "strictsimvp_style":
        return _RGBPredictor(shapes, depth=3)
    if name in {"gru", "lstm", "tcn", "wayformer_style"}:
        return _TrajectoryPredictor(shapes, name)
    if name == "convgru":
        return _FineCSIPredictor(shapes, depth=1)
    if name == "temporal3d":
        return _FineCSIPredictor(shapes, depth=2)
    if name == "channeltransformer10_style":
        return _ChannelTransformerPredictor(shapes)
    if name in {"gmu", "crossattn", "decisionfusion"}:
        return _BeamPredictor(shapes, name, num_beams)
    if name == "rgb_csi_mmbp_style":
        return _RGBCSIMMBPPredictor(shapes, num_beams)
    raise ValueError(f"unknown baseline {name!r}")
