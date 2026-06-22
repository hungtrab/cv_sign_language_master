"""Zero-DCE++ lightweight low-light image enhancement.

Paper: "Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation"
Architecture matches: github.com/Li-Chongyi/Zero-DCE_extension/Zero-DCE++/model.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_URL = "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE_extension/main/Zero-DCE%2B%2B/snapshots_Zero_DCE%2B%2B/Epoch99.pth"
WEIGHTS_PATH = Path("weights/zero_dce_pp.pth")


class CSDN(nn.Module):
    """Depthwise separable convolution block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, groups=1)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


class DCENet(nn.Module):
    """Zero-DCE++ enhance_net_nopool — exact match to official implementation."""
    def __init__(self, scale_factor: int = 1):
        super().__init__()
        self.scale_factor = scale_factor
        self.relu = nn.ReLU(inplace=True)

        n = 32
        self.e_conv1 = CSDN(3, n)
        self.e_conv2 = CSDN(n, n)
        self.e_conv3 = CSDN(n, n)
        self.e_conv4 = CSDN(n, n)
        self.e_conv5 = CSDN(n * 2, n)
        self.e_conv6 = CSDN(n * 2, n)
        self.e_conv7 = CSDN(n * 2, 3)

    def forward(self, x):
        if self.scale_factor != 1:
            x_down = F.interpolate(x, scale_factor=1.0 / self.scale_factor,
                                   mode="bilinear", align_corners=True)
        else:
            x_down = x

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        if self.scale_factor != 1:
            x_r = F.interpolate(x_r, size=(x.shape[2], x.shape[3]),
                                mode="bilinear", align_corners=True)

        # Iterative curve application (8 iterations)
        enhanced = x
        for _ in range(8):
            enhanced = enhanced + x_r * (torch.pow(enhanced, 2) - enhanced)

        return enhanced


def _ensure_weights() -> str:
    if WEIGHTS_PATH.exists():
        return str(WEIGHTS_PATH)
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Zero-DCE++ weights -> {WEIGHTS_PATH}")
    urllib.request.urlretrieve(WEIGHTS_URL, str(WEIGHTS_PATH))
    return str(WEIGHTS_PATH)


_model_cache = {}


def get_zero_dce_model(device: str = "cpu") -> DCENet:
    if device in _model_cache:
        return _model_cache[device]

    model = DCENet(scale_factor=1)
    weights_path = _ensure_weights()
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model.eval()
    model.to(device)
    _model_cache[device] = model
    return model


def apply_zero_dce(img_rgb: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Enhance an RGB uint8 image using Zero-DCE++."""
    model = get_zero_dce_model(device)
    tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced = model(tensor)

    enhanced = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(enhanced * 255, 0, 255).astype(np.uint8)
