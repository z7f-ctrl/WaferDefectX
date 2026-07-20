"""Lightweight CNN for wafer defect classification (Phase A1).

MobileNetV3-Small adapted for 1-channel grayscale input.
~1.5M parameters, designed for MPS training on Apple Silicon.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from cnn_model import AbstractClassifier, IDX_TO_LABEL


class WaferMobileNetV3(nn.Module, AbstractClassifier):
    """MobileNetV3-Small for wafer defect ROI classification."""

    NUM_CLASSES = 3

    def __init__(self, num_classes: int = 3, patch_size: int = 96):
        nn.Module.__init__(self)
        self.num_classes = num_classes
        self.patch_size = patch_size

        self.backbone = models.mobilenet_v3_small(weights=None)
        self.backbone.features[0][0] = nn.Conv2d(
            1, 16, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.backbone.classifier[3] = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict(self, X) -> np.ndarray:
        if isinstance(X, np.ndarray):
            if X.ndim == 3:
                X = X[:, np.newaxis]
            tensor = torch.from_numpy(X.astype(np.float32) / 255.0)
        else:
            tensor = X

        self.eval()
        with torch.no_grad():
            logits = self.forward(tensor)
            indices = logits.argmax(dim=1).cpu().numpy()
        return np.array([IDX_TO_LABEL[i] for i in indices])

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu"))
        self.eval()

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = WaferMobileNetV3()
    params = WaferMobileNetV3.param_count(model)
    print(f"WaferMobileNetV3 parameters: {params:,}")
    dummy = torch.randn(2, 1, 96, 96)
    out = model(dummy)
    print(f"Forward pass output shape: {out.shape}")
