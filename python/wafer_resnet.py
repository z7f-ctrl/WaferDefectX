"""Upgraded CNN with ResNet-18 backbone for WM-811K (M3a).

WaferResNet18: ResNet-18 adapted for 1-channel 64×64 grayscale input.
~11M parameters, significantly more powerful than the 75K WaferCNN.
"""
from __future__ import annotations

import abc
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from cnn_model import AbstractClassifier, IDX_TO_LABEL, PATCH_SIZE


class WaferResNet18(nn.Module, AbstractClassifier):
    """ResNet-18 adapted for wafer defect ROI classification.

    Modifications from standard ResNet-18:
      - conv1: 1 channel input (grayscale), 3×3 kernel, stride=1
        (avoids aggressive 4× downsampling for small patches)
      - maxpool removed (preserves spatial resolution)
      - fc: Linear(512 → num_classes)
    """

    NUM_CLASSES = 3

    def __init__(self, num_classes: int = 3, pretrained: bool = False,
                 patch_size: int = 64):
        nn.Module.__init__(self)
        self.num_classes = num_classes
        self.patch_size = patch_size

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        if pretrained:
            with torch.no_grad():
                orig_weight = models.resnet18().conv1.weight
                self.backbone.conv1.weight.copy_(
                    orig_weight.mean(dim=1, keepdim=True)
                )

        self.backbone.maxpool = nn.Identity()

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

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
        print(f"[WaferResNet18] Saved weights -> {path}")

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu"))
        self.eval()
        print(f"[WaferResNet18] Loaded weights <- {path}")

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = WaferResNet18()
    params = WaferResNet18.param_count(model)
    print(f"WaferResNet18 parameters: {params:,}")
    dummy = torch.randn(2, 1, PATCH_SIZE, PATCH_SIZE)
    out = model(dummy)
    print(f"Forward pass output shape: {out.shape}")
