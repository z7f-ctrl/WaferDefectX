"""
cnn_model.py — WaferDefectX Deep Learning Branch

Defines:
  - AbstractClassifier  : Dependency Inversion base (SOLID)
  - WaferCNN            : Lightweight custom CNN for defect classification
  - WaferPatchDataset   : torch.utils.data.Dataset over synthetic wafer PNGs

Architecture (WaferCNN):
  Input  : (B, 1, 64, 64) grayscale patch
  Block1 : Conv2d(1→16, 3×3) → BN → ReLU → MaxPool2d(2)      → (B, 16, 32, 32)
  Block2 : Conv2d(16→32, 3×3) → BN → ReLU → MaxPool2d(2)     → (B, 32, 15, 15)
  Block3 : Conv2d(32→64, 3×3) → BN → ReLU → AdaptiveAvgPool  → (B, 64, 4, 4)
  FC     : Flatten → Linear(1024→128) → ReLU → Dropout(0.3) → Linear(128→3)
  Classes: 0=good, 1=particle, 2=scratch
"""

import abc
import os
import glob

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# SOLID: Dependency Inversion — abstract classifier interface
# ---------------------------------------------------------------------------

class AbstractClassifier(abc.ABC):
    """Abstract base for all classifiers (RF, SVM, CNN, …).
    
    Applying the Dependency Inversion Principle: high-level pipeline code
    depends on this abstraction, not on concrete implementations.
    """

    @abc.abstractmethod
    def predict(self, X) -> np.ndarray:
        """Return predicted class labels for input X."""

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Persist the model to disk."""

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """Restore the model from disk."""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

LABEL_MAP = {"good": 0, "particle": 1, "scratch": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
PATCH_SIZE = 64  # CNN input spatial size (64×64 grayscale)


class WaferPatchDataset(Dataset):
    """Reads synthetic wafer PNG files; returns (patch_tensor, label_idx).

    The label is parsed directly from the filename convention:
        wafer_<id>_<label>.png  →  label ∈ {good, particle, scratch}

    Each image is centre-cropped / resized to PATCH_SIZE×PATCH_SIZE and
    normalised to [0, 1].
    """

    def __init__(self, data_dir: str):
        self.samples: list[tuple[str, int]] = []
        paths = glob.glob(os.path.join(data_dir, "*.png"))
        for p in paths:
            label_str = os.path.basename(p).split("_")[-1].split(".")[0]
            if label_str in LABEL_MAP:
                self.samples.append((p, LABEL_MAP[label_str]))

        if not self.samples:
            raise FileNotFoundError(
                f"No labelled PNG files found in '{data_dir}'. "
                "Run data_generator.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label_idx = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Cannot read image: {path}")

        # Resize to PATCH_SIZE × PATCH_SIZE
        patch = cv2.resize(img, (PATCH_SIZE, PATCH_SIZE),
                           interpolation=cv2.INTER_AREA)

        # Normalise [0, 255] → [0.0, 1.0] and add channel dim
        tensor = torch.from_numpy(patch.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor, label_idx


# ---------------------------------------------------------------------------
# CNN Architecture
# ---------------------------------------------------------------------------

class WaferCNN(nn.Module, AbstractClassifier):
    """Lightweight CNN for wafer defect classification.

    Designed for edge deployment:
      - Small parameter count (~75 K params)
      - Fully convolutional feature extractor + compact FC head
      - Compatible with 8-bit post-training quantization & ONNX export
    """

    NUM_CLASSES = 3  # good / particle / scratch

    def __init__(self):
        nn.Module.__init__(self)

        # --- Feature Extractor ---
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),         # 64 → 32

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),         # 32 → 16

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4), # → 4×4
        )

        # --- Classifier Head ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, self.NUM_CLASSES),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    # ------------------------------------------------------------------
    # AbstractClassifier interface (used when loading .pth for inference)
    # ------------------------------------------------------------------

    def predict(self, X) -> np.ndarray:
        """Accept a numpy array (N, 64, 64) or (N, 1, 64, 64) and return
        string label array, e.g. ['scratch', 'good', ...]."""
        if isinstance(X, np.ndarray):
            if X.ndim == 3:            # (N, H, W)
                X = X[:, np.newaxis]   # → (N, 1, H, W)
            tensor = torch.from_numpy(X.astype(np.float32) / 255.0)
        else:
            tensor = X                 # already a torch.Tensor

        self.eval()
        with torch.no_grad():
            logits = self.forward(tensor)
            indices = logits.argmax(dim=1).numpy()
        return np.array([IDX_TO_LABEL[i] for i in indices])

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        print(f"[WaferCNN] Saved weights → {path}")

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu"))
        self.eval()
        print(f"[WaferCNN] Loaded weights ← {path}")

    @staticmethod
    def param_count() -> int:
        m = WaferCNN()
        return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    print(f"WaferCNN total parameters: {WaferCNN.param_count():,}")
    dummy = torch.randn(2, 1, PATCH_SIZE, PATCH_SIZE)
    out = WaferCNN()(dummy)
    print(f"Forward pass output shape : {out.shape}")  # (2, 3)
