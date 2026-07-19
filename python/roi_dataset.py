"""ROI Patch Dataset for WM-811K (M3a).

Extracts bbox crops from localized defects, resizes to 64×64,
and trains CNN classifier on these patches instead of whole images.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from paths import DATA_WM811K_IMAGES, RESULTS_DIR

LABEL_MAP = {"good": 0, "particle": 1, "scratch": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
PATCH_SIZE = 64
MIN_AREA = 10


class WaferROIPatchDataset(Dataset):
    """Extracts ROI patches from wafer images using localization pipeline.

    For each image:
      1. Preprocess (CLAHE)
      2. Localize defects (Canny + morphology → contours)
      3. Crop each contour's bounding box
      4. Resize to PATCH_SIZE × PATCH_SIZE
      5. Label from filename convention
    """

    def __init__(
        self,
        data_dir: str | Path,
        patch_size: int = PATCH_SIZE,
        min_area: int = MIN_AREA,
        max_patches_per_image: int = 10,
        augment: bool = False,
    ):
        from preprocessing import Preprocessor
        from defect_localization import DefectLocalizer

        self.patch_size = patch_size
        self.augment = augment
        self.preprocessor = Preprocessor()
        self.localizer = DefectLocalizer()
        self.min_area = min_area
        self.max_patches_per_image = max_patches_per_image

        self.samples: list[Tuple[np.ndarray, int]] = []
        self._build(str(data_dir))

    def _build(self, data_dir: str) -> None:
        paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        print(f"[ROI Dataset] Scanning {len(paths)} images...")

        for img_path in paths:
            filename = os.path.basename(img_path)
            label_str = filename.split("_")[-1].split(".")[0]
            if label_str not in LABEL_MAP:
                continue

            image = cv2.imread(img_path)
            if image is None:
                continue

            enhanced, _ = self.preprocessor.process_pipeline(image)
            loc = self.localizer.localize(enhanced)
            contours = loc["contours"]
            bboxes = loc["bboxes"]

            if not contours:
                if label_str == "good":
                    full_patch = cv2.resize(
                        enhanced, (self.patch_size, self.patch_size),
                        interpolation=cv2.INTER_AREA,
                    )
                    self.samples.append((full_patch, LABEL_MAP[label_str]))
                continue

            valid = [(c, b) for c, b in zip(contours, bboxes)
                     if cv2.contourArea(c) >= self.min_area]

            if not valid:
                if label_str == "good":
                    full_patch = cv2.resize(
                        enhanced, (self.patch_size, self.patch_size),
                        interpolation=cv2.INTER_AREA,
                    )
                    self.samples.append((full_patch, LABEL_MAP[label_str]))
                continue

            valid.sort(key=lambda x: cv2.contourArea(x[0]), reverse=True)
            valid = valid[:self.max_patches_per_image]

            h, w = enhanced.shape[:2]
            for _, (x, y, bw, bh) in valid:
                x0 = max(0, x - 4)
                y0 = max(0, y - 4)
                x1 = min(w, x + bw + 4)
                y1 = min(h, y + bh + 4)

                crop = enhanced[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                patch = cv2.resize(
                    crop, (self.patch_size, self.patch_size),
                    interpolation=cv2.INTER_AREA,
                )
                self.samples.append((patch, LABEL_MAP[label_str]))

        print(f"[ROI Dataset] Extracted {len(self.samples)} patches")
        from collections import Counter
        counts = Counter(lbl for _, lbl in self.samples)
        for idx, name in IDX_TO_LABEL.items():
            print(f"              {name:>10s}: {counts.get(idx, 0)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        patch, label = self.samples[idx]

        if self.augment:
            patch = self._augment(patch)

        tensor = torch.from_numpy(patch.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor, label

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        import random
        if random.random() > 0.5:
            patch = cv2.flip(patch, 1)
        if random.random() > 0.5:
            patch = cv2.flip(patch, 0)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            h, w = patch.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            patch = cv2.warpAffine(patch, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if random.random() > 0.5:
            beta = random.uniform(-20, 20)
            patch = np.clip(patch.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        if random.random() > 0.5:
            sigma = random.uniform(0.5, 2.0)
            noise = np.random.normal(0, sigma, patch.shape).astype(np.float32)
            patch = np.clip(patch.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return patch
