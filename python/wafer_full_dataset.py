"""Full-image wafer dataset for CNN classification (Phase A1).

Instead of running classical CV localization (which fails on 3-value wafer maps),
this loads the full wafer map image and resizes it for direct CNN classification.

The wafer map encodes die status spatially:
  - 40  = background (outside wafer disk)
  - 140 = good die
  - 240 = defective die

The CNN learns spatial patterns that distinguish good/particle/scratch.

Usage:
    PYTHONPATH=python python3 python/wafer_full_dataset.py --build
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from paths import DATA_WM811K

LABEL_MAP = {"good": 0, "particle": 1, "scratch": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

CACHE_DIR = DATA_WM811K / "full_cache"


class WaferFullImageDataset(Dataset):
    """Full-image wafer map dataset with disk caching.

    Loads 800×800 wafer map PNGs, resizes to target size,
    normalizes to [0,1], caches as numpy arrays.
    """

    def __init__(
        self,
        data_dir: str | Path,
        target_size: int = 96,
        augment: bool = False,
        cache_dir: str | Path | None = None,
        max_per_class: int = 0,
        oversample_scratch: int = 1,
    ):
        self.target_size = target_size
        self.augment = augment
        self.oversample_scratch = oversample_scratch

        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR / f"sz{target_size}"
        self.cache_meta = self.cache_dir / "meta.json"

        self.samples: list[Tuple[np.ndarray, int]] = []

        if self.cache_meta.exists():
            self._load_cache(max_per_class)
        else:
            self._build(str(data_dir), max_per_class)

    def _build(self, data_dir: str, max_per_class: int = 0) -> None:
        paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        print(f"[Full Dataset] Scanning {len(paths)} images...")

        images = []
        labels = []
        per_class = {}

        for i, img_path in enumerate(paths):
            filename = os.path.basename(img_path)
            label_str = filename.split("_")[-1].split(".")[0]
            if label_str not in LABEL_MAP:
                continue

            cls = LABEL_MAP[label_str]
            if max_per_class > 0 and per_class.get(cls, 0) >= max_per_class:
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            resized = cv2.resize(img, (self.target_size, self.target_size),
                                 interpolation=cv2.INTER_NEAREST)
            images.append(resized)
            labels.append(cls)
            per_class[cls] = per_class.get(cls, 0) + 1

            if (i + 1) % 20000 == 0:
                print(f"  ... {i+1:,} / {len(paths):,}")

        print(f"[Full Dataset] Loaded {len(images):,} images")
        for idx, name in IDX_TO_LABEL.items():
            print(f"              {name:>10s}: {per_class.get(idx, 0):,}")

        self.samples = list(zip(images, labels))
        self._save_cache(images, labels)

    def _save_cache(self, images: list, labels: list) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        img_arr = np.stack(images)
        lbl_arr = np.array(labels, dtype=np.int32)

        np.save(str(self.cache_dir / "images.npy"), img_arr)
        np.save(str(self.cache_dir / "labels.npy"), lbl_arr)

        meta = {
            "target_size": self.target_size,
            "num_samples": len(images),
            "label_counts": {IDX_TO_LABEL[i]: int(np.sum(lbl_arr == i))
                             for i in range(3)},
        }
        self.cache_meta.write_text(json.dumps(meta, indent=2))
        print(f"[Full Dataset] Cache saved to {self.cache_dir}")
        print(f"  Shape: {img_arr.shape}")

    def _load_cache(self, max_per_class: int = 0) -> None:
        print(f"[Full Dataset] Loading cache from {self.cache_dir}...")
        img_arr = np.load(str(self.cache_dir / "images.npy"))
        lbl_arr = np.load(str(self.cache_dir / "labels.npy"))

        meta = json.loads(self.cache_meta.read_text())
        print(f"  Cache: {meta['num_samples']:,} samples")
        for name, cnt in meta["label_counts"].items():
            print(f"    {name:>10s}: {cnt:,}")

        if max_per_class > 0:
            selected_idx = []
            per_class = {}
            for i, lbl in enumerate(lbl_arr):
                if per_class.get(lbl, 0) < max_per_class:
                    selected_idx.append(i)
                    per_class[lbl] = per_class.get(lbl, 0) + 1
            selected_idx = np.array(selected_idx)
            img_arr = img_arr[selected_idx]
            lbl_arr = lbl_arr[selected_idx]
            print(f"  Subsampled to {len(lbl_arr):,} (max_per_class={max_per_class})")

        self.samples = [(img_arr[i], int(lbl_arr[i])) for i in range(len(lbl_arr))]

        if self.oversample_scratch > 1:
            scratch_idx = [i for i, (_, lbl) in enumerate(self.samples) if lbl == 2]
            scratch_data = [self.samples[i] for i in scratch_idx]
            for _ in range(self.oversample_scratch - 1):
                self.samples.extend(scratch_data)
            print(f"  Scratch oversampled ×{self.oversample_scratch}")

        from collections import Counter
        counts = Counter(lbl for _, lbl in self.samples)
        print(f"  Final distribution:")
        for idx, name in IDX_TO_LABEL.items():
            print(f"    {name:>10s}: {counts.get(idx, 0):,}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.samples[idx]

        if self.augment:
            img = self._augment(img)

        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor, label

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)
        if np.random.random() > 0.5:
            img = cv2.flip(img, 0)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if np.random.random() > 0.5:
            gamma = np.random.uniform(0.8, 1.2)
            img = np.clip(255.0 * (img.astype(np.float32) / 255.0) ** gamma, 0, 255).astype(np.uint8)
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 1.0, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img
