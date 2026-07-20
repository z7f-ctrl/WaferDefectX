"""ROI Patch Dataset with disk caching (M3a + A1).

Extracts bbox crops from localized defects, resizes to 64×64,
and caches patches to disk to avoid re-running localization.

Two modes:
  - build(): Run localization on images, save patches to cache
  - load():  Read pre-computed patches from cache (fast)

Usage:
    PYTHONPATH=python python3 python/roi_dataset.py --build --data-dir data/wm811k/full_images
    PYTHONPATH=python python3 python/roi_dataset.py --build --data-dir data/wm811k/full_images --max-per-class 25000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from paths import DATA_WM811K, RESULTS_DIR

LABEL_MAP = {"good": 0, "particle": 1, "scratch": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
PATCH_SIZE = 64
MIN_AREA = 10

CACHE_DIR = DATA_WM811K / "patch_cache"


class WaferROIPatchDataset(Dataset):
    """ROI Patch Dataset with disk caching.

    First call: build() to extract patches and save to cache.
    Subsequent calls: load() from cache (seconds instead of hours).
    """

    def __init__(
        self,
        data_dir: str | Path,
        patch_size: int = PATCH_SIZE,
        augment: bool = False,
        cache_dir: str | Path | None = None,
        max_per_class: int = 0,
        oversample_scratch: int = 1,
    ):
        self.patch_size = patch_size
        self.augment = augment
        self.oversample_scratch = oversample_scratch

        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR / f"ps{patch_size}"
        self.cache_meta = self.cache_dir / "meta.json"
        self.cache_labels = self.cache_dir / "labels.npy"

        self.samples: list[Tuple[np.ndarray, int]] = []

        if self.cache_meta.exists():
            self._load_cache(max_per_class)
        else:
            self._build(str(data_dir), max_per_class)

    def _build(self, data_dir: str, max_per_class: int = 0) -> None:
        from preprocessing import Preprocessor
        from defect_localization import DefectLocalizer

        preprocessor = Preprocessor()
        localizer = DefectLocalizer()

        paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        print(f"[ROI Dataset] Scanning {len(paths)} images...")

        patches = []
        labels = []
        per_class = {}
        skipped = 0

        for i, img_path in enumerate(paths):
            filename = os.path.basename(img_path)
            label_str = filename.split("_")[-1].split(".")[0]
            if label_str not in LABEL_MAP:
                continue

            cls = LABEL_MAP[label_str]
            if max_per_class > 0 and per_class.get(cls, 0) >= max_per_class:
                continue

            image = cv2.imread(img_path)
            if image is None:
                skipped += 1
                continue

            enhanced, _ = preprocessor.process_pipeline(image)
            loc = localizer.localize(enhanced)
            contours = loc["contours"]
            bboxes = loc["bboxes"]

            if not contours:
                if label_str == "good":
                    full_patch = cv2.resize(
                        enhanced, (self.patch_size, self.patch_size),
                        interpolation=cv2.INTER_AREA,
                    )
                    patches.append(full_patch)
                    labels.append(cls)
                    per_class[cls] = per_class.get(cls, 0) + 1
                continue

            valid = [(c, b) for c, b in zip(contours, bboxes)
                     if cv2.contourArea(c) >= MIN_AREA]

            if not valid:
                if label_str == "good":
                    full_patch = cv2.resize(
                        enhanced, (self.patch_size, self.patch_size),
                        interpolation=cv2.INTER_AREA,
                    )
                    patches.append(full_patch)
                    labels.append(cls)
                    per_class[cls] = per_class.get(cls, 0) + 1
                continue

            valid.sort(key=lambda x: cv2.contourArea(x[0]), reverse=True)
            valid = valid[:10]

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
                patches.append(patch)
                labels.append(cls)
                per_class[cls] = per_class.get(cls, 0) + 1

            if (i + 1) % 5000 == 0:
                print(f"  ... {i+1:,} / {len(paths):,} | patches: {len(patches):,}")

        print(f"[ROI Dataset] Extracted {len(patches):,} patches (skipped {skipped})")
        for idx, name in IDX_TO_LABEL.items():
            print(f"              {name:>10s}: {per_class.get(idx, 0):,}")

        self.samples = list(zip(patches, labels))
        self._save_cache(patches, labels)

    def _save_cache(self, patches: list, labels: list) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        patches_arr = np.stack(patches)
        labels_arr = np.array(labels, dtype=np.int32)

        np.save(str(self.cache_dir / "patches.npy"), patches_arr)
        np.save(str(self.cache_dir / "labels.npy"), labels_arr)

        meta = {
            "patch_size": self.patch_size,
            "num_patches": len(patches),
            "label_counts": {IDX_TO_LABEL[i]: int(np.sum(labels_arr == i))
                             for i in range(3)},
        }
        self.cache_meta.write_text(json.dumps(meta, indent=2))
        print(f"[ROI Dataset] Cache saved to {self.cache_dir}")
        print(f"  Patches: {patches_arr.shape}, Labels: {labels_arr.shape}")

    def _load_cache(self, max_per_class: int = 0) -> None:
        print(f"[ROI Dataset] Loading cache from {self.cache_dir}...")
        patches_arr = np.load(str(self.cache_dir / "patches.npy"))
        labels_arr = np.load(str(self.cache_dir / "labels.npy"))

        meta = json.loads(self.cache_meta.read_text())
        print(f"  Cache contains {meta['num_patches']:,} patches")
        for name, cnt in meta["label_counts"].items():
            print(f"    {name:>10s}: {cnt:,}")

        if max_per_class > 0:
            selected_idx = []
            per_class = {}
            for i, lbl in enumerate(labels_arr):
                if per_class.get(lbl, 0) < max_per_class:
                    selected_idx.append(i)
                    per_class[lbl] = per_class.get(lbl, 0) + 1
            selected_idx = np.array(selected_idx)
            patches_arr = patches_arr[selected_idx]
            labels_arr = labels_arr[selected_idx]
            print(f"  Subsampled to {len(labels_arr):,} patches (max_per_class={max_per_class})")

        self.samples = [(patches_arr[i], int(labels_arr[i]))
                        for i in range(len(labels_arr))]

        if self.oversample_scratch > 1:
            scratch_idx = [i for i, (_, lbl) in enumerate(self.samples) if lbl == 2]
            scratch_patches = [self.samples[i] for i in scratch_idx]
            for _ in range(self.oversample_scratch - 1):
                self.samples.extend(scratch_patches)
            print(f"  Oversampled scratch ×{self.oversample_scratch}: "
                  f"{len(scratch_idx)} → {len(scratch_idx) * self.oversample_scratch}")

        from collections import Counter
        counts = Counter(lbl for _, lbl in self.samples)
        print(f"  Final distribution:")
        for idx, name in IDX_TO_LABEL.items():
            print(f"    {name:>10s}: {counts.get(idx, 0):,}")

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
        if random.random() > 0.7:
            gamma = random.uniform(0.7, 1.5)
            patch = np.clip(255.0 * (patch.astype(np.float32) / 255.0) ** gamma, 0, 255).astype(np.uint8)
        if random.random() > 0.8:
            h, w = patch.shape
            rh = int(h * random.uniform(0.05, 0.2))
            rw = int(w * random.uniform(0.05, 0.2))
            ry = random.randint(0, max(0, h - rh))
            rx = random.randint(0, max(0, w - rw))
            patch = patch.copy()
            patch[ry:ry+rh, rx:rx+rw] = 0
        return patch

    def _elastic_deform(self, patch: np.ndarray, strength: int = 3) -> np.ndarray:
        h, w = patch.shape
        dx = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32), (21, 21), 5) * strength
        dy = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32), (21, 21), 5) * strength
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = np.clip((x + dx).astype(np.float32), 0, w - 1)
        map_y = np.clip((y + dy).astype(np.float32), 0, h - 1)
        return cv2.remap(patch, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def _cutout(self, patch: np.ndarray, max_ratio: float = 0.2) -> np.ndarray:
        import random
        h, w = patch.shape
        rw = int(w * random.uniform(0.05, max_ratio))
        rh = int(h * random.uniform(0.05, max_ratio))
        rx = random.randint(0, max(0, w - rw))
        ry = random.randint(0, max(0, h - rh))
        patch = patch.copy()
        patch[ry:ry+rh, rx:rx+rw] = 0
        return patch


def main():
    parser = argparse.ArgumentParser(description="Build or load ROI patch cache")
    parser.add_argument("--build", action="store_true", help="Build cache from images")
    parser.add_argument("--data-dir", type=str, default=str(DATA_WM811K / "full_images"))
    parser.add_argument("--max-per-class", type=int, default=0, help="0=all samples")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--oversample-scratch", type=int, default=1)
    args = parser.parse_args()

    if args.build:
        ds = WaferROIPatchDataset(
            data_dir=args.data_dir,
            patch_size=args.patch_size,
            max_per_class=args.max_per_class,
            oversample_scratch=args.oversample_scratch,
        )
        print(f"\nTotal patches: {len(ds)}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
