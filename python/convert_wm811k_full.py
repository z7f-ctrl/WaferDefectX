"""Convert full WM-811K LSWMD.pkl to PNG images (all labeled samples).

Reads LSWMD.pkl (811K total, ~172K labeled), applies class simplification,
and writes PNG images to data/wm811k/full_images/.

Usage:
    PYTHONPATH=python python3 python/convert_wm811k_full.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from dataset_parser import SIMPLIFY_MAP
from paths import DATA_WM811K, ensure_dir

OUTPUT_DIR = DATA_WM811K / "full_images"
VAL_MAP = np.array([40, 140, 240], dtype=np.uint8)


def convert():
    pkl_path = DATA_WM811K / "LSWMD.pkl"
    if not pkl_path.exists():
        print(f"[ERROR] {pkl_path} not found")
        sys.exit(1)

    print(f"[1/4] Loading {pkl_path}...")
    df = pd.read_pickle(str(pkl_path))
    print(f"  Total samples: {len(df):,}")

    labeled_mask = df["failureType"].apply(
        lambda x: x.size > 0 if hasattr(x, "size") else len(x) > 0
    )
    df_labeled = df[labeled_mask].copy()
    df_labeled["raw_label"] = df_labeled["failureType"].apply(lambda x: x.flatten()[0])
    df_labeled["label"] = df_labeled["raw_label"].map(SIMPLIFY_MAP).fillna("unknown")
    df_labeled = df_labeled[df_labeled["label"] != "unknown"]

    print(f"  Labeled samples: {len(df_labeled):,}")
    print(f"  Class distribution:")
    for lbl, cnt in df_labeled["label"].value_counts().items():
        print(f"    {lbl:>10s}: {cnt:,}")

    ensure_dir(OUTPUT_DIR)
    print(f"\n[2/4] Converting to PNG images -> {OUTPUT_DIR}...")

    idx = 0
    label_counts = {}
    for _, row in df_labeled.iterrows():
        wm = row["waferMap"]
        label = str(row["label"])

        wm_resized = cv2.resize(wm, (800, 800), interpolation=cv2.INTER_NEAREST)
        wm_gray = VAL_MAP[wm_resized]

        h, w = wm_gray.shape
        center = (w // 2, h // 2)
        radius = min(h, w) // 2 - 10
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        wm_gray[mask == 0] = 40

        wm_bgr = cv2.cvtColor(wm_gray, cv2.COLOR_GRAY2BGR)
        fname = f"wafer_{idx:06d}_{label}.png"
        cv2.imwrite(str(OUTPUT_DIR / fname), wm_bgr)

        label_counts[label] = label_counts.get(label, 0) + 1
        idx += 1

        if idx % 10000 == 0:
            print(f"  ... {idx:,} / {len(df_labeled):,}")

    print(f"\n[3/4] Done. Total images: {idx:,}")
    print(f"  Class distribution:")
    for lbl, cnt in sorted(label_counts.items()):
        print(f"    {lbl:>10s}: {cnt:,}")

    print(f"\n[4/4] Output: {OUTPUT_DIR}")
    total_mb = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.png")) / (1024 ** 2)
    print(f"  Total size: {total_mb:.0f} MB")


if __name__ == "__main__":
    convert()
