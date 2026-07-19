"""Real wafer map dataset parsers (P1-08).

Supports:
  - WM-811K raw .pkl (pandas pickle) → conversion to images
  - WM-811K converted image directory (data/wm811k/images/)
  - Generic directory layout matching synthetic convention
  - CSV/JSON label files with per-image annotations
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

LABEL_MAP = {
    0: "edge-loc",
    1: "edge-ring",
    2: "center",
    3: "scratch",
    4: "random",
    5: "none",
    6: "near-full",
    7: "horiz-line",
    8: "vert-line",
    9: "local-spot",
}

SIMPLIFY_MAP = {
    "edge-loc": "particle",
    "edge-ring": "particle",
    "center": "particle",
    "scratch": "scratch",
    "random": "particle",
    "none": "good",
    "near-full": "good",
    "horiz-line": "scratch",
    "vert-line": "scratch",
    "local-spot": "particle",
    "Good": "good",
    "Edge-Ring": "particle",
    "Edge-Loc": "particle",
    "Center": "particle",
    "Loc": "particle",
    "Random": "particle",
    "Scratch": "scratch",
    "Donut": "particle",
    "Near-full": "good",
}


@dataclass
class WaferSample:
    image: np.ndarray
    label: str
    wafer_id: str = ""
    map_index: int = 0
    path: str = ""


def parse_wm811k_mat(mat_path: Path) -> List[WaferSample]:
    """Parse a single WM-811K .mat file containing one wafer map.

    Expects scipy .mat with key 'wm_map' (2-D ndarray) and optional 'label'.
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy is required for WM-811K .mat parsing: pip install scipy")

    mat = loadmat(str(mat_path))
    wm = mat.get("wm_map")
    if wm is None:
        for k in mat:
            if not k.startswith("_"):
                wm = mat[k]
                break
    if wm is None:
        return []

    if isinstance(wm, np.ndarray) and wm.ndim == 0:
        wm = wm.item()
    if isinstance(wm, dict):
        wm = wm.get("map", list(wm.values())[0])

    wm = np.asarray(wm, dtype=np.uint8)
    label_key = mat.get("label", mat.get("failureType", None))
    if label_key is not None:
        if isinstance(label_key, np.ndarray) and label_key.size > 0:
            raw = label_key.flat[0]
            if isinstance(raw, str):
                label = raw
            elif isinstance(raw, np.ndarray) and raw.size > 0:
                label = str(raw.flat[0])
            else:
                label = str(raw)
        else:
            label = "unknown"
    else:
        label = "unknown"

    label = SIMPLIFY_MAP.get(label, label)

    wm_color = cv2.cvtColor(wm * 30, cv2.COLOR_GRAY2BGR) if wm.max() <= 1 else cv2.cvtColor(wm, cv2.COLOR_GRAY2BGR)

    return [WaferSample(
        image=wm_color,
        label=label,
        wafer_id=mat_path.stem,
        path=str(mat_path),
    )]


def parse_directory(
    data_dir: Path,
    label_file: Optional[Path] = None,
    ext: str = "*.png",
) -> List[WaferSample]:
    """Parse a directory of wafer images.

    Label resolution priority:
      1. label_file (CSV with columns: filename, label)
      2. Filename convention: wafer_<id>_<label>.<ext>
    """
    label_map = {}
    if label_file and label_file.exists():
        with open(label_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_map[row["filename"]] = row.get("label", "unknown")

    samples = []
    for img_path in sorted(data_dir.glob(ext)):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        filename = img_path.name
        if filename in label_map:
            label = label_map[filename]
        else:
            parts = filename.rsplit(".", 1)[0]
            label = parts.split("_")[-1]

        samples.append(WaferSample(
            image=image,
            label=label,
            wafer_id=filename,
            path=str(img_path),
        ))

    return samples


def parse_json_labels(json_path: Path, image_dir: Path) -> List[WaferSample]:
    """Parse a JSON label file with structure:
    [
      {"filename": "wafer_001.png", "label": "scratch", "bbox": [x,y,w,h]},
      ...
    ]
    """
    with open(json_path) as f:
        annotations = json.load(f)

    samples = []
    for ann in annotations:
        img_path = image_dir / ann["filename"]
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        samples.append(WaferSample(
            image=image,
            label=ann.get("label", "unknown"),
            wafer_id=ann["filename"],
            path=str(img_path),
        ))

    return samples


def samples_to_training_data(
    samples: List[WaferSample],
    preprocessor=None,
    localizer=None,
    extractor=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert parsed samples into feature matrix and labels.

    Uses the same preprocessing→localization→feature extraction pipeline
    as the synthetic path for consistency.
    """
    from defect_localization import DefectLocalizer
    from features import FeatureExtractor
    from preprocessing import Preprocessor

    preprocessor = preprocessor or Preprocessor()
    localizer = localizer or DefectLocalizer()
    extractor = extractor or FeatureExtractor()

    X, y = [], []
    for sample in samples:
        enhanced, _ = preprocessor.process_pipeline(sample.image)
        loc = localizer.localize(enhanced)
        contours = loc["contours"]
        if not contours:
            continue
        for contour in contours:
            feats = extractor.extract_features(enhanced, contour)
            X.append(feats)
            y.append(sample.label)

    return np.array(X) if X else np.empty((0, 7)), np.array(y) if y else np.empty(0)


def parse_wm811k_pkl(pkl_path: Path, max_per_class: int = 2000) -> List[WaferSample]:
    """Parse WM-811K raw .pkl file and convert to WaferSample list.

    This reads the original LSWMD.pkl from Kaggle and converts wafer maps
    to grayscale images. Use parse_wm811k_directory() for pre-converted images.

    Parameters
    ----------
    pkl_path : Path
        Path to LSWMD.pkl
    max_per_class : int
        Maximum samples per simplified class to keep (for memory/speed).
    """
    import pandas as pd

    df = pd.read_pickle(str(pkl_path))

    labeled_mask = df["failureType"].apply(
        lambda x: x.size > 0 if hasattr(x, "size") else len(x) > 0
    )
    df_labeled = df[labeled_mask].copy()
    df_labeled["raw_label"] = df_labeled["failureType"].apply(lambda x: x.flatten()[0])
    df_labeled["label"] = df_labeled["raw_label"].map(SIMPLIFY_MAP).fillna("unknown")

    # Balance classes
    samples = []
    for lbl in df_labeled["label"].unique():
        subset = df_labeled[df_labeled["label"] == lbl]
        if len(subset) > max_per_class:
            subset = subset.sample(max_per_class, random_state=42)
        samples.append(subset)
    df_sample = pd.concat(samples)

    VAL_MAP = np.array([40, 140, 240], dtype=np.uint8)
    result = []
    for idx, row in df_sample.iterrows():
        wm = row["waferMap"]
        wm_resized = cv2.resize(wm, (800, 800), interpolation=cv2.INTER_NEAREST)
        wm_gray = VAL_MAP[wm_resized]

        h, w = wm_gray.shape
        center = (w // 2, h // 2)
        radius = min(h, w) // 2 - 10
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        wm_gray[mask == 0] = 40

        wm_bgr = cv2.cvtColor(wm_gray, cv2.COLOR_GRAY2BGR)
        result.append(WaferSample(
            image=wm_bgr,
            label=str(row["label"]),
            wafer_id=f"wm811k_{idx}",
            path=str(pkl_path),
        ))

    return result


def parse_wm811k_directory(
    data_dir: Path,
    ext: str = "*.png",
) -> List[WaferSample]:
    """Parse WM-811K pre-converted image directory.

    Expects directory structure: data/wm811k/images/wafer_NNNNN_<label>.png
    Also reads labels.csv if present for original WM-811K label mapping.
    """
    label_csv = data_dir.parent / "labels.csv"
    label_map = {}
    if label_csv.exists():
        with open(label_csv) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                fname = f"wafer_{i:05d}_{SIMPLIFY_MAP.get(row.get('label', ''), row.get('label', 'unknown'))}.png"
                label_map[fname] = row.get("simple_label", row.get("label", "unknown"))

    samples = []
    for img_path in sorted(data_dir.glob(ext)):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        filename = img_path.name
        if filename in label_map:
            label = label_map[filename]
        else:
            parts = filename.rsplit(".", 1)[0]
            label = parts.split("_")[-1]

        samples.append(WaferSample(
            image=image,
            label=label,
            wafer_id=filename,
            path=str(img_path),
        ))

    return samples
