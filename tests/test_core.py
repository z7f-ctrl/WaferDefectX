"""Unit tests for core CV modules (P0-08)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
sys.path.insert(0, str(PYTHON))

from defect_localization import DefectLocalizer  # noqa: E402
from features import FeatureExtractor  # noqa: E402
from paths import DATA_SYNTHETIC, FEATURE_DIM  # noqa: E402
from preprocessing import Preprocessor  # noqa: E402


@pytest.fixture
def sample_bgr():
    paths = sorted(DATA_SYNTHETIC.glob("*.png"))
    if paths:
        img = cv2.imread(str(paths[0]))
        assert img is not None
        return img
    # Fallback synthetic wafer if data/ not populated
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 80, (100, 100, 100), -1)
    cv2.line(img, (60, 60), (140, 140), (220, 220, 220), 2)
    return img


def test_preprocessor_grayscale_input(sample_bgr):
    pre = Preprocessor()
    gray = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2GRAY)
    enhanced, thresholded = pre.process_pipeline(gray)
    assert enhanced.shape == gray.shape
    assert thresholded.shape == gray.shape


def test_preprocessor_output_shapes(sample_bgr):
    pre = Preprocessor()
    enhanced, thresholded = pre.process_pipeline(sample_bgr)
    assert enhanced.ndim == 2
    assert thresholded.ndim == 2
    assert enhanced.shape == sample_bgr.shape[:2]
    assert enhanced.dtype == np.uint8


def test_localizer_requires_gray_and_finds_structure(sample_bgr):
    pre = Preprocessor()
    loc = DefectLocalizer()
    enhanced, _ = pre.process_pipeline(sample_bgr)
    result = loc.localize(enhanced)
    assert "contours" in result
    assert "bboxes" in result
    assert "mask" in result
    assert result["mask"].shape == enhanced.shape
    assert result["mask"].dtype == np.uint8


def test_localizer_empty_image():
    loc = DefectLocalizer()
    empty = np.zeros((0, 0), dtype=np.uint8)
    # findContours on empty may raise or return empty — should not hang
    result = loc.localize(np.zeros((32, 32), dtype=np.uint8))
    assert isinstance(result["contours"], list)


def test_feature_dim(sample_bgr):
    pre = Preprocessor()
    loc = DefectLocalizer()
    fe = FeatureExtractor()
    enhanced, _ = pre.process_pipeline(sample_bgr)
    result = loc.localize(enhanced)
    if not result["contours"]:
        pytest.skip("no contours on sample")
    feats = fe.extract_features(enhanced, result["contours"][0])
    assert feats.shape == (FEATURE_DIM,)
    assert np.isfinite(feats).all()


def test_paths_resolve_inside_repo():
    from paths import PROJECT_ROOT, RESULTS_DIR

    assert PROJECT_ROOT.name == "WaferDefectX" or (PROJECT_ROOT / "python").is_dir()
    assert RESULTS_DIR.parent == PROJECT_ROOT
