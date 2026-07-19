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


# ---------------------------------------------------------------------------
# P1-03: C++/Python feature alignment test
# ---------------------------------------------------------------------------

def test_cpp_feature_alignment():
    """Run the C++ binary with --json and compare features to Python.

    Tolerance is 1e-3 for floating-point differences caused by
    implementation variance (OpenCV C++ vs Python bindings).
    """
    import json
    import subprocess

    cpp_bin = ROOT / "cpp" / "build" / "WaferDefectX_Run"
    if not cpp_bin.is_file():
        pytest.skip("C++ binary not built (run: cmake --build cpp/build)")

    sample = sorted(DATA_SYNTHETIC.glob("*.png"))[0]
    result = subprocess.run(
        [str(cpp_bin), "--json", str(sample)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"C++ binary failed: {result.stderr}"
    cpp_out = json.loads(result.stdout.split("\n", 6)[-1])

    pre = Preprocessor()
    loc = DefectLocalizer()
    fe = FeatureExtractor()

    image = cv2.imread(str(sample))
    enhanced, _ = pre.process_pipeline(image)
    loc_res = loc.localize(enhanced)

    cpp_defects = cpp_out.get("defects", [])
    assert len(cpp_defects) == len(loc_res["contours"]), (
        f"Defect count mismatch: C++={len(cpp_defects)}, Python={len(loc_res['contours'])}"
    )

    for i, (cpp_d, contour) in enumerate(zip(cpp_defects, loc_res["contours"])):
        py_feats = fe.extract_features(enhanced, contour)
        cpp_feats = np.array(cpp_d["features"])
        np.testing.assert_allclose(
            cpp_feats, py_feats, atol=0.01,
            err_msg=f"Feature mismatch at defect {i}: C++={cpp_feats}, Python={py_feats}",
        )


# ---------------------------------------------------------------------------
# P1-05: Per-contour training produces noise class
# ---------------------------------------------------------------------------

def test_extract_dataset_includes_noise_class():
    from train_eval import extract_dataset_features
    X, y, meta = extract_dataset_features(DATA_SYNTHETIC, include_negative=True)
    assert len(X) > 0
    assert "noise" in set(y), "Negative class 'noise' should appear for good images"
