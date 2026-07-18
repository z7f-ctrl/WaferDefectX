"""Tests for model metadata contract (P0-05/06)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from model_meta import classes_from_meta, load_model_meta, write_model_meta  # noqa: E402
from paths import FEATURE_DIM  # noqa: E402


def test_write_and_load_meta(tmp_path):
    model = tmp_path / "rf_model.onnx"
    model.write_bytes(b"fake")
    out = write_model_meta(model, ["particle", "scratch"], extra={"backend": "onnx"})
    assert out.name == "rf_model.meta.json"
    meta = load_model_meta(model, required=True)
    assert meta["feature_dim"] == FEATURE_DIM
    assert classes_from_meta(meta) == ["particle", "scratch"]


def test_missing_meta_raises(tmp_path):
    model = tmp_path / "missing.onnx"
    model.write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        load_model_meta(model, required=True)


def test_classifier_rejects_unknown_type():
    from classifier import DefectClassifier

    with pytest.raises(ValueError, match="Unknown model_type"):
        DefectClassifier(model_type="not-a-model")


def test_classifier_onnx_requires_path():
    from classifier import DefectClassifier

    with pytest.raises(ValueError, match="onnx_path"):
        DefectClassifier(model_type="onnx")


def test_openvino_loads_classes_from_meta(tmp_path, monkeypatch):
    """Ensure classes are not hard-coded when metadata is present."""
    xml = tmp_path / "rf_model.xml"
    xml.write_text("<net/>")
    write_model_meta(xml, ["scratch", "particle"])

    # Skip if openvino not installed
    pytest.importorskip("openvino")

    # openvino will fail to read fake xml — we only assert resolve_classes path
    from classifier import DefectClassifier

    clf = DefectClassifier.__new__(DefectClassifier)
    classes = DefectClassifier._resolve_classes(
        clf, str(xml), classes=None, meta_path=None
    )
    assert classes == ["scratch", "particle"]
