"""Model contract metadata (classes, feature schema) for deploy backends."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from paths import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FEATURE_VERSION,
    ONNX_INPUT_NAME,
)


def meta_path_for(model_path: str | Path) -> Path:
    path = Path(model_path)
    return path.with_suffix(path.suffix + ".meta.json") if path.suffix else path.with_name(
        path.name + ".meta.json"
    )


def write_model_meta(
    model_path: str | Path,
    classes: Sequence[str],
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write sidecar metadata next to a model artifact."""
    path = Path(model_path)
    out = path.with_name(path.stem + ".meta.json")
    payload: Dict[str, Any] = {
        "feature_version": FEATURE_VERSION,
        "feature_dim": FEATURE_DIM,
        "feature_names": list(FEATURE_NAMES),
        "input_name": ONNX_INPUT_NAME,
        "classes": [str(c) for c in classes],
        "model_file": path.name,
    }
    if extra:
        payload.update(extra)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def load_model_meta(
    model_path: str | Path,
    *,
    required: bool = True,
) -> Optional[Dict[str, Any]]:
    """Load sidecar metadata for a model. Tries stem.meta.json then path.meta.json."""
    path = Path(model_path)
    candidates = [
        path.with_name(path.stem + ".meta.json"),
        path.with_suffix(path.suffix + ".meta.json"),
    ]
    # Also try sibling for OpenVINO: rf_model.xml -> rf_model.meta.json
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    if required:
        tried = ", ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"Model metadata not found for {path}. Tried: {tried}. "
            "Re-export the model or provide classes via classes= / meta_path=."
        )
    return None


def classes_from_meta(meta: Dict[str, Any]) -> List[str]:
    classes = meta.get("classes")
    if not classes:
        raise ValueError("Model metadata missing non-empty 'classes' list")
    return [str(c) for c in classes]
