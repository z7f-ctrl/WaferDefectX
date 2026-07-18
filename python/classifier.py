"""Defect classifier factory with deploy backends and model-contract checks."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC

from model_meta import classes_from_meta, load_model_meta, write_model_meta
from paths import FEATURE_DIM

try:
    from cnn_model import WaferCNN

    _CNN_AVAILABLE = True
except ImportError:
    _CNN_AVAILABLE = False

SUPPORTED_MODEL_TYPES = ("rf", "svm", "onnx", "openvino", "cnn")


class DefectClassifier:
    def __init__(
        self,
        model_type: str = "rf",
        onnx_path: Optional[str] = None,
        cnn_path: Optional[str] = None,
        classes: Optional[Sequence[str]] = None,
        meta_path: Optional[str] = None,
    ):
        """Factory constructor — one backend per model_type.

        Parameters
        ----------
        model_type : str
            One of 'rf' | 'svm' | 'onnx' | 'openvino' | 'cnn'
        onnx_path : str, optional
            Path to .onnx or OpenVINO .xml/.onnx (for 'onnx'/'openvino').
        cnn_path : str, optional
            Path to WaferCNN .pth weights (for 'cnn').
        classes : sequence of str, optional
            Explicit class labels (required for openvino if no metadata).
        meta_path : str, optional
            Override path to *.meta.json contract file.
        """
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unknown model_type={model_type!r}. "
                f"Choose one of {SUPPORTED_MODEL_TYPES}"
            )

        self.model_type = model_type
        self.classes_: Optional[List[str]] = None
        self.model = None
        self.input_name = None
        self.label_name = None
        self.output_layer = None

        if model_type == "rf":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        elif model_type == "svm":
            self.model = SVC(kernel="rbf", probability=True, random_state=42)

        elif model_type == "onnx":
            self._init_onnx(onnx_path, classes=classes, meta_path=meta_path)

        elif model_type == "openvino":
            self._init_openvino(onnx_path, classes=classes, meta_path=meta_path)

        elif model_type == "cnn":
            self._init_cnn(cnn_path)

    def _resolve_classes(
        self,
        model_path: str,
        classes: Optional[Sequence[str]],
        meta_path: Optional[str],
    ) -> List[str]:
        if classes is not None:
            return [str(c) for c in classes]
        if meta_path is not None:
            meta = load_model_meta(meta_path, required=True)
            return classes_from_meta(meta)
        meta = load_model_meta(model_path, required=True)
        return classes_from_meta(meta)

    def _init_onnx(
        self,
        onnx_path: Optional[str],
        classes: Optional[Sequence[str]],
        meta_path: Optional[str],
    ) -> None:
        if not onnx_path:
            raise ValueError(
                "onnx_path must be provided for model_type='onnx' "
                "(path to a .onnx model file)"
            )
        path = Path(onnx_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")

        try:
            import onnxruntime as rt
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for model_type='onnx'. "
                "Install with: pip install onnxruntime"
            ) from exc

        self.model = rt.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.model.get_inputs()[0].name
        self.label_name = self.model.get_outputs()[0].name
        # Labels may be embedded in ZipMap-free string output; still load contract.
        try:
            self.classes_ = self._resolve_classes(str(path), classes, meta_path)
        except FileNotFoundError:
            self.classes_ = list(classes) if classes is not None else None

    def _init_openvino(
        self,
        onnx_path: Optional[str],
        classes: Optional[Sequence[str]],
        meta_path: Optional[str],
    ) -> None:
        if not onnx_path:
            raise ValueError(
                "openvino requires onnx_path (path to .xml or .onnx model)"
            )
        path = Path(onnx_path)
        if not path.is_file():
            raise FileNotFoundError(f"OpenVINO/ONNX model not found: {path}")

        try:
            import openvino as ov
        except ImportError as exc:
            raise ImportError(
                "openvino is required for model_type='openvino'. "
                "Install with: pip install openvino"
            ) from exc

        core = ov.Core()
        ov_model = core.read_model(str(path))
        self.model = core.compile_model(ov_model, "CPU")
        self.output_layer = self.model.output(0)
        self.classes_ = self._resolve_classes(str(path), classes, meta_path)

    def _init_cnn(self, cnn_path: Optional[str]) -> None:
        if not _CNN_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Run: pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cpu"
            )
        self.model = WaferCNN()
        if cnn_path is not None:
            path = Path(cnn_path)
            if not path.is_file():
                raise FileNotFoundError(f"CNN weights not found: {path}")
            self.model.load(str(path))
        else:
            print(
                "[DefectClassifier] CNN initialised with random weights. "
                "Pass cnn_path= to load trained weights."
            )

    @staticmethod
    def _ensure_feature_matrix(X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(
                f"Expected 2D feature matrix [N, {FEATURE_DIM}], got shape {arr.shape}"
            )
        if arr.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"Expected feature_dim={FEATURE_DIM}, got {arr.shape[1]}"
            )
        return arr

    def train(self, X, y):
        if self.model_type not in ("rf", "svm"):
            raise TypeError(
                f"train() is only supported for rf/svm, not {self.model_type!r}"
            )
        print(f"Training {self.model_type.upper()} classifier...")
        self.model.fit(X, y)
        self.classes_ = [str(c) for c in self.model.classes_]

    def predict(self, X):
        if self.model_type == "cnn":
            return self.model.predict(X)

        if self.model_type == "onnx":
            X = self._ensure_feature_matrix(X)
            return self.model.run([self.label_name], {self.input_name: X})[0]

        if self.model_type == "openvino":
            if not self.classes_:
                raise RuntimeError(
                    "OpenVINO backend has no classes_; provide metadata or classes="
                )
            X = self._ensure_feature_matrix(X)
            res = self.model([X])
            pred_idx = res[self.output_layer][0]
            if isinstance(pred_idx, np.ndarray):
                pred_idx = int(pred_idx.item() if pred_idx.size == 1 else pred_idx[0])
            else:
                pred_idx = int(pred_idx)
            if pred_idx < 0 or pred_idx >= len(self.classes_):
                raise IndexError(
                    f"Predicted class index {pred_idx} out of range for "
                    f"classes_={self.classes_}"
                )
            return self.classes_[pred_idx]

        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        if self.model_type not in ("rf", "svm"):
            raise TypeError(
                f"evaluate() is only supported for rf/svm, not {self.model_type!r}"
            )
        y_pred = self.model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

    def save_model(self, path: Union[str, Path]):
        if self.model_type not in ("rf", "svm"):
            raise TypeError(
                f"save_model() is only supported for rf/svm, not {self.model_type!r}"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        classes = self.classes_ or [str(c) for c in getattr(self.model, "classes_", [])]
        write_model_meta(path, classes, extra={"backend": self.model_type})

    def load_model(self, path: Union[str, Path]):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(path)
        self.model_type = "rf"  # joblib path is sklearn
        if hasattr(self.model, "classes_"):
            self.classes_ = [str(c) for c in self.model.classes_]
        else:
            meta = load_model_meta(path, required=False)
            self.classes_ = classes_from_meta(meta) if meta else None
