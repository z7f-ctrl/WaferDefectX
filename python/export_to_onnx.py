import os

import joblib
import numpy as np
import onnxruntime as rt
from model_meta import write_model_meta
from paths import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FEATURE_VERSION,
    ONNX_INPUT_NAME,
    RESULTS_DIR,
    ensure_dir,
)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def main():
    ensure_dir(RESULTS_DIR)
    model_path = RESULTS_DIR / "rf_model.pkl"
    onnx_model_path = RESULTS_DIR / "rf_model.onnx"

    if not model_path.is_file():
        print(f"Model file {model_path} not found. Please train the model first.")
        return

    print(f"Loading scikit-learn model from {model_path}...")
    model = joblib.load(model_path)
    classes = [str(c) for c in getattr(model, "classes_", [])]

    initial_type = [(ONNX_INPUT_NAME, FloatTensorType([None, FEATURE_DIM]))]

    print("Converting model to ONNX format...")
    options = {id(model): {"zipmap": False}}
    onx = convert_sklearn(
        model, initial_types=initial_type, target_opset=12, options=options
    )

    with open(onnx_model_path, "wb") as f:
        f.write(onx.SerializeToString())

    meta_out = write_model_meta(
        onnx_model_path,
        classes,
        extra={
            "backend": "onnx",
            "opset": 12,
            "source_model": model_path.name,
            "feature_version": FEATURE_VERSION,
            "feature_names": list(FEATURE_NAMES),
        },
    )
    print(f"ONNX model saved to {onnx_model_path}")
    print(f"Metadata saved to {meta_out}")

    print("\n--- Verifying ONNX Inference ---")
    sess = rt.InferenceSession(
        str(onnx_model_path), providers=["CPUExecutionProvider"]
    )

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    dummy_input = np.random.rand(1, FEATURE_DIM).astype(np.float32)
    pred_onnx = sess.run([label_name], {input_name: dummy_input})[0]

    print(f"Dummy Input: {dummy_input}")
    print(f"ONNX Prediction Output: {pred_onnx}")
    print("Inference successful! ONNX runtime integration is working.")


if __name__ == "__main__":
    main()
