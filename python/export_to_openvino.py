import shutil

import numpy as np
import openvino as ov
from model_meta import load_model_meta, write_model_meta
from paths import FEATURE_DIM, RESULTS_DIR, ensure_dir


def main():
    ensure_dir(RESULTS_DIR)
    onnx_model_path = RESULTS_DIR / "rf_model.onnx"
    ov_model_dir = ensure_dir(RESULTS_DIR / "ov_model")
    ov_xml_path = ov_model_dir / "rf_model.xml"

    if not onnx_model_path.is_file():
        print(
            f"ONNX model file {onnx_model_path} not found. "
            "Please convert the model to ONNX first."
        )
        return

    print(f"Loading ONNX model from {onnx_model_path}...")
    print("Converting model to OpenVINO IR format...")
    ov_model = ov.convert_model(str(onnx_model_path))
    ov.save_model(ov_model, str(ov_xml_path))
    print(f"OpenVINO model saved to {ov_model_dir}")

    # Propagate class / feature contract from ONNX sidecar (or pkl meta).
    meta = load_model_meta(onnx_model_path, required=False)
    if meta is None:
        meta = load_model_meta(RESULTS_DIR / "rf_model.pkl", required=False)
    if meta and meta.get("classes"):
        write_model_meta(
            ov_xml_path,
            meta["classes"],
            extra={
                "backend": "openvino",
                "source_model": onnx_model_path.name,
                "feature_dim": meta.get("feature_dim", FEATURE_DIM),
                "feature_version": meta.get("feature_version"),
                "feature_names": meta.get("feature_names"),
            },
        )
        # Also copy next to onnx for convenience if missing
        onnx_meta = onnx_model_path.with_name("rf_model.meta.json")
        if not onnx_meta.is_file():
            shutil.copy(ov_xml_path.with_name("rf_model.meta.json"), onnx_meta)
        print(f"Metadata saved to {ov_xml_path.with_name('rf_model.meta.json')}")
    else:
        print(
            "WARNING: No classes metadata found. "
            "OpenVINO inference via DefectClassifier will require classes= or meta."
        )

    print("\n--- Verifying OpenVINO Inference ---")
    core = ov.Core()
    compiled_model = core.compile_model(ov_model, "CPU")
    dummy_input = np.random.rand(1, FEATURE_DIM).astype(np.float32)
    res = compiled_model([dummy_input])
    output_layer = compiled_model.output(0)
    pred_ov = res[output_layer][0]

    print(f"Dummy Input: {dummy_input}")
    print(f"OpenVINO Prediction Output: {pred_ov}")
    print("Inference successful! OpenVINO integration is working.")


if __name__ == "__main__":
    main()
