import joblib
import numpy as np
import openvino as ov
import torch
from hummingbird.ml import convert
from model_meta import write_model_meta
from paths import FEATURE_DIM, RESULTS_DIR, ensure_dir


def main():
    ensure_dir(RESULTS_DIR)
    model_path = RESULTS_DIR / "rf_model.pkl"
    onnx_path = RESULTS_DIR / "hb_rf_model.onnx"
    ov_model_dir = ensure_dir(RESULTS_DIR / "ov_model")
    ov_xml_path = ov_model_dir / "rf_model.xml"

    if not model_path.is_file():
        print(f"Model file {model_path} not found. Please train first.")
        return

    print("Loading scikit-learn model...")
    model = joblib.load(model_path)
    original_classes = [str(c) for c in model.classes_]
    model.classes_ = np.arange(len(original_classes))

    print("Converting RF to PyTorch tensors using Hummingbird...")
    hb_model = convert(model, "torch")

    print("Exporting PyTorch model to standard ONNX...")
    dummy_input = torch.randn(1, FEATURE_DIM)
    torch.onnx.export(
        hb_model.model,
        dummy_input,
        str(onnx_path),
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"Hummingbird ONNX model saved to {onnx_path}")

    print("Converting ONNX to OpenVINO IR format...")
    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(ov_xml_path))
    print(f"OpenVINO model saved to {ov_model_dir}")

    write_model_meta(
        onnx_path,
        original_classes,
        extra={"backend": "hummingbird-onnx", "source_model": model_path.name},
    )
    write_model_meta(
        ov_xml_path,
        original_classes,
        extra={"backend": "openvino-hb", "source_model": onnx_path.name},
    )

    print("\n--- Verifying OpenVINO Inference ---")
    core = ov.Core()
    compiled_model = core.compile_model(ov_model, "CPU")
    dummy_ov_input = np.random.rand(1, FEATURE_DIM).astype(np.float32)
    res = compiled_model([dummy_ov_input])

    print(f"Dummy Input: {dummy_ov_input}")
    print("OpenVINO Outputs:")
    for key, value in res.items():
        print(f"  {key}: {value}")

    print("Inference successful! OpenVINO integration via Hummingbird is working.")


if __name__ == "__main__":
    main()
