import joblib
import torch
from hummingbird.ml import convert
import openvino as ov
import numpy as np

def main():
    model_path = "WaferDefectX/results/rf_model.pkl"
    onnx_path = "WaferDefectX/results/hb_rf_model.onnx"
    ov_model_dir = "WaferDefectX/results/ov_model"

    print("Loading scikit-learn model...")
    model = joblib.load(model_path)
    # Hummingbird requires integer labels for Random Forest Classifier
    original_classes = model.classes_
    model.classes_ = np.arange(len(original_classes))
    
    # 1. Convert to Hummingbird (PyTorch)
    print("Converting RF to PyTorch tensors using Hummingbird...")
    hb_model = convert(model, 'torch')
    
    # 2. Export PyTorch model to standard ONNX
    print("Exporting PyTorch model to standard ONNX...")
    dummy_input = torch.randn(1, 7)
    torch.onnx.export(
        hb_model.model, 
        dummy_input, 
        onnx_path, 
        opset_version=12, 
        input_names=['input'], 
        output_names=['output']
    )
    print(f"Hummingbird ONNX model saved to {onnx_path}")
    
    # 3. Convert standard ONNX to OpenVINO IR
    print("Converting ONNX to OpenVINO IR format...")
    ov_model = ov.convert_model(onnx_path)
    
    import os
    if not os.path.exists(ov_model_dir):
        os.makedirs(ov_model_dir)
        
    ov.save_model(ov_model, os.path.join(ov_model_dir, "rf_model.xml"))
    print(f"OpenVINO model saved to {ov_model_dir}")
    
    # 4. Verify OpenVINO inference
    print("\n--- Verifying OpenVINO Inference ---")
    core = ov.Core()
    compiled_model = core.compile_model(ov_model, "CPU")
    
    dummy_ov_input = np.random.rand(1, 7).astype(np.float32)
    res = compiled_model([dummy_ov_input])
    
    # Getting output
    # hummingbird typically outputs a tuple of (predictions, probabilities)
    # The first output is usually the class predictions (numeric)
    
    print(f"Dummy Input: {dummy_ov_input}")
    print(f"OpenVINO Outputs:")
    for key, value in res.items():
        print(f"  {key}: {value}")
        
    print("Inference successful! OpenVINO integration via Hummingbird is working.")

if __name__ == "__main__":
    main()
