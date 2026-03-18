import os
import openvino as ov

def main():
    onnx_model_path = "WaferDefectX/results/rf_model.onnx"
    ov_model_dir = "WaferDefectX/results/ov_model"
    
    if not os.path.exists(onnx_model_path):
        print(f"ONNX model file {onnx_model_path} not found. Please convert the model to ONNX first.")
        return
        
    print(f"Loading ONNX model from {onnx_model_path}...")
    
    # Convert ONNX model to OpenVINO model
    print("Converting model to OpenVINO IR format...")
    # modern openvino conversion
    # equivalent to: ov_model = ov.tools.mo.convert_model(onnx_model_path)
    # openvino.convert_model is preferred in newest versions
    ov_model = ov.convert_model(onnx_model_path)
    
    # Save the OpenVINO model
    if not os.path.exists(ov_model_dir):
        os.makedirs(ov_model_dir)
        
    ov.save_model(ov_model, os.path.join(ov_model_dir, "rf_model.xml"))
    print(f"OpenVINO model saved to {ov_model_dir}")
    
    # Verify inference
    print("\n--- Verifying OpenVINO Inference ---")
    core = ov.Core()
    compiled_model = core.compile_model(ov_model, "CPU")
    
    # Get the input layer
    input_layer = compiled_model.input(0)
    
    # Create dummy data matching the feature shape (num_features = 7)
    import numpy as np
    dummy_input = np.random.rand(1, 7).astype(np.float32)
    
    # Run inference
    res = compiled_model([dummy_input])
    
    # Getting output
    output_layer = compiled_model.output(0)
    pred_ov = res[output_layer][0]
    
    print(f"Dummy Input: {dummy_input}")
    print(f"OpenVINO Prediction Output: {pred_ov}")
    print("Inference successful! OpenVINO integration is working.")

if __name__ == "__main__":
    main()
