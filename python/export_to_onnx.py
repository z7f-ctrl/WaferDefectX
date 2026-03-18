import os
import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

def main():
    model_path = "WaferDefectX/results/rf_model.pkl"
    onnx_model_path = "WaferDefectX/results/rf_model.onnx"
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
        
    print(f"Loading scikit-learn model from {model_path}...")
    model = joblib.load(model_path)
    
    # The feature extractor currently produces 7 features:
    # [area, perimeter, aspect_ratio, rectangularity, circularity, mean_intensity, std_intensity]
    num_features = 7
    initial_type = [('float_input', FloatTensorType([None, num_features]))]
    
    print("Converting model to ONNX format...")
    options = {id(model): {'zipmap': False}}
    onx = convert_sklearn(model, initial_types=initial_type, target_opset=12, options=options)
    
    with open(onnx_model_path, "wb") as f:
        f.write(onx.SerializeToString())
        
    print(f"ONNX model saved to {onnx_model_path}")
    
    # Verify the ONNX model can be loaded and run via onnxruntime
    print("\n--- Verifying ONNX Inference ---")
    sess = rt.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    
    # Create dummy data matching the feature shape
    dummy_input = np.random.rand(1, num_features).astype(np.float32)
    
    # Run inference
    pred_onnx = sess.run([label_name], {input_name: dummy_input})[0]
    
    print(f"Dummy Input: {dummy_input}")
    print(f"ONNX Prediction Output: {pred_onnx}")
    print("Inference successful! ONNX runtime integration is working.")

if __name__ == "__main__":
    main()
