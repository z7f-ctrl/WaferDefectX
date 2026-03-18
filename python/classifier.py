import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

class DefectClassifier:
    def __init__(self, model_type='rf', onnx_path=None):
        self.model_type = model_type
        if model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.model = SVC(kernel='rbf', probability=True, random_state=42)
        elif model_type == 'onnx':
            import onnxruntime as rt
            if onnx_path is None:
                raise ValueError("onnx_path must be provided for onnx model type")
            self.model = rt.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            self.input_name = self.model.get_inputs()[0].name
            self.label_name = self.model.get_outputs()[0].name
        elif model_type == 'openvino':
            import openvino as ov
            if onnx_path is None:
                raise ValueError("openvino requires onnx_path (path to .xml or .onnx) to load model")
            core = ov.Core()
            ov_model = core.read_model(onnx_path)
            self.model = core.compile_model(ov_model, "CPU")
            self.output_layer = self.model.output(0)
            self.classes_ = ['good', 'particle', 'scratch']
        else:
            raise ValueError("Unknown model type. Choose 'rf', 'svm', 'onnx', or 'openvino'")
            
    def train(self, X, y):
        print(f"Training {self.model_type.upper()} classifier...")
        self.model.fit(X, y)
        
    def predict(self, X):
        if self.model_type == 'onnx':
            import numpy as np
            X = np.array(X, dtype=np.float32)
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            pred = self.model.run([self.label_name], {self.input_name: X})[0]
            return pred
        elif self.model_type == 'openvino':
            import numpy as np
            X = np.array(X, dtype=np.float32)
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            res = self.model([X])
            pred_idx = res[self.output_layer][0]
            if isinstance(pred_idx, np.ndarray):
                pred_idx = pred_idx.item()
            return self.classes_[pred_idx]
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm
        }

    def save_model(self, path):
        joblib.dump(self.model, path)
        
    def load_model(self, path):
        self.model = joblib.load(path)
