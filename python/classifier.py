import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

class DefectClassifier:
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        if model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.model = SVC(kernel='rbf', probability=True, random_state=42)
        else:
            raise ValueError("Unknown model type. Choose 'rf' or 'svm'")
            
    def train(self, X, y):
        print(f"Training {self.model_type.upper()} classifier...")
        self.model.fit(X, y)
        
    def predict(self, X):
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
