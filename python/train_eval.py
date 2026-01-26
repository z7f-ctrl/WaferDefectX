import cv2
import os
import glob
import numpy as np
import pandas as pd
from preprocessing import Preprocessor
from defect_localization import DefectLocalizer
from features import FeatureExtractor
from classifier import DefectClassifier

def extract_dataset_features(data_dir):
    preprocessor = Preprocessor()
    localizer = DefectLocalizer()
    extractor = FeatureExtractor()
    
    image_paths = glob.glob(os.path.join(data_dir, "*.png"))
    
    X = []
    y = []
    
    print(f"Extracting features from {len(image_paths)} images...")
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        # Parse label from filename: wafer_0000_particle.png -> particle
        label = filename.split('_')[-1].split('.')[0]
        
        if label == "good":
            # For 'good' wafers, we might skip classification or treat them as negative class
            # But the localizer might find noise as defects. 
            # If localizer finds nothing, we skip.
            # If it finds something, it's a False Positive candidate or we label it 'noise'?
            # For this MVP, let's assume we are classifying detected defects.
            # If it is 'good', any detection is effectively a False Positive / background noise.
            # We can label "noise" or "background" if we want to train the classifier to reject checks.
            pass

        image = cv2.imread(img_path)
        if image is None: continue
        
        # Pipeline
        enhanced, _ = preprocessor.process_pipeline(image)
        loc_results = localizer.localize(enhanced)
        
        contours = loc_results['contours']
        
        if not contours:
            continue
            
        # For simplicity in this prototype, we take the largest contour as the "main defect"
        # in a real system we would classify each contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Extract features
        feats = extractor.extract_features(enhanced, largest_contour)
        
        X.append(feats)
        y.append(label)
        
    return np.array(X), np.array(y)

def main():
    data_dir = "WaferDefectX/data/synthetic"
    
    # 1. Feature Extraction
    X, y = extract_dataset_features(data_dir)
    print(f"Extracted {len(X)} samples.")
    
    if len(X) == 0:
        print("No defects found to train on. Check localization or data.")
        return

    # Encode labels if necessary, but sklearn handles strings fine usually
    print(f"Classes: {np.unique(y)}")
    
    # 2. Split Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 3. Train Classifier
    clf = DefectClassifier(model_type='rf')
    clf.train(X_train, y_train)
    
    # 4. Evaluate
    results = clf.evaluate(X_test, y_test)
    
    print("\n--- Classification Results ---")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("Classification Report:")
    print(results['report'])
    print("Confusion Matrix:")
    print(results['confusion_matrix'])
    
    # Save Model
    if not os.path.exists("WaferDefectX/results"):
        os.makedirs("WaferDefectX/results")
    clf.save_model("WaferDefectX/results/rf_model.pkl")
    print("Model saved to WaferDefectX/results/rf_model.pkl")

if __name__ == "__main__":
    main()
