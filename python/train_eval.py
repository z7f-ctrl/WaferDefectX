import glob
import os

import cv2
import numpy as np
from classifier import DefectClassifier
from defect_localization import DefectLocalizer
from features import FeatureExtractor
from paths import DATA_SYNTHETIC, RESULTS_DIR, ensure_dir
from preprocessing import Preprocessor


def extract_dataset_features(data_dir):
    preprocessor = Preprocessor()
    localizer = DefectLocalizer()
    extractor = FeatureExtractor()

    image_paths = glob.glob(os.path.join(str(data_dir), "*.png"))

    X = []
    y = []

    print(f"Extracting features from {len(image_paths)} images...")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        label = filename.split("_")[-1].split(".")[0]

        if label == "good":
            pass

        image = cv2.imread(img_path)
        if image is None:
            continue

        enhanced, _ = preprocessor.process_pipeline(image)
        loc_results = localizer.localize(enhanced)

        contours = loc_results["contours"]
        if not contours:
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        feats = extractor.extract_features(enhanced, largest_contour)

        X.append(feats)
        y.append(label)

    return np.array(X), np.array(y)


def main():
    data_dir = DATA_SYNTHETIC

    X, y = extract_dataset_features(data_dir)
    print(f"Extracted {len(X)} samples.")

    if len(X) == 0:
        print("No defects found to train on. Check localization or data.")
        return

    print(f"Classes: {np.unique(y)}")

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    clf = DefectClassifier(model_type="rf")
    clf.train(X_train, y_train)

    results = clf.evaluate(X_test, y_test)

    print("\n--- Classification Results ---")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("Classification Report:")
    print(results["report"])
    print("Confusion Matrix:")
    print(results["confusion_matrix"])

    ensure_dir(RESULTS_DIR)
    model_path = RESULTS_DIR / "rf_model.pkl"
    clf.save_model(model_path)
    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {model_path.with_name('rf_model.meta.json')}")


if __name__ == "__main__":
    main()
