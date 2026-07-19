"""Training and evaluation pipeline with per-contour extraction and metrics logging.

P1-05: Each contour is an independent sample (no longest-only).
P1-06: 'good' images with detected contours → negative class 'noise'.
P1-07: Accuracy + per-class P/R/F1 + confusion matrix saved to disk.
"""
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from classifier import DefectClassifier
from defect_localization import DefectLocalizer
from features import FeatureExtractor
from paths import DATA_SYNTHETIC, DATA_WM811K_IMAGES, RESULTS_DIR, ensure_dir
from preprocessing import Preprocessor

NEGATIVE_LABEL = "noise"
MIN_AREA = 10


def extract_dataset_features(data_dir, include_negative=True):
    """Extract features from ALL contours per image.

    Parameters
    ----------
    data_dir : Path
        Directory with wafer_*.png images.
    include_negative : bool
        If True, contours found on 'good' images are labeled NEGATIVE_LABEL.

    Returns
    -------
    X : np.ndarray (N, 7)
    y : np.ndarray (N,)
    metadata : list[dict]
        Per-sample metadata: {file, contour_id, area, label}.
    """
    preprocessor = Preprocessor()
    localizer = DefectLocalizer()
    extractor = FeatureExtractor()

    image_paths = sorted(data_dir.glob("*.png"))
    print(f"Extracting features from {len(image_paths)} images...")

    X, y, metadata = [], [], []

    for img_path in image_paths:
        filename = img_path.name
        label = filename.split("_")[-1].split(".")[0]

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [WARN] Could not read {filename}, skipping")
            continue

        enhanced, _ = preprocessor.process_pipeline(image)
        loc_results = localizer.localize(enhanced)
        contours = loc_results["contours"]

        if not contours:
            continue

        for cid, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < MIN_AREA:
                continue

            if label == "good":
                if not include_negative:
                    continue
                sample_label = NEGATIVE_LABEL
            else:
                sample_label = label

            feats = extractor.extract_features(enhanced, contour)
            X.append(feats)
            y.append(sample_label)
            metadata.append({
                "file": filename,
                "contour_id": cid,
                "area": float(area),
                "label": sample_label,
            })

    return np.array(X), np.array(y), metadata


def save_metrics(results, y_test, y_pred, output_dir):
    """Save classification metrics to JSON and CSV (P1-07)."""
    output_dir = ensure_dir(output_dir)

    # JSON report
    report_dict = {
        "accuracy": float(results["accuracy"]),
        "classification_report": results["report"],
        "confusion_matrix": results["confusion_matrix"].tolist(),
        "labels": sorted(set(y_test)),
    }
    json_path = output_dir / "eval_metrics.json"
    json_path.write_text(json.dumps(report_dict, indent=2))
    print(f"Metrics JSON saved to {json_path}")

    # CSV per-sample predictions
    csv_path = output_dir / "eval_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "predicted_label"])
        for true, pred in zip(y_test, y_pred):
            writer.writerow([true, pred])
    print(f"Predictions CSV saved to {csv_path}")

    return json_path


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_SYNTHETIC

    X, y, metadata = extract_dataset_features(data_dir, include_negative=True)
    print(f"Extracted {len(X)} samples from {len(set(m['file'] for m in metadata))} images.")
    print(f"Classes: {np.unique(y, return_counts=True)}")

    if len(X) == 0:
        print("No samples found. Check data or localization parameters.")
        return

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    clf = DefectClassifier(model_type="rf")
    clf.train(X_train, y_train)

    results = clf.evaluate(X_test, y_test)
    y_pred = clf.model.predict(X_test)

    print("\n--- Classification Results ---")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("Classification Report:")
    print(results["report"])
    print("Confusion Matrix:")
    print(results["confusion_matrix"])

    ensure_dir(RESULTS_DIR)
    model_path = RESULTS_DIR / "rf_model.pkl"
    clf.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    save_metrics(results, y_test, y_pred, RESULTS_DIR)


if __name__ == "__main__":
    main()
