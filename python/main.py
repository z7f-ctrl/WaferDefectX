import glob
import os

import cv2
import numpy as np
from defect_localization import DefectLocalizer
from paths import DATA_SYNTHETIC, RESULTS_DIR, ensure_dir
from preprocessing import Preprocessor


def main():
    data_dir = DATA_SYNTHETIC
    results_dir = ensure_dir(RESULTS_DIR)

    preprocessor = Preprocessor()
    localizer = DefectLocalizer()

    image_paths = glob.glob(os.path.join(str(data_dir), "*.png"))[:5]

    if not image_paths:
        print(f"No images found in {data_dir}. Run data_generator.py first.")
        return

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"Processing {filename}...")

        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load {img_path}")
            continue

        enhanced_image, _thresholded = preprocessor.process_pipeline(image)
        loc_results = localizer.localize(enhanced_image)

        output_img = image.copy()
        for x, y, w, h in loc_results["bboxes"]:
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        h, w = image.shape[:2]

        res_enhanced = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
        res_edges = cv2.cvtColor(loc_results["edges"], cv2.COLOR_GRAY2BGR)
        res_mask = cv2.cvtColor(loc_results["mask"], cv2.COLOR_GRAY2BGR)

        cv2.putText(image, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(
            res_enhanced, "Enhanced (CLAHE)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.putText(res_edges, "Canny Edges", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(res_mask, "Defect Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(output_img, "Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        scale = 0.5
        small_size = (int(w * scale), int(h * scale))
        imgs = [image, res_enhanced, res_edges, res_mask, output_img]
        imgs_resized = [cv2.resize(img, small_size) for img in imgs]
        collage = np.hstack(imgs_resized)

        save_path = results_dir / f"result_{filename}"
        cv2.imwrite(str(save_path), collage)
        print(f"Saved result to {save_path}")


if __name__ == "__main__":
    main()
