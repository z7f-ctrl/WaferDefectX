import cv2
import os
import glob
import numpy as np
from preprocessing import Preprocessor
from defect_localization import DefectLocalizer

def main():
    data_dir = "WaferDefectX/data/synthetic"
    results_dir = "WaferDefectX/results"
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    # Initialize modules
    preprocessor = Preprocessor()
    localizer = DefectLocalizer()
    
    # Process a few sample images
    image_paths = glob.glob(os.path.join(data_dir, "*.png"))[:5]
    
    for i, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        print(f"Processing {filename}...")
        
        # Load Image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load {img_path}")
            continue
            
        # 1. Preprocessing
        enhanced_image, thresholded_image = preprocessor.process_pipeline(image)
        
        # 2. Localization
        # Pass the enhanced image (gray) to localizer
        loc_results = localizer.localize(enhanced_image)
        
        # 3. Visualize and Save
        # Draw Bounding Boxes on original image
        output_img = image.copy()
        for (x, y, w, h) in loc_results['bboxes']:
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Create a collage: Original | Enhanced | Edges | Mask | Final Result
        # Convert all to 3-channel for concatenation
        h, w = image.shape[:2]
        
        res_enhanced = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
        res_edges = cv2.cvtColor(loc_results['edges'], cv2.COLOR_GRAY2BGR)
        res_mask = cv2.cvtColor(loc_results['mask'], cv2.COLOR_GRAY2BGR)
        
        # Add labels
        cv2.putText(image, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(res_enhanced, "Enhanced (CLAHE)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(res_edges, "Canny Edges", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(res_mask, "Defect Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(output_img, "Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Resize for collage if needed
        scale = 0.5
        small_size = (int(w*scale), int(h*scale))
        
        imgs = [image, res_enhanced, res_edges, res_mask, output_img]
        imgs_resized = [cv2.resize(img, small_size) for img in imgs]
        
        collage = np.hstack(imgs_resized)
        
        save_path = os.path.join(results_dir, f"result_{filename}")
        cv2.imwrite(save_path, collage)
        print(f"Saved result to {save_path}")

if __name__ == "__main__":
    main()
