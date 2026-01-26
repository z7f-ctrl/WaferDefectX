import cv2
import numpy as np

class DefectLocalizer:
    def __init__(self):
        pass

    def detect_edges(self, image, low_threshold=50, high_threshold=150):
        """Canny Edge Detection."""
        return cv2.Canny(image, low_threshold, high_threshold)

    def apply_morphology(self, binary_map, kernel_size=(5, 5)):
        """Closing operation to connect broken edge segments."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        # Dilate then Erode (Close) to fill gaps
        closed = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
        return closed

    def find_defect_contours(self, binary_map, min_area=10):
        """Find contours and filter by minimum area."""
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        bounding_boxes = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                valid_contours.append(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                bounding_boxes.append((x, y, w, h))
                
        return valid_contours, bounding_boxes

    def create_mask(self, shape, contours):
        """Create a binary mask from contours."""
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
        return mask

    def localize(self, preprocessed_image):
        """
        Full localization pipeline. 
        Expects a preprocessed grayscale image (e.g. from CLAHE).
        """
        # 1. Edge Detection
        edges = self.detect_edges(preprocessed_image)
        
        # 2. Morphology to clean up
        clean_map = self.apply_morphology(edges)
        
        # 3. Find Contours
        contours, bboxes = self.find_defect_contours(clean_map)
        
        # 4. Generate Mask
        mask = self.create_mask(preprocessed_image.shape, contours)
        
        return {
            "edges": edges,
            "clean_map": clean_map,
            "contours": contours,
            "bboxes": bboxes,
            "mask": mask
        }
