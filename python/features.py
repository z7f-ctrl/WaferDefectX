import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self):
        pass

    def extract_geometric_features(self, contour):
        """
        Extracts geometric properties:
        - Area
        - Perimeter
        - Aspect Ratio
        - Rectangularity
        - Circularity
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            circularity = 0
        else:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        rectangularity = area / (w * h) if w * h > 0 else 0
        
        return [area, perimeter, aspect_ratio, rectangularity, circularity]

    def extract_texture_features(self, image, mask):
        """
        Extracts simple texture statistics from the region defined by mask.
        - Mean Intensity
        - Standard Deviation of Intensity
        """
        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
        mean_val, std_dev = cv2.meanStdDev(image, mask=mask)
        
        return [mean_val[0][0], std_dev[0][0]]

    def extract_features(self, image, contour):
        """
        Extracts all features for a single defect candidate.
        Expects 'image' to be the original or grayscale value image.
        """
        # Create a local mask for this contour to extract texture
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        geo_feats = self.extract_geometric_features(contour)
        tex_feats = self.extract_texture_features(image, mask)
        
        # Concatenate features
        return np.array(geo_feats + tex_feats)
