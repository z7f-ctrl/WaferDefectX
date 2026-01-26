import cv2
import numpy as np

class Preprocessor:
    def __init__(self, blur_ksize=(5, 5), median_ksize=5):
        self.blur_ksize = blur_ksize
        self.median_ksize = median_ksize
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def apply_gaussian_blur(self, image):
        """Reduces high-frequency noise."""
        return cv2.GaussianBlur(image, self.blur_ksize, 0)

    def apply_median_blur(self, image):
        """Effective for salt-and-pepper noise."""
        return cv2.medianBlur(image, self.median_ksize)

    def apply_clahe(self, image):
        """Contrast Limited Adaptive Histogram Equalization."""
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(image)

    def apply_adaptive_threshold(self, image):
        """Adaptive Gaussian Thresholding to handle varying lighting."""
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Inverted because defects (scratches/particles) are often brighter/different than background
        # or we might want to capture edges.
        # Adjusted blockSize and C for generic surface defect detection
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        return thresh

    def process_pipeline(self, image):
        """Full preprocessing pipeline."""
        # 1. Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 2. Denoise
        blurred = self.apply_gaussian_blur(gray)
        median = self.apply_median_blur(blurred)
        
        # 3. Enhance Contrast
        enhanced = self.apply_clahe(median)
        
        # 4. Threshold (Optional, can be used by localization or returned separately)
        thresholded = self.apply_adaptive_threshold(enhanced)
        
        return enhanced, thresholded
