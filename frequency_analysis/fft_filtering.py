"""
FFT-based Filtering for Wafer Noise Suppression

This module implements frequency-domain filtering techniques for 
removing periodic noise patterns common in semiconductor wafer images.
"""

import cv2
import numpy as np


class FFTFilter:
    """Frequency-domain filtering for wafer image denoising."""
    
    def __init__(self):
        pass
    
    def compute_fft(self, image):
        """Compute 2D FFT and shift zero frequency to center."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply windowing to reduce spectral leakage
        rows, cols = image.shape
        window = np.outer(np.hanning(rows), np.hanning(cols))
        windowed = image * window
        
        # Compute FFT
        f_transform = np.fft.fft2(windowed.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        
        return f_shift
    
    def compute_magnitude_spectrum(self, f_shift):
        """Compute log magnitude spectrum for visualization."""
        magnitude = np.abs(f_shift)
        # Log scale for better visualization
        spectrum = 20 * np.log10(magnitude + 1)
        return spectrum
    
    def create_lowpass_filter(self, shape, cutoff_freq=30):
        """Create Gaussian low-pass filter."""
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        
        # Create meshgrid for distance calculation
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        U, V = np.meshgrid(v, u)
        
        # Gaussian low-pass filter
        D = np.sqrt(U**2 + V**2)
        H = np.exp(-(D**2) / (2 * (cutoff_freq**2)))
        
        return H
    
    def create_highpass_filter(self, shape, cutoff_freq=30):
        """Create Gaussian high-pass filter for edge enhancement."""
        return 1 - self.create_lowpass_filter(shape, cutoff_freq)
    
    def create_bandstop_filter(self, shape, freq_center, bandwidth=10):
        """Create band-stop (notch) filter to remove periodic noise."""
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        U, V = np.meshgrid(v, u)
        D = np.sqrt(U**2 + V**2)
        
        # Band-stop filter
        H = 1 - np.exp(-((D**2 - freq_center**2)**2) / (D**2 * bandwidth**2 + 1e-6))
        
        return H
    
    def apply_filter(self, image, filter_mask):
        """Apply frequency-domain filter and reconstruct image."""
        f_shift = self.compute_fft(image)
        
        # Apply filter in frequency domain
        filtered_shift = f_shift * filter_mask
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(filtered_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalize to 8-bit
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        
        return img_back.astype(np.uint8)
    
    def denoise_wafer(self, image, cutoff=50):
        """
        Apply low-pass filtering for wafer noise suppression.
        
        This removes high-frequency noise while preserving 
        defect structures that typically have lower spatial frequencies.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create low-pass filter
        lpf = self.create_lowpass_filter(gray.shape, cutoff)
        
        # Apply filter
        denoised = self.apply_filter(gray, lpf)
        
        return denoised
    
    def enhance_defects(self, image, low_cutoff=5, high_cutoff=80):
        """
        Apply band-pass filtering to enhance defect visibility.
        
        Removes both low-frequency background variations and 
        high-frequency noise, isolating defect-relevant frequencies.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create band-pass filter (low-pass - very-low-pass)
        lpf_high = self.create_lowpass_filter(gray.shape, high_cutoff)
        lpf_low = self.create_lowpass_filter(gray.shape, low_cutoff)
        bandpass = lpf_high - lpf_low
        
        enhanced = self.apply_filter(gray, bandpass)
        
        return enhanced


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '../python')
    
    # Test with synthetic wafer
    img = cv2.imread('../data/synthetic/wafer_0033_scratch.png', cv2.IMREAD_GRAYSCALE)
    
    if img is not None:
        fft_filter = FFTFilter()
        
        # Denoise
        denoised = fft_filter.denoise_wafer(img, cutoff=40)
        
        # Enhance defects
        enhanced = fft_filter.enhance_defects(img, low_cutoff=5, high_cutoff=60)
        
        cv2.imwrite('../results/fft_denoised.png', denoised)
        cv2.imwrite('../results/fft_enhanced.png', enhanced)
        print("FFT filtering complete. Results saved.")
    else:
        print("Could not load test image.")
