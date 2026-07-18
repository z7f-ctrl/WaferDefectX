"""
FFT-based Filtering for Wafer Noise Suppression

This module implements frequency-domain filtering techniques for 
removing periodic noise patterns common in semiconductor wafer images.
Optimized using OpenCV's DFT implementation.
"""

import cv2
import numpy as np


class FFTFilter:
    """Frequency-domain filtering for wafer image denoising using OpenCV."""
    
    def __init__(self):
        pass
    
    def compute_fft(self, image):
        """Compute 2D DFT using OpenCV optimization."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Optimize size for DFT efficiency
        rows, cols = image.shape
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        
        # Pad right and bottom
        padded = cv2.copyMakeBorder(image, 0, nrows - rows, 0, ncols - cols, 
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # Apply windowing to reduce spectral leakage (Hanning)
        # Note: Windowing slightly alters signal, used here for visualization stability
        # For strict reconstruction, might skip or use careful overlap-add
        pad_rows, pad_cols = padded.shape
        # Create 2D Hanning window manually or use minimal windowing
        # window = np.outer(np.hanning(pad_rows), np.hanning(pad_cols))
        # padded = padded * window # skipping window for robust filtering reconstruction
        
        # Compute DFT (Output is dual channel: Real + Imaginary)
        planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
        complex_img = cv2.merge(planes)
        
        dft = cv2.dft(complex_img, flags=cv2.DFT_COMPLEX_OUTPUT)
        
        # Shift zero frequency to center
        dft_shift = np.fft.fftshift(dft)
        
        return dft_shift, (rows, cols)
    
    def compute_magnitude_spectrum(self, dft_shift):
        """Compute log magnitude spectrum for visualization."""
        # Split channels
        planes = cv2.split(dft_shift)
        magnitude = cv2.magnitude(planes[0], planes[1])
        
        # Log scale
        spectrum = 20 * np.log(magnitude + 1)
        
        # Normalize for visualization
        spectrum_norm = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX)
        return spectrum_norm.astype(np.uint8)
    
    def create_lowpass_filter(self, shape, cutoff_freq=30):
        """Create Gaussian low-pass filter mask (2-channel for OpenCV DFT)."""
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        
        # Meshgrid
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        U, V = np.meshgrid(v, u)
        
        D = np.sqrt(U**2 + V**2)
        H = np.exp(-(D**2) / (2 * (cutoff_freq**2)))
        
        # Duplicate for 2 channels (Real, Imag)
        H_2ch = np.dstack([H, H])
        return H_2ch
    
    def create_bandstop_filter(self, shape, freq_center, bandwidth=10):
        """Create band-stop (notch) filter."""
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        U, V = np.meshgrid(v, u)
        D = np.sqrt(U**2 + V**2)
        
        H = 1 - np.exp(-((D**2 - freq_center**2)**2) / (D**2 * bandwidth**2 + 1e-6))
        
        return np.dstack([H, H])
    
    def apply_filter(self, image, filter_mask):
        """Apply frequency-domain filter and reconstruct image."""
        dft_shift, (orig_rows, orig_cols) = self.compute_fft(image)
        
        # Resize filter to match padded DFT size
        dft_rows, dft_cols = dft_shift.shape[:2]
        filter_rows, filter_cols = filter_mask.shape[:2]
        
        # If sizes differ (due to optimization padding), resize mask
        if (dft_rows, dft_cols) != (filter_rows, filter_cols):
             # Resize each channel
             m0 = cv2.resize(filter_mask[:,:,0], (dft_cols, dft_rows))
             m1 = cv2.resize(filter_mask[:,:,1], (dft_cols, dft_rows))
             filter_mask = cv2.merge([m0, m1])

        # Apply filter
        fshift = dft_shift * filter_mask
        
        # Inverse Shift
        f_ishift = np.fft.ifftshift(fshift)
        
        # Inverse DFT
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        
        # Crop back to original size
        img_back = img_back[0:orig_rows, 0:orig_cols]
        
        # Normalize
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        return img_back.astype(np.uint8)
    
    def denoise_wafer(self, image, cutoff=50):
        """Low-pass filtering using OpenCV DFT."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Get padded size for filter creation needed? 
        # Actually create_lowpass_filter typically takes image shape. 
        # But we need optimization. Let's do it inside apply or separate.
        # For simplicity, we calculate optimal size here too to match filter
        rows, cols = gray.shape
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        
        lpf = self.create_lowpass_filter((nrows, ncols), cutoff)
        return self.apply_filter(gray, lpf)
    
    def enhance_defects(self, image, low_cutoff=5, high_cutoff=80):
        """Band-pass filtering using OpenCV DFT."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        rows, cols = gray.shape
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        
        lpf_high = self.create_lowpass_filter((nrows, ncols), high_cutoff)
        lpf_low = self.create_lowpass_filter((nrows, ncols), low_cutoff)
        bandpass = lpf_high - lpf_low
        
        return self.apply_filter(gray, bandpass)

if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    img_path = root / "data" / "synthetic" / "wafer_0033_scratch.png"
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.append(str(root / "python"))

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    if img is not None:
        fft_filter = FFTFilter()
        denoised = fft_filter.denoise_wafer(img, cutoff=40)
        enhanced = fft_filter.enhance_defects(img, low_cutoff=5, high_cutoff=60)

        cv2.imwrite(str(out_dir / "fft_denoised_cv.png"), denoised)
        cv2.imwrite(str(out_dir / "fft_enhanced_cv.png"), enhanced)
        print("OpenCV FFT filtering complete.")
    else:
        print(f"Test image not found at {img_path}")
