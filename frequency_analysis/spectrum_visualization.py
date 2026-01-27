"""
Spectrum Visualization for Wafer Image Analysis

Provides visualization tools for understanding frequency-domain 
characteristics of wafer images and defect patterns.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from fft_filtering import FFTFilter


class SpectrumVisualizer:
    """Visualize frequency spectrum of wafer images."""
    
    def __init__(self):
        self.fft_filter = FFTFilter()
    
    def plot_spectrum(self, image, title="Frequency Spectrum", save_path=None):
        """Plot magnitude spectrum of an image."""
        f_shift = self.fft_filter.compute_fft(image)
        spectrum = self.fft_filter.compute_magnitude_spectrum(f_shift)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Original image
        if len(image.shape) == 3:
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(image, cmap='gray')
        axes[0].set_title("Spatial Domain", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Frequency spectrum
        im = axes[1].imshow(spectrum, cmap='hot')
        axes[1].set_title("Frequency Spectrum (Log Magnitude)", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved: {save_path}")
        
        plt.close()
        return spectrum
    
    def plot_filter_comparison(self, image, save_path=None):
        """Compare original, denoised, and enhanced images."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply filters
        denoised = self.fft_filter.denoise_wafer(gray, cutoff=40)
        enhanced = self.fft_filter.enhance_defects(gray, low_cutoff=5, high_cutoff=60)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Images
        axes[0, 0].imshow(gray, cmap='gray')
        axes[0, 0].set_title("Original", fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(denoised, cmap='gray')
        axes[0, 1].set_title("Low-Pass Filtered (Denoised)", fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(enhanced, cmap='gray')
        axes[0, 2].set_title("Band-Pass Filtered (Defect Enhanced)", fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Spectra
        for idx, (img, label) in enumerate([(gray, "Original"), 
                                             (denoised, "Denoised"), 
                                             (enhanced, "Enhanced")]):
            f_shift = self.fft_filter.compute_fft(img)
            spectrum = self.fft_filter.compute_magnitude_spectrum(f_shift)
            axes[1, idx].imshow(spectrum, cmap='hot')
            axes[1, idx].set_title(f"{label} Spectrum", fontsize=12)
            axes[1, idx].axis('off')
        
        plt.suptitle("Frequency-Domain Filtering for Wafer Noise Suppression", 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_filter_response(self, shape=(256, 256), save_path=None):
        """Visualize filter frequency responses."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Low-pass filter
        lpf = self.fft_filter.create_lowpass_filter(shape, cutoff_freq=30)
        axes[0].imshow(lpf, cmap='gray')
        axes[0].set_title("Low-Pass Filter\n(Noise Suppression)", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # High-pass filter
        hpf = self.fft_filter.create_highpass_filter(shape, cutoff_freq=30)
        axes[1].imshow(hpf, cmap='gray')
        axes[1].set_title("High-Pass Filter\n(Edge Enhancement)", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Band-pass filter
        lpf_high = self.fft_filter.create_lowpass_filter(shape, cutoff_freq=60)
        lpf_low = self.fft_filter.create_lowpass_filter(shape, cutoff_freq=10)
        bpf = lpf_high - lpf_low
        axes[2].imshow(bpf, cmap='gray')
        axes[2].set_title("Band-Pass Filter\n(Defect Isolation)", fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle("Frequency-Domain Filter Responses", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved: {save_path}")
        
        plt.close()


if __name__ == "__main__":
    # Generate visualization for wafer image
    img = cv2.imread('../data/synthetic/wafer_0033_scratch.png', cv2.IMREAD_GRAYSCALE)
    
    if img is not None:
        viz = SpectrumVisualizer()
        
        # Generate visualizations
        viz.plot_spectrum(img, title="Wafer Image Frequency Analysis",
                         save_path='../results/spectrum_analysis.png')
        
        viz.plot_filter_comparison(img, save_path='../results/fft_filter_comparison.png')
        
        viz.plot_filter_response(save_path='../results/filter_responses.png')
        
        print("Spectrum visualizations complete!")
    else:
        print("Could not load test image.")
