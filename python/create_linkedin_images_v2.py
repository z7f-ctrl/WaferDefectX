import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, 'WaferDefectX/python')

from preprocessing import Preprocessor
from defect_localization import DefectLocalizer
from features import FeatureExtractor

def generate_visible_defect_wafer():
    """Generate a wafer with clearly visible defects for demonstration"""
    width, height = 800, 800
    wafer_radius = 350
    center = (400, 400)
    
    # Create wafer background (cleaner, less noise)
    img = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(img, center, wafer_radius, 120, -1)  # Gray wafer
    
    # Add subtle texture (less aggressive noise)
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add VISIBLE defects
    defect_info = []
    
    # Scratch 1 - Long prominent scratch
    cv2.line(img, (250, 200), (450, 500), 200, thickness=4)
    defect_info.append(("Scratch", (220, 180, 260, 340)))
    
    # Scratch 2 - Another scratch
    cv2.line(img, (500, 300), (600, 450), 190, thickness=3)
    defect_info.append(("Scratch", (490, 290, 120, 170)))
    
    # Particle 1 - Large bright spot
    cv2.circle(img, (350, 550), 15, 220, -1)
    defect_info.append(("Particle", (330, 530, 40, 40)))
    
    # Particle 2 - Another particle
    cv2.circle(img, (550, 250), 12, 210, -1)
    defect_info.append(("Particle", (535, 235, 30, 30)))
    
    # Dark defect (void)
    cv2.circle(img, (280, 350), 10, 50, -1)
    defect_info.append(("Void", (265, 335, 30, 30)))
    
    return img, defect_info

def create_linkedin_image1_improved():
    """Create before/after defect detection with VISIBLE defects"""
    # Generate wafer with visible defects
    wafer_img, defect_info = generate_visible_defect_wafer()
    
    # Convert to BGR for consistency
    image = cv2.cvtColor(wafer_img, cv2.COLOR_GRAY2BGR)
    
    preprocessor = Preprocessor()
    localizer = DefectLocalizer()
    
    # Process
    enhanced, _ = preprocessor.process_pipeline(image)
    loc_results = localizer.localize(enhanced)
    
    # Create detection output with clear annotations
    output_img = image.copy()
    
    # Draw all detected contours with thick green boxes
    for i, (x, y, w, h) in enumerate(loc_results['bboxes']):
        # Color code by size
        if w * h > 500:
            color = (0, 0, 255)  # Red for large defects
            label = "MAJOR"
        else:
            color = (0, 255, 255)  # Yellow for small
            label = "MINOR"
        
        cv2.rectangle(output_img, (x-5, y-5), (x+w+5, y+h+5), color, 3)
        cv2.putText(output_img, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Add defect count
    cv2.putText(output_img, f"Defects Found: {len(loc_results['bboxes'])}", 
                (20, 750), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original with arrows pointing to defects
    original_annotated = image.copy()
    cv2.putText(original_annotated, "Scratches", (180, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.arrowedLine(original_annotated, (270, 185), (300, 250), (255, 255, 255), 2)
    cv2.putText(original_annotated, "Particles", (360, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.arrowedLine(original_annotated, (370, 600), (355, 565), (255, 255, 255), 2)
    
    axes[0].imshow(cv2.cvtColor(original_annotated, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input: Wafer with Surface Defects", fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Output: Automated Defect Localization", fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle("WaferDefectX: Real-time Defect Detection System", fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("WaferDefectX/results/linkedin_image1_detection_v2.png", dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: linkedin_image1_detection_v2.png")
    
    # Also save just the detection result
    cv2.imwrite("WaferDefectX/results/demo_wafer_original.png", image)
    cv2.imwrite("WaferDefectX/results/demo_wafer_detected.png", output_img)

def create_linkedin_image2_improved():
    """Create feature extraction pipeline with visible defects"""
    wafer_img, _ = generate_visible_defect_wafer()
    image = cv2.cvtColor(wafer_img, cv2.COLOR_GRAY2BGR)
    
    preprocessor = Preprocessor()
    localizer = DefectLocalizer()
    extractor = FeatureExtractor()
    
    # Process
    enhanced, thresh = preprocessor.process_pipeline(image)
    loc_results = localizer.localize(enhanced)
    
    # Create multi-stage visualization
    fig = plt.figure(figsize=(16, 10))
    
    # Use GridSpec for better layout
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.2)
    
    # Row 1: Pipeline stages
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title("1. Input Wafer", fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(enhanced, cmap='gray')
    ax2.set_title("2. CLAHE Enhanced", fontsize=11, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(loc_results['edges'], cmap='gray')
    ax3.set_title("3. Canny Edges", fontsize=11, fontweight='bold')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(loc_results['mask'], cmap='gray')
    ax4.set_title("4. Defect Mask", fontsize=11, fontweight='bold')
    ax4.axis('off')
    
    # Row 2: Features and output
    ax5 = fig.add_subplot(gs[1, 0:2])
    if loc_results['contours']:
        # Get features for multiple defects
        all_feats = []
        for cnt in loc_results['contours'][:3]:  # Top 3 defects
            feats = extractor.extract_features(enhanced, cnt)
            all_feats.append(feats)
        
        feature_names = ['Area', 'Perimeter', 'Aspect\nRatio', 'Rect.', 'Circular.', 'Mean\nInt.', 'Std\nDev']
        x = np.arange(len(feature_names))
        width = 0.25
        
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        for i, feats in enumerate(all_feats):
            normalized = feats / (np.max(feats) + 1e-6)
            ax5.bar(x + i*width, normalized, width, label=f'Defect {i+1}', color=colors[i], alpha=0.8)
        
        ax5.set_xticks(x + width)
        ax5.set_xticklabels(feature_names, fontsize=9)
        ax5.set_ylabel('Normalized Value', fontsize=10)
        ax5.set_title("5. Feature Extraction (7 Features per Defect)", fontsize=11, fontweight='bold')
        ax5.legend(loc='upper right', fontsize=9)
        ax5.set_ylim(0, 1.3)
    
    ax6 = fig.add_subplot(gs[1, 2:4])
    output_img = image.copy()
    defect_types = ['SCRATCH', 'SCRATCH', 'PARTICLE', 'PARTICLE', 'VOID']
    for i, (x, y, w, h) in enumerate(loc_results['bboxes'][:5]):
        color = (0, 255, 0) if i < 2 else (0, 165, 255) if i < 4 else (255, 0, 0)
        cv2.rectangle(output_img, (x-3, y-3), (x+w+3, y+h+3), color, 2)
        if i < len(defect_types):
            cv2.putText(output_img, defect_types[i], (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    ax6.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    ax6.set_title("6. Classification: Multi-class Defect Typing", fontsize=11, fontweight='bold')
    ax6.axis('off')
    
    plt.suptitle("WaferDefectX: Feature-based Defect Classification Pipeline", fontsize=16, fontweight='bold')
    plt.savefig("WaferDefectX/results/linkedin_image2_pipeline_v2.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: linkedin_image2_pipeline_v2.png")

if __name__ == "__main__":
    create_linkedin_image1_improved()
    create_linkedin_image2_improved()
    print("\nImproved LinkedIn images created!")
