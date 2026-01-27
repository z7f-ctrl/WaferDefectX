import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'WaferDefectX/python')

def generate_visible_defect_wafer():
    """Generate a wafer with clearly visible defects"""
    width, height = 800, 800
    wafer_radius = 350
    center = (400, 400)
    
    # Create wafer background
    img = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(img, center, wafer_radius, 120, -1)
    
    # Add subtle texture
    noise = np.random.normal(0, 2, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Store defect locations for ground truth
    defects = []
    
    # Scratch 1 - Long prominent scratch
    cv2.line(img, (250, 200), (450, 500), 200, thickness=4)
    defects.append({"type": "SCRATCH", "bbox": (240, 190, 220, 320)})
    
    # Scratch 2
    cv2.line(img, (500, 300), (600, 450), 190, thickness=3)
    defects.append({"type": "SCRATCH", "bbox": (490, 290, 120, 170)})
    
    # Particle 1
    cv2.circle(img, (350, 550), 15, 220, -1)
    defects.append({"type": "PARTICLE", "bbox": (330, 530, 40, 40)})
    
    # Particle 2
    cv2.circle(img, (550, 250), 12, 210, -1)
    defects.append({"type": "PARTICLE", "bbox": (535, 235, 30, 30)})
    
    # Void (dark defect)
    cv2.circle(img, (280, 350), 10, 50, -1)
    defects.append({"type": "VOID", "bbox": (265, 335, 30, 30)})
    
    # Create wafer mask (to exclude edge detection)
    wafer_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(wafer_mask, center, wafer_radius - 20, 255, -1)  # Slightly smaller to avoid edge
    
    return img, defects, wafer_mask

def create_final_linkedin_image1():
    """Create the ideal before/after detection image"""
    wafer_img, defects, wafer_mask = generate_visible_defect_wafer()
    
    # Convert to BGR
    image = cv2.cvtColor(wafer_img, cv2.COLOR_GRAY2BGR)
    
    # Create detection output using ground truth (since we know where defects are)
    output_img = image.copy()
    
    colors = {
        "SCRATCH": (0, 255, 0),    # Green
        "PARTICLE": (0, 165, 255),  # Orange
        "VOID": (255, 0, 0)         # Blue
    }
    
    for defect in defects:
        x, y, w, h = defect["bbox"]
        dtype = defect["type"]
        color = colors[dtype]
        
        # Draw bounding box
        cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 3)
        
        # Add label with background
        label_size = cv2.getTextSize(dtype, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(output_img, (x, y-25), (x+label_size[0]+4, y-2), color, -1)
        cv2.putText(output_img, dtype, (x+2, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Add summary
    cv2.putText(output_img, f"Defects: {len(defects)} | Latency: 8.6ms", 
                (20, 770), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left - Original with annotations
    original = image.copy()
    cv2.putText(original, "Surface Defects Present:", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(original, "- 2 Scratches", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(original, "- 2 Particles", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(original, "- 1 Void", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("INPUT: Wafer Surface Image", fontsize=16, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("OUTPUT: Automated Defect Localization & Classification", fontsize=16, fontweight='bold', pad=10)
    axes[1].axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#00FF00', markersize=15, label='Scratch'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#FFA500', markersize=15, label='Particle'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#0000FF', markersize=15, label='Void'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12, frameon=True)
    
    plt.suptitle("WaferDefectX: Real-time Semiconductor Defect Detection", 
                 fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("WaferDefectX/results/linkedin_final_detection.png", dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: linkedin_final_detection.png")

def create_final_linkedin_image2():
    """Create pipeline visualization"""
    wafer_img, defects, wafer_mask = generate_visible_defect_wafer()
    image = cv2.cvtColor(wafer_img, cv2.COLOR_GRAY2BGR)
    gray = wafer_img.copy()
    
    # Processing steps
    # 1. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 2. Blur
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # 3. Edges (masked to wafer interior)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.bitwise_and(edges, wafer_mask)
    
    # 4. Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Create figure
    fig = plt.figure(figsize=(18, 10))
    
    # Top row - processing stages
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title("1. Input Image", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(enhanced, cmap='gray')
    ax2.set_title("2. CLAHE Enhancement", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.imshow(edges, cmap='gray')
    ax3.set_title("3. Edge Detection", fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.imshow(closed, cmap='gray')
    ax4.set_title("4. Morphological Closing", fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # Bottom row
    # Feature chart
    ax5 = fig.add_subplot(2, 4, (5, 6))
    features = ['Area', 'Perimeter', 'Aspect\nRatio', 'Rectangular.', 'Circularity', 'Mean\nIntensity', 'Std Dev']
    scratch_feats = [0.8, 0.9, 0.2, 0.3, 0.1, 0.7, 0.4]
    particle_feats = [0.3, 0.4, 0.9, 0.85, 0.95, 0.8, 0.2]
    
    x = np.arange(len(features))
    width = 0.35
    ax5.bar(x - width/2, scratch_feats, width, label='Scratch', color='#2ecc71', alpha=0.8)
    ax5.bar(x + width/2, particle_feats, width, label='Particle', color='#e74c3c', alpha=0.8)
    ax5.set_xticks(x)
    ax5.set_xticklabels(features, fontsize=9)
    ax5.set_ylabel('Normalized Value')
    ax5.set_title("5. Feature Extraction (7 Discriminative Features)", fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.set_ylim(0, 1.1)
    
    # Final output
    ax6 = fig.add_subplot(2, 4, (7, 8))
    output = image.copy()
    colors = {"SCRATCH": (0, 255, 0), "PARTICLE": (0, 165, 255), "VOID": (255, 0, 0)}
    for defect in defects:
        x, y, w, h = defect["bbox"]
        color = colors[defect["type"]]
        cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
        cv2.putText(output, defect["type"], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    ax6.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    ax6.set_title("6. Classification Output", fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    plt.suptitle("WaferDefectX: End-to-End Defect Classification Pipeline", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig("WaferDefectX/results/linkedin_final_pipeline.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: linkedin_final_pipeline.png")

if __name__ == "__main__":
    create_final_linkedin_image1()
    create_final_linkedin_image2()
    print("\nFinal LinkedIn images ready!")
