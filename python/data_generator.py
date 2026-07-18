import cv2
import numpy as np
import os
import random

class WaferDataGenerator:
    def __init__(self, output_dir="data/synthetic", width=800, height=800):
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.wafer_radius = int(min(width, height) * 0.45)
        self.center = (width // 2, height // 2)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_wafer_background(self):
        # Create black background
        img = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Draw wafer disk (gray)
        cv2.circle(img, self.center, self.wafer_radius, (100, 100, 100), -1)
        
        # Add some grain/noise
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        return img

    def add_scratch(self, img):
        # Random start and end points within the wafer
        angle = random.uniform(0, 2 * np.pi)
        r = random.uniform(0, self.wafer_radius * 0.8)
        start_x = int(self.center[0] + r * np.cos(angle))
        start_y = int(self.center[1] + r * np.sin(angle))
        
        length = random.uniform(20, 100)
        angle_scratch = random.uniform(0, 2 * np.pi)
        end_x = int(start_x + length * np.cos(angle_scratch))
        end_y = int(start_y + length * np.sin(angle_scratch))
        
        # Draw scratch (brighter than background)
        cv2.line(img, (start_x, start_y), (end_x, end_y), (200, 200, 200), thickness=random.randint(1, 3))
        return img, "scratch"

    def add_particle(self, img):
        # Random position
        angle = random.uniform(0, 2 * np.pi)
        r = random.uniform(0, self.wafer_radius * 0.8)
        cx = int(self.center[0] + r * np.cos(angle))
        cy = int(self.center[1] + r * np.sin(angle))
        
        # Draw particle (bright spot)
        radius = random.randint(2, 6)
        cv2.circle(img, (cx, cy), radius, (220, 220, 220), -1)
        return img, "particle"

    def generate_dataset(self, num_images=20):
        print(f"Generating {num_images} synthetic images in {self.output_dir}...")
        for i in range(num_images):
            img = self.generate_wafer_background()
            
            defect_type = "good"
            if random.random() < 0.7: # 70% chance of defect
                if random.random() < 0.5:
                    img, defect_type = self.add_scratch(img)
                else:
                    img, defect_type = self.add_particle(img)
            
            filename = f"wafer_{i:04d}_{defect_type}.png"
            cv2.imwrite(os.path.join(self.output_dir, filename), img)
        print("Done.")

if __name__ == "__main__":
    from paths import DATA_SYNTHETIC, ensure_dir

    output_dir = str(ensure_dir(DATA_SYNTHETIC))
    generator = WaferDataGenerator(output_dir=output_dir)
    generator.generate_dataset(num_images=50)
