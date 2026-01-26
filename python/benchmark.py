import time
import subprocess
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import Preprocessor
from defect_localization import DefectLocalizer
from tqdm import tqdm

def benchmark_python(image_paths):
    preprocessor = Preprocessor()
    localizer = DefectLocalizer()
    
    latencies = []
    
    for img_path in tqdm(image_paths, desc="Benchmarking Python"):
        image = cv2.imread(img_path)
        if image is None: continue
        
        start_time = time.time()
        
        # Pipeline
        enhanced, _ = preprocessor.process_pipeline(image)
        _ = localizer.localize(enhanced)
        
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000) # ms
        
    return latencies

def benchmark_cpp(image_paths, executable_path):
    latencies = []
    
    for img_path in tqdm(image_paths, desc="Benchmarking C++"):
        # Run C++ executable
        # The C++ code prints "Processing Time: X ms"
        result = subprocess.run([executable_path, img_path], capture_output=True, text=True)
        
        output = result.stdout
        for line in output.split('\n'):
            if "Processing Time:" in line:
                try:
                    time_ms = float(line.split(':')[1].strip().split(' ')[0])
                    latencies.append(time_ms)
                except:
                    pass
                    
    return latencies

def main():
    data_dir = "WaferDefectX/data/synthetic"
    cpp_exec = "./WaferDefectX/cpp/build/WaferDefectX_Run"
    
    image_paths = glob.glob(os.path.join(data_dir, "*.png"))
    
    if not image_paths:
        print("No images found.")
        return

    print("--- Running Benchmarks ---")
    
    py_times = benchmark_python(image_paths)

    if not os.path.exists(cpp_exec):
        print(f"C++ executable not found at {cpp_exec}. Skipping C++ benchmark.")
        print("Note: C++ compilation failed due to missing OpenCV dependencies in this environment.")
        
        avg_py = np.mean(py_times)
        print("\n--- Results ---")
        print(f"Python Average Latency: {avg_py:.2f} ms")
        return
    cpp_times = benchmark_cpp(image_paths, cpp_exec)
    
    avg_py = np.mean(py_times)
    avg_cpp = np.mean(cpp_times)
    
    print("\n--- Results ---")
    print(f"Python Average Latency: {avg_py:.2f} ms")
    print(f"C++ Average Latency:    {avg_cpp:.2f} ms")
    print(f"Speedup: {avg_py / avg_cpp:.2f}x")
    
    # Optional: Save plot
    plt.figure(figsize=(10, 6))
    plt.boxplot([py_times, cpp_times], labels=['Python', 'C++'])
    plt.ylabel('Latency (ms)')
    plt.title('Defect Detection Latency Comparison')
    plt.grid(True)
    if not os.path.exists("WaferDefectX/benchmarks"):
        os.makedirs("WaferDefectX/benchmarks")
    plt.savefig("WaferDefectX/benchmarks/latency_comparison.png")
    print("Plot saved to WaferDefectX/benchmarks/latency_comparison.png")

if __name__ == "__main__":
    main()
