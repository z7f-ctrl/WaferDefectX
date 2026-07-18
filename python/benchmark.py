import glob
import os
import subprocess
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from defect_localization import DefectLocalizer
from paths import BENCHMARKS_DIR, CPP_EXECUTABLE, DATA_SYNTHETIC, ensure_dir
from preprocessing import Preprocessor
from tqdm import tqdm


def benchmark_python(image_paths):
    preprocessor = Preprocessor()
    localizer = DefectLocalizer()
    latencies = []

    for img_path in tqdm(image_paths, desc="Benchmarking Python"):
        image = cv2.imread(img_path)
        if image is None:
            continue

        start_time = time.time()
        enhanced, _ = preprocessor.process_pipeline(image)
        _ = localizer.localize(enhanced)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)

    return latencies


def benchmark_cpp(image_paths, executable_path):
    latencies = []

    for img_path in tqdm(image_paths, desc="Benchmarking C++"):
        result = subprocess.run(
            [executable_path, img_path], capture_output=True, text=True
        )
        for line in result.stdout.split("\n"):
            if "Processing Time:" in line:
                try:
                    time_ms = float(line.split(":")[1].strip().split(" ")[0])
                    latencies.append(time_ms)
                except ValueError:
                    pass

    return latencies


def main():
    data_dir = DATA_SYNTHETIC
    cpp_exec = str(CPP_EXECUTABLE)

    image_paths = glob.glob(os.path.join(str(data_dir), "*.png"))
    if not image_paths:
        print(f"No images found in {data_dir}.")
        return

    print("--- Running Benchmarks ---")
    py_times = benchmark_python(image_paths)

    if not os.path.exists(cpp_exec):
        print(f"C++ executable not found at {cpp_exec}. Skipping C++ benchmark.")
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

    ensure_dir(BENCHMARKS_DIR)
    plt.figure(figsize=(10, 6))
    plt.boxplot([py_times, cpp_times], labels=["Python", "C++"])
    plt.ylabel("Latency (ms)")
    plt.title("Defect Detection Latency Comparison")
    plt.grid(True)
    out_path = BENCHMARKS_DIR / "latency_comparison.png"
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
