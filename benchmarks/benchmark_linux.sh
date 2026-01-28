#!/bin/bash

# Benchmarking Script for WaferDefectX
# Measures: Real Time (Latency), CPU Usage, Max Resident Set Size (Memory)

OUTPUT_FILE="benchmark_report_linux.txt"
DATA_IMAGE="WaferDefectX/data/synthetic/wafer_0033_scratch.png"
CPP_EXEC="./WaferDefectX/cpp/build/WaferDefectX_Run"
PYTHON_SCRIPT="WaferDefectX/python/main.py"

echo "========================================" > "$OUTPUT_FILE"
echo "   WaferDefectX Performance Benchmark   " >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "Date: $(date)" >> "$OUTPUT_FILE"
echo "System: $(uname -sr)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# check if image exists
if [ ! -f "$DATA_IMAGE" ]; then
    echo "Error: Test image $DATA_IMAGE not found. Generating data..."
    python3 WaferDefectX/python/data_generator.py
fi

# Function to run benchmark
run_benchmark() {
    NAME="$1"
    CMD="$2"
    
    echo "Benchmarking $NAME..."
    echo "----------------------------------------" >> "$OUTPUT_FILE"
    echo "Pipeline: $NAME" >> "$OUTPUT_FILE"
    echo "Command: $CMD" >> "$OUTPUT_FILE"
    
    # Use /usr/bin/time -v for detailed stats if available, else fallback to basic time
    if [ -f "/usr/bin/time" ]; then
        /usr/bin/time -v $CMD 2>> "$OUTPUT_FILE"
    else
        echo "Warning: /usr/bin/time not found, using shell time"
        { time $CMD; } 2>> "$OUTPUT_FILE"
    fi
    echo "" >> "$OUTPUT_FILE"
}

# 1. Benchmark Python Pipeline
# We run the main.py which processes ~5 images. 
# For better per-image stats, we might want a loop, but this gives system-level view.
run_benchmark "Python Prototype (OpenCV + scikit-learn)" "python3 $PYTHON_SCRIPT"

# 2. Benchmark C++ Pipeline
if [ -f "$CPP_EXEC" ]; then
    run_benchmark "C++ Production Core" "$CPP_EXEC $DATA_IMAGE"
else
    echo "----------------------------------------" >> "$OUTPUT_FILE"
    echo "Pipeline: C++ Production Core" >> "$OUTPUT_FILE"
    echo "Status: Executable not found (Build skipped in this env)" >> "$OUTPUT_FILE"
    echo "Note: To build, ensure OpenCV C++ libs are installed and run 'cmake .. && make' in cpp/build" >> "$OUTPUT_FILE"
fi

echo "Benchmark complete. Results saved to $OUTPUT_FILE"
cat "$OUTPUT_FILE"
