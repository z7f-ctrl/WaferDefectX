#!/usr/bin/env bash
# Benchmarking Script for WaferDefectX
# Run from anywhere; resolves to repo root.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_FILE="$ROOT/benchmarks/benchmark_report_linux.txt"
DATA_IMAGE="$ROOT/data/synthetic/wafer_0033_scratch.png"
CPP_EXEC="$ROOT/cpp/build/WaferDefectX_Run"
PYTHON_SCRIPT="$ROOT/python/main.py"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================" > "$OUTPUT_FILE"
echo "   WaferDefectX Performance Benchmark   " >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "Date: $(date)" >> "$OUTPUT_FILE"
echo "System: $(uname -sr)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

if [ ! -f "$DATA_IMAGE" ]; then
    echo "Test image not found. Generating data..."
    python3 "$ROOT/python/data_generator.py"
fi

run_benchmark() {
    NAME="$1"
    shift
    echo "Benchmarking $NAME..."
    echo "----------------------------------------" >> "$OUTPUT_FILE"
    echo "Pipeline: $NAME" >> "$OUTPUT_FILE"
    echo "Command: $*" >> "$OUTPUT_FILE"

    if [ -f "/usr/bin/time" ]; then
        /usr/bin/time -v "$@" 2>> "$OUTPUT_FILE"
    else
        echo "Warning: /usr/bin/time not found, using shell time"
        { time "$@"; } 2>> "$OUTPUT_FILE"
    fi
    echo "" >> "$OUTPUT_FILE"
}

run_benchmark "Python Prototype (OpenCV + scikit-learn)" python3 "$PYTHON_SCRIPT"

if [ -x "$CPP_EXEC" ]; then
    run_benchmark "C++ Production Core" "$CPP_EXEC" "$DATA_IMAGE"
else
    echo "----------------------------------------" >> "$OUTPUT_FILE"
    echo "Pipeline: C++ Production Core" >> "$OUTPUT_FILE"
    echo "Status: Executable not found (Build skipped in this env)" >> "$OUTPUT_FILE"
fi

echo "Benchmark complete. Results saved to $OUTPUT_FILE"
cat "$OUTPUT_FILE"
