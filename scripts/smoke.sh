#!/usr/bin/env bash
# Smoke test: run from any cwd; resolves to repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
PY="$ROOT/python"
export PYTHONPATH="${PY}${PYTHONPATH:+:$PYTHONPATH}"

echo "==> Repo root: $ROOT"

echo "==> [1/5] Ensure synthetic data exists"
if ! compgen -G "$ROOT/data/synthetic/*.png" > /dev/null; then
  "$PYTHON" "$PY/data_generator.py"
else
  echo "    data/synthetic already populated"
fi

SAMPLE="$(ls "$ROOT"/data/synthetic/*.png | head -n 1)"
echo "    sample: $SAMPLE"

echo "==> [2/5] Python detection demo (main.py)"
"$PYTHON" "$PY/main.py"

echo "==> [3/5] Unit tests"
"$PYTHON" -m pytest "$ROOT/tests" -q

echo "==> [4/5] ONNX export check (uses existing rf_model.pkl if present)"
if [[ -f "$ROOT/results/rf_model.pkl" ]]; then
  "$PYTHON" "$PY/export_to_onnx.py"
else
  echo "    skip: results/rf_model.pkl missing (run python/train_eval.py to train)"
fi

echo "==> [5/5] C++ build + run"
if command -v cmake >/dev/null 2>&1; then
  mkdir -p "$ROOT/cpp/build"
  cmake -S "$ROOT/cpp" -B "$ROOT/cpp/build"
  cmake --build "$ROOT/cpp/build" --config Release
  CPP_BIN="$ROOT/cpp/build/WaferDefectX_Run"
  if [[ -x "$CPP_BIN" ]]; then
    "$CPP_BIN" "$SAMPLE"
  else
    echo "ERROR: C++ binary not found at $CPP_BIN" >&2
    exit 1
  fi
else
  echo "    skip: cmake not found"
fi

echo ""
echo "Smoke OK."
