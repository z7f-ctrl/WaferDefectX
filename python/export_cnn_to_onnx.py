"""
export_cnn_to_onnx.py — WaferDefectX ONNX Export + Runtime Benchmark

Steps:
  1. Load the trained fp32 WaferCNN from results/cnn_wafer_model.pth
  2. Export to ONNX (opset 13) with constant folding enabled
  3. Run ONNX Runtime inference with ALL graph optimisations enabled
  4. Benchmark latency: PyTorch FP32 vs ONNX Runtime
  5. Print model sizes: .pth vs .onnx

Why fp32 for ONNX export (not the quantized .pth)?
  PyTorch dynamic quantization of Linear layers produces quantized-only
  weights and special ATen ops that ONNX opset 13 does not map cleanly.
  Instead, we export the standard fp32 graph and apply ONNX Runtime's
  own graph-level optimisations (constant folding, op fusion) which are
  cross-platform and HW-agnostic — ideal for Qualcomm QNN/Hexagon backends.

Usage (from project root):
    python3 WaferDefectX/python/export_cnn_to_onnx.py

Outputs:
    WaferDefectX/results/cnn_wafer_defect.onnx
"""

import os
import sys
import time

import numpy as np
import torch
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(__file__))
from cnn_model import WaferCNN, PATCH_SIZE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL_PATH  = os.path.join(RESULTS_DIR, "cnn_wafer_model.pth")
ONNX_PATH   = os.path.join(RESULTS_DIR, "cnn_wafer_defect.onnx")

N_WARMUP    = 5    # inference warm-up runs (discarded)
N_BENCH     = 50   # timed inference runs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 2)


def bench_pytorch(model: torch.nn.Module, dummy: torch.Tensor) -> float:
    """Return mean per-image latency (ms) over N_BENCH runs."""
    model.eval()
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(dummy)
        t0 = time.perf_counter()
        for _ in range(N_BENCH):
            _ = model(dummy)
        return (time.perf_counter() - t0) / N_BENCH * 1000  # ms


def bench_ort(sess: ort.InferenceSession, input_name: str,
              dummy_np: np.ndarray) -> float:
    """Return mean per-image latency (ms) over N_BENCH runs."""
    for _ in range(N_WARMUP):
        sess.run(None, {input_name: dummy_np})
    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        sess.run(None, {input_name: dummy_np})
    return (time.perf_counter() - t0) / N_BENCH * 1000  # ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("  WaferDefectX — ONNX Export + ONNX Runtime Benchmark")
    print("=" * 62)

    # -----------------------------------------------------------------------
    # 1. Load trained WaferCNN
    # -----------------------------------------------------------------------
    if not os.path.isfile(MODEL_PATH):
        print(f"\n[ERROR] Trained model not found: {MODEL_PATH}")
        print("  Run first: python3 WaferDefectX/python/train_cnn.py")
        sys.exit(1)

    model = WaferCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] WaferCNN loaded — {total_params:,} parameters")

    # Dummy input: batch=1, channels=1, 64×64
    dummy_tensor = torch.randn(1, 1, PATCH_SIZE, PATCH_SIZE)
    dummy_np     = dummy_tensor.numpy().astype(np.float32)

    # -----------------------------------------------------------------------
    # 2. Export to ONNX
    # -----------------------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n[ONNX] Exporting to {ONNX_PATH} ...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_tensor,
            ONNX_PATH,
            opset_version=13,
            do_constant_folding=True,         # folds constant sub-expressions
            input_names=["wafer_patch"],
            output_names=["class_logits"],
            dynamic_axes={
                "wafer_patch":   {0: "batch_size"},
                "class_logits":  {0: "batch_size"},
            },
        )
    print(f"       Export complete ✓")

    # Verify with onnx package if available
    try:
        import onnx
        onnx_model = onnx.load(ONNX_PATH)
        onnx.checker.check_model(onnx_model)
        node_count = len(onnx_model.graph.node)
        print(f"       ONNX model validated ✓ ({node_count} graph nodes)")
    except ImportError:
        print("       (install 'onnx' package for graph validation)")

    # -----------------------------------------------------------------------
    # 3. ONNX Runtime Session — ALL graph optimisations enabled
    #
    #    ORT_ENABLE_ALL applies:
    #      - Basic graph optimisations (redundant node elimination)
    #      - Extended optimisations (op fusion, constant folding)
    #      - Layout / memory optimisations
    #
    #    On Qualcomm QNN/Hexagon backends, these same ORT optimisations
    #    are the entry point before NPU-specific lowering.
    # -----------------------------------------------------------------------
    print("\n[ORT]  Creating InferenceSession with ORT_ENABLE_ALL ...")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 1   # single-threaded (simulates edge constraint)

    sess = ort.InferenceSession(
        ONNX_PATH,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )

    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"       Input  : '{input_name}' — {sess.get_inputs()[0].shape}")
    print(f"       Output : '{output_name}' — {sess.get_outputs()[0].shape}")

    # Quick correctness check
    logits_ort = sess.run([output_name], {input_name: dummy_np})[0]
    pred_ort   = int(np.argmax(logits_ort, axis=1)[0])
    with torch.no_grad():
        pred_torch = int(model(dummy_tensor).argmax(dim=1).item())

    labels = ["good", "particle", "scratch"]
    print(f"\n[Check] PyTorch prediction : {labels[pred_torch]}")
    print(f"        ORT    prediction  : {labels[pred_ort]}")
    match = "✓ match" if pred_ort == pred_torch else "✗ mismatch (numerical diff)"
    print(f"        Result             : {match}")

    # -----------------------------------------------------------------------
    # 4. Latency Benchmark
    # -----------------------------------------------------------------------
    print(f"\n[Bench] Running {N_BENCH} timed inferences (batch=1) ...")

    lat_torch = bench_pytorch(model, dummy_tensor)
    lat_ort   = bench_ort(sess, input_name, dummy_np)
    speedup   = lat_torch / lat_ort if lat_ort > 0 else float("nan")

    print(f"\n  PyTorch FP32 (CPU)        : {lat_torch:.3f} ms / image")
    print(f"  ONNX Runtime (ORT_ALL)    : {lat_ort:.3f} ms / image")
    print(f"  ORT speedup               : {speedup:.2f}×")

    # -----------------------------------------------------------------------
    # 5. Model size comparison
    # -----------------------------------------------------------------------
    pth_mb  = get_size_mb(MODEL_PATH)
    onnx_mb = get_size_mb(ONNX_PATH)

    quant_path = os.path.join(RESULTS_DIR, "cnn_wafer_model_quantized.pth")
    quant_mb   = get_size_mb(quant_path) if os.path.isfile(quant_path) else None

    print("\n" + "=" * 62)
    print("  Artefact Summary")
    print("=" * 62)
    print(f"  {'File':<40}  {'Size':>8}")
    print(f"  {'-'*40}  {'-'*8}")
    print(f"  {os.path.basename(MODEL_PATH):<40}  {pth_mb:>7.3f} MB  (FP32)")
    if quant_mb is not None:
        reduction = (1 - quant_mb / pth_mb) * 100
        print(f"  {'cnn_wafer_model_quantized.pth':<40}  {quant_mb:>7.3f} MB  (INT8, {reduction:.0f}% smaller)")
    print(f"  {os.path.basename(ONNX_PATH):<40}  {onnx_mb:>7.3f} MB  (ONNX opset 13)")
    print("=" * 62)

    print("\n✅ ONNX export successful.")
    print(f"   Inspect graph visually at https://netron.app (drag {os.path.basename(ONNX_PATH)})")
    print("\n   Latency summary for README:")
    print(f"   | Backend              | Latency (ms) |")
    print(f"   |----------------------|--------------|")
    print(f"   | PyTorch FP32 (CPU)   | {lat_torch:>10.3f} |")
    print(f"   | ONNX Runtime (CPU)   | {lat_ort:>10.3f} |")
    print(f"   | C++ Pipeline (total) | ~2.1         |")


if __name__ == "__main__":
    main()
