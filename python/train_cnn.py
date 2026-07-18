"""
train_cnn.py — WaferDefectX CNN Training + 8-bit Quantization

Steps:
  1. Load WaferPatchDataset from synthetic wafer images
  2. Train WaferCNN (20 epochs, Adam, CrossEntropyLoss) — CPU-friendly
  3. Evaluate accuracy on held-out test split
  4. Apply post-training DYNAMIC quantization (8-bit, torch.qint8)
  5. Report model size: full fp32 vs quantized int8
  6. Save both checkpoints

Usage (from project root):
    python3 python/train_cnn.py   # from repo root

Outputs:
    results/cnn_wafer_model.pth           <- fp32 weights
    results/cnn_wafer_model_quantized.pth <- int8 quantized
"""

import os
import sys
import time
import platform

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Allow running from project root or from python/
sys.path.insert(0, os.path.dirname(__file__))
from cnn_model import WaferCNN, WaferPatchDataset, IDX_TO_LABEL, PATCH_SIZE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")

EPOCHS       = 20
BATCH_SIZE   = 16
LR           = 1e-3
VAL_SPLIT    = 0.2
SEED         = 42

MODEL_PATH   = os.path.join(RESULTS_DIR, "cnn_wafer_model.pth")
QUANT_PATH   = os.path.join(RESULTS_DIR, "cnn_wafer_model_quantized.pth")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_model_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 2)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    print("=" * 60)
    print("  WaferDefectX — CNN Training + 8-bit Quantization")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Dataset
    # -----------------------------------------------------------------------
    data_dir = os.path.abspath(DATA_DIR)
    if not os.path.isdir(data_dir):
        print(f"\n[ERROR] Data directory not found: {data_dir}")
        print("  Run: python3 python/data_generator.py")
        sys.exit(1)

    dataset = WaferPatchDataset(data_dir)
    print(f"\n[Data] {len(dataset)} labelled samples from: {data_dir}")

    # Class distribution
    from collections import Counter
    label_counts = Counter(lbl for _, lbl in dataset.samples)
    for idx, name in IDX_TO_LABEL.items():
        print(f"       {name:>10s}: {label_counts.get(idx, 0)} samples")

    # Train / val split
    torch.manual_seed(SEED)
    val_size   = max(1, int(len(dataset) * VAL_SPLIT))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # -----------------------------------------------------------------------
    # 2. Model, Loss, Optimizer
    # -----------------------------------------------------------------------
    device = torch.device("cpu")  # CPU-only; no GPU required
    model  = WaferCNN().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] WaferCNN — {total_params:,} parameters")
    print(f"        Input: (B, 1, {PATCH_SIZE}, {PATCH_SIZE}) grayscale patch")
    print(f"        Output: 3 classes (good / particle / scratch)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    # -----------------------------------------------------------------------
    # 3. Training Loop
    # -----------------------------------------------------------------------
    print(f"\n[Train] {EPOCHS} epochs | batch={BATCH_SIZE} | lr={LR} | device=CPU\n")
    print(f"  {'Epoch':>5}  {'Loss':>8}  {'Train Acc':>10}  {'Val Acc':>8}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*8}")

    best_val_acc = 0.0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct      += (logits.argmax(dim=1) == labels).sum().item()
            total        += labels.size(0)

        scheduler.step()

        epoch_loss  = running_loss / total
        train_acc   = correct / total
        val_acc     = evaluate(model, val_loader, device)

        print(f"  {epoch:>5}  {epoch_loss:>8.4f}  {train_acc:>10.4f}  {val_acc:>8.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            # Save best fp32 model
            os.makedirs(RESULTS_DIR, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)

    elapsed = time.time() - t0
    print(f"\n[Train] Done in {elapsed:.1f}s | Best val accuracy: {best_val_acc:.4f}")

    # -----------------------------------------------------------------------
    # 4. Load best model for final evaluation
    # -----------------------------------------------------------------------
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    fp32_acc = evaluate(model, val_loader, device)
    print(f"\n[Eval]  FP32 model validation accuracy : {fp32_acc:.4f}")

    print(f"        FP32 model saved → {MODEL_PATH}")

    # -----------------------------------------------------------------------
    # 5. Post-Training Dynamic Quantization (8-bit)
    #
    #    torch.quantization.quantize_dynamic() converts Linear layers to
    #    int8, reducing model size by ~75% with minimal accuracy degradation.
    #    This is the standard first step for edge/NPU deployment at Qualcomm.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Applying 8-bit Post-Training Dynamic Quantization")
    print("=" * 60)

    # Set quantization backend:
    #   - 'qnnpack' : ARM / Apple Silicon / mobile (Qualcomm-relevant)
    #   - 'fbgemm'  : x86 / Intel (server)
    is_arm = platform.machine() in ("arm64", "aarch64")
    qengine = "qnnpack" if is_arm else "fbgemm"
    torch.backends.quantized.engine = qengine
    print(f"[Quant] Backend: {qengine} ({'ARM/mobile' if is_arm else 'x86/server'})")

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={torch.nn.Linear},  # quantize all Linear layers
        dtype=torch.qint8,               # 8-bit signed integers
    )
    quantized_model.eval()

    # Evaluate quantized model
    quant_acc = evaluate(quantized_model, val_loader, device)
    print(f"\n[Quant] INT8 model validation accuracy : {quant_acc:.4f}")
    print(f"        Accuracy delta (fp32 - int8)    : {fp32_acc - quant_acc:+.4f}")

    # Save quantized model
    torch.save(quantized_model.state_dict(), QUANT_PATH)
    print(f"\n[Quant] INT8 model saved → {QUANT_PATH}")

    # -----------------------------------------------------------------------
    # 6. Model size comparison
    # -----------------------------------------------------------------------
    fp32_mb  = get_model_size_mb(MODEL_PATH)
    quant_mb = get_model_size_mb(QUANT_PATH)
    reduction = (1 - quant_mb / fp32_mb) * 100 if fp32_mb > 0 else 0

    print("\n" + "=" * 60)
    print("  Model Size Comparison (Edge Deployment Story)")
    print("=" * 60)
    print(f"  FP32 model  : {fp32_mb:.3f} MB")
    print(f"  INT8 model  : {quant_mb:.3f} MB")
    print(f"  Reduction   : {reduction:.1f}%")
    print(f"\n  → {reduction:.0f}% smaller footprint with <{abs(fp32_acc - quant_acc)*100:.1f}% accuracy loss")
    print("    Suitable for edge devices with memory-constrained NPUs.")
    print("=" * 60)

    print("\n✅ All done. Next step:")
    print("   python3 python/export_cnn_to_onnx.py")


if __name__ == "__main__":
    train()
