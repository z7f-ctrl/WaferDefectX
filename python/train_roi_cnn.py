"""Train WaferResNet18 on WM-811K ROI patches (M3a).

Features:
  - ROI patch extraction (bbox crop → 64×64)
  - ResNet-18 backbone (1-channel grayscale)
  - Class-weighted CrossEntropyLoss for imbalance
  - Cosine annealing LR scheduler
  - Data augmentation (flip, rotate, brightness, noise)
  - Early stopping on validation loss
  - Confusion matrix + per-class metrics to JSON

Usage (from repo root):
    PYTHONPATH=python python3 python/train_roi_cnn.py

Outputs:
    results/wafer_resnet18_roi.pth
    results/roi_cnn_eval_metrics.json
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(__file__))

from paths import DATA_WM811K_IMAGES, RESULTS_DIR, ensure_dir
from roi_dataset import WaferROIPatchDataset, IDX_TO_LABEL, LABEL_MAP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EPOCHS = 30
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
PATIENCE = 5
PATCH_SIZE = 64

MODEL_PATH = RESULTS_DIR / "wafer_resnet18_roi.pth"
METRICS_PATH = RESULTS_DIR / "roi_cnn_eval_metrics.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_class_weights(dataset: WaferROIPatchDataset) -> torch.Tensor:
    from collections import Counter
    counts = Counter(lbl for _, lbl in dataset.samples)
    total = sum(counts.values())
    num_classes = len(LABEL_MAP)
    weights = torch.zeros(num_classes)
    for cls_idx in range(num_classes):
        if counts.get(cls_idx, 0) > 0:
            weights[cls_idx] = total / (num_classes * counts[cls_idx])
        else:
            weights[cls_idx] = 1.0
    return weights


def evaluate(model, loader, device, criterion=None):
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            if criterion is not None:
                total_loss += criterion(logits, labels).item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if criterion is not None else 0.0
    return acc, avg_loss, np.array(all_preds), np.array(all_labels)


def build_metrics(y_true, y_pred, accuracy, labels_map):
    from sklearn.metrics import classification_report, confusion_matrix
    target_names = [labels_map[i] for i in sorted(labels_map.keys())]
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": target_names,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  WaferResNet18 ROI Training on WM-811K")
    print("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"[Device] {device}")

    # --- Dataset ---
    data_dir = str(DATA_WM811K_IMAGES)
    if not os.path.isdir(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}")
        sys.exit(1)

    print("\n[1/5] Extracting ROI patches...")
    dataset = WaferROIPatchDataset(data_dir, augment=False)
    if len(dataset) == 0:
        print("[ERROR] No patches extracted")
        sys.exit(1)

    # --- Split ---
    torch.manual_seed(SEED)
    n = len(dataset)
    n_test = max(1, int(n * TEST_SPLIT))
    n_val = max(1, int(n * VAL_SPLIT))
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_ds_aug = WaferROIPatchDataset(data_dir, augment=True)
    train_indices = train_ds.indices
    train_ds_aug.samples = [dataset.samples[i] for i in train_indices]
    train_ds_aug.augment = True

    train_loader = DataLoader(train_ds_aug, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n[Split] Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # --- Model ---
    from wafer_resnet import WaferResNet18
    model = WaferResNet18(num_classes=3).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] WaferResNet18 — {params:,} parameters")

    # --- Loss / Optimizer / Scheduler ---
    class_weights = compute_class_weights(dataset).to(device)
    print(f"[Loss] Class weights: {class_weights.cpu().numpy().round(3)}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # --- Training ---
    print(f"\n[4/5] Training {EPOCHS} epochs...\n")
    print(f"  {'Epoch':>5}  {'Loss':>8}  {'Train Acc':>10}  {'Val Acc':>8}  {'LR':>10}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}")

    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    t0 = time.time()

    ensure_dir(RESULTS_DIR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        epoch_loss = running_loss / total
        train_acc = correct / total
        val_acc, val_loss, _, _ = evaluate(model, val_loader, device, criterion)
        lr = optimizer.param_groups[0]["lr"]

        marker = ""
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(MODEL_PATH))
            marker = " *"

        print(f"  {epoch:>5}  {epoch_loss:>8.4f}  {train_acc:>10.4f}  {val_acc:>8.4f}  {lr:>10.2e}{marker}")

        if val_acc >= best_val_acc:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    elapsed = time.time() - t0
    print(f"\n[Train] Done in {elapsed:.1f}s | Best val accuracy: {best_val_acc:.4f}")

    # --- Test evaluation ---
    print("\n[5/5] Final test evaluation...")
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
    model.eval()
    test_acc, _, y_pred, y_true = evaluate(model, test_loader, device)
    print(f"[Test] Accuracy: {test_acc:.4f}")

    metrics = build_metrics(y_true, y_pred, test_acc, IDX_TO_LABEL)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"[Test] Metrics saved to {METRICS_PATH}")
    print("\nClassification Report:")
    print(metrics["classification_report"])

    # --- Quantization ---
    print("=" * 60)
    print("  8-bit Dynamic Quantization")
    print("=" * 60)
    import platform as plat
    is_arm = plat.machine() in ("arm64", "aarch64")
    torch.backends.quantized.engine = "qnnpack" if is_arm else "fbgemm"

    model_cpu = model.cpu()
    test_loader_cpu = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    quantized = torch.quantization.quantize_dynamic(
        model_cpu, qconfig_spec={nn.Linear}, dtype=torch.qint8,
    )
    q_acc, _, _, _ = evaluate(quantized, test_loader_cpu, torch.device("cpu"))
    print(f"  FP32 test acc: {test_acc:.4f}")
    print(f"  INT8 test acc: {q_acc:.4f}")
    print(f"  Delta: {test_acc - q_acc:+.4f}")

    quant_path = RESULTS_DIR / "wafer_resnet18_roi_quantized.pth"
    torch.save(quantized.state_dict(), str(quant_path))
    print(f"  INT8 model saved -> {quant_path}")

    fp32_mb = MODEL_PATH.stat().st_size / (1024 ** 2)
    quant_mb = quant_path.stat().st_size / (1024 ** 2)
    print(f"  FP32 size: {fp32_mb:.2f} MB | INT8 size: {quant_mb:.2f} MB | Reduction: {(1-quant_mb/fp32_mb)*100:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
