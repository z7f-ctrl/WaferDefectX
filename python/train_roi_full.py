"""Train ResNet-18 on full WM-811K wafer maps (Phase A1).

Full-image CNN classification with MixUp regularization.
ResNet-18 (11M params) on 96×96 wafer maps.

Usage:
    PYTHONPATH=python python3 python/train_roi_full.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))

from paths import DATA_WM811K, RESULTS_DIR, ensure_dir
from wafer_full_dataset import WaferFullImageDataset, IDX_TO_LABEL, LABEL_MAP

EPOCHS = 30
BATCH_SIZE = 48
LR = 1e-3
WEIGHT_DECAY = 1e-3
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
SEED = 42
PATIENCE = 10
TARGET_SIZE = 96
MAX_PER_CLASS = 15000
MIXUP_ALPHA = 0.4

FULL_DATA_DIR = str(DATA_WM811K / "full_images")
MODEL_PATH = RESULTS_DIR / "wafer_resnet18_full.pth"
METRICS_PATH = RESULTS_DIR / "resnet18_full_metrics.json"

CLASS_WEIGHTS = torch.tensor([1.0, 1.0, 10.0])


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class AugmentedSubset(Dataset):
    def __init__(self, base_dataset, indices):
        self.samples = [(base_dataset.samples[i][0].copy(), base_dataset.samples[i][1])
                        for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)
        if np.random.random() > 0.5:
            img = cv2.flip(img, 0)
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor, label


class EvalSubset(Dataset):
    def __init__(self, base_dataset, indices):
        self.samples = [(base_dataset.samples[i][0].copy(), base_dataset.samples[i][1])
                        for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor, label


def stratified_split(dataset, val_ratio, test_ratio, seed=42):
    rng = np.random.RandomState(seed)
    by_class = {}
    for i, (_, lbl) in enumerate(dataset.samples):
        by_class.setdefault(lbl, []).append(i)

    train_idx, val_idx, test_idx = [], [], []
    for lbl, indices in by_class.items():
        indices = np.array(indices)
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_val - n_test
        train_idx.extend(indices[:n_train].tolist())
        val_idx.extend(indices[n_train:n_train+n_val].tolist())
        test_idx.extend(indices[n_train+n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    raw_acc = (preds_arr == labels_arr).mean()
    per_class_recall = []
    for c in range(3):
        mask = labels_arr == c
        if mask.sum() > 0:
            per_class_recall.append((preds_arr[mask] == c).sum() / mask.sum())
    balanced_acc = np.mean(per_class_recall) if per_class_recall else 0.0
    return balanced_acc, raw_acc, preds_arr, labels_arr


def main():
    print("=" * 60)
    print("  ResNet-18 — Full WM-811K Training (Phase A1)")
    print("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"[Device] {device}")
    print(f"[Config] BS={BATCH_SIZE}, LR={LR}, MixUp=α{MIXUP_ALPHA}")

    if not os.path.isdir(FULL_DATA_DIR):
        print(f"[ERROR] Full dataset not found: {FULL_DATA_DIR}")
        sys.exit(1)

    print(f"\n[1/5] Loading full wafer map dataset (target_size={TARGET_SIZE})...")
    dataset = WaferFullImageDataset(
        data_dir=FULL_DATA_DIR,
        target_size=TARGET_SIZE,
        augment=False,
        max_per_class=MAX_PER_CLASS,
        oversample_scratch=1,
    )
    if len(dataset) == 0:
        print("[ERROR] No images loaded")
        sys.exit(1)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("\n[2/5] Stratified train/val/test split...")
    train_idx, val_idx, test_idx = stratified_split(dataset, VAL_SPLIT, TEST_SPLIT, SEED)
    print(f"  Train: {len(train_idx):,}, Val: {len(val_idx):,}, Test: {len(test_idx):,}")

    train_labels = [dataset.samples[i][1] for i in train_idx]
    print(f"  Train class dist: {dict(Counter(train_labels))}")

    train_aug = AugmentedSubset(dataset, train_idx)
    val_ds = EvalSubset(dataset, val_idx)
    test_ds = EvalSubset(dataset, test_idx)

    train_loader = DataLoader(train_aug, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    print("\n[3/5] Building ResNet-18 model...")
    backbone = models.resnet18(weights=None)
    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc = nn.Linear(backbone.fc.in_features, 3)
    model = backbone.to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  ResNet-18 — {params:,} parameters")

    cw = CLASS_WEIGHTS.to(device)
    print(f"  Class weights: {cw.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=cw)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )

    print(f"\n[4/5] Training {EPOCHS} epochs...\n")
    print(f"  {'Epoch':>5}  {'Loss':>8}  {'Train Acc':>10}  {'Bal Acc':>8}  {'Raw Acc':>8}  {'LR':>10}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}")
    sys.stdout.flush()

    best_bal_acc = 0.0
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

            imgs_mixed, targets_a, targets_b, lam = mixup_data(imgs, labels)
            logits = model(imgs_mixed)
            loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += imgs.size(0)

        scheduler.step()
        epoch_loss = running_loss / total
        train_acc = correct / total

        bal_acc, raw_acc, y_pred, y_true = evaluate(model, val_loader, device)

        lr = optimizer.param_groups[0]["lr"]

        marker = ""
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            torch.save(model.state_dict(), str(MODEL_PATH))
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1

        print(f"  {epoch:>5}  {epoch_loss:>8.4f}  {train_acc:>10.4f}  {bal_acc:>8.4f}  {raw_acc:>8.4f}  {lr:>10.2e}{marker}")
        sys.stdout.flush()

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    elapsed = time.time() - t0
    print(f"\n[Train] Done in {elapsed:.1f}s | Best balanced accuracy: {best_bal_acc:.4f}")

    print("\n[5/5] Final test evaluation...")
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
    model.eval()
    test_bal_acc, test_raw_acc, y_pred, y_true = evaluate(model, test_loader, device)
    print(f"[Test] Balanced accuracy: {test_bal_acc:.4f}")
    print(f"[Test] Raw accuracy: {test_raw_acc:.4f}")

    from sklearn.metrics import classification_report, confusion_matrix
    target_names = [IDX_TO_LABEL[i] for i in range(3)]
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    metrics = {
        "balanced_accuracy": float(test_bal_acc),
        "raw_accuracy": float(test_raw_acc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": target_names,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\n[Metrics] Saved to {METRICS_PATH}")

    onnx_path = RESULTS_DIR / "wafer_resnet18_full.onnx"
    model.cpu()
    dummy = torch.randn(1, 1, TARGET_SIZE, TARGET_SIZE)
    torch.onnx.export(model, dummy, str(onnx_path), opset_version=17,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    fp32_mb = MODEL_PATH.stat().st_size / (1024 ** 2)
    onnx_mb = onnx_path.stat().st_size / (1024 ** 2)
    print(f"  FP32: {fp32_mb:.2f} MB | ONNX: {onnx_mb:.2f} MB")

    print("\nDone.")


if __name__ == "__main__":
    main()
