# WaferDefectX 测试报告

| 字段 | 内容 |
|------|------|
| 报告版本 | 1.0 |
| 项目名称 | WaferDefectX |
| 测试日期 | 2026-07-19 |
| 测试环境 | macOS ARM64, Python 3.9.6, PyTorch 2.8.0, OpenCV 5.0.0 |
| 数据集 | WM-811K（Kaggle: qingyi/wm811k-wafer-map） |

---

## 1. 测试概览

本报告覆盖 WaferDefectX 项目的三项核心测试：

| 测试类型 | 说明 | 结果 |
|----------|------|------|
| 单元测试 | Python 核心模块 + 模型契约 | ✅ 13/13 通过 |
| 冒烟测试 | 端到端流水线（数据→训练→导出→C++） | ✅ 5/5 通过 |
| 性能测试 | CNN 分类准确率（WM-811K 真实数据集） | ✅ 93.83%（目标 90%） |

---

## 2. 单元测试

```bash
PYTHONPATH=python python3 -m pytest tests -v
```

| 测试用例 | 模块 | 结果 |
|----------|------|------|
| test_preprocessor_grayscale_input | preprocessing | ✅ PASSED |
| test_preprocessor_output_shapes | preprocessing | ✅ PASSED |
| test_localizer_requires_gray_and_finds_structure | defect_localization | ✅ PASSED |
| test_localizer_empty_image | defect_localization | ✅ PASSED |
| test_feature_dim | features | ✅ PASSED |
| test_paths_resolve_inside_repo | paths | ✅ PASSED |
| test_cpp_feature_alignment | C++/Python 对齐 | ✅ PASSED |
| test_extract_dataset_includes_noise_class | train_eval | ✅ PASSED |
| test_write_and_load_meta | model_meta | ✅ PASSED |
| test_missing_meta_raises | model_meta | ✅ PASSED |
| test_classifier_rejects_unknown_type | classifier | ✅ PASSED |
| test_classifier_onnx_requires_path | classifier | ✅ PASSED |
| test_openvino_loads_classes_from_meta | classifier | ✅ PASSED |

**结果：13 passed, 0 failed, 0 skipped**

---

## 3. 冒烟测试

```bash
bash scripts/smoke.sh
```

| 步骤 | 内容 | 结果 |
|------|------|------|
| 1/5 | 合成数据存在性检查 | ✅ 79 张 PNG |
| 2/5 | Python 检测 Demo（main.py） | ✅ 5 张图处理成功 |
| 3/5 | 单元测试（pytest） | ✅ 13 passed |
| 4/5 | ONNX 导出验证 | ✅ 推理成功 |
| 5/5 | C++ 构建 + 运行 | ✅ 编译零 warning，推理 6.3ms |

**结果：5/5 通过**

---

## 4. C++ 编译与运行测试

### 4.1 编译

```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build --clean-first
```

| 检查项 | 结果 |
|--------|------|
| 编译 warning | 0 |
| 链接错误 | 0 |
| 产物 | libwafer_core.a + WaferDefectX_Run |

### 4.2 运行时测试

| 测试场景 | 输入 | 输出 | EXIT |
|----------|------|------|------|
| 正常 particle | wafer_0000_particle.png | 1 defect, 12.6ms | 0 |
| 正常 scratch | wafer_0003_scratch.png | 1 defect, 5.5ms | 0 |
| 正常 good | wafer_0001_good.png | 1 defect, 4.8ms | 0 |
| JSON 输出 | --json wafer_0000_particle.png | JSON 含 features | 0 |
| 无参数 | （空） | Usage 提示 | 255 |
| 文件不存在 | nonexistent.png | 错误信息 | 253 |
| 非法输入 | /dev/null | 错误信息 | 253 |

### 4.3 C++/Python 特征对齐

| 特征 | C++ | Python | 差异 |
|------|-----|--------|------|
| area | 636087 | 636087.0 | 0.0 |
| perimeter | 3755.19 | 3755.186 | 0.004 |
| aspect_ratio | 1.0 | 1.0 | 0.0 |
| rectangularity | 0.9939 | 0.9939 | <0.001 |
| circularity | 0.5668 | 0.5668 | <0.001 |
| mean_intensity | 140.26 | 140.26 | <0.01 |
| std_intensity | 46.61 | 46.61 | <0.01 |

**最大差异：0.004（perimeter），满足 atol=0.01 容差**

---

## 5. CNN 分类性能测试

### 5.1 测试配置

| 项 | 值 |
|----|-----|
| 模型 | WaferResNet18（ResNet-18，1-ch 64×64） |
| 参数量 | 11,171,779 |
| 数据集 | WM-811K（5,193 张平衡采样） |
| 划分 | Train: 3,637 / Val: 778 / Test: 778 |
| 训练 | 30 epochs, batch=32, lr=3e-4, AdamW |
| 增强 | flip, rotation(±15°), brightness(±20), noise(σ≤2) |
| 损失 | Class-weighted CrossEntropy（scratch 权重 1.45×） |
| 调度 | Cosine annealing LR（3e-4 → 1e-6） |
| 设备 | Apple Silicon MPS |

### 5.2 训练曲线

| Epoch | Train Loss | Train Acc | Val Acc | LR | 备注 |
|-------|-----------|-----------|---------|-----|------|
| 1 | 0.8737 | 0.5988 | 0.5913 | 2.99e-04 | 初始化 |
| 5 | 0.4304 | 0.8359 | 0.8111 | 2.80e-04 | |
| 7 | 0.3498 | 0.8691 | 0.8715 | 2.62e-04 | 首次 >85% |
| 15 | 0.1934 | 0.9318 | 0.9049 | 1.50e-04 | 首次 >90% |
| 25 | 0.0804 | 0.9684 | 0.9280 | 2.10e-05 | 接近收敛 |
| 30 | 0.0594 | 0.9788 | 0.9293 | 1.00e-06 | 最佳 val |

### 5.3 测试集结果

| 指标 | 值 |
|------|-----|
| **Accuracy** | **0.9383** |
| Macro F1 | 0.93 |
| Weighted F1 | 0.94 |

#### 每类详细

| 类别 | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
| good | 0.94 | 0.95 | 0.94 | 300 |
| particle | 0.95 | 0.95 | 0.95 | 315 |
| scratch | 0.91 | 0.90 | 0.91 | 163 |

#### 混淆矩阵

```
              Predicted
           good  particle  scratch
Actual
good        285      10        5
particle     10     299        6
scratch      10      7       146
```

### 5.4 与 RF 基线对比

| 指标 | RF + 7-dim | ResNet-18 ROI | 提升 |
|------|-----------|---------------|------|
| Accuracy | 0.7009 | **0.9383** | +33.9% |
| good Recall | 0.80 | 0.95 | +18.8% |
| particle Recall | 0.76 | 0.95 | +25.0% |
| scratch Recall | 0.44 | **0.90** | +104.5% |
| Macro F1 | 0.67 | **0.93** | +38.8% |

### 5.5 目标达成情况

| 目标 | 阈值 | 实际 | 状态 |
|------|------|------|------|
| Accuracy ≥ 90% | 0.90 | 0.94 | ✅ 达成 |
| scratch Recall ≥ 80% | 0.80 | 0.90 | ✅ 达成 |
| Macro F1 ≥ 85% | 0.85 | 0.93 | ✅ 达成 |

---

## 6. C++ ONNX Runtime 集成测试

### 6.1 编译

```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build --clean-first
```

| 检查项 | 结果 |
|--------|------|
| 编译 warning | 0 |
| 链接错误 | 0 |
| C++ 标准 | C++17（ONNX Runtime 1.27 要求） |
| 产物 | libwafer_core.a + libonnx_classifier.a + WaferDefectX_Run |

### 6.2 分类验证

| 输入图片 | 真实类别 | 预测类别 | 置信度 | 延迟 |
|----------|----------|----------|--------|------|
| wafer_00002_good.png | good | **good** | 97.2% | 18.8ms |
| wafer_02000_particle.png | particle | **particle** | 100.0% | 13.2ms |
| wafer_04000_scratch.png | scratch | **scratch** | 100.0% | 12.0ms |

**分类准确率：3/3（100%）**

### 6.3 JSON 输出格式

```json
{
  "image": "wafer_00002_good.png",
  "resolution": [800, 800],
  "channels": 3,
  "processing_ms": 18.77,
  "model": "wafer_resnet18_roi.onnx",
  "defect_count": 1,
  "defects": [{
    "id": 0,
    "bbox": [10, 10, 781, 781],
    "features": [471866, 2736.49, 1, 0.7736, 0.791844, 150.076, 32.9805],
    "classification": {
      "label": "good",
      "label_id": 0,
      "confidence": 0.972272
    }
  }]
}
```

---

## 7. 延迟基准

### 7.1 C++ 预处理+定位+特征（不含分类）

| 指标 | 值 |
|------|-----|
| 平均 | 5.8 ms |
| P50 | 5.4 ms |
| P95 | 11.9 ms |

### 7.2 端到端延迟（含 ONNX Runtime 分类）

| 组件 | 延迟 |
|------|------|
| C++ 预处理+定位+特征 | ~6 ms |
| ONNX Runtime 推理（ResNet-18） | ~1 ms |
| **端到端总计** | **12–19 ms** |

---

## 8. 已知问题与限制

| 编号 | 问题 | 影响 | 状态 |
|------|------|------|------|
| 1 | 8-bit 量化对 ResNet-18 无压缩效果（Conv2d 为主） | 模型体积 42.7MB | 已知，可接受 |
| 2 | WM-811K 仅 5,193 张平衡采样（原集 172,950） | 未充分利用数据 | 可扩展 |
| 3 | 合成数据集 accuracy 仅 42%（三类含 noise） | 仅用于冒烟 | 设计如此 |
| 4 | ~~C++ 路径不含 ONNX 推理~~ | ✅ 已集成 ONNX Runtime | 已解决 |

---

## 9. 结论

WaferDefectX 在 WM-811K 真实数据集上通过 ResNet-18 ROI 分类器达到 **93.83% 准确率**，超过 90% 目标。scratch 召回率从 RF 的 44% 提升至 90%，Macro F1 从 0.67 提升至 0.93。所有单元测试（13/13）和冒烟测试（5/5）通过，C++ 编译零 warning，C++ ONNX Runtime 端到端集成验证通过（good/particle/scratch 三类分类全部正确）。

### 9.1 完整指标

| 指标 | RF 基线 | ResNet-18 ROI | 目标 | 状态 |
|------|---------|---------------|------|------|
| Accuracy | 0.70 | **0.94** | ≥0.90 | ✅ |
| scratch Recall | 0.44 | **0.90** | ≥0.80 | ✅ |
| Macro F1 | 0.67 | **0.93** | ≥0.85 | ✅ |
| C++ 端到端 | 不含分类 | **12-19ms** | — | ✅ |
| ONNX 导出 | — | **42.61MB** | — | ✅ |

### 9.2 项目里程碑

| 阶段 | 状态 |
|------|------|
| P0（工程基础） | ✅ 10/10 |
| P1（生产流水线） | ✅ 10/10 |
| M3a（CNN ROI 分类） | ✅ 6/6 |
| **C++ ONNX 集成** | ✅ 完成 |

项目已具备生产部署条件。

---

*报告生成时间：2026-07-19 | 测试执行者：自动化流水线*
