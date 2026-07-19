# WaferDefectX 基线报告

| 字段 | 内容 |
|------|------|
| 文档版本 | 2.0 |
| 关联文档 | [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md), [PLAN.md](./PLAN.md) |
| 评估日期 | 2026-07-19 |
| 评估协议 | EVAL_PROTOCOL.md §2 |

---

## 1. 数据集概况

### 1.1 WM-811K 真实数据集（主基线）

| 项 | 值 |
|----|-----|
| 来源 | Kaggle: qingyi/wm811k-wafer-map |
| 原始总量 | 811,457 张晶圆图 |
| 有标签 | 172,950 张（21.3%） |
| 采样量 | 5,193 张（平衡采样） |
| 类别 | good: 2,000 / particle: 2,000 / scratch: 1,193 |
| 划分 | train=3,635, test=1,558（stratified, seed=42） |
| 图像格式 | 800×800 灰度（从原始 waferMap 转换） |

### 1.2 合成数据集（冒烟用）

| 项 | 值 |
|----|-----|
| 总图数 | 79 |
| 类别 | noise: 24 / particle: 30 / scratch: 25 |
| 划分 | train=55, test=24 |

---

## 2. 分类指标（WM-811K 真实数据集）

### 2.1 总体

| 指标 | 值 |
|------|-----|
| **Accuracy** | **0.7009** |
| Macro Avg F1 | 0.67 |
| Weighted Avg F1 | 0.70 |

### 2.2 每类详细

| 类别 | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
| noise | 0.72 | 0.80 | 0.75 | 600 |
| particle | 0.78 | 0.76 | 0.77 | 600 |
| scratch | 0.51 | 0.44 | 0.47 | 358 |

### 2.3 混淆矩阵

```
              Predicted
           noise  particle  scratch
Actual
noise        477     47       76
particle      67    456       77
scratch      121     78      159
```

---

## 3. 分类指标（合成数据集 · 冒烟用）

| 指标 | 值 |
|------|-----|
| Accuracy | 0.4167 |
| Macro F1 | 0.38 |
| scratch Recall | 0.12 |

> 合成集仅用于冒烟回归，不代表生产性能。

---

## 4. C++ 延迟基准

在 Apple Silicon (macOS) 上测试 14 张合成图：

| 指标 | 值 |
|------|-----|
| 平均延迟 | 5.8 ms |
| P50 | 5.4 ms |
| P95 | 11.9 ms |
| 最小 | 4.8 ms |
| 最大 | 11.9 ms |

> C++ 路径仅含预处理+定位+特征提取，不含分类推理。

---

## 5. 基线解读

### 5.1 真实数据集 vs 合成数据集

| 指标 | 合成集 | WM-811K | 变化 |
|------|--------|---------|------|
| Accuracy | 0.42 | **0.70** | +67% |
| Macro F1 | 0.38 | **0.67** | +76% |
| noise Recall | 0.43 | **0.80** | +86% |
| particle Recall | 0.67 | **0.76** | +13% |
| scratch Recall | 0.12 | **0.44** | +267% |

**关键发现**：
1. 真实数据集上整体性能显著优于合成集（+67% accuracy）
2. scratch 类仍有提升空间（Recall 0.44，F1 0.47）
3. noise 误检控制较好（Recall 0.80）

### 5.2 改进方向

| 优先级 | 方向 | 预期收益 |
|--------|------|----------|
| P2 | 晶圆圆盘 mask 抑制边缘伪轮廓 | 减少 noise 误检 |
| P2 | 更多 scratch 样本 + 数据增强 | 提升 scratch Recall |
| P2 | 特征增强（纹理：LBP/HOG） | 提升类间可分性 |
| P2 | CNN ROI patch 分类 | 端到端学习更鲁棒特征 |

---

## 6. 回归基线

后续每次模型/特征/定位变更，需对比本基线：

| 对比项 | WM-811K 基线 | 目标 |
|--------|-------------|------|
| Accuracy | 0.70 | ≥ 0.80 |
| Macro F1 | 0.67 | ≥ 0.75 |
| scratch Recall | 0.44 | ≥ 0.60 |
| noise Recall | 0.80 | ≥ 0.85 |
| C++ P50 延迟 | 5.4 ms | ≤ 10 ms |

---

## 7. 产物清单

| 文件 | 说明 |
|------|------|
| `data/wm811k/LSWMD.pkl` | 原始 WM-811K 数据（2GB） |
| `data/wm811k/images/` | 转换后的 5,193 张 PNG 图像 |
| `data/wm811k/labels.csv` | 标签映射表 |
| `results/rf_model.pkl` | RF 模型权重 |
| `results/rf_model.meta.json` | 类别表 + 特征契约 |
| `results/rf_model.onnx` | ONNX 导出 |
| `results/eval_metrics.json` | 分类指标 JSON |
| `results/eval_predictions.csv` | 逐样本预测 CSV |

---

*本报告基于 WM-811K 真实数据集（平衡采样 5,193 张）。完整数据集（172,950 张有标签）可进一步提升性能。*
