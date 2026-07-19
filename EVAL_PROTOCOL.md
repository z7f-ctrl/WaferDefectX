# WaferDefectX 评估协议

| 字段 | 内容 |
|------|------|
| 文档版本 | 1.0 |
| 关联文档 | [DESIGN.md](./DESIGN.md), [PLAN.md](./PLAN.md) |
| 目标 | 定义检测与分类的评估规则、指标计算方式、数据集划分约定 |
| 状态 | P1 完成（2026-07-19） |

---

## 1. 评估范围

本协议覆盖以下两个层级的评估：

| 层级 | 定义 | 关键指标 |
|------|------|----------|
| **定位（Localization）** | 在图像上找到缺陷候选区域 | 定位召回率、误报数/图 |
| **分类（Classification）** | 对定位到的 ROI 判定缺陷类型 | Accuracy、每类 P/R/F1、混淆矩阵 |

> 当前系统定位与分类串联评估；定位失败则该缺陷无法进入分类。

---

## 2. 数据集划分

### 2.1 合成数据集（当前默认）

- **来源**：`data/synthetic/wafer_*.png`，由 `WaferDataGenerator` 生成
- **划分方式**：`train_test_split(test_size=0.3, random_state=42, stratify=y)`
  - stratify 按类别标签分层
  - random_state=42 保证可复现
- **注意**：合成集 accuracy **不可作为生产 KPI**，仅用于冒烟和回归

### 2.2 真实数据集（规划中）

- **来源**：WM-811K 或内部标注样张
- **划分方式**：按 wafer_id 分组划分（避免同一晶圆的多张图同时出现在 train/test）
  - 推荐 80/10/10 train/val/test
  - 测试集独立于训练集中的所有晶圆
- **最小规模**：测试集 ≥ 50 张含缺陷图，≥ 20 张良品图

---

## 3. 定位评估规则

### 3.1 命中（Hit）定义

对于一张图像中的**每个标注缺陷区域**（ground-truth bbox），若系统输出的任一预测 bbox 与其 IoU ≥ 阈值，则视为**定位命中**。

```
IoU = area(pred ∩ gt) / area(pred ∪ gt)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| IoU 阈值 | 0.3 | 晶圆缺陷边界模糊，不宜过严 |

### 3.2 指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **定位召回率** | 命中数 / GT 缺陷总数 | 衡量漏检 |
| **每图误报数** | 预测数 - 命中数（按图统计） | 衡量虚警 |
| **定位精确率** | 命中数 / 预测总数 | 衡量误报占比 |

### 3.3 特殊情况

| 场景 | 处理方式 |
|------|----------|
| 同一 GT 被多个 pred 命中 | 仅计一次命中，多余 pred 计为误报 |
| 同一 pred 命中多个 GT | 每个 GT 分别计一次命中 |
| GT 与 pred 无交集 | GT 计为漏检，pred 计为误报 |
| good 图上无 GT | 所有 pred 均计为误报（false positive） |

---

## 4. 分类评估规则

### 4.1 样本定义

每个**定位到的轮廓**为一个独立分类样本（P1-05）。

| 图像标签 | 轮廓来源 | 分类标签 |
|----------|----------|----------|
| `particle` | 定位输出 | `particle` |
| `scratch` | 定位输出 | `scratch` |
| `good` | 定位输出（误检） | `noise`（负类，P1-06） |

### 4.2 指标

| 指标 | 说明 |
|------|------|
| **Overall Accuracy** | 正确分类样本数 / 总样本数 |
| **每类 Precision** | 该类正确预测数 / 该类预测总数 |
| **每类 Recall** | 该类正确预测数 / 该类实际总数 |
| **每类 F1** | Precision 与 Recall 的调和均值 |
| **Macro Avg** | 各类指标的算术平均（不考虑样本量） |
| **Weighted Avg** | 各类指标的样本量加权平均 |
| **混淆矩阵** | N×N 矩阵，行=真实标签，列=预测标签 |

### 4.3 生产门禁（推荐）

| 指标 | 合格线 | 说明 |
|------|--------|------|
| Overall Accuracy | ≥ 0.80 | 合成集冒烟基准 |
| 每类 Recall | ≥ 0.60 | 不可有严重漏检类别 |
| Macro F1 | ≥ 0.70 | 类别均衡表现 |

> 合成集仅作冒烟；真实集指标需在 P1-10 基线报告中建立。

---

## 5. 端到端评估流程

```text
1. 加载数据集
2. 对每张图执行: Preprocess → Localize → Extract Features → Classify
3. 收集:
   - 定位层: 预测 bbox 列表 vs GT bbox 列表
   - 分类层: 每个轮廓的预测标签 vs 真实标签
4. 计算:
   - 定位召回率 / 误报数
   - 分类 Accuracy / P/R/F1 / 混淆矩阵
5. 输出:
   - results/eval_metrics.json (分类指标)
   - results/eval_predictions.csv (逐样本预测)
   - 控制台打印定位统计
```

---

## 6. 可复现性约定

| 项 | 约定 |
|----|------|
| 数据划分 | `random_state=42`，`stratify=y` |
| 模型随机种子 | RF: `random_state=42` |
| 特征版本 | `FEATURE_VERSION = "1.0"`（改特征必须 bump） |
| 评估脚本 | `python/train_eval.py`（自动输出 eval_metrics.json） |
| C++ 对齐 | `tests/test_core.py::test_cpp_feature_alignment` |

---

## 7. 后续演进

| 阶段 | 新增指标 |
|------|----------|
| P2 | 定位+分类一体化: mAP@IoU、F1@IoU |
| P2 | 分阶段耗时: P50/P95（preprocess/localize/feature/infer） |
| P3 | 吞吐量: WPH（wafers per hour） |

---

*本文档定义当前评估规则；若实现与本文冲突，以代码为准并回写更新。*
