# WaferDefectX 生产化改进计划 v2.0

| 字段 | 内容 |
|------|------|
| 文档版本 | 2.0 |
| 关联文档 | [DESIGN.md](./DESIGN.md), [PLAN.md](./PLAN.md) |
| 目标 | 准确率≥95% + 工业显微镜对接 + 1000-2000张/分钟吞吐 |
| 状态 | P0 已完成；P1+ 规划中 |
| 更新日期 | 2026-07-20 |

---

## 1. 本版目标（完成标准）

| 维度 | 验收标准 |
|------|----------|
| 准确率 | ≥95%（WM-811K 全量测试集） |
| 吞吐量 | ≥1000张/分钟（GPU模式） |
| 工业接口 | C API + gRPC + Python SDK |
| GPU加速 | 预处理+定位+推理全链路 CUDA |
| 部署 | TensorRT/ONNX Runtime GPU 可用 |

---

## 2. 优先级总览

```
P0  准确率提升：EfficientNet-B0 + 全量数据 + 训练调优
P1  GPU 加速：CUDA 预处理/定位 + TensorRT 推理
P2  工业接口：C API + gRPC + Python SDK + 相机对接
P3  吞吐量优化：多线程流水线 + Batch 推理 + 内存优化
```

---

## 3. P0 — 准确率提升（93.83% → 95%）

目标：通过模型升级和训练策略优化，将准确率提升至95%以上。

### 3.1 数据增强

- [ ] **P0-01** 增大全量数据使用（当前仅用 5193 张，应使用全量 172,950 张）
- [ ] **P0-02** 高分辨率输入 64×64 → 128×128
- [ ] **P0-03** 增强策略：MixUp(α=0.2) + RandomErasing(p=0.3) + ColorJitter
- [ ] **P0-04** Hard Example Mining：聚焦易混淆样本

### 3.2 模型架构

- [ ] **P0-05** 实现 EfficientNet-B0（灰度转3通道 + ImageNet 预训练）
- [ ] **P0-06** 可选：ResNet-18 + SE 注意力模块
- [ ] **P0-07** 模型集成框架（支持多模型投票）

### 3.3 训练策略

- [ ] **P0-08** 损失函数：Label Smoothing CE (smoothing=0.1)
- [ ] **P0-09** 优化器：AdamW (lr=1e-4, weight_decay=0.05)
- [ ] **P0-10** 调度器：CosineAnnealing with Warmup
- [ ] **P0-11** 训练 60-100 epochs，全量数据

### 3.4 推理增强

- [ ] **P0-12** TTA（Test-Time Augmentation）：水平翻转 + 旋转
- [ ] **P0-13** 置信度阈值调优
- [ ] **P0-14** 3模型集成推理

**P0 退出条件**：WM-811K 测试集准确率≥95%，Macro F1≥0.94。

---

## 4. P1 — GPU 加速（CUDA 全链路）

目标：预处理、定位、推理全链路支持 GPU 加速。

### 4.1 预处理 GPU 化

- [ ] **P1-01** 实现 `cpp/preprocess_cuda.cu`：Gray Convert + GaussianBlur + CLAHE
- [ ] **P1-02** CUDA Morphology 操作
- [ ] **P1-03** GpuMat 零拷贝传输优化

### 4.2 定位 GPU 化

- [ ] **P1-04** 实现 `cpp/localization_cuda.cu`：Canny 边缘检测
- [ ] **P1-05** CUDA 轮廓发现与过滤
- [ ] **P1-06** GPU 特征提取（并行计算7维特征）

### 4.3 推理引擎

- [ ] **P1-07** TensorRT FP16 推理引擎
- [ ] **P1-08** ONNX Runtime CUDA Provider 集成
- [ ] **P1-09** Batch 推理支持（batch=32-64）
- [ ] **P1-10** 动态 Batching（自动收集 ROI 后提交）

### 4.4 性能验证

- [ ] **P1-11** GPU 预处理延迟基准测试
- [ ] **P1-12** TensorRT 推理延迟基准测试
- [ ] **P1-13** 端到端 GPU 吞吐量测试

**P1 退出条件**：GPU 模式下单图预处理≤0.3ms，推理≤1ms。

---

## 5. P2 — 工业接口（显微镜对接）

目标：提供完整的工业级 API 接口，支持显微镜和相机 SDK 集成。

### 5.1 C API 设计

- [ ] **P2-01** 设计 `wafer_defect_api.h` 头文件
- [ ] **P2-02** 同步接口：`wafer_detect(buffer, result)`
- [ ] **P2-03** 异步接口：`wafer_detect_async(buffer, callback)`
- [ ] **P2-04** 批量接口：`wafer_detect_batch(buffers[], results[])`
- [ ] **P2-05** 模型热加载：`wafer_reload_model(path)`
- [ ] **P2-06** 配置管理：`wafer_set_config(key, value)`

### 5.2 gRPC 服务

- [ ] **P2-07** 设计 `wafer_defect.proto` 接口定义
- [ ] **P2-08** 实现 `Detect` RPC（单张检测）
- [ ] **P2-09** 实现 `DetectStream` RPC（流式检测）
- [ ] **P2-10** 实现 `ReloadModel` RPC（模型热更新）
- [ ] **P2-11** 健康检查与状态查询接口

### 5.3 Python SDK

- [ ] **P2-12** 封装 `WaferDetector` 类
- [ ] **P2-13** 单张检测：`detect(image) → Result`
- [ ] **P2-14** 批量检测：`detect_batch(images) → List[Result]`
- [ ] **P2-15** 流式检测：`detect_stream(camera) → Iterator`
- [ ] **P2-16** 相机配置工具：`from_camera(config)`

### 5.4 相机对接

- [ ] **P2-17** GenICam/GigE Vision 对接示例
- [ ] **P2-18** TWAIN 数据源对接示例
- [ ] **P2-19** 共享内存零拷贝传输
- [ ] **P2-20** 坐标标定（pixel → μm 转换）

**P2 退出条件**：C API + gRPC + Python SDK 可用，GenICam 示例可运行。

---

## 6. P3 — 吞吐量优化（1000-2000张/分钟）

目标：通过流水线并行和内存优化，达到目标吞吐量。

### 6.1 流水线架构

- [ ] **P3-01** 实现生产者-消费者流水线架构
- [ ] **P3-02** 并发队列：`ConcurrentQueue<T>`
- [ ] **P3-03** 多阶段流水线：采集→预处理→定位→推理→回调

### 6.2 内存优化

- [ ] **P3-04** Pinned Memory（锁定内存）用于 DMA 传输
- [ ] **P3-05** 内存池复用（避免频繁分配/释放）
- [ ] **P3-06** GpuMat 对象池

### 6.3 并发优化

- [ ] **P3-07** 多线程预处理（利用多核 CPU）
- [ ] **P3-08** 异步推理（Pipeline 并行）
- [ ] **P3-09** 动态 Batch 调度器

### 6.4 性能验证

- [ ] **P3-10** 吞吐量基准测试（目标：≥1000张/分钟）
- [ ] **P3-11** 延迟分布测试（P50/P95/P99）
- [ ] **P3-12** 内存占用监控
- [ ] **P3-13** CPU/GPU 利用率监控

**P3 退出条件**：GPU 模式吞吐量≥1000张/分钟，P95 延迟≤50ms。

---

## 7. 里程碑

| 里程碑 | 包含 | 产出 | 预计周期 |
|--------|------|------|----------|
| M0 | P0 全部 | 95%+ 准确率模型 | 1-2周 |
| M1 | P1 全部 | GPU 全链路加速 | 2-3周 |
| M2 | P2 核心 | C API + Python SDK | 1-2周 |
| M3 | P2 剩余 | gRPC + 相机示例 | 1周 |
| M4 | P3 全部 | 1000+张/分钟吞吐 | 1-2周 |

---

## 8. Todo 索引（按 ID）

### P0

| ID | Todo | 状态 |
|----|------|------|
| P0-01 | 全量数据训练 | 待办 |
| P0-02 | 高分辨率输入 128×128 | 待办 |
| P0-03 | 增强策略 MixUp/RandomErasing | 待办 |
| P0-04 | Hard Example Mining | 待办 |
| P0-05 | EfficientNet-B0 实现 | 待办 |
| P0-06 | ResNet-18 + SE 注意力 | 待办 |
| P0-07 | 模型集成框架 | 待办 |
| P0-08 | Label Smoothing CE | 待办 |
| P0-09 | AdamW 优化器 | 待办 |
| P0-10 | CosineAnnealing + Warmup | 待办 |
| P0-11 | 全量数据训练 60-100 epochs | 待办 |
| P0-12 | TTA 推理 | 待办 |
| P0-13 | 置信度阈值调优 | 待办 |
| P0-14 | 3模型集成推理 | 待办 |

### P1

| ID | Todo | 状态 |
|----|------|------|
| P1-01 | CUDA 预处理实现 | 待办 |
| P1-02 | CUDA Morphology | 待办 |
| P1-03 | GpuMat 零拷贝 | 待办 |
| P1-04 | CUDA Canny 定位 | 待办 |
| P1-05 | CUDA 轮廓发现 | 待办 |
| P1-06 | GPU 特征提取 | 待办 |
| P1-07 | TensorRT FP16 引擎 | 待办 |
| P1-08 | ONNX Runtime CUDA | 待办 |
| P1-09 | Batch 推理支持 | 待办 |
| P1-10 | 动态 Batching | 待办 |
| P1-11 | GPU 预处理基准测试 | 待办 |
| P1-12 | TensorRT 基准测试 | 待办 |
| P1-13 | 端到端 GPU 吞吐测试 | 待办 |

### P2

| ID | Todo | 状态 |
|----|------|------|
| P2-01 | wafer_defect_api.h 设计 | 待办 |
| P2-02 | 同步检测接口 | 待办 |
| P2-03 | 异步检测接口 | 待办 |
| P2-04 | 批量检测接口 | 待办 |
| P2-05 | 模型热加载 | 待办 |
| P2-06 | 配置管理接口 | 待办 |
| P2-07 | proto 接口定义 | 待办 |
| P2-08 | Detect RPC | 待办 |
| P2-09 | DetectStream RPC | 待办 |
| P2-10 | ReloadModel RPC | 待办 |
| P2-11 | 健康检查接口 | 待办 |
| P2-12 | Python SDK 封装 | 待办 |
| P2-13 | 单张检测 | 待办 |
| P2-14 | 批量检测 | 待办 |
| P2-15 | 流式检测 | 待办 |
| P2-16 | 相机配置工具 | 待办 |
| P2-17 | GenICam 示例 | 待办 |
| P2-18 | TWAIN 示例 | 待办 |
| P2-19 | 共享内存传输 | 待办 |
| P2-20 | 坐标标定 | 待办 |

### P3

| ID | Todo | 状态 |
|----|------|------|
| P3-01 | 生产者-消费者流水线 | 待办 |
| P3-02 | 并发队列实现 | 待办 |
| P3-03 | 多阶段流水线 | 待办 |
| P3-04 | Pinned Memory | 待办 |
| P3-05 | 内存池复用 | 待办 |
| P3-06 | GpuMat 对象池 | 待办 |
| P3-07 | 多线程预处理 | 待办 |
| P3-08 | 异步推理 | 待办 |
| P3-09 | 动态 Batch 调度器 | 待办 |
| P3-10 | 吞吐量基准测试 | 待办 |
| P3-11 | 延迟分布测试 | 待办 |
| P3-12 | 内存占用监控 | 待办 |
| P3-13 | CPU/GPU 利用率监控 | 待办 |

---

## 9. 维护约定

1. 完成某项时将对应 `- [ ]` 改为 `- [x]`，并同步 §8 状态为「完成」。
2. 若实现与 DESIGN 冲突，先改代码与测试，再回写 DESIGN，最后更新本 PLAN。
3. 新增工作项时分配新 ID（如 `P0-15`），勿复用已完成 ID。
4. 生产默认路径变更（模型架构、特征维度、类别）必须新增或更新 ADR。

---

*本文档是面向「95%准确率 + 工业显微镜对接 + 高吞吐量」的执行计划；算法细节与现状以 DESIGN.md 为准。*