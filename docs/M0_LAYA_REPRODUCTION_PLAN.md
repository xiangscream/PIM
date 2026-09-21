# M0 — Laya Reproduction

## 1. 目标

M0 不训练摄影模型。目标是亲手跑通并理解一个完整的 typed probabilistic decision 模型工作流：

```text
state + typed questions
        ↓
tokenization / option rendering
        ↓
ModernBERT encoder + decision head
        ↓
logits / probability distribution
        ↓
training
        ↓
post-training calibration
        ↓
evaluation / inference
```

M0 完成后，应能解释 Laya 在训练什么、typed decision 如何编码、概率如何产生、calibration 为什么必要，以及这些机制哪些可以迁移到 PIM。

Laya 是 PIM 的第一个实验 backend，不等于 PIM 本身。

## 2. 固定上游基线

为避免上游快速变化导致复现漂移，M0 以以下版本为基线：

- upstream repository: `NandhaKishorM/laya`
- upstream commit: `42626c348753fbb17572a813127df2278a1ec527`
- package version at baseline: `0.3.4`
- model: `convaiinnovations/laya`
- benchmark dataset: `LocalLLaMA/typed-decisions`
- official reproduction notebook:
  `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`

不要在 M0 过程中静默升级 Laya、Transformers 或训练逻辑。任何偏离都单独记录。

## 3. 官方训练事实

当前官方 notebook 的复现路径：

- base model: Laya, 421M parameters
- training set: 1,200 cases / 6,000 typed decisions
- test set: 400 cases / 2,000 typed decisions
- accelerator: 2 × NVIDIA T4
- distributed training: DDP / `torchrun --nproc_per_node=2`
- epochs: 4
- micro batch: 8 / GPU
- gradient accumulation: 4
- effective batch: 64 sequences
- encoder LR: `2.5e-5`
- head LR: `1.0e-4`
- objective: RLCD-style policy term + soft cross-entropy guidance
- post-training calibration: separate temperatures for choice / score / noul

这些值在 M0 第一轮复现中保持不变。

## 4. 阶段

### M0-A — Upstream Anatomy

目标：不训练，先能解释数据怎样进入模型。

任务：

1. checkout 固定 upstream commit；
2. 建立独立 Python 环境；
3. 安装固定版本的 Laya 与依赖；
4. 阅读并标注：
   - `build_sequence`
   - `render_options`
   - `build_model`
   - `proper_reward`
   - `Agent.predict`
   - temperature / calibration 相关代码；
5. 取 typed-decisions 中 1 个 case，手工追踪：
   `state → question → rendered options → token sequence → marker positions → logits → probabilities`。

交付：

- `docs/m0/LAYA_ANATOMY.md`
- 一个最小 inspect 脚本，能打印单 case 的中间表示；
- 不修改上游模型逻辑。

Gate A：

用户能够用自己的话解释：
- choice / score / noul 有什么区别；
- marker positions 在做什么；
- 为什么它不是 autoregressive generation；
- logits 怎样变成 typed probability。

### M0-B — Base Inference Baseline

目标：先证明本地能够正确加载和调用模型。

任务：

1. 加载 `convaiinnovations/laya`；
2. 运行官方 Quickstart；
3. 对 benchmark 抽样 20–50 cases；
4. 记录：
   - latency；
   - prediction；
   - confidence；
   - gold；
   - correctness；
5. 保存环境信息：Python / torch / CUDA / GPU / transformers / laya。

交付：

- `experiments/m0_base_inference/`
- `results/m0/base_inference.json`
- `docs/m0/BASELINE.md`

Gate B：

同一固定输入可重复得到稳定结构输出，评测脚本能够独立计算 accuracy 与至少一种概率指标。

### M0-C — Official Fine-tune Reproduction

目标：不创新，先复现官方训练。

优先直接运行固定 notebook 的训练逻辑。

注意：

- 官方复现使用 2×T4；
- 如果本机硬件无法忠实运行，保留本地 inference / inspect，将正式训练放到等价双 GPU 环境；
- 不为了“跑起来”擅自改成 LoRA、单卡、减数据、减 epoch，然后仍称为官方复现；
- 若做缩小版 smoke test，必须标记为 `smoke`，不能作为 reproduction result。

交付：

- 完整训练日志；
- checkpoint；
- 训练配置快照；
- 环境快照；
- loss / reward progression 摘要。

Gate C：

官方训练流程从原始 dataset 到可加载 checkpoint 完整跑通，且没有未记录的训练逻辑偏离。

### M0-D — Evaluation & Calibration

目标：理解“预测正确”和“概率可信”不是一回事。

至少计算：

- accuracy；
- Brier score；
- ECE；
- p50 / p95 latency；
- score MAE（对 score primitive）；
- calibration 前后对比。

做一个小型 reliability table / diagram：

```text
confidence bucket → empirical accuracy
```

同时人工检查若干：
- high-confidence correct；
- high-confidence wrong；
- low-confidence correct；
- ambiguous cases。

交付：

- `results/m0/reproduction_report.json`
- `docs/m0/CALIBRATION.md`

Gate D：

能够说明 temperature scaling 改变了什么、没有改变什么，以及为什么未来 PIM 的自动执行阈值不能只看 argmax accuracy。

### M0-E — PIM Transfer Note

M0 结束时才讨论摄影。

回答：

1. Laya 哪些机制可以直接复用到 PIM？
2. 哪些机制只是 Laya 当前实现细节？
3. Photography State 应怎样对应 Laya 的 state？
4. Verifier 最适合映射成 noul / score / choice 中哪种？
5. `WAIT / REFRAME / ESCALATE` 最适合怎样定义？
6. 哪些摄影问题如果规则系统就能解决，不应该交给 Laya？

交付：

- `docs/m0/PIM_TRANSFER.md`

Gate M0：

进入 M1 前，必须同时满足：
- 能运行；
- 能训练；
- 能评测；
- 能解释；
- 能指出 Laya 的局限；
- 尚未把 Tail2-specific 字段写进 PIM 的长期接口。

## 5. M0 明确不做

- 不训练真正 Photography PIM；
- 不定义最终 Photography State schema；
- 不接 Tail2 真机闭环；
- 不加入图像输入；
- 不改 Laya 架构；
- 不因为指标漂亮而跳过 calibration；
- 不以官方 README 数字代替自己的复现结果。

## 6. DeepSeek / OpenCode 的职责

本地 Agent 作为 executor / investigator：

- 环境搭建；
- 固定 upstream；
- 运行代码；
- 收集日志与环境事实；
- 写最小 inspect/eval 工具；
- 报告失败与差异；
- 不自行改变研究问题和 Gate。

网页侧负责：

- 架构与实验设计；
- 结果审查；
- 概念解释；
- 判断偏离是否仍可称为 reproduction；
- 每个 Gate 后做一次 Feynman 式复盘。

## 7. 每轮 handoff 格式

本地 Agent 每轮只需汇报：

```text
Commit / Environment
What ran
Observed result
Expected vs actual
Artifacts
Open questions / blockers
No silent deviations
```

失败同样是有效结果，不要为获得绿色状态而修改基线。
