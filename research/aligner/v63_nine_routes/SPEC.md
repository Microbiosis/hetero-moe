# V63.0 — 九条路线执行矩阵

> **文档 ID**: HMF-NINE-ROUTES-1.0
> **日期**: 2026-11-12（补登记；工作完成于 2026-11-03）
> **状态**: 代码就位，**Kaggle 批量验证待执行**
> **归属**: aligner 线，接 v62
> **一句话**: v62 把「共享空间该多大」查清了，v63 一次性铺开九条自修改/增强路线，回答「拿到这个空间之后能拿它做什么」。

---

## 0. 本代性质（先说清楚，避免误引）

**本代没有端到端数字。**

九条路线中多数需要重训（批 B）或外部 API（LLM 解码，批 A 之一），Kaggle 批量验证**尚未执行**。按项目纪律「无支撑的数字不得对外引用」，本代：

- `claim.asserts` 写的是**机制可加载且接口自洽**，不是精度提升；
- `results/metrics.json` 只记录可本地复算的结构性事实；
- 端到端产物的落盘位置已在 §5 写明，等验证后回填。

这与 v62.0–v62.2 的处理方式一致：那三代也因 router bug 而使端到端数字作废，但**作废的数字仍被保留在 SPEC 里并标注原因**——未验证不是没发生。

---

## 1. 九条路线

| # | 路线 | 类别 | 代码 | 状态 |
|---|---|---|---|---|
| 1 | Q 槽自修复 | 路由 | `v62/self_repair.py` | 前提不成立，见 §3 |
| 2 | 元投影网络（HyperLinear） | 路由 | `v62/hyper_proj.py` | 代码就位 `USE_HYPER=1` |
| 3 | 快速权重专家 | 路由 | `v62/fusion.py` | 代码就位 `USE_FAST=1` |
| 4 | 硬约束路由 | 路由 | `routing/hard_router.py` | 本包内 |
| 5 | 单模态鲁棒性 | 鲁棒性 | `robustness/modality_dropout.py` | 本包内 |
| 6 | 少样本迁移 | 鲁棒性 | `robustness/transfer_eval.py` | 本包内 |
| 7 | LLM 语义解码 | 可解释 | `interpret/decode_slots.py` | 本包内 |
| 8 | 动态路由 | 路由 | `routing/dynamic_router.py` | 本包内 |
| 9 | INT4 量化 | 部署 | `deploy/quantize_int4.py` | 本包内 |
| 10 | 轻量底座 | 部署 | `deploy/lite_backbones.py` | 本包内 |
| 11 | 自动化审计 | 审计 | `audit/audit_v63.py` | 本包内 |

> 注：README 里列了 11 行（含 #2/#3 复用 v62 模块），「九条」指本代**新写**的路线集合。

---

## 2. 两个路由器的机制语义（本代唯一可本地证伪的部分）

### 2.1 HardLoadRouter（路线 4）

替代 v62 的纯熵正则软约束：先做 top-k 硬选择，再加**批次级容量惩罚**。

```text
capacity    = capacity_factor / K              # 每槽上限（和归一化）
mean_p      = softmax(logits / T).mean(0,1)   # 每槽期望负载
penalty     = relu(mean_p - capacity).sum()
gate        = top-k(probs) 归一化
```

已验证（本地，见 `tests/test_package.py`）：

- 每 token 恰好 2 槽非零，权重和为 1；
- 偏斜输入 ⇒ 惩罚 > 0（实测 0.6874）；完全均衡 ⇒ 惩罚 = 0（实测 0.00e+00）。

### 2.2 DynamicSlotRouter（路线 8）

逐 token 自适应槽数：高不确定性 token 用更多槽，确定性 token 稀疏化。

- 训练态：Gumbel-sigmoid，可微（实测梯度回传非零）；
- 推理态：阈值化，确定性（两次前向逐位一致）；
- 激活槽数恒在 `[min_slots, max_slots]` 内，权重和为 1。

---

## 3. 一条已证伪的前提

**「Q 槽塌缩到接近 0，需要自修复机制介入」——前提不成立。**

v62.3 / v62.4 实测 Q 槽 EMA 全程 **0.11–0.25**，从未跌破 `self_repair.py` 的 0.01 触发阈值，**自修复机制在真实轨迹上一次都没触发**。

补充一条更一般的证伪：槽塌缩**不是**普遍现象，只出现在 v62.0–v62.2（router lazy 创建 / 槽上做 top-k 两种 bug），v62.3 修好后四槽全部正常分工（槽门控熵 0.647）。⇒ 自修复是针对特定 bug 的保险，不是通用机制。

---

## 4. 审计脚本固化 v62 的关键判据（路线 11）

`audit/audit_v63.py` 把人工切换的评价器与协议固化为可复用模块，8 项判据：

| 项 | 判据 | 阈值来源 |
|---|---|---|
| A1 | 底座间 CKA | > 零分布上限 |
| A2 | 有效秩 / 联合 PCA 足迹重叠 | < 0.06 |
| A3 | 路由负载均衡系数 | > 0.3（v62 实测 0.612） |
| A4 | 槽门控熵 | > 0.4（v62 实测 0.647） |
| A5 | held-out 准确率 | > 0.85（v62.3 实测 0.9076） |
| A6 | 投影保真度（rank=256） | ≈ 1.0（v62 实测 0.9998） |
| A7 | 跨底座余弦一致性 | > 0.5 |
| A8 | 泄漏检查 | 训练/测试不重叠 |

A3/A4 的阈值直接取自 v62 实测值，不是拍脑袋。`cka` 与 `effective_rank` 两个数学工具已在本地单测中验证（cka(X,X)=1、effective_rank 对 rank-4 矩阵给 3）。

---

## 5. 待执行：Kaggle 批量验证

**批 A（评估类，10 分钟内，无需重训）**
```bash
python robustness/modality_dropout.py --ckpt v62.4.pt --data capture.npz
python audit/audit_v63.py --ckpt v62.4.pt --data capture.npz
python interpret/decode_slots.py --ckpt v62.4.pt --out slot_prompts.jsonl
```

**批 B（训练类，各 10–15 分钟）**
```bash
USE_HYPER=1 USE_FAST=1        # B1 元投影 + 快速权重
# B2 硬约束路由（替换 proj_space.router）
# B3 动态路由（替换 router）
```

**产物落盘位置**
- 评估类 → `research/aligner/v63_nine_routes/results/*.json`
- 训练类 → HF `redmingwei/HeteroMoEProjSpace-v2/checkpoints/v63.*`
- 审计报告 → `research/aligner/v63_nine_routes/audit/AUDIT.md`

验证完成后应回填 `results/metrics.json` 的 `pending` 段，并把 `status` 从 `code_ready_pending_batch_run` 改为实测结论。

---

## 6. 复现

```bash
python research/aligner/v63_nine_routes/tests/test_package.py     # 8/8，本地 CPU
```

本代测试**不触碰 v62 权重**（需 GPU + 已训练 ckpt），只验证机制语义。

---

*本代无端到端数字；任何引用不得把「代码就位」说成「已验证有效」。*
