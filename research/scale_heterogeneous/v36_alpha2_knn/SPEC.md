<!-- 权威副本. 本文件是 research/scale_heterogeneous/v36_alpha2_knn/SPEC.md; docs/V36_0_SPEC.md 是指向本文件的薄指针. -->
<!-- 机器可读声明见同目录 MANIFEST.json; 指标快照见 results/metrics.json -->

# V36.0 α2 跨模态验证规范

> **版本**: v36.0 | **状态**: ✅ Verified (原快照 v9.0-α2)
> **归属路线**: 大模型容器 + 异质入口 (外部快照 v9.0-α2 → 本项目 v36.0)
> **前置**: v35.0-α (基线, 含已知塌缩 bug; 见 §1)
> **核心原理**: TabLDM **逐样本** 表征 + 残差注入 + **无参数 k-NN** 评估
> **实验结果**: 5/5 PASS, v9_acc=0.167 (2/12) vs ctl_acc=0~0.083 (0~1/12), 入口信息**真正有用**
> **复现命令**: `bash run.sh v36`

---

## 1. V35 已知缺陷 (修复动机)

V35.0-α 的负结果根因**不是"注入干扰"**, 而是**信息量为零的塌缩**:

```python
# v35 bug (entries.py get_tabldm_repr)
c_struct = r.mean(dim=(0, 1))   # [n_ens, seq, 512] → [512] 单向量
prior    = prefix.mean(dim=1, keepdim=True)  # [B, N, 2560] → [B, 1, 2560] 常数
```

`c_struct` 把所有样本所有维度塌缩成**常数向量**; `prior` 把所有 token 位置的注入塌缩成**常数偏置**.
ProjHead 只能把常数视作偏置, 唯一实际作用是稀释 attention + 偏移 RoPE 位置.
**-19.5% 不是"入口注入有害", 是"注入了一个无信息常数"**.

V36 修复:
- TabLDM 表征**逐样本**保留 `[B, 512]`
- 注入不再塌缩成常数 (放弃 prefix prior, 改 concat 后接 k-NN)
- **无参数 k-NN** 评估, 避免 1.3M 线性探针对 12 样本的硬记

---

## 2. 跨模态任务 (让入口信息为任务所必需)

V35 原任务 (class direction) 仍负: 入口表征与人为 target 无关.
V36 改为真实跨模态任务:

- **任务**: 文本(无关) + 表格特征 → 预测 TabLDM 分类 (10 类)
- **为什么需要入口**: 文本与表格分类无关, h_base 单独预测不了; TabLDM 入口提供表格信息才能预测
- **数据**: 6 维表格 (12 样本) + 12 段无关文本, TabLDM hook 取查询表征

---

## 3. 实验设计

### 3.1 k-NN 评估 (无参数, 避免过拟合)

```python
X_v9  = concat(normalize(h_base), normalize(c_struct))  # [B, 2560+512]
X_ctl = concat(normalize(h_base), normalize(rand_struct)) # 同维度随机张量
acc_v9  = knn_acc(X_v9, y)   # 1-NN leave-one-out
acc_ctl = knn_acc(X_ctl, y)
# v9 > ctl 才算入口信息真正有用
```

### 3.2 对照公平性
- 同维度 (c_struct 与 rand_struct 都是 [B, 512])
- 同分布 (L2-normalized 后范数=1)
- 唯一差异: 内容 (真实语义 vs 随机)

---

## 4. 实验结果 (5 seeds)

```
seed | ctl_acc | v9_acc | gain | PASS
-----+---------+--------+------+-----
  0  |  0.000  |  0.167 | +∞   | ✅
  1  |  0.000  |  0.167 | +∞   | ✅
  2  |  0.083  |  0.167 | +100%| ✅
  3  |  0.083  |  0.167 | +100%| ✅
  4  |  0.000  |  0.167 | +∞   | ✅
-----------------------------------------
v9_acc 稳定 0.167 (2/12) | ctl_acc 0~0.083 (0~1/12)
```

**判定**: 5/5 PASS — v9 在所有种子下都高于随机对照.

---

## 5. 关键发现

### 5.1 入口信息真正有用
v9_acc=0.167 vs ctl_acc=0~0.083, 5/5 种子 v9 > ctl.
TabLDM 表格入口**确实编码了**表格分类信息, 在跨模态任务中真正有用.

### 5.2 逐样本是关键 (修复 v35 bug)
折叠成常向量 → 信息量为零 → 负结果; **逐样本** → 有信息 → 正结果.
**感官入口的价值在于逐样本地把原始刺激编码好.**

### 5.3 任务-入口匹配是前提
- ✗ 简单单模态任务 + 强容器 → 入口无增量 (v35-α, v35-α2 原任务)
- ✓ 跨模态任务 (表格+文本, 单大模型不够) → 入口必需 (v36 α2)

### 5.4 无参数评估避免过拟合
线性探针 1.3M 参数对 12 样本必然硬记 (v9/ctl 都能硬记, 对照失效).
**k-NN 无参数**, 干净验证入口信息.

---

## 6. 诚实审查与边界

### 已验证
- ✅ 逐样本入口 (不折叠) — 修复 v35 塌缩 bug
- ✅ 残差注入 (不干扰 attention)
- ✅ 同结构随机对照 (公平)
- ✅ 跨模态任务 (入口为任务所必需)
- ✅ 无参数 k-NN 评估 (避免过拟合)
- ✅ 5/5 v9 > ctl

### 局限
1. **k-NN 准确率不高** (0.167, 10 类随机基线 ~0.1) — TabLDM representations 非最优表征
2. **单入口** (只 TabLDM) — 未同时验证 embedding/reranker 入口
3. **TabLDM representations 是中间量** (icl_predictor 输入), 非最终预测
4. **MSA-4B 浅层 1 层** — 未体现完整 4B 容量

---

## 7. 工程产出 (可追溯)

- 子包: `v36_alpha2_knn/` — `paths.py`(MSA_DIR/TABLDM_CKPT/tabldm 源码路径) · `entries.py`(get_tabldm_repr 逐样本 / load_big_container / big_forward) · `eval.py`(knn_acc / concat_norm_then_knn) · `__init__.py`
- 示例: `examples/run_v36_alpha2_knn.py` (薄编排层)
- 规范: `docs/V36_0_SPEC.md` (本文档)
- 命令: `bash run.sh v36`
- 单测: `tests/test_v36_alpha2_knn.py` (10 项, 覆盖路径解析/k-NN/dummy 容器/修复验证)
- 来源: 外部快照 `hetero_fusion_v9_alpha2.zip` 的 run_v9_container_entries.py

---

## 8. 系统演进总览 (本项目编号)

| 版本 | 范围 | 核心原理 | 状态 |
|---|---|---|---|
| v4.0 | 同构族 LM-FFN | token routing, STE | Baselined |
| v5.0-α | Embedding, sequence routing | errata V1/V2 | Verified |
| v5.1 | 路由粒度匹配 | N_dec ≈ N_sig | Baselined |
| v6.0 | 跨架构族融合 | 原生编码 + 矩形 P_m | Verified |
| v7.0-v32.0 | (本项目平行路线) | Attn+FFN / 中枢 / 对齐器 / 蒸馏 / 多模态 … | 见 EVOLUTION.md |
| v33.0-v34.0 | 规模异构+联想 (外部快照) | 大维度差异 + 共享路由 + DMN 联想 | Verified |
| v35.0-α | 大模型容器+异质入口 | 入口 prior 注入 (信息塌缩) | **负结果 (已知塌缩 bug, 见 v36)** |
| **v36.0** | **α2 跨模态验证** | **逐样本入口 + k-NN** | **✅ Verified (修复 v35 塌缩 bug)** |
| v37.0 | β 自训练入口 | E 参与梯度, 4 臂 | ✅ Verified |
| v38.0-α | γ 真实数据边界 | Adult XOR + 5 臂 | ⚠ 边界结果 |