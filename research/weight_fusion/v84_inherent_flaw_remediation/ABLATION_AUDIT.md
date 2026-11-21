# V84.0 附录 — Exp4/Exp5 消融复现与「β 墙」归因审计

> 本文件是 v84 包的**外部复现附录**（addendum），不替代 [`SPEC.md`](SPEC.md) 与
> [`results/metrics.json`](results/metrics.json) 的权威记录。本轮**未改动**任何上游产物、
> 断言或脚本，只新增本文件、可复现脚本与一份原始输出。

| 入口 | 位置 |
|---|---|
| 复现脚本 | [`tools/ablation_v84_exp4_exp5_replication.py`](../../../tools/ablation_v84_exp4_exp5_replication.py) |
| 原始输出 | [`results/ablation_exp4_exp5_replication.json`](results/ablation_exp4_exp5_replication.json) |
| 被审计对象 | [`audit_remediation.py`](audit_remediation.py)（Exp4 / Exp5）· [`MANIFEST.json`](MANIFEST.json) |

**环境**：repo HEAD `7ca0c12`（含上游 `772e0c3`）· torch `2.10.0+cpu` · Python `3.12.3` · Linux aarch64
· 复现脚本 `SEED = 42`，与 `audit_remediation.py` 同值，且逐行复刻其数据生成、三集切分、
专家底座训练与门禁协议，只替换「专家个数 / 专家是否可训练 / 门控 / LoRA / 跨层梯度」。

---

## 1. 先证明复刻是忠实的

| 实验臂 | 本复现 | 官方 `results/metrics.json` | 一致性 |
|---|---|---|---|
| Exp4 v5（2 冻结切片专家 + gate8 + LoRA6） | `test_acc = 1.0000` | `fusion.accuracy = 1.0` | ✅ |
| Exp4 v4（gate0 / LoRA0 / `detach_input=True`） | `0.8666666746139526` | `v4_legacy_frozen_test_acc = 0.8666666746139526` | ✅ 逐位一致 |
| Exp5 v5（2 冻结 LM 专家 + gate6 + LoRA4） | `nll 0.8377615809440613` / `ppl 2.3111877755735972` | `nll 0.8352254033088684` / `ppl 2.3053336195541645` | ⚠️ 差 0.0059（+0.26%） |

Exp4 两个官方臂**逐位复现**，说明消融与官方跑的是同一份数据与同一条训练协议。
Exp5 在本机存在 +0.26% 的偏差（CPU 线程归约顺序），**不改变结论方向**，但也意味着
官方 PPL 数字不是逐位可复现的（见 §6）。

---

## 2. 发现 A：上游自己的 K=1 路由修复抬高了「冻结基线」，宣称增益缩水 42%

`772e0c3` 修掉了 `core/router.py` 里 `top_k=1` + `renorm_topk=True` 的零梯度死锁
（改用 STE 直通）。但 Exp4 的 **v4「冻结」臂用的正是 `top_k=1` 的 `HeteroFusionLayer`**，
而 `core/fusion.py::trainable_parameters` 恒含 `W_router`：

- 修复前：`W_router` 拿不到任何梯度 → 停在随机初始化 → v4 臂 0.7667；
- 修复后：`W_router` 开始训练 → v4 臂自己涨到 0.8667。

A/B（只换 `core/router.py`，其余完全不动，跑权威 `audit_remediation.py`）：

| `core/router.py` | v4 冻结臂 | v5 臂 | Δ(v5 − v4) |
|---|---|---|---|
| `76e1402`（修复前） | `0.7666666507720947` | `0.9944444298744202` | **+22.78pp** |
| `772e0c3`（修复后，当前 HEAD） | `0.8666666746139526` | `1.0000` | **+13.33pp** |

**结论**：官方口径的「Δ=+22.8%」是在「v4 的路由也被冻着」的条件下取得的。同一份代码在
上游自己的修复落地后，v4 基线自己涨了 10pp，Δ 缩到 +13.3pp（缩水 42%）。同时
`v4_legacy_frozen_test_acc` 这个字段名已经名不副实——它的 `W_router` 并不 frozen。

```bash
# 复现 A/B（记得还原工作树）
git show 76e1402:hetero_moe/core/router.py > hetero_moe/core/router.py
python research/weight_fusion/v84_inherent_flaw_remediation/audit_remediation.py   # → v4 0.7667 / v5 0.9944
git checkout hetero_moe/core/router.py
python research/weight_fusion/v84_inherent_flaw_remediation/audit_remediation.py   # → v4 0.8667 / v5 1.0000
```

---

## 3. 发现 B：Exp4 消融 —— 单专家、零融合，能力与 v5 持平

同数据、同切分、同专家底座、同训练循环，只改专家个数与可训练性：

| 臂 | 专家 | 底座 | 门控 / LoRA | test_acc | 可训练参数 |
|---|---|---|---|---|---|
| v5 上游 | 2（各看 10 维切片） | 冻结 | gate8 + LoRA6 | `1.0000` | 3818 |
| v4 冻结 | 2（各看 10 维切片） | 冻结 | 无 | `0.8667` | 1946 |
| 单专家·全输入·可训练 | 1（看全 36 维） | 可训练 | 无 | `0.9889` | 7917 |
| **单专家·全输入·随机冻结底座·只训线性头** | 1（看全 36 维） | 冻结 | 无 | **`1.0000`** | **1261** |
| 单专家·全输入·冻结 + LoRA6 | 1（看全 36 维） | 冻结 | LoRA6 | `1.0000` | 1669 |

官方 β 墙基线 `best_single = 0.5222`（两个**只看 10 维切片**的冻结专家）。而一个
**1261 参数、单专家、底座随机冻结、只训一个线性头**的模型就打到了 `1.0000`，与 v5 持平。

**所谓「β 墙」不是融合机制的性质**，而是「把专家视野限死在 10 维切片 + 冻结底座」这一
人工构造的性质；一旦把视野放开、把底座解冻，墙就不存在了。

---

## 4. 发现 C：Exp5 消融 —— 单专家、零融合，PPL 反而优于 v5

| 臂 | test_nll | test_ppl | 可训练参数 |
|---|---|---|---|
| v5 上游（2 LM 专家 + gate6 + LoRA4） | `0.8377615809440613` | `2.3111877755735972` | 3378 |
| **单专家·全输入·可训练·无融合** | **`0.7876035571098328`** | **`2.198122434490068`** | 6241 |
| 单专家·冻结 + LoRA4 | `0.8608787059783936` | `2.365238129729316` | 353 |
| （官方 best_single：冻结 LM 专家 + 线性头） | `1.029990792274475` | `2.801040043372436` | — |

v5 的 PPL 收益同样不来自异构融合：**单专家无融合的 PPL 2.198 优于 v5 的 2.311**。
Exp5 与 Exp4 是同一个模式——多出来的能力来自「多了一个可训练的非线性」，而不是融合机制。

---

## 5. 对官方主张的修正

| 官方主张（SPEC.md / MANIFEST.json / metrics.json） | 本附录的复现结论 |
|---|---|
| 「v5 融合打破纯冻结 β 墙，Δ=+22.8%」 | 增益在当前 HEAD 上是 **+13.3pp**（+22.8pp 只在修复前的 router 下成立）；且**单专家零融合**即可达到同等准确率 → **归因不成立** |
| 「v5 显著优于 best_single (0.5222)」 | `best_single` 是「两个切片视野冻结专家」这一特定构造；换成全输入单专家线性头即 `1.0000` → **基线不具代表性** |
| 「LM 门禁 v5 PPL 2.801 → 2.305（−17.7%）」 | **单专家无融合 PPL 2.198**，优于 v5 → **归因不成立** |
| 真稀疏分派 / Phase 1 `z_loss` 零污染 / `renorm_topk` 训练-推理幅值对齐 | 本附录未涉及，不受影响 |
| K=1 路由零梯度修复（`772e0c3`） | **修复本身有效且显著**：v4 臂 +10pp、`W_router` 梯度非零；但它同时抬高了对照基线，使对外宣称的增益缩水 42% |

---

## 6. 附带观察（本轮不处理，仅登记）

1. **MANIFEST.json 与同包 metrics.json 数字漂移**：`MANIFEST.json` 的 `claim.asserts` 仍写
   旧值（v5 `0.9944` / v4 `0.7667` / Δ `+22.8%` / PPL `2.338` / 降幅 `16.5%`），而同包
   `results/metrics.json` 已是 v5 `1.0000` / v4 `0.8667` / PPL `2.305` / 降幅 `17.7%`。
2. **Exp5 的 PPL 非逐位可复现**：仓库记录 `2.3053`，本机复现 `2.3112`（+0.26%）。
3. **三个已知失败本轮按指示未动**：① SHA256 锁是 Windows CRLF 哈希（hetero-moe 无
   `.gitattributes`）→ 2 个测试在 LF checkout 必挂；② `research/vqc/` 不在仓库 → 1 个测试
   只在作者本机留着未提交 `.venv` 时才过；③ `audit_remediation.py` 直接写
   `results/metrics.json`，在本机跑完 `verdict` 翻成 `REMEDIATION_FAIL`（`_fidelity`
   `0.9994 → 0.5271` 跨过 `0.99` 阈值）。

---

## 7. 复现命令

```bash
cd <repo root>
python tools/ablation_v84_exp4_exp5_replication.py        # → §3 §4 的两张表
# §2 的 A/B 见该节内命令
```

脚本输出即 `results/ablation_exp4_exp5_replication.json`（随本附录分发）。
