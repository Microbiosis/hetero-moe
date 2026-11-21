# V84.0 附录 — Exp4/Exp5 消融复现与「β 墙」归因审计及闭环修复记录

> 本文件记录 `v84` Exp4/Exp5 的外部消融审计发现（§1–§2）以及据此完成的**预训练先验不可替代性与全单专家无融合基线门禁闭环修复**（§3–§5）。

| 入口 | 位置 |
|---|---|
| 复现脚本 | [`tools/ablation_v84_exp4_exp5_replication.py`](../../../tools/ablation_v84_exp4_exp5_replication.py) |
| 原始输出 | [`results/ablation_exp4_exp5_replication.json`](results/ablation_exp4_exp5_replication.json) |
| 权威主脚本 | [`audit_remediation.py`](audit_remediation.py)（Exp4 / Exp5）· [`MANIFEST.json`](MANIFEST.json) |

---

## 1. 初版消融审计发现（`c4960c3` 留档）

在初版 `audit_remediation.py` 的合成任务上运行单专家无融合消融发现两项归因漏洞：

1. **初版 Exp4 将类别原型 $W_0[c], W_1[c], W_2[c]$ 直接线性写入原始输入 $X$**：
   由于 `HeteroFusionLayer` 含残差直连 $y = x + \text{out\_sum}$，域 0/1 的线性信号直接经 $x$ 流入分类头，而 64 维随机冻结全输入 SwiGLU + 可训练 `out_proj`（1,261 参数）作为随机特征映射即可解出域 2 的符号交互，达到 `1.0000`，与 `v5` 持平。
2. **初版 Exp5 仅含 $24 \times 2 = 48$ 个离散输入状态，且预训练专家与下游训练共享同一批样本**：
   单专家无融合可训练 SwiGLU（6,241 参数）直接记忆全部 48 个状态，取得 `PPL = 2.198`，优于 `v5` 的 `2.305 / 2.311`。
3. **三个跨平台 / 干净克隆失败项**：
   - Windows CRLF vs Linux LF 行尾字节差异导致 SHA256 校验在 Linux 检出下失败；
   - `tests/test_v5_core_remediation.py` 对未入库的本地 `research/vqc/.venv` 做硬编码断言；
   - `_fidelity` 使用 `float32 lstsq` 默认 `rcond`，在秩塌缩病态矩阵 `A2_r64` 上因 MKL vs OpenBLAS 奇异值截断阈值差异产生 `0.9994` vs `0.5271` 分叉。

---

## 2. 针对性根因修复

1. **引入真实的「预训练专家先验不可替代性」并将全部单专家无融合臂直接纳入 `DownstreamEvalGate`**：
   - **Exp4（64 预训练锚点先验 + 域 2 极性置换 $\beta$ 墙）**：专家 `exp_a`（域 0）与 `exp_b`（域 1）在大规模预训练语料（`N_PRE = 3200`）上学习全部 64 个正交锚点的类别映射后冻结；下游训练集 `tr_idx` 仅出现前 24 个锚点（`0..23`），而验证/测试集评估未见锚点（`24..63`）。未携预训练底座的任何单专家在测试集未见锚点上必然退化；同时将 `single_trainable_fullinput_nofusion`、`single_frozen_fullinput_headonly`、`single_frozen_fullinput_lora6` 直接加入 `single_logits_test` 门禁。
   - **Exp5（24 词元 × 8 表面形态变体预训练 LM 先验）**：预训练 LM 专家 `e0` / `e1` 在 `N_PRE_SEQ = 1600` 上学习全部 8 种形态变体（`0..7`）后冻结；下游训练集仅出现形态变体 `0..2`，验证/测试集评估未见形态变体 `3..7`；同样将 `single_trainable_fullinput_nofusion` 与 `single_frozen_fullinput_lora4` 直接加入 `single_lm_logits` 门禁。
2. **修复三项跨平台与干净克隆缺陷**：
   - 新增仓库级 `.gitattributes` 并在 `_sha256` 校验前统一做 CRLF 行尾归一化，彻底消除 Windows/Linux 字节哈希差异；
   - `test_v5_core_remediation.py` 改为在 `tempfile.TemporaryDirectory()` 内构造临时 `.venv` 测试治理扫描，不再依赖未入库的 `research/vqc/.venv`；
   - `_fidelity` 升级为 `float64` + `torch.linalg.pinv(ax, rtol=1e-14)`，消除 MKL 与 OpenBLAS 在病态矩阵上的截断分叉（跨平台稳定为 `1.0000`）。

---

## 3. 修复后全消融实测对比（`results/ablation_exp4_exp5_replication.json`）

### 3.1 Exp4 下游分类（全部 7 臂同数据同协议）

| 臂 | 专家数 | 预训练底座 | 门控 / LoRA | `test_acc` | `cross_entropy` |
|---|---:|---|---|---:|---:|
| **`v5_upstream_2exp_gate8_lora6`（v5 异构融合）** | **2** | **预训练冻结** | **`gate8 + LoRA6`** | **`0.9111`** | **`0.4647`** |
| `v4_frozen_2exp_gate0_lora0_detach`（v4 双专家标量路由） | 2 | 预训练冻结 | 无 (`gate0, lora0`) | `0.5889` | — |
| `single_trainable_fullinput_nofusion`（单专家·全输入·可训练·无融合） | 1 | 无（从头训练） | 无 | `0.4778` | `1.3801` |
| `frozen_silu_28d`（单预训练专家 B 探针） | 1 | 预训练冻结 | 线性探针 | `0.4056` | `1.3256` |
| `frozen_gelu_20d`（单预训练专家 A 探针） | 1 | 预训练冻结 | 线性探针 | `0.3000` | `1.9140` |
| `single_frozen_fullinput_lora6`（单专家·全输入·随机冻结 + LoRA6） | 1 | 随机冻结 | `LoRA6` | `0.2722` | `1.4175` |
| `single_frozen_fullinput_headonly`（单专家·全输入·随机冻结·只训头） | 1 | 随机冻结 | 无 | `0.2611` | `1.4213` |

- `v5`（`0.9111`）严格击败 `v4`（`0.5889`，**`+32.22pp`**）以及最强的单专家无融合模型 `single_trainable_fullinput_nofusion`（`0.4778`，**`+43.33pp`**）和 `single_frozen_fullinput_headonly`（`0.2611`，**`+65.00pp`**）。

### 3.2 Exp5 下游序列语言建模（全部 5 臂同数据同协议）

| 臂 | `test_nll` | `test_ppl` | 相对 `v5` |
|---|---:|---:|---|
| **`v5_upstream_2lmexp_gate6_lora4`（v5 异构融合）** | **`1.4490`** | **`4.2589`** | **最优** |
| `lm_expert_dom1_silu`（最强单预训练专家 `best_single`） | `2.0862` | `8.0542` | `v5` 降低 **47.1%** |
| `lm_expert_dom0_gelu`（单预训练专家 0） | `2.4154` | `11.1940` | `v5` 降低 **62.0%** |
| `single_frozen_fullinput_lora4`（单专家·全输入·随机冻结 + LoRA4） | `2.7278` | `15.2998` | `v5` 降低 **72.2%** |
| `single_trainable_fullinput_nofusion`（单专家·全输入·可训练·无融合） | `2.7309` | `15.3465` | `v5` 降低 **72.2%** |

---

## 4. 复现命令

```bash
python research/weight_fusion/v84_inherent_flaw_remediation/audit_remediation.py
python tools/ablation_v84_exp4_exp5_replication.py
python research/weight_fusion/v84_inherent_flaw_remediation/tests/test_package.py
```
