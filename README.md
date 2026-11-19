# 异构 Mixture-of-Experts (Hetero-MoE)

> 异构架构底座做专家的 MoE — Heterogeneous-Bases Mixture-of-Experts
> 核心主包: **hetero_moe v5.0**（向下兼容 v4.0 Final，系统修复伪异构/伪稀疏/路由低秩瓶颈/β 墙/显存梯度/评测门禁等固有缺陷）
> 研究演进: **v4.0 → v84 (9 条研究线, 78 个世代包: research 75 + archive 3)**
> GitHub: https://github.com/Microbiosis/hetero-moe

**产物入口**

| 找什么 | 去哪 |
|---|---|
| **可加载的模型** | [HeteroMoEProjSpace-v2](https://huggingface.co/redmingwei/HeteroMoEProjSpace-v2) · 645.19M |
| **HF 全部产物** | [Hetero-MoE 研究档案 Collection](https://huggingface.co/collections/redmingwei/hetero-moe-6ae6b67d79b0423227575c99) · 7 个仓库（4 model + 3 dataset） |
| **仓库逐个说明** | [`docs/HF_ARTIFACTS.md`](docs/HF_ARTIFACTS.md) |
| **最新一代（v84.0）** | [`research/weight_fusion/v84_inherent_flaw_remediation/`](research/weight_fusion/v84_inherent_flaw_remediation/) — **全历史固有缺陷实证审计与 `hetero_moe v5.0` 系统修复（裁决 `REMEDIATION_PASS`）**：揭露并修复 `v62` 无约束 `fit_shared` 随机矩阵假象（`2-W CKA=0.9714`、`eff_rank=5.36/64`）、`v83` P5d 标签覆盖循环论证（人为压低 `85.5%` 串扰）与 `O(256)` 欠定性；在主包 `v5.0` 实现真异构专家、真稀疏分派、`grad_checkpoint` 真省显存、低秩向量门控（路由 Jacobian 秩 `1 → 17` 破 `v55` 定理 3）与可选 LoRA 破纯冻结 $\beta$ 墙；首次在严格三集防泄漏下游门禁 `DownstreamEvalGate` 上实现分类（`0.9944` vs `v4_frozen 0.7667` / `best_single 0.5222`）与序列 LM PPL（`2.338` vs `best_single 2.801`, `-16.5%`）双 `PASS` |
| **上一代（v83.0）** | [`research/weight_fusion/v83_s4_aligned_pipe/`](research/weight_fusion/v83_s4_aligned_pipe/) — S4-AP 管道（5/6 机制通过，仅 P2′ `p=0.1256` 未过，裁决 `S4AP_FAIL`；见 [`docs/V83_0_S4_ALIGNED_PIPE.md`](docs/V83_0_S4_ALIGNED_PIPE.md)） |
| **主线第一代** | [`research/weight_fusion/v75_weight_conjugation/`](research/weight_fusion/v75_weight_conjugation/) — 槽级共轭的权重级有效性检验，**负结果**（G2 未成立，见 §4.7） |
| **v63–v74 登记批次** | [`research/aligner/`](research/aligner/) · 12 代补齐为世代包（SPEC + MANIFEST + 指标快照 + 权威测试） |
| **主线路线（正本）** | [`docs/MAINLINE_ROUTE.md`](docs/MAINLINE_ROUTE.md) — 晶壁系几何 → 工程语言映射、逐条主张状态表、第一代实验设计 |
| **历史子包勘误总账** | [`docs/PAST_SUBPACKAGES_ERRATA.md`](docs/PAST_SUBPACKAGES_ERRATA.md) — `v5_alpha–v83` 全量 77 个历史子包 BUG 排查、公共组件修复清单与封存代科研勘误 |
| **S3 预登记协议（正本）** | [`docs/S3_PROTOCOL.md`](docs/S3_PROTOCOL.md) — A1/S3 跑前写死的口径：文本对 64 块（门槛 ≥32）、token 级探针 + 逐维标准化、α=0.01、P1a/P1b/P1c/P2/P3/P4 |
| **三维机制（可交互）** | [v62-space3d](https://huggingface.co/spaces/redmingwei/v62-space3d) — 拖动旋转看 v62 的三条正交轴 |

## 5 秒理解

- **干什么**: 把多个异构底座（不同宽度、不同架构、可冻结或 LoRA 轻解冻）通过稀疏高带宽路由融合，并在严格防泄漏的下游分类/LM PPL 门禁上验证净增量
- **核心结论（已剔除早期被证伪项）**:
  - 早期纯冻结随机投影增益（`v6/v7/v33`）与单师蒸馏 `+47.5%/+50%`（`v16/v21`）已分别被 `v50/v51/v54` 与 `v32/v42` 审计撤回；`v62` 无约束 `fit_shared` 也被 `v84` 证伪为随机矩阵可复现的秩塌缩伪像
  - `hetero_moe v5.0` + `v84` 通过**真异构专家 + 真稀疏分派 + 低秩向量门控（破 `v55` 定理 3）+ 可选 LoRA（破纯冻结 $\beta$ 墙）+ 统一三集防泄漏门禁 `DownstreamEvalGate`**，在真实下游分类与 LM PPL 基准上显著击败最强单底座与纯冻结基线
- **测试状态**: 核心主包 `tests/test_{fusion,router,losses,quant,train_infer,7_2_isolation,7_3_convergence,v5_core_remediation}.py` + `v84` 权威测试全绿；由 `./run.sh` 动态扫描 `tests/test_*.py`
- **结构与自包含校验**: `python tools/validate_generations.py` → **78 包 / 错误 0**（含非治理子域/内嵌 `.venv`/膨胀包透明审计）；`python tools/check_self_containment.py --strict` → **75/75**（含静态 + 动态加载全扫描，历史 SHA256 锁定包的动态依赖公开登记在案，新增未登记动态跨线直接判红）
- **泄漏审计**: **75/75 全部结清**（`research/` 下 75 个世代包全有 `leakage_audit.status`，无 pending；`archive/` 下 3 个老包不计入）。见 [`docs/LEAKAGE_AUDIT_CHAIN.md`](docs/LEAKAGE_AUDIT_CHAIN.md)
- **注**(2026-10-11 更新):`research/scale_heterogeneous/`、`results/`、`archive/`、`docs/_archive/` 原为公开排除(理由是依赖专有辅助模型)。经核实所依赖的四个模型 (EverMind-AI/MSA-4B、skillcorpus-embedding-0.6b、skillcorpus-reranker-0.6b、occams/Xiaomi-TabLDM) **全部公开可下**,排除理由已不成立,四个目录均已恢复入库。
- **论文**(2026-10-12 更新):`microfusion` + `paper` 已合并重构为单一长稿 `docs/research_paper/reconstruct.tex` / `reconstruct_zh.tex`(23 / 24 页),旧稿三份 `.tex` 标记 DEPRECATED。v47 审计使摘要里一条无支撑的「规模异构增益存活」主张暴露出来,已补 §6.3 实验。入口见 [`docs/research_paper/PAPER_README.md`](docs/research_paper/PAPER_README.md)。
- **注**(2026-10-11 更新):v47 样本外审计在补 held-out 时挖出**五个此前无人发现的缺陷**—— v12 / v19 / v35 的 runner **从未真正跑起来** (函数体用了未导入的裸名)、v19 声称的「容量对照臂」与基线逐位同构、attn 专家的输出投影不受种子控制 (跨进程不可复现)。均已修复并记录在案,详见 [`docs/V47_0_HELDOUT_AUDIT.md`](docs/V47_0_HELDOUT_AUDIT.md)。
- **注**(2026-10-11 更新):`tabldm` 的权重在 HuggingFace `occams/Xiaomi-TabLDM`,但**源码在 GitHub `xiaomi-research/xiaomi-tabldm`** ——两者不是一个仓库,只看 HF 会误判成「无可安装源码」。现由 `paths.py` 自动浅克隆到 `$XDG_CACHE_HOME/hetero_moe/`,无需手工安装。

## 项目结构

```
hetero_moe/                          # v5.0 主包（兼容 v4.0）— 真异构/真稀疏/高带宽/防泄漏门禁
├── core/                               # 融合核心 (experts/fusion/router/quant/losses/parallel)
├── train/                              # 两阶段训练器 (Phase 1 零 z_loss 污染)
├── infer/                              # 容量感知推理器 (单次路由乘 + 校准多层 skip_rate)
└── eval/                               # 统一防泄漏下游评测门禁 (SplitGuard/IdentifiabilityGuard/DownstreamEvalGate)

research/                               # 9 条研究线 — 每条解决一个独立研究问题
├── routing_evolution/                  # 线 1: 路由机制演化 (v7/v8/v9)
├── central_diagnostics/                # 线 2: C-1 失败根因诊断 (v19/v20)
├── central_stability/                  # 线 3: 中枢稳定性修复 (v22/v30/v31)
├── aligner/                            # 线 4: 语义对齐器与投影空间审计 (v10–v74)
├── distillation/                       # 线 5: 蒸馏与目标泄漏审计 (v16/v17/v18/v32/v40)
├── multimodal/                         # 线 6: 多模态扩展 (v21/v23/v24/v42)
├── corpus_scale/                       # 线 7: 真实语料下中枢稳定性 (v25-v29)
├── scale_heterogeneous/                # 线 8: 规模异构 + 容器范式 (v33-v39, v56)
└── weight_fusion/                      # 线 9: 权重级合并 → 管道审计 → v5.0 系统修复 (v75→v84)

archive/                                # 负面发现 + 前基线
├── pre_baseline/v5_alpha/              # 序列级路由前基线 (与 v6 范式不同)
└── negative_findings/
    ├── v12_compose/                    # v9+v10 组合无叠加
    └── v23b_hard_vqa/                  # 8 类陷阱 toy 全部 = 随机

tests/                                  # 430 个 V&V 测试
examples/                               # 40+ 端到端脚本
docs/                                   # 研究规范文档
└── research_paper/                   # 论文:reconstruct.tex(合并稿)/ 中英 PDF + 评审记录
    └── PAPER_README.md               # ★ 论文入口(产物 / 贡献 / 编译 / 权重 / 自引)
```

→ 详见 [research/README.md](./research/README.md) 看 9 条研究线的全景

## 包结构变化 (2026-04 重构)

旧的 `v5_alpha/`、`v7_attention/` ... `v39_g2_interaction/` 35+ 平铺兄弟包已重组成:
- `research/<line>/v*_<name>/` — 9 条研究线
- `archive/<reason>/v*_<name>/` — 负面发现 + 前基线
- `hetero_moe.{core,train,infer}` — 主包内部分层

**导入示例**:
```python
# 旧 (已废弃)
from v9_gate_central import GateStyleWorkspace

# 新
from research.routing_evolution.v9_gate_central import GateStyleWorkspace
from hetero_moe.core.fusion import HeteroFusionLayer
from hetero_moe.train.trainer import HeteroTrainer
```

## 9 条研究线全景

| # | 研究线 | 关键版本 | 核心发现 |
|---|---|---|---|
| 1 | routing_evolution | v7→v8→v9 | gate-style 中枢 +88%; 中枢必须有结构 |
| 2 | central_diagnostics | v19→v20 | C-1 失败主因是 H1 U 训练扰动; H5 位置不重要 |
| 3 | central_stability | v22→v30→v31 | softplus +3.0%, grad-scale +1.9% |
| 4 | aligner | v10→v13→v15 | mixture > attn > linear; 嵌套天花板 |
| 5 | distillation | v16→v17→v18→**v32**→**v40** | v16 单教师 +47.5% 已由 **v32 证伪为目标泄漏**（纯蒸馏不超教师）；多教师反而退步；v40 证明换分派形式仍无法迁移 13% headroom |
| 6 | multimodal | v21→v23→v24→**v42** | v21 表观 +50% 已由 **v42 证伪为目标泄漏**；v23b 8 类陷阱 toy = 随机 |
| 7 | corpus_scale | v25→v26→v27→v28→v29→**v45** | 语料规模边界样本外未成立 (v45)；v27/v28 patch/架构变体全负 |
| 8 | scale_heterogeneous | v33→v39→**v56** | v33 表观 +26.7% 已由 v50/v54 证实为冻结随机投影伪像；v56 证实 2560d 容器有效秩仅 1.32% |
| 9 | weight_fusion | v75→**v84** | v75 证伪 G2 权重共轭；v76–v83 推进 A1 激活管道至 5/6 (`S4AP_FAIL`)；**v84 全面审计并修复 v4–v83 固有缺陷**（`hetero_moe v5.0` 破路由带宽墙与纯冻结 β 墙，真实下游分类与 LM PPL 防泄漏门禁双 `PASS` ⇒ `REMEDIATION_PASS`） |

→ 详见 [research/README.md](./research/README.md)

## 当前有效工程入口（v5.0 / v84.0）

```
hetero_moe v5.0  真异构专家 + 真稀疏分派 + 低秩向量门控 + 可选 LoRA + grad_checkpoint
DownstreamEvalGate  强制 train/val/test 三集零重叠 + 单底座/零参数/阴性对照 + Bootstrap CI
v84 分类门禁     test_acc 0.9944 vs v4_frozen 0.7667 / best_single 0.5222 (PASS)
v84 LM PPL 门禁  test_ppl 2.338  vs best_single 2.801 (-16.5%, PASS)
（历史 v6/v13/v16/v21 合成 MSE 读数已由 v32/v42/v50/v54 审计归档，不再作为能力指标）
```

## 操作入口

```bash
python tools/eval_gate.py --self-check                 # 运行防泄漏下游门禁自检
python tools/validate_generations.py                   # 78 世代包 L2 校验 + 治理范围审计
python tools/check_self_containment.py --strict        # 75 研究包静态+动态自包含校验
python -m pytest tests/test_v5_core_remediation.py -v  # v5.0 固有缺陷修复回归套件 (14 项)
python tests/test_v84_inherent_flaw_remediation.py     # v84 权威测试 (10 项)
bash run.sh test                                       # 全仓测试套件
```

## 关键约束

| 维度 | 内容 |
|---|---|
| 核心依赖 | `torch>=2.0`、`numpy>=1.24`; 可选 `transformers>=4.30`、`Pillow>=9.0` |
| 沙箱维度 | D ≤ 128 |
| §8.1 真实验收 | 需 A100-80G + 0.5B 模型 (本沙箱不替代) |
| 运行底座 | Linux container (Python 3.11+, PyTorch 2.x, CPU-only) |
| 测试 | 430/430 ALL GREEN ✓ |

## V&V 矩阵

| 类别 | 测试 | 状态 |
|---|---|---|
| 规范 §8.1 Phase1/Phase2 | `test_train_infer.py` | ✅ 7 PASS |
| 规范 §8.2 量化 | `test_quant.py` | ✅ 6 PASS |
| 规范 §8.2 路由 | `test_router.py` | ✅ 6 PASS |
| 规范 §8.2 fusion | `test_fusion.py` | ✅ 7 PASS |
| 规范 §8.3 推理 C=1.0/1.25/1.5 | `test_train_infer.py` | ✅ |
| 规范 §8.4 Triton Kernel | — | ⏳ GPU 性能优化 (CPU 沙箱外) |
| V4.0 §7/§8 baseline | test_fusion/losses/quant/router/train_infer | ✅ |
| V5-v8 | test_7_*/test_v7_attention/test_v8_central | ✅ |
| V9-v22 | `test_v9_gate_central` … `test_v22_central_theory` (9 个转发层) | ✅ 12+8+8+8+8+8+8+8+14 PASS |
| V30-v32 | `test_v30_gate_softplus` / `test_v31_gate_gradient_scale` / `test_v32_leakage_audit` | ✅ 9+9+8 PASS |
| V40-v46 | `test_v40_dispatch` … `test_v46_attn_path_audit` (7 个转发层) | ✅ 12+8+10+8+8+7+7 PASS |
| V47 | `test_v47_heldout_audit` (样本外判定算子) | ✅ 9 PASS |
| V23-v29 | test_v23-v29 (含 multimodal, corpus, central-stability) | ✅ |
| V33-v39 | test_v33-v39 (scale-heterogeneous) | ✅ 10+10+10+10+10+10+10 PASS |

## 设计哲学 (4 条)

1. **异构粒度要可分可合**: v6/v7/v8/v9 每一层都允许只启用部分组件
2. **失败模式比成功模式更有信息量**: 7 个负结果共同勾勒"地图边界"
3. **神经科学类比是设计灵感, 不是验证标准**: 验证靠工程指标 (fuse MSE)
4. **增量扩展优于重写**: v7 复用 v6 P_m+SwiGLU+INT4+K=1 STE; v8 只加 4 个新组件

→ 历史 README.md (1162 行) + PROJECT_REPORT.md (282 行) 已合并入本 README + [research/README.md](./research/README.md) + 各 research/*/README.md