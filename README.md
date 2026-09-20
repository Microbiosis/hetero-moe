# 异构微观融合系统 (HeteroFusion)

> 异构底座的微观融合 — MoE-Style Heterogeneous Micro-Fusion
> 规范基线: **v4.0 Final (冻结)**
> 工程实现: **v4.0 → v39 (8 条研究线, 35+ 子包)**

## 5 秒理解

- **干什么**: 把多个异构底座 (BERT + LLaMA + ViT 等不同架构) 通过共享路由融合, 然后蒸馏到单一独立模型
- **核心结论**: 融合层 (mixture aligner) 比纯线性投影 MSE -77%, 单教师蒸馏可超教师 +47.5%, 最优中枢模式依赖数据规模
- **测试状态**: **327/327 ALL GREEN** ✓
- **注**: `research/scale_heterogeneous/` 研究线（v33-v39）的代码使用机构内部辅助模型（proprietary auxiliary models）；该目录从公开仓库中排除，权重和代码可通过邮件向作者索取。其他 7 条研究线（routing_evolution / central_diagnostics / central_stability / aligner / distillation / multimodal / corpus_scale）和主包 `hetero_fusion/` 全部公开。

## 项目结构

```
hetero_fusion/                          # v4.0/v6 主包 — baseline + 内部分层
├── core/                               # 融合核心 (fusion/router/quant/losses/parallel)
├── train/                              # 训练器
└── infer/                              # 推理

research/                               # 8 条研究线 — 每条解决一个独立研究问题
├── routing_evolution/                  # 线 1: 路由机制演化 (v7/v8/v9)
├── central_diagnostics/                # 线 2: C-1 失败根因诊断 (v19/v20)
├── central_stability/                  # 线 3: 中枢稳定性修复 (v22/v30/v31)
├── aligner/                            # 线 4: 语义对齐器 (v10/v13/v15)
├── distillation/                       # 线 5: 蒸馏到独立模型 (v16/v17/v18)
├── multimodal/                         # 线 6: 多模态扩展 (v21/v23/v24)
├── corpus_scale/                       # 线 7: 真实语料下中枢稳定性 (v25-v29)
└── scale_heterogeneous/                # 线 8: 规模异构 + 容器范式 (v33-v39, 独立范式)

archive/                                # 负面发现 + 前基线
├── pre_baseline/v5_alpha/              # 序列级路由前基线 (与 v6 范式不同)
└── negative_findings/
    ├── v12_compose/                    # v9+v10 组合无叠加
    └── v23b_hard_vqa/                  # 8 类陷阱 toy 全部 = 随机

tests/                                  # 327 个 V&V 测试
examples/                               # 40+ 端到端脚本
docs/                                   # 研究规范文档
```

→ 详见 [research/README.md](./research/README.md) 看 8 条研究线的全景

## 包结构变化 (2026-04 重构)

旧的 `v5_alpha/`、`v7_attention/` ... `v39_g2_interaction/` 35+ 平铺兄弟包已重组成:
- `research/<line>/v*_<name>/` — 8 条研究线
- `archive/<reason>/v*_<name>/` — 负面发现 + 前基线
- `hetero_fusion.{core,train,infer}` — 主包内部分层

**导入示例**:
```python
# 旧 (已废弃)
from v9_gate_central import GateStyleWorkspace

# 新
from research.routing_evolution.v9_gate_central import GateStyleWorkspace
from hetero_fusion.core.fusion import HeteroFusionLayer
from hetero_fusion.train.trainer import HeteroTrainer
```

## 8 条研究线全景

| # | 研究线 | 关键版本 | 核心发现 |
|---|---|---|---|
| 1 | routing_evolution | v7→v8→v9 | gate-style 中枢 +88%; 中枢必须有结构 |
| 2 | central_diagnostics | v19→v20 | C-1 失败主因是 H1 U 训练扰动; H5 位置不重要 |
| 3 | central_stability | v22→v30→v31 | softplus +3.0%, grad-scale +1.9% |
| 4 | aligner | v10→v13→v15 | mixture > attn > linear; 嵌套天花板 |
| 5 | distillation | v16→v17→v18 | 单教师 +47.5%; 多教师反而退步 |
| 6 | multimodal | v21→v23→v24 | BERT+ViT 蒸馏可; toy 不可行 |
| 7 | corpus_scale | v25→v26→v27→v28→v29 | 最优中枢模式依赖数据规模; v27/v28 patch/架构变体全负 |
| 8 | scale_heterogeneous | v33→v34→v35→v36→v37→v38→v39 | **独立范式**: 1大+N小 + 大模型容器 |

→ 详见 [research/README.md](./research/README.md)

## 当前最强组合

```
v6   linear P_m         fuse 1.14   (基线)
v10  attn aligner       fuse 0.34   (-70%)
v13  mixture            fuse 0.26   (-77%)   ← 教师当前最强
v16+BERT (真实预训练)   fuse 0.135  ← 当前最强单底座独立模型
v21+BERT+ViT (多模态)   fuse 0.16   ← 当前最强多模态独立模型
```

## 操作入口

```bash
bash run.sh setup       # 一键重建 (幂等)
bash run.sh test        # 全部 327 V&V 测试
bash run.sh v8          # v8.0 三中枢路线
bash run.sh v33         # v33 规模异构
bash run.sh v39         # v39 加性 vs 交互
bash run.sh e2e         # §8.1-8.3 端到端仿真
bash run.sh smoke       # 最小冒烟 (验证依赖+包)
bash run.sh clean       # 清理 __pycache__
```

## 关键约束

| 维度 | 内容 |
|---|---|
| 核心依赖 | `torch>=2.0`、`numpy>=1.24`; 可选 `transformers>=4.30`、`Pillow>=9.0` |
| 沙箱维度 | D ≤ 128 |
| §8.1 真实验收 | 需 A100-80G + 0.5B 模型 (本沙箱不替代) |
| 运行底座 | Linux container (Python 3.11+, PyTorch 2.x, CPU-only) |
| 测试 | 327/327 ALL GREEN ✓ |

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
| V9-v22 | test_v9_gate/test_v10-v22 | ✅ 14+12+8+8+8+8+8+8+8+14 PASS |
| V23-v29 | test_v23-v29 (含 multimodal, corpus, central-stability) | ✅ |
| V33-v39 | test_v33-v39 (scale-heterogeneous) | ✅ 10+10+10+10+10+10+10 PASS |

## 设计哲学 (4 条)

1. **异构粒度要可分可合**: v6/v7/v8/v9 每一层都允许只启用部分组件
2. **失败模式比成功模式更有信息量**: 7 个负结果共同勾勒"地图边界"
3. **神经科学类比是设计灵感, 不是验证标准**: 验证靠工程指标 (fuse MSE)
4. **增量扩展优于重写**: v7 复用 v6 P_m+SwiGLU+INT4+K=1 STE; v8 只加 4 个新组件

→ 历史 README.md (1162 行) + PROJECT_REPORT.md (282 行) 已合并入本 README + [research/README.md](./research/README.md) + 各 research/*/README.md