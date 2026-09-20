# 研究线 7: corpus-scale (真实语料下中枢稳定性)

**研究问题**: research/central_stability/ 的 v22 中枢模式在真实语料下是否仍然稳定? 数据规模影响如何?

**父版本**: research/central_stability/v22_central_theory (4 mode) + research/distillation/v18_real_corpus (MiniCorpus)
**子版本**: 无 (这是 v22 路线的语料分支, 已收敛)

## 成员

| 版本 | 贡献 | 关键模块 |
|---|---|---|
| v25 | v22 + v18 跨版本集成 (首个跨版本包) | `layer.RealCorpusTheoryLayer`, `build_v25_param_groups` |
| v26 | fully-real-corpus (text+code+image 全部真实) | `code_corpus.MiniCodeCorpus`, `image_corpus.MiniImageCorpus` |
| v27 | router-norm / adaptive-ema held-out 修正补丁 (4 patch × 4 mode × 5 seeds = 80 run) | `patches.apply_l2_to_c`, `router_dropout`, `extend_warmup_steps`, `noise_inject_c` |
| v28 | 架构层面修正 (11 架构变体 × 5 seeds = 55 run) | `router_norm_variants.RouterNormAffineFalse/ZeroU/RMS/Pure`, `adaptive_ema_variants.Clamped/Bounded/Gate` |
| v29 | 大型真实语料 (64 text + 32 code + 50 image, 4×) | `text_corpus.LargeTextCorpus`, `code_corpus.LargeCodeCorpus`, `image_corpus.LargeImageCorpus` |

## 关键发现

| mode | v22 合成 | v25 真实 (16+6) | v26 真实 (16+6) | v29 大语料 (64+32) |
|---|---:|---:|---:|---:|
| broadcast | 0.3283 | — | 0.4624 | 0.4548 |
| gate (v9) | 0.3006 | 0.3562 | **0.4516** | 0.5253 |
| router-norm | 0.2875 | 0.4125 | 0.4859 (退步) | 0.5763 (退步) |
| adaptive-ema | **0.2972** | **0.3094** | 0.4931 (退步) | **0.4309** |

**v29 最强发现**: **最优中枢模式依赖数据规模**
- 小语料 (16+6): adaptive-ema / gate 最佳
- 大语料 (64+32): adaptive-ema 最佳 (+12.6%)

## 负面发现 (重要)

- **v27**: 4 patch (L2/Dropout/warmup/noise) **全部失败**, v22 patch 已局部最优
- **v28**: 7 架构变体 **全部失败**, 架构层救不了数据规模问题

> 这两个负结果是本项目最有信息量的发现之一: "v22 已有 patch 已局部最优, 架构层救不了数据规模问题"

## 跨线复用

- 复用 research/central_stability/v22 (CentralTheory 基类)
- 复用 research/distillation/v18 (MiniCorpus 真实语料接口)
- 复用 research/routing_evolution/v7 (AttnPool)
- 复用 research/aligner/v10 (CrossArchAttnAligner)

## 测试

- test_v25_corpus_central.py (10 PASS)
- test_v26_real_corpus.py (11 PASS)
- test_v27_router_norm_fix.py (12 PASS)
- test_v28_architectural_fix.py (16 PASS)
- test_v29_large_corpus.py (12 PASS)