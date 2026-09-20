# research/_primitives/ — 共享原语层

5 个**自包含**的共享原语包, 被整个 research/ 内的实验包消费。
这些原语**不依赖任何研究线内代码**, 只依赖 `torch`、`numpy` 等通用库。

## 5 个原语包

| 原语 | 路径 | 被多少研究包复用 | 关键 API |
|---|---|---:|---|
| **attention** | `research._primitives.attention` | 4 | `AttnPool`, `manual_multi_head_attention` |
| **central_mechanism** | `research._primitives.central_mechanism` | 6+ | `GateStyleWorkspace`, `BroadcastWorkspace`, `make_workspace`, `freeze_broadcast_matrix` |
| **central_theory** | `research._primitives.central_theory` | 3 | `CentralTheory`, `RouterNormCentral`, `AdaptiveEMACentral`, `freeze_u/c`, `set_u_init_scale`, `gate_style_analysis` |
| **real_corpus** | `research._primitives.real_corpus` | 2 | `MiniCorpus`, `cycle_batches` |
| **scale_hetero_loaders** | `research._primitives.scale_hetero_loaders` | 6+ | `MSA_DIR`, `EMB_DIR`, `RER_DIR`, `TABLDM_CKPT`, `load_msa_big`, `load_small`, `load_tokenizer`, `encode`, `load_big_container`, `big_forward`, `get_tabldm_repr`, `encode_baseline_text`, `EntryEncoder`, `gen_table_classification`, `container_text_repr`, `tabldm_passive_repr`, `knn_acc`, `concat_norm_then_knn`, `train_arm` |

## 原始位置 → 提升后位置

```
原位置                                                  新位置
─────────────────────────────────────────────         ───────────────────────────────────────
research/routing_evolution/v7_attention/attn_pool.py   →  research/_primitives/attention/attn_pool.py
research/routing_evolution/v9_gate_central/workspace.py →  research/_primitives/central_mechanism/workspace.py
research/central_stability/v22_central_theory/theory.py →  research/_primitives/central_theory/theory.py
research/distillation/v18_real_corpus/corpus.py       →  research/_primitives/real_corpus/corpus.py
research/scale_heterogeneous/v33_scale_hetero/paths.py  →  research/_primitives/scale_hetero_loaders/paths.py
research/scale_heterogeneous/v33_scale_hetero/encoders.py → research/_primitives/scale_hetero_loaders/encoders.py
research/scale_heterogeneous/v36_alpha2_knn/{paths,entries,eval}.py
                                                      →  research/_primitives/scale_hetero_loaders/{paths,entries,eval}.py
research/scale_heterogeneous/v37_b_selftrained_entry/{entry_encoder,arm,paths}.py
                                                      →  research/_primitives/scale_hetero_loaders/{entry_encoder,arm,paths}.py
```

## 自包含性

每个原语包**只依赖**:
- `torch` / `torch.nn` / `torch.nn.functional`
- `numpy`
- `safetensors` / `transformers` (仅 `scale_hetero_loaders` 实际加载模型时)

**不依赖**:
- `research.<line>.*` (任何研究线内代码)
- `archive.*`
- `hetero_fusion.*` (主包)

这意味着每个原语可以**独立测试、独立发布、独立升级**,不被研究实验污染。

## 验证

```bash
python3 tools/check_self_containment.py
```

输出: 29/29 research 包全部通过自包含性检查 — 没有任何跨研究线的外部依赖。

## 设计动机

之前 (重构前):
- v7.AttnPool 被 v8/v10/v12/v25 复用 → 必须装 4 个包才能用其中一个
- v33.{paths,encoders} 被 v34-v39 全部依赖 → v33 是"必经之路"
- v22.CentralTheory 是中枢机制的基类,被 3 个包依赖
- v18.MiniCorpus 是真实语料,被 v25/v26 依赖

这意味着研究包**没有自包含独立性**: 取走任何一个, 它就断了一半。

现在:
- 5 个原语是**基础设施层**,与具体研究实验解耦
- 实验包只依赖原语层 + 同线内允许的兄弟包
- 每个原语可以单独升级 (例如 AttnPool 加新优化,不影响 v8/v10 实验包)
- 未来加新研究线 (v40+, v41+) 只需 import 必要的原语即可

## 实验包成为 thin shim 的情况

由于原语被抽走,以下研究包变成 thin re-export shim (自身只剩 `__init__.py`):
- `research.central_stability.v22_central_theory/` — CentralTheory 已提升
- `research.distillation.v18_real_corpus/` — MiniCorpus 已提升

为保持向后兼容,这两个包的 `__init__.py` 仍 re-export 原语。
带 `__is_re_export_shim__ = True` 标记。
未来可以考虑删掉这两个目录, 但现在保留以减少破坏性变化。