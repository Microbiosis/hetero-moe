# V47.0 — 「in-sample 优势能否活到样本外」批次审计

> **本文件是薄指针 (thin pointer), 非权威副本.**
> 权威规范: [`research/central_stability/v47_heldout_audit/SPEC.md`](../research/central_stability/v47_heldout_audit/SPEC.md)
> 机器可读声明: [`research/central_stability/v47_heldout_audit/MANIFEST.json`](../research/central_stability/v47_heldout_audit/MANIFEST.json)
> 指标快照: [`research/central_stability/v47_heldout_audit/results/metrics.json`](../research/central_stability/v47_heldout_audit/results/metrics.json)

## 一句话结论

> 16 个 `leakage_audit=pending` 的包**全部结清, 无一遗留**: 7 个据实核实其**原生就有留出协议**
> (此前 MANIFEST 里『本包无 held-out 评估』的说法是错的, 已更正), 9 个补跑三个口径的 held-out 实测
> (三口径全部同向)。逐包数字见 `results/metrics.json` 的 `per_package` —— 聚合数字不作结论引用。

## 实测读数 (5 seeds, 配对逐种子中位数)

| 包 | in_sample | heldout_noise | heldout_stimulus | 判定 |
|---|---:|---:|---:|:---:|
| v33 融合 vs 单大 | +26.7% | +26.7% | +26.5% | clean |
| v34 DMN vs 无 DMN | +6.4% | +6.4% | +6.5% | clean |
| v19 长训练 vs 短训练 | +4.3% | +1.5% | +2.4% | clean (幅度只剩 34%~54%) |
| v8 C-1 vs 无中枢 | −306.7% | −51.0% | −50.6% | clean (负向复现) |
| v12 attn 对齐器 + gate | −43.0% | −41.4% | −69.4% | clean (无叠加增益复现) |
| v20 中枢放 attn | −117.6% | −137.4% | −153.9% | clean (更差, 复现) |
| v30 softplus gate | −0.4% | −0.4% | −0.4% | clean (对融合 MSE 无差别) |
| v31 梯度重缩放 gate | −0.1% | −0.1% | −0.1% | clean (同上) |
| v35 异质入口注入 | −11.1% | −11.1% | −3.3% | clean (否定结论复现) |

## 顺手修掉的三个「从未跑起来」

1. `examples/run_v12_compose.py:37` 给 `CentralAugmentedFusionLayer` 传了它不接受的 `c1_alpha`
2. `research/_primitives/central_mechanism/workspace.py` 的 `make_workspace` 对 broadcast 分支也传 `alpha`
   ⇒ **任何 `c1_mode="broadcast"` 的层构造即失败**, v19 全部五臂与 v12 的 broadcast 对照臂从未执行过
3. `examples/run_v19_c1_revisit.py` 定义了 `C1OnlyModel` 并接受 `capacity` 参数但**两者都没被使用**
   ⇒ 声称的 H4「学生容量」对照臂与 `c1-100` 完全同构, 那一半结论在构造上没有支撑

## 复现

```bash
python3 examples/run_v47_central_heldout.py       # v8 v12 v19 v20 v30 v31
python3 examples/run_v47_scale_hetero_heldout.py  # v33 v34 v35
```
