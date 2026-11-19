# V80.0 — S2：注入算子下沉到真模型层边界（**先导失败 → λ 标定 → 闸门重跑：全过**）

> 本文件是指针。权威规范在
> [`research/weight_fusion/v80_inject_operator/SPEC.md`](../research/weight_fusion/v80_inject_operator/SPEC.md)

**一句话**：S3 的注入算子真做出来了（共享空间载体、**不学任何学出来的映射**），在真模型层边界用 `forward_pre_hook` 注入，四臂在全部 84 个块上跑通；先导运行 λ=0.1 被证伪「注入把 Qwen 中层打坏 55–80%」⇒ **S2-1 标定选出 λ* = 0.03** ⇒ 闸门重跑**六项全过**。

| 入口 | 位置 |
|---|---|
| 权威规范 | `research/weight_fusion/v80_inject_operator/SPEC.md` |
| 机器可读声明 | `research/weight_fusion/v80_inject_operator/MANIFEST.json` |
| 指标快照（λ=0.03 重跑） | `research/weight_fusion/v80_inject_operator/results/metrics.json` |
| λ 标定快照 | `research/weight_fusion/v80_inject_operator/results/lambda_calibration.json` |
| 先导运行（λ=0.1，FAIL，留档） | `research/weight_fusion/v80_inject_operator/results/run_lambda0p1.json`（源码同址归档，sha256 逐位吻合） |
| 权威测试 | `research/weight_fusion/v80_inject_operator/tests/test_package.py` |

## 三步链

| 步 | λ | 裁决 |
|---|---|---|
| (1) 先导运行 | 0.1 | ❌ `S2_OPERATOR_FAIL` — C4 被证伪：36/252 臂块 final 扰动 > 0.5（最大 0.8011），全部在 BERT→Qwen 的 j ∈ {3, 6, 12} |
| (2) S2-1 标定 | 候选 {0.01…0.05} | ✅ `CALIB_LAMBDA_SELECTED` ⇒ **λ* = 0.03**（写死候选集 + 写死选择规则：max final ≤ 0.3 且 min readout ≥ 1e-3，取最大可行） |
| (3) 闸门重跑 | **0.03** | ✅ `S2_OPERATOR_READY` — 六项闸门全过（退出码 0） |

**阈值一条未放宽**：六项判据与先导运行完全相同，只换了 λ；而换 λ 是被先导的证伪逼出来的「重新预登记」（协议 §2.3 修订记录第 6 条）。

## 最终结果（λ = 0.03）

- **G1** 零注入**逐位等价**（17 个注入点 × 逐层 rel_err **恰为 0.0**）⇒ 协议 §4 开跑前清单第 2 条**已满足**。
- **C2** 注入范数比与 λ 的最大偏差 **2.51e-7**。
- **C3** 读出层扰动全部落在 (1e-3, 0.5)（min **0.00479** / 中位 0.0295 / max 0.0806）。
- **C4** 全部 252 臂块 final 扰动 **≤ 0.3000**（上界 0.5），且**无一** < 1e-3。
- **C5** arm3 去相关：最大 |cos| 0.2448，违例 **0**。
- **C6** 全部有限。
- **交叉验证**：标定在 λ=0.03 算出的 max final = **0.260999233**，闸门重跑 arm1 最大值 = **0.260999233** ⇒ 两条独立路径逐位相同 ✓。

## 引用本代必须带上

1. **λ=0.1 的失败未被撤销**：只引 λ=0.03 的成功会掩盖「0.1 会把 Qwen 中层打坏 55–80%」这条事实 ⇒ **两次运行必须成对引用**。
2. **机理**：Qwen 残差流在**第 3 层**量级爆炸（v77 B4 实测 RMS 0.277 → 2.092），落在爆炸层的注入绝对幅度被放大 **117 倍**；失效**内容无关**（三臂一样）。
3. **本代不产生任何 P1a/P1b/P2/P3 语义判据**，不得被引用成语义层证据；A1 整体**仍未判定**。

复现步骤见 SPEC.md §6（标定约 22 分钟 + 闸门重跑约 15 分钟，本机 CPU）。
