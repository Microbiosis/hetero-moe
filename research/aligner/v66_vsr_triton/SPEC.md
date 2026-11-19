# V66.0 — VSR 真视觉任务 + MoA Triton 融合内核库

> **文档 ID**: HMF-VSR-TRITON-1.0
> **日期**: 2026-11-12（补登记；工作完成于 2026-11-04，commit `e6709ca` / `c4e32a2` + 5 个修复）
> **状态**: 内核位级正确性与 HBM 流量账**已本地实测**；端到端精度**待 Kaggle 批量验证**
> **归属**: aligner 线，接 v65
> **一句话**: 一代两件事——VSR 真视觉空间关系任务（caption 只断言空间关系、必须看图判断真假），以及四个 MoA Triton 融合内核，附 numpy 逐 program 位级验证器。

---

## 0. 本代性质

**本代是 v63–v74 中唯一能在本地 CPU 上复算硬指标的一代。**

`TritonTileSimulator` 是 numpy 写的**逐 program 模拟器**：它把四个 `@triton.jit` 内核的分块策略、SRAM 寄存器累加、在线 softmax 重标定、块稀疏跳过逐 program 复算，于是：

- 「内核写对了没有」可以在**没有 GPU** 的情况下给出**位级**答案；
- 「融合省了多少 HBM 流量」可以给出**精确字节账**，而不是估算。

但本代**仍然没有端到端精度数字**（融合内核要接进训练才能测精度，需 Kaggle 批量跑）。所以 `claim` 分两层写清楚：位级正确性与流量账是**实测**；端到端收益是**待验证**。

---

## 1. VSR 真视觉任务

`extract_vsr_v66.py`：caption **只断言空间关系**（如「杯子在盘子左边」），不描述场景；标签要求**看图判断真假**。⇒ 纯语言模型靠先验猜不了，必须用视觉证据。

配套修掉一个真实 bug（commit `aa8a8d7`）：ViT hidden 曾**全零**——原因是图像下载失败时静默填了占位图却没报错。修法三条：请求加 User-Agent、失败时填灰图占位、并加**下载成功率断言**。

---

## 2. 四个 Triton 融合内核

| 内核 | 融合了什么 | 省的是 |
|---|---|---|
| K1 | Branch-RMSNorm + LayerScale（前向+反向） | 14 遍张量扫描 → 5 遍 |
| K2 | 多 proposer GEMM + SwiGLU（3D 分块） | Gate/Up 中间张量**不回写 HBM** |
| K3 | 稀疏 Top-K gather-reduce 聚合 | 只读 K 个 proposer 而非 N 个 |
| K4 | 块稀疏 MoA flash attention（在线 softmax） | 整块被掩掉时**跳过 K/V 的 HBM 载入与 GEMM** |

K4 的三个头模式按 `h % 3` 轮转：sink+local、dilated、full-causal。

---

## 3. 权威测试：8 项（本地 CPU，全部位级）

| # | 判据 | 实测 |
|---|---|---|
| 1 | K1 前向与反向均与 eager 参考一致 | max diff **4.44e-16** |
| 2 | K1 带宽下降 | **63.4%** |
| 3 | K2 分块累加与 eager einsum 一致 | max diff **2.84e-14** |
| 4 | K2 中间张量不回写（流量账） | 省 **9728 字节**激活流量 |
| 5 | K3 稀疏聚合与 dense 参考一致 | max diff **0.00e+00** |
| 6 | K3 只读 K 个 proposer | 流量比 **0.389**（K/N = 2/8） |
| 7 | K4 前向与参考一致且真跳过块 | max diff **5.55e-16**，跳过 tile **13.9%** |
| 8 | 四个内核入口齐备 | 4 个 `simulate_*` |

第 1/3/5/7 项是位级一致性（与 eager 参考实现的 max diff），第 2/4/6 是精确 HBM 字节账。

---

## 4. 本代踩过的坑（记下来，避免重犯）

### 4.1 SSM 的 launch-bound 与显存三角

`moa_fusion.py` 的 SSM 分支在这一代被反复折腾，留下的结论是：

| 尝试 | 结局 | commit |
|---|---|---|
| O(L) 串行扫描 | launch-bound，GPU 利用率极低 | `d57b7c5` |
| Chunkwise SSD（O(1) 并行扫描） | Q×Q 中间张量 4GB → OOM | `bc7bd46` |
| Kogge-Stone（O(log L)） | D256×N16 状态下 1.5GB 中间张量 → OOM | `05cc0c1` |
| 回退串行 + d_state 压缩 | `MOA_DSTATE` 默认 4（压缩 4 倍） | `05cc0c1` / `e4a2373` |

⇒ **SSM 分支最终在 v67 被整个去掉**（见 `v67_three_branch`）。本代记录这段是因为：它证明了「算子能写对」和「算子能在 T4 上跑」是两件事。

### 4.2 numpy 模拟器的形状契约

`simulate_fused_branch_rmsnorm_scale` 的 `scale` 形状是 **`[N, D]`**（每分支一组 LayerScale），不是 `[N]`；`simulate_block_sparse_moa_flash_attention` 返回 **3 个值**（`out, stats, report`），不是 2 个。这两条已固化进权威测试。

---

## 5. 待执行：Kaggle 批量验证

```bash
python extract_vsr_v66.py          # 提 VSR 数据
USE_MOA=1 python train_v2.py        # 接融合内核进训练
```

**产物落盘位置**：数据 → HF dataset `redmingwei/hetero-moe-cache-v66`；训练产物 → HF `redmingwei/HeteroMoEProjSpace-v2/checkpoints/v66.*`；评估产物 → `research/aligner/v66_vsr_triton/results/*.json`。

验证完成后应回填 `results/metrics.json` 的 `pending` 段。

---

## 6. 复现

```bash
python research/aligner/v66_vsr_triton/tests/test_package.py     # 8/8，本地 CPU
```

---

*本代内核正确性与流量账是**实测**；端到端精度收益**待验证**。引用时不得把前者说成后者。*
