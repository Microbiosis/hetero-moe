# V68.0 — 架构图 + zero-OOM SSM 四分支 + 单模态消融

> **文档 ID**: HMF-ARCH-ABLATION-1.0
> **日期**: 2026-11-12（补登记；工作完成于 2026-11-04，commit `6f21312` / `0b95cdf` / `f71389d`）
> **状态**: 内核机制**已本地实测**；端到端精度**已被后续实测否掉**（见 §3）
> **归属**: aligner 线，接 v67
> **一句话**: 恢复四分支并配 zero-OOM SSM 路径（autograd checkpointing + 状态 tile），交付三模态异构融合 pipeline 架构图，加单模态消融开关。

---

## 0. 本代性质

本代三件事，其中**两件成功、一件失败**：

- ✅ zero-OOM SSM 路径：机制**已本地实测**（前向位级一致、反向与有限差分一致、峰值显存低于两种旧实现）；
- ✅ 架构图与单模态消融开关：代码就位；
- ❌ 四分支路线本身：**被后续实测否掉**（§3）。

失败的那条必须留在记录里——否则下一个人会以为"四分支是待验证的更好方案"。

---

## 1. zero-OOM SSM 机制

`ssm_zero_oom_checkpointed_forward` / `..._backward` 的三条设计：

1. **绝不分配整序列 `[B, L, D, d_state]`** 也不分配 6 维 SSD 张量 `[B, n_chunks, Q, Q, D, d_state]`；
2. **状态微切块**：`d_state` 切成宽度 `state_tile ≤ 4` 的微块逐块处理 ⇒ 峰值工作显存与 `d_state=4` 同量级；
3. **只保存 chunk 边界状态** `h_boundaries [B, n_chunks, D, d_state]`，反向时在 `(chunk, state_tile)` 窗口内**即时重算**块内状态。

commit `0b95cdf` 记录：`d_state=16` 时显存仅 **4.6MB**。

---

## 2. 架构图与单模态消融

| 项 | 内容 |
|---|---|
| 架构图 | `hetero_arch_v68.png`（2385×1260，紫色论文风，三模态异构融合 pipeline） |
| 消融开关 | `ABL=all`（融合，默认）/ `text`（只文本，拿掉 ViT）/ `image`（只视觉） |
| 实现 | `train_v2.py` 的 `_feats(hq, hb, hv)` helper，按 `ABL` 决定是否喂 ViT |

消融开关存在的理由：四槽把三模态预混合进 canonical，**增益到底来自视觉通道还是来自多一个文本底座**，不消融就分不清。

---

## 3. 本代路线被否掉（负结果，保留）

| 配置 | held-out | 来源 |
|---|---|---|
| 3 分支（Attn/Conv/SwiGLU） | **0.64** | commit `c496d6d`（v71 记录） |
| 4 分支 + zero-OOM SSM | 0.53 | 同上 |

⇒ 后续 v67 去掉 SSM 分支。**zero-OOM 的实现本身是对的**（§4 的测试证明前向反向都对），
但**把它接回 proposer 会掉 0.11**。这两件事必须分开说：机制成立 ≠ 路线值得。

---

## 4. 权威测试：6 项（本地 CPU）

| # | 判据 | 实测 |
|---|---|---|
| 1 | zero-OOM 前向与串行参考实现一致 | max diff **5.55e-17** |
| 2 | 解析反向与**数值有限差分**一致（u/dt/A/B/C 五个参数全查） | 最大相对误差 **3.57e-09** |
| 3 | 峰值显存低于 SSD 与 Kogge-Stone 两种旧实现 | new < ssd 且 new < ks |
| 4 | 架构图是合法 PNG 且尺寸足够 | 2385×1260，252521 字节 |
| 5 | ABL 消融开关真能拿掉 ViT 通道 | all/text/image 三档，默认 all |
| 6 | 四分支实验底座完整保留 | 4 分支轮转 + 4 个 SSM 核齐备 |

第 2 项是本代最硬的证据：**解析梯度对五个参数分别做中心差分验证**，相对误差 3.6e-09。

---

## 5. 复现

```bash
python research/aligner/v68_arch_ablation/tests/test_package.py     # 6/6，本地 CPU
```

---

*本代的机制已实测，但本代的路线已被否掉；两者都要说。*
