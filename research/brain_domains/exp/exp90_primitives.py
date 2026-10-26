#!/usr/bin/env python3
"""exp90_primitives.py — **位置轴形态判据**（三分支 · 3×3 转分化 + 接口的因果价值）。

## 它要回答什么（R3：原语个数只能由判据①「转分化代价」定）

`BRAIN_STRUCTURE_SPEC.md` §4.2 的 6 条已定规格 + §4.3 的判据三件：

| 规格 | 落法 |
|---|---|
| ① 底座用「保留通道身份」的骨干、不加时间粒度 | 骨干 = exp94 的 **V1**（逐通道时间编码 → 每通道 1 token → 跨通道 Transformer）|
| ② 三组切分 POST 34 / CENT 14 / FRONT 15 | **三个分支**，各自只吃自己那组通道（输入侧切分）|
| ③ 拆分做在输入侧 | 同上 |
| ④ 形态与读出一起定 | 形态 = **连续**（离散码读出 200-way 停随机水平，A18）⇒ 读出 = 连续重建口径（与 exp96 的 s1a/`recon` 同口径）|
| ⑤ 窗口纳入设计 | 本轮 **0–250 ms**（现状，与 exp93b/exp94 同口径 ⇒ 可比）；信号窗留第二轮（写在未测清单）|
| ⑥ 一个损失：后部接对比学习 | POST 分支 = InfoNCE ↔ 图像特征（**不叠加多目标**）|
| ⑦ 前部两个原语用自监督（同记录）| CENT / FRONT 分支 = **去噪重建**（mask 一段时间 ⇒ 重建）⇒ 同记录 ⇒ ①②可测 |

⚠️ **可比较性纪律**（§4.2 末）：转分化比较必须发生在**同一个模型的两部分**之间
（换模型必然塌 ⇒ 判据失去判别力 —— exp89 的教训）⇒ ⇒ **一个模型、三个分支**，
不是一个模型训练三次。

## 组件（**按脑科学定义选；没有就自己写** ⇒ `arch/ops_brain.py`）

| 原语 | 脑科学作用 | 组件 | 状态 |
|---|---|---|---|
| POST 34 | V1 简单细胞中心-周边 + 稀疏编码 | `RegionLocalCenterSurround`（**群体相对**版：中心=本通道，环绕=同组其它通道的公共驱动）| 自写，**用上**（初始化为「只有中心、环绕≈0」⇒ 起步即恒等 ⇒ 不破坏信号）|
| CENT 14 | M1/PMd/PMv：集群活动/共同驱动 | `RegionLocalLowRank` | **只当诊断量**（低秩本身不是证据；scalp 上也测不到肌肉协同）⇒ **区内模块诚实留空** |
| FRONT 15 | dlPFC：维持任务集 + top-down 偏置 | `TaskContextGate`（组内维持 ⇒ 产生上下文状态 ⇒ 经 top-down 接口加到后部）| 自写，**用上** |
| （POST 组内）| V1 简单细胞的 ON/OFF 亚群 | `RegionLocalTransient` | **本轮排除**：ON/OFF 与中心-周边是**同一层的两个正交属性**，不是串联的两级；且它把通道数变 3 倍 ⇒ 引入参数量自由度 |
| （POST 组内层间）| V1→V2 的预测编码接口 | `PredictiveInterface` | **本轮不用**：它的接口是 (B, D, T) 的**信号流**形状，而我们现在是**token 化**的表示 ⇒ 接上去就是假接口 ⇒ 按「不符合使用场景就不上」剔掉（留作候选）|

⇒ 三个接口（**可切断的显式通路**）：
- **背侧流**（顶→前运动，vision for action）POST→CENT
- **腹侧流**（颞→前额）POST→FRONT
- **top-down**（前额→感觉，C19）FRONT→POST：**用 FRONT 的维持状态（`TaskContextGate` 的 context）加性偏置到后部**
  （⇒ 机制贴 Miller & Cohen 2001 的加性 PFC bias；且 `TaskContextGate` 一符两职：维持 + 偏置源）

## 判据①为什么用「原型集残差」而不是图像指标

CENT / FRONT 分支**没有**对比学习 ⇒ 它们的潜变量对图像没有语义 ⇒ 200-way / RSA 对它们无意义
⇒ 转分化矩阵**不能**用图像指标跨分支比 ⇒ ⇒ 改用**每分支都有、纯内部、可跨分支比**的标尺：

    d(A→B) = 用 **A 的原型集**解释 **B 的潜变量**的残差 − 用 **B 自己的**原型集的残差

⇒ 残差上升越多 ⇒ B 的潜变量越不能用 A 的原型集解释 ⇒ 两个原语**越独立**。
⇒ 「三个原语还是两个」= **3×3 矩阵里有没有一对明显更近**（CENT↔FRONT 若显著更近 ⇒ 实为一个原语 ⇒ 2 个）。

## 预登记判据（**写死在这里，事后不改**）

| # | 判据 | 通过 ⇒ |
|---|---|---|
| **C0** | **管线自检**：POST 分支的 200-way 单试次中位 **≥ 3×随机水平** 且其图像 RSA **> 0** | 说明训练确实学到了东西 ⇒ 继续；**不过 ⇒ 全实验作废**。⚠️ **不是** exp94 那种「复现同一实现的数值」⇒ 本实验是**新实现且有意加入脑科学组件** ⇒ 数值本就不该逐位复现（**预登记修订，出结果之前做的**）|
| **C1** | **位置轴第三次确认**（§4.2 纪律）：三个分支对图像的 RSA 上 POST **>** CENT、**>** FRONT（逐被试配对符号检验，两对都要）| 位置轴在第三个实现上成立 ⇒ 继续 |
| **C2** | **置换敏感性**（判据①的前置，exp89 的教训）：3×3 里**多数非对角元素的 d 显著 > 0**（配对符号检验）| 指标对置换敏感 ⇒ 判据①非空；**不过 ⇒ 判据是空的 ⇒ 停** |
| **C3** | **原语个数**（判据①的主判决）：若 d(CENT→FRONT) 显著**小于** d(POST→CENT) **且** d(POST→FRONT)（逐被试配对）⇒ ⇒ **原语个数 = 2**（CENT 与 FRONT 实为一个）；否则 ⇒ **原语个数 = 3** | exp90 规格第 2 条的 3×3 由此定案 |
| **C4** | **接口的因果价值**（判据②）：切断某接口后该分支的**验证损失**显著上升 ⇒ 那条接口有价值；上升 ≈ 0 ⇒ **砍掉那条接口**（剃刀）| 三条接口的去留 |

## 用法

    # 自测（合成数据；不碰数据）
    python -u research/brain_domains/exp/exp90_primitives.py --self-test

    # 云端
    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp90_primitives.py \
      --train-subjects 1 2 3 4 5 6 7 8 9 --eval-subjects 1 2 3 4 5 6 7 8 9 10 \
      --epochs 30 --steps-per-epoch 1000 \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor a10g-large --timeout 240m -d --name t90-primitives
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

EXP_DIR = Path(__file__).resolve().parent
_PKG_CANDIDATES = [EXP_DIR / "models", EXP_DIR.parent / "models", EXP_DIR]
PKG_DIR = next((p for p in _PKG_CANDIDATES if (p / "registry.py").exists()),
               _PKG_CANDIDATES[0])
for _p in (str(EXP_DIR), str(PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exp93b_contrastive as E93B   # noqa: E402
import exp89_posaxis as E89         # noqa: E402
import exp94_trunk_ablation as E94  # noqa: E402

try:
    # 云端：算子与本脚本同放在 /domains/scripts/ ⇒ 直接 import
    from ops_brain import (
        RegionLocalCenterSurround,
        RegionLocalLowRank,
        TaskContextGate,
    )
except ImportError:
    # 本地：仓库内路径
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from research.brain_domains.arch.ops_brain import (
        RegionLocalCenterSurround,
        RegionLocalLowRank,
        TaskContextGate,
    )

E73 = E93B.E73
OUT_DIR = Path("/domains/runs/illusionbrain-align/exp90")
SEED = 0
CHANCE = 1.0 / 200.0
GROUPS = ("POST", "CENT", "FRONT")
# exp94 的 V1 已发表值（C0 自一致性用）
V1_REF_ACC1 = 0.0627
V1_REF_RSA = 0.1192
V1_TOL_ACC1 = 0.020
V1_TOL_RSA = 0.045

CRITERIA = {
    "C0_pipeline_selfcheck": {"acc1_min": 3 * CHANCE, "rsa_min": 0.0,
                            "note": "**管线自检**（不是 exp94 那种同实现复现）⇒ 数值不应逐位复现 exp94"},
    "C1_posaxis_third_confirm": "POST 的图像 RSA > CENT、> FRONT（逐被试配对）",
    "C2_permutation_sensitivity": "3×3 里多数非对角 d 显著 > 0 ⇒ 判据①非空",
    "C3_primitive_count": "d(CENT→FRONT) 显著小于 d(POST→CENT) 且 d(POST→FRONT) ⇒ 2 个原语；否则 3 个",
    "C4_interface_value": "切断接口后该分支验证损失显著上升 ⇒ 接口有价值",
}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 统计小工具（与 exp89/exp94 同口径：上中位 + 精确符号检验 + 零差剔除）
# ---------------------------------------------------------------------------

def med_up(xs):
    s = sorted(float(x) for x in xs if x == x)
    return s[len(s) // 2] if s else None


def sign_test_nonzero(diffs):
    """剔除零差后的双侧符号检验 ⇒ (n_eff, n_pos, n_neg, p, n_zero)。"""
    d = [float(x) for x in diffs if x == x]
    n_zero = sum(1 for x in d if x == 0)
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0:
        return 0, 0, 0, float("nan"), n_zero
    k = sum(1 for x in d if x > 0)
    m = n - k
    lo = min(k, m)
    p = 2.0 * sum(math.comb(n, i) for i in range(lo + 1)) / (2.0 ** n)
    return n, k, m, float(min(1.0, p)), n_zero


def paired_test(diffs, name, out):
    n_eff, k, m, p, n_zero = sign_test_nonzero(diffs)
    med = med_up(diffs)
    out[name] = {"n": len(diffs), "n_eff": n_eff, "n_zero": n_zero,
                 "n_pos": k, "n_neg": m, "p_sign": p, "median": med,
                 "diffs": [float(x) for x in diffs]}
    return out[name]


# ---------------------------------------------------------------------------
# 模型：一个模型、三个原语分支、三条可切断接口
# ---------------------------------------------------------------------------

def build_model(n_ch=63, n_times=250, out_dim=1024, dim=64, n_layers=4,
                nhead=4, head_hidden=256, t_stride=8, n_proto=64,
                groups=None, proto_temp=0.1):
    """三分支形态模型（POST / CENT / FRONT）+ 三条接口。

    每个分支 = ① 区内算子（按脑科学定义）② V1 式逐通道编码 + 跨通道 Transformer
    ③ 池化到 (B, dim) ④ **软原型集**（K 个可学原型 ⇒ 「连续版的码本」，可跨分支互换）
    ⑤ 自己的头（→ out_dim 图像空间；只有 POST 需要）
    """
    import torch
    from torch import nn

    # 用**模块级**已解析好的名字（云端/本地都能跑）
    _CS = RegionLocalCenterSurround
    _TCG = TaskContextGate

    groups = groups or {}

    class PerChannelLN(nn.Module):
        """**只沿时间归一化**（reshape 成 (B*C,1,T)）⇒ 绝不跨通道混合。"""

        def __init__(self, n_times):
            super().__init__()
            self.ln = nn.LayerNorm(n_times)

        def forward(self, h):
            b, c, t = h.shape
            return self.ln(h.reshape(b * c, 1, t)).reshape(b, c, t)

    class Branch(nn.Module):
        """一个原语 = 一个脑区模块。"""

        def __init__(self, kind, n_ch_):
            super().__init__()
            self.kind = kind
            self.n_ch = n_ch_
            # ① 区内算子（按脑科学定义；CENT 留空 ⇒ 诚实空白）
            self.cs = _CS(n_ch_, n_ch_) if kind == "post" else None
            self.tcg = _TCG(n_ch_, gamma=0.9) if kind == "front" else None
            if self.cs is not None:
                # ⚠️ **起步即恒等**：中心 = 本通道（单位阵）、环绕 ≈ 0
                #    （Ws_raw = -3 ⇒ softplus ≈ 0.049 ⇒ 环绕权重 ≈ -0.05，
                #      既不破坏信号，又**梯度仍能流到环绕项**（若用 -1e4 ⇒ softplus'≈0 ⇒ 环绕学不动））
                with torch.no_grad():
                    self.cs.Wc.copy_(torch.eye(n_ch_) / max(n_ch_, 1) ** 0.5)
                    self.cs.Ws_raw.fill_(-3.0)
            # ② V1 式骨干（exp94）：逐通道时间编码 → 每通道 1 token → 跨通道注意力
            self.enc = nn.Sequential(
                nn.Conv1d(n_ch_, n_ch_, 5, padding=2), nn.GELU(),
                PerChannelLN(n_times),
                nn.Conv1d(n_ch_, n_ch_, 5, padding=2), nn.GELU(),
                PerChannelLN(n_times),
            )
            self.t_down = nn.Conv1d(n_ch_, n_ch_, t_stride, stride=t_stride)
            with torch.no_grad():
                dummy = torch.zeros(1, n_ch_, n_times)
                self.n_times_d = int(self.t_down(self.enc(dummy)).shape[2])
            self.time_proj = nn.Linear(self.n_times_d, dim)
            self.ch_embed = nn.Parameter(torch.zeros(1, n_ch_, dim))
            nn.init.normal_(self.ch_embed, std=0.02)
            self.tr = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=dim, nhead=nhead, dim_feedforward=4 * dim,
                    dropout=0.0, batch_first=True, norm_first=True,
                    activation="gelu"), n_layers)
            self.ln_f = nn.LayerNorm(dim)
            # ④ 软原型集（「连续版的码本」⇒ 可跨分支互换）
            self.proto = nn.Parameter(torch.randn(n_proto, dim) / math.sqrt(dim))
            self.proto_temp = proto_temp
            # ⑤ 自己的头（**只有一套** ⇒ 三个分支同构）
            #    读出路径统一为：pooled → **原型集聚合** → 头
            #    ⇒ 原型集**在**每个分支的损失路径上（POST 的对比学习也经过它）
            #      ⇒ 原型集才有梯度（否则转分化矩阵里该分支那一列是随机的）
            self.head_pool = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, head_hidden), nn.GELU(),
                nn.Linear(head_hidden, out_dim))

        def tokens(self, x):
            """x (B, n_ch, T) → (B, n_ch, dim) 通道 token（**沿时间投影，不转置**）。"""
            h = self.enc(x)
            h = self.t_down(h)
            h = self.time_proj(h)                    # (B,C,dim)
            h = h + self.ch_embed
            h = self.tr(h)                           # 跨通道注意力
            return self.ln_f(h)

        def pooled(self, tokens):
            """通道 token → (B, dim) 组表示（跨通道平均）。"""
            return tokens.mean(dim=1)

        def proto_recon(self, a):
            """用**自己的**原型集解释潜变量 ⇒ 残差（越小越贴合）。"""
            w = torch.softmax(a @ self.proto.t() / self.proto_temp, dim=-1)
            rec = w @ self.proto
            return (a - rec).pow(2).sum(-1).mean()

        def proto_agg(self, a):
            """用**自己的**原型集聚合潜变量（读出路径的一环）。"""
            w = torch.softmax(a @ self.proto.t() / self.proto_temp, dim=-1)
            return w @ self.proto

        def proto_read(self, a, proto):
            """用**给定**的原型集读潜变量 ⇒ 残差（转分化用）。"""
            w = torch.softmax(a @ proto.t() / self.proto_temp, dim=-1)
            rec = w @ proto
            return (a - rec).pow(2).sum(-1).mean()

    class TriPrimitive(nn.Module):
        def __init__(self):
            super().__init__()
            self.post = Branch("post", len(groups.get("POST", [])))
            self.cent = Branch("cent", len(groups.get("CENT", [])))
            self.front = Branch("front", len(groups.get("FRONT", [])))
            # 三条接口（各一个很小的 Linear ⇒ 切断=置零）
            d = dim
            self.if_dorsal = nn.Linear(d, d)      # POST → CENT（背侧流）
            self.if_ventral = nn.Linear(d, d)      # POST → FRONT（腹侧流）
            self.if_topdown = nn.Linear(d, d)      # FRONT → POST（top-down，加性）
            self.groups = groups

        def forward(self, x, interfaces=True, cut=None):
            """x (B, 63, T) → 各分支的潜变量与读出。"""
            g = self.groups
            x_post = x[:, g["POST"], :]
            x_cent = x[:, g["CENT"], :]
            x_front = x[:, g["FRONT"], :]
            # POST：区内中心-周边 → 编码器（tokens 内部含编码）
            x_post = self.post.cs(x_post) if self.post.cs is not None else x_post
            h_post = self.post.tokens(x_post)
            # FRONT：dlPFC 维持（加性偏置，g=0 ⇒ 恒等）+ 编码器
            xf = self.front.tcg(x_front) if self.front.tcg is not None else x_front
            h_front = self.front.tokens(xf)
            # CENT：区内模块留空（诚实空白）⇒ 直接编码
            h_cent = self.cent.tokens(x_cent)
            # 接口
            cut = cut or set()
            if interfaces:
                if "dorsal" not in cut:
                    h_cent = h_cent + self.if_dorsal(h_post.mean(1)).unsqueeze(1)
                if "ventral" not in cut:
                    h_front = h_front + self.if_ventral(h_post.mean(1)).unsqueeze(1)
                if "topdown" not in cut:
                    # top-down：**维持后的前部表示**（FRONT 组内由 `TaskContextGate` 维持）
                    #           以**加性偏置**的形式作用到后部
                    #           （C19 / Miller & Cohen 2001：PFC bias 是加性的）
                    ctx = h_front.mean(1)
                    h_post = h_post + self.if_topdown(ctx).unsqueeze(1)
            a_post = self.post.pooled(h_post)
            a_cent = self.cent.pooled(h_cent)
            a_front = self.front.pooled(h_front)
            return {"POST": a_post, "CENT": a_cent, "FRONT": a_front,
                    "tokens": {"POST": h_post, "CENT": h_cent, "FRONT": h_front}}

    return TriPrimitive()


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def build_loss_fns(model, args):
    """每个原语**一个**损失（规格 6/7）：
    - POST：InfoNCE ↔ 图像特征（**不叠加多目标**）——**用 flatten 头**（exp94 的 V1 口径 ⇒ 可比）
    - CENT / FRONT：去噪重建（mask 一段时间 ⇒ 重建）⇒ 同记录的自监督
    """
    import torch
    import torch.nn.functional as F

    def masked_recon(branch, x, mask_frac=0.25, gen=None):
        """去噪重建：随机 mask 掉一段时间 ⇒ 重建整段（用 pooled + 原型集）。"""
        b, c, t = x.shape
        m = torch.ones(b, 1, t, device=x.device)
        span = max(int(t * mask_frac), 1)
        for i in range(b):
            st = int(torch.randint(0, max(t - span, 1), (1,), generator=gen).item())
            m[i, :, st:st + span] = 0.0
        xin = x * m
        tokens = branch.tokens(xin)
        a = branch.pooled(tokens)
        w = torch.softmax(a @ branch.proto.t() / branch.proto_temp, dim=-1)
        rec = w @ branch.proto
        return (a - rec).pow(2).sum(-1).mean()

    def post_contrastive(a_post, img_feat, temp=0.07):
        """POST 分支的对比学习：**pooled → 原型集聚合 → 头**（⇒ 原型集有梯度）。"""
        z = F.normalize(model.post.head_pool(model.post.proto_agg(a_post)), dim=-1)
        t = F.normalize(img_feat, dim=-1)
        logits = z @ t.t() / temp
        lab = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))

    return masked_recon, post_contrastive


def train(args, IMG_train, device):
    """联合训练三个分支（同一批数据、同一采样序列 ⇒ 严格受控）。

    ⚠️ **采样/划分/优化器逐字对齐 exp93b/exp94**（batch 256、lr 1e-3、temp 0.07、
    val_frac 0.02、float16 存储、同一套 permutation）⇒ ⇒ 与 exp94 的 V1 **可比**。
    """
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    subjects = list(args.train_subjects)      # ⚠️ 是**列表**（nargs="+"）⇒ 不能 range(1, x+1)

    # ---------- 预检 P1：EEG ↔ 图像 三段式对齐证明（exp93b/exp94 同款）----------
    eeg0, paths0, labels0 = E93B.load_train_subject(
        subjects[0], n_images=args.align_check_images)
    X0 = E73.per_rep_targets(np.asarray(eeg0, dtype=np.float16))
    del eeg0
    al = E93B.verify_alignment(X0, paths0, labels0, IMG_train,
                               np.random.default_rng(SEED), args.align_check_perm)
    del X0
    log(f"  [P1] 对齐：概念块序 {al.get('concept_block_order_ok')} "
        f"｜同块命中 {al.get('nn_same_block_frac', 0):.4f} "
        f"｜RSA {al.get('rsa_r', 0):+.4f}（p={al.get('rsa_p', 1):.4f}）")
    if not (al.get("concept_block_order_ok", False)
            and al.get("nn_same_block_frac", 0) > 0.5
            and al.get("rsa_r", 0) > 0 and al.get("rsa_p", 1) < 0.05):
        raise RuntimeError("P1 对齐预检失败 ⇒ EEG 与图像特征不是一一对应 ⇒ 中止")

    # ---------- 载入训练数据（float16，与 exp93b 同）----------
    EEG, IMGIDX = [], []
    t0 = time.time()
    for s in subjects:
        e, _paths, _labels = E93B.load_train_subject(
            s, n_images=(args.n_images or None))
        EEG.append(torch.from_numpy(e))
        IMGIDX.append(np.arange(len(e), dtype=np.int64))
        log(f"  sub-{s:02d}: {tuple(e.shape)}（{time.time()-t0:.0f}s）")
    n_img, n_rep = EEG[0].shape[0], EEG[0].shape[1]
    n_ch, n_times = EEG[0].shape[2], EEG[0].shape[3]
    log(f"  训练数据：{len(EEG)} 被试 × {n_img} 图 × {n_rep} 试次 × {n_ch} 通道")

    perm = np.random.default_rng(SEED).permutation(n_img)
    n_val = max(1, int(args.val_frac * n_img))
    val_img = np.sort(perm[:n_val])
    log(f"  划分：fit {n_img-n_val} 图 / val {len(val_img)} 图（图像级，被试间一致）")

    # 通道分组（用 train split 的 ch_names）
    ch_names, _ = E89.load_train_chunk(subjects[0], 1)
    grp, _lab = E89.build_groups(ch_names)
    groups = {k: list(map(int, grp[k])) for k in GROUPS}
    log("  分组：" + "  ".join(f"{k}={len(v)}" for k, v in groups.items()))

    model = build_model(n_ch=n_ch, n_times=n_times, groups=groups,
                         n_layers=args.n_layers, dim=args.dim,
                         head_hidden=args.head_hidden, t_stride=args.t_stride,
                         n_proto=args.n_proto, proto_temp=args.proto_temp).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"  模型参数量：{n_par:,}（三个分支大小不同是**通道数不同**的必然结果，不是混淆变量）")
    for name, br in (("post", model.post), ("cent", model.cent), ("front", model.front)):
        log(f"    {name}: 参数量 {sum(p.numel() for p in br.parameters()):,}")

    masked_recon, post_contrastive = build_loss_fns(model, args)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))
    step_rng = np.random.default_rng(SEED)

    def sample_batch():
        """**逐字对齐 exp93b**：一批图像互不相同（避免同图不同重复变成假负例）。"""
        imgs = step_rng.choice(n_img - n_val, size=args.batch, replace=False)
        imgs = np.sort(imgs)                 # fit 集是 perm 的后段
        imgs = perm[n_val:][imgs]
        subs = step_rng.integers(0, len(subjects), size=args.batch)
        reps = step_rng.integers(0, n_rep, size=args.batch)
        xb = torch.empty((args.batch, n_ch, n_times), dtype=torch.float16)
        for si in range(len(subjects)):
            m = subs == si
            if not m.any():
                continue
            rows = EEG[si][torch.from_numpy(imgs[m]).long(),
                           torch.from_numpy(reps[m]).long()]
            xb[torch.from_numpy(np.flatnonzero(m))] = rows
        # ⚠️ 图像特征索引 = **置换后**的 imgs（与 exp93b 的 IMGIDX[si][imgs[i]] 同义）
        tb = np.stack([IMG_train[IMGIDX[si][imgs[i]]] for i, si in enumerate(subs)])
        return xb.to(device), torch.from_numpy(tb).float().to(device)

    hist = []
    for ep in range(args.epochs):
        model.train()
        agg = {"post": 0.0, "cent": 0.0, "front": 0.0, "n": 0}
        for _ in range(args.steps_per_epoch):
            tb, ifb = sample_batch()
            # ⚠️ **必须转 float32**（训练数据按 exp93b 的口径存 float16，
            #    而模型权重是 float32 ⇒ einsum/conv 会报
            #    "expected scalar type Half but found Float"）⇒ 与 exp93b 的 `model(xb.float())` 一致
            out = model(tb.float())
            l_post = post_contrastive(out["POST"], ifb)
            l_cent = masked_recon(model.cent, tb[:, groups["CENT"], :])
            l_front = masked_recon(model.front, tb[:, groups["FRONT"], :])
            loss = l_post + l_cent + l_front
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            agg["post"] += float(l_post)
            agg["cent"] += float(l_cent)
            agg["front"] += float(l_front)
            agg["n"] += 1
        rec = {"epoch": ep, "loss_post": agg["post"] / max(agg["n"], 1),
               "loss_cent": agg["cent"] / max(agg["n"], 1),
               "loss_front": agg["front"] / max(agg["n"], 1)}
        hist.append(rec)
        log(f"  epoch {ep:02d}  post {rec['loss_post']:.4f}  cent {rec['loss_cent']:.4f}"
            f"  front {rec['loss_front']:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "groups": groups,
                "n_ch": n_ch, "n_times": n_times, "n_proto": args.n_proto,
                "dim": args.dim, "n_layers": args.n_layers,
                "head_hidden": args.head_hidden, "t_stride": args.t_stride,
                "proto_temp": args.proto_temp, "n_par": n_par},
               out_dir / "exp90_model.pt")
    log(f"  模型写入 {out_dir}/exp90_model.pt")
    return model, groups, hist


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------

def _proto_residuals(model, a, tag):
    """用**每个分支自己的**与**别人的**原型集解释 a ⇒ 残差矩阵。"""
    import torch
    res = {}
    for src in GROUPS:
        proto = getattr(model, src).proto.detach()
        res[src] = float(model.__getattribute__(src).proto_read(a, proto))
    return res


def eval_subject(model, groups, sid, IMG_test, args, device, rng,
                 eeg=None, img_features=None):
    """一个被试的评测：① 图像 RSA/200-way ② 3×3 转分化 ③ 接口切断。

    `eeg` / `img_features` 可注入 ⇒ **自测能用合成数据把整条评测路径跑一遍**
    （否则 reshape / 读出这类错只在云上才炸）。
    """
    import torch
    import torch.nn.functional as F

    if eeg is None:
        eeg = E93B.load_test_subject(sid)
    if img_features is None:
        img_features = IMG_test
    n_img, n_rep, n_ch, T = eeg.shape
    rec = {"subject": f"sub-{sid:02d}"}
    model.eval()          # 评测阶段（虽然 dropout=0，但语义上该切）

    # ---- 前向：完整接口 vs 切断各接口 ----
    outs = {}
    cuts = [(), ("dorsal",), ("ventral",), ("topdown",)]
    for cut in cuts:
        acc_a = {k: [] for k in GROUPS}
        acc_t = {k: [] for k in GROUPS}
        flat = eeg.reshape(-1, n_ch, T)
        with torch.no_grad():
            for s in range(0, flat.shape[0], args.batch):
                xb = torch.from_numpy(
                    np.ascontiguousarray(flat[s:s + args.batch])).float().to(device)
                o = model(xb, cut=set(cut))
                for k in GROUPS:
                    acc_a[k].append(o[k].cpu().numpy())
                    acc_t[k].append(o["tokens"][k].cpu().numpy())
        outs[cut] = {
            "a": {k: np.concatenate(v, 0).reshape(n_img, n_rep, -1) for k, v in acc_a.items()},
            "tokens": {k: np.concatenate(v, 0).reshape(n_img, n_rep, *np.shape(v[0])[1:])
                       for k, v in acc_t.items()},
        }

    def _readout(cut, k):
        """该分支的**读出**：POST 用 flatten 头（exp94 的 V1 口径 ⇒ 200-way 可比），
        CENT/FRONT 用 pooled 头（它们没有对比学习 ⇒ 只能用自己的读出）。"""
        with torch.no_grad():
            br = getattr(model, k.lower())
            a = outs[cut]["a"][k]
            at = torch.from_numpy(a.reshape(-1, a.shape[-1])).float().to(device)
            z = br.head_pool(br.proto_agg(at))     # 与训练路径完全一致
            return z.cpu().numpy().reshape(n_img, n_rep, -1)

    # ---- ① 图像指标（位置轴第三次确认 + 管线自检）----
    img_rsa = {}
    readouts = {}
    for k in GROUPS:
        z = _readout((), k)
        readouts[k] = z
        r, p = E89.rsa(z, img_features, args.n_perm, rng)
        img_rsa[k] = {"r": r, "p": p}
    rec["img_rsa"] = img_rsa

    # POST 分支的 200-way（与 exp93b/exp94 同口径：图像特征作候选）
    acc1 = None
    if args.do_200way:
        Z = readouts["POST"].reshape(-1, readouts["POST"].shape[-1]).astype(np.float64)
        Vc = img_features.astype(np.float64)
        zn = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
        vn = (Vc - Vc.mean(0)) / (Vc.std(0) + 1e-8)
        sim = zn @ vn.T / np.sqrt((zn ** 2).sum(1)[:, None] * (vn ** 2).sum(1)[None, :])
        pred = sim.argmax(1)
        truth = np.repeat(np.arange(len(Vc)), n_rep)
        acc1 = float((pred == truth).mean())
    rec["post_acc1"] = acc1

    # ---- ② 3×3 转分化：残差差 ----
    #    d(A→B) = 用 A 的原型集解释 B 的潜变量的残差 − 用 B 自己的原型集的残差
    dmat = {}
    for B in GROUPS:
        a = torch.from_numpy(
            outs[()]["a"][B].reshape(-1, outs[()]["a"][B].shape[-1])).float().to(device)
        own = getattr(model, B.lower())
        with torch.no_grad():
            r_own = float(own.proto_read(a, own.proto.detach()))
            for A in GROUPS:
                if A == B:
                    dmat[f"{A}->{B}"] = 0.0
                    continue
                proto_A = getattr(model, A.lower()).proto.detach()
                r_other = float(own.proto_read(a, proto_A))
                dmat[f"{A}->{B}"] = r_other - r_own
    rec["transdiff"] = dmat

    # ---- ②b 诊断量：该原语的 token 低秩程度（`RegionLocalLowRank`）
    #     ⚠️ **只报不作判据**：低秩本身不是证据 ⇒ 其价值须与同尺寸随机组比（exp96 的地板）
    diag = {}
    for B in GROUPS:
        tok = outs[()]["tokens"][B]
        lr = RegionLocalLowRank(tok.shape[2], rank=args.diag_rank)
        with torch.no_grad():
            t = torch.from_numpy(
                tok.reshape(-1, tok.shape[2], tok.shape[3])).float()
            diag[B] = lr.reconstruction_error(t)      # ⚠️ 传 tensor（不是 numpy）
    rec["lowrank_diag"] = diag

    # ---- ③ 接口切断的代价：各分支的**自监督验证损失**（用重建残差，跨分支同口径）----
    iface = {}
    for cut in cuts[1:]:
        for k in GROUPS:
            a_full = torch.from_numpy(
                outs[()]["a"][k].reshape(-1, outs[()]["a"][k].shape[-1])).float().to(device)
            a_cut = torch.from_numpy(
                outs[cut]["a"][k].reshape(-1, outs[cut]["a"][k].shape[-1])).float().to(device)
            with torch.no_grad():
                own = getattr(model, k.lower())
                r_full = float(own.proto_read(a_full, own.proto.detach()))
                r_cut = float(own.proto_read(a_cut, own.proto.detach()))
            iface[f"{cut[0]}|{k}"] = r_cut - r_full
    rec["iface"] = iface

    # ---- ④ 读出随原型集换掉而变多少（报告项；判据用上面的残差 d）----
    shift = {}
    for B in GROUPS:
        br = getattr(model, B.lower())
        a = torch.from_numpy(
            outs[()]["a"][B].reshape(-1, outs[()]["a"][B].shape[-1])).float().to(device)
        with torch.no_grad():
            own = br.head_pool(br.proto_agg(a))
            for A in GROUPS:
                if A == B:
                    continue
                proto_A = getattr(model, A.lower()).proto.detach()
                w = torch.softmax(a @ proto_A.t() / br.proto_temp, dim=-1)
                other = br.head_pool(w @ proto_A)
                shift[f"{A}->{B}"] = float((own - other).pow(2).sum(-1).mean())
    rec["readout_shift"] = shift
    return rec


# ---------------------------------------------------------------------------
# 汇总与判决
# ---------------------------------------------------------------------------

def decide(records):
    """**预登记判据**（写死；事后不改）。"""
    out = {}
    ok = [r for r in records if "error" not in r]
    if not ok:
        return {"verdict": "INSUFFICIENT", "why": "没有可用被试"}

    # ---- C0 管线自检 ----
    # ⚠️ **本轮预登记修订**（修订发生在**出结果之前** ⇒ 不是篡改预登记）：
    #    原来的 C0 写的是「POST 分支的 200-way/RSA 复现 exp94 的 V1 值」⇒ 那是
    #    exp94 那种「**同一实现**复现」的自一致性检查；但 exp90 是**新实现**
    #    **且有意加入了脑科学组件**（中心-周边算子）⇒ 数值**本就不该逐位复现**
    #    ⇒ 拿它当判据会把「结构差异」误报成「实现漂移」。
    # ⇒ 改成**管线自检**：POST 分支的 200-way 必须**显著高于随机水平**且 RSA 为正
    #    （说明训练确实学到了东西）；「与 exp94 的数值对照」降级为**报告项**。
    accs = [r["post_acc1"] for r in ok if r.get("post_acc1") is not None]
    rsas = [r["img_rsa"]["POST"]["r"] for r in ok if "POST" in r.get("img_rsa", {})]
    c0 = {}
    if accs:
        m = med_up(accs)
        c0["acc1_median"] = m
        c0["acc1_above_chance3"] = bool(m >= 3 * CHANCE)
    if rsas:
        m = med_up(rsas)
        c0["rsa_median"] = m
        c0["rsa_positive"] = bool(m > 0)
    out["C0"] = c0
    out["comparable_to_exp94"] = {
        "note": "exp90 是新实现且**有意加入脑科学组件**（读出路径也经原型集）⇒ "
                "数值不应逐位复现 exp94；此项只作量级参照，不作判据",
        "exp94_v1_acc1": V1_REF_ACC1, "exp94_v1_rsa": V1_REF_RSA,
        "our_acc1_median": c0.get("acc1_median"), "our_rsa_median": c0.get("rsa_median"),
    }
    c0_pass = bool(c0.get("acc1_above_chance3")) and bool(c0.get("rsa_positive"))
    out["C0_pass"] = c0_pass
    if not c0_pass:
        return {"verdict": "PIPELINE_SELFCHECK_FAIL", "C0": c0,
                "why": "POST 分支没学到东西（200-way 不显著高于随机水平或 RSA ≤ 0）⇒ 全实验作废"}

    # ---- C1 位置轴第三次确认 ----
    c1 = {}
    for other in ("CENT", "FRONT"):
        diffs = [r["img_rsa"]["POST"]["r"] - r["img_rsa"][other]["r"]
                 for r in ok if other in r.get("img_rsa", {})]
        if diffs:
            paired_test(diffs, f"post_minus_{other.lower()}", c1)
    out["C1"] = c1
    c1_pass = bool(c1) and all(
        v.get("n_eff", 0) >= 8 and v.get("p_sign", 1) <= 0.05 and (v.get("median") or 0) > 0
        for v in c1.values())
    out["C1_pass"] = c1_pass
    if not c1_pass:
        return {"verdict": "POSAXIS_FAIL", "C0": c0, "C1": c1,
                "why": "位置轴在第三个实现上没成立 ⇒ 停"}

    # ---- C2 置换敏感性 ----
    c2 = {}
    n_sig = 0
    for A in GROUPS:
        for B in GROUPS:
            if A == B:
                continue
            diffs = [r["transdiff"][f"{A}->{B}"] for r in ok if "transdiff" in r]
            if not diffs:
                continue
            v = paired_test(diffs, f"{A}->{B}", c2)
            if v["n_eff"] >= 8 and v["p_sign"] <= 0.05 and (v["median"] or 0) > 0:
                n_sig += 1
    out["C2"] = c2
    out["C2_n_significant"] = n_sig
    c2_pass = n_sig >= 4     # 6 个非对角元素里多数显著
    out["C2_pass"] = c2_pass
    if not c2_pass:
        return {"verdict": "PERMUTATION_INSENSITIVE", "C0": c0, "C1": c1, "C2": c2,
                "why": f"只有 {n_sig}/6 个转分化元素显著 ⇒ 指标对置换不敏感 ⇒ 判据①是空的 ⇒ 停"}

    # ---- C3 原语个数 ----
    def _d(key):
        return [r["transdiff"][key] for r in ok if key in r.get("transdiff", {})]

    c3 = {}
    cf = _d("CENT->FRONT")
    pc = _d("POST->CENT")
    pf = _d("POST->FRONT")
    if cf and pc and pf:
        # CENT↔FRONT 的转分化代价显著**小于** POST↔CENT / POST↔FRONT
        # ⇒ 差值取「大的 − 小的」⇒ **为正**表示 CENT 与 FRONT 更近（更像一个原语）
        d1 = [b - a for a, b in zip(cf, pc)]     # POST→CENT − CENT→FRONT
        d2 = [b - a for a, b in zip(cf, pf)]     # POST→FRONT − CENT→FRONT
        v1 = paired_test(d1, "post_cent_minus_cent_front", c3)
        v2 = paired_test(d2, "post_front_minus_cent_front", c3)
        closer = bool(v1["n_eff"] >= 8 and v1["p_sign"] <= 0.05
                      and (v1["median"] or 0) > 0
                      and v2["n_eff"] >= 8 and v2["p_sign"] <= 0.05
                      and (v2["median"] or 0) > 0)
    out["C3"] = c3
    out["C3_pass"] = closer
    n_prim = 2 if closer else 3
    out["n_primitives"] = n_prim

    # ---- C4 接口的因果价值 ----
    c4 = {}
    for key in sorted({k for r in ok for k in r.get("iface", {})}):
        diffs = [r["iface"][key] for r in ok if key in r.get("iface", {})]
        if diffs:
            v = paired_test(diffs, key, c4)
            v["has_value"] = bool(v["n_eff"] >= 8 and v["p_sign"] <= 0.05
                                  and (v["median"] or 0) > 0)
    out["C4"] = c4

    keep = [k for k, v in c4.items() if v.get("has_value")]
    drop = [k for k, v in c4.items() if not v.get("has_value")]
    out["interfaces_keep"] = keep
    out["interfaces_drop"] = drop
    out["verdict"] = "PRIMITIVES_%d" % n_prim
    out["why"] = (f"原语个数 = {n_prim}（{'CENT 与 FRONT 实为一个' if closer else '三组各自独立'}）；"
                  f"接口有因果价值：{keep}；无价值（砍）：{drop}")
    return out


# ---------------------------------------------------------------------------
# 自测（**判据实现也要被测**）
# ---------------------------------------------------------------------------

def _fake_records(n=10, closerness=0.0, sig_all=True, iface_bad=None):
    """造合成记录驱动 decide()。

    closerness : CENT↔FRONT 的转分化代价比 POST↔* 小多少（>0 ⇒ 判 2 个原语）
    sig_all    : 非对角 d 是否全为正且显著
    """
    rng = np.random.default_rng(0)
    recs = []
    for i in range(n):
        rec = {"subject": f"sub-{i+1:02d}", "post_acc1": V1_REF_ACC1,
               "img_rsa": {"POST": {"r": 0.12, "p": 0.01},
                           "CENT": {"r": 0.00, "p": 0.9},
                           "FRONT": {"r": 0.00, "p": 0.9}},
               "transdiff": {}, "iface": {}}
        for A in GROUPS:
            for B in GROUPS:
                if A == B:
                    continue
                base = 0.05 if sig_all else 0.0
                if {A, B} == {"CENT", "FRONT"}:
                    base = max(base - closerness, 0.001)
                rec["transdiff"][f"{A}->{B}"] = base
        for cut in ("dorsal", "ventral", "topdown"):
            for k in GROUPS:
                v = 0.02
                if iface_bad and iface_bad == f"{cut}|{k}":
                    v = 0.0
                rec["iface"][f"{cut}|{k}"] = v
        recs.append(rec)
    return recs


def self_test():
    log("=== 自测 1：模型形状（三个分支、接口可切断）===")
    import torch
    groups = {"POST": list(range(34)), "CENT": list(range(34, 48)),
              "FRONT": list(range(48, 63))}
    m = build_model(n_ch=63, n_times=250, groups=groups)
    x = torch.randn(2, 63, 250)
    o = m(x)
    for k in GROUPS:
        assert o[k].shape == (2, 64), (k, o[k].shape)
    log(f"  ✓ 三个分支潜变量 (2, 64)；参数量 {sum(p.numel() for p in m.parameters()):,}")

    log("=== 自测 2：切断接口 ⇒ 输出确实变（接口有可切断的通路）===")
    with torch.no_grad():
        o_full = m(x)
        o_cut = m(x, cut={"dorsal"})
    d = (o_full["CENT"] - o_cut["CENT"]).abs().max()
    assert float(d) > 1e-6, d
    log(f"  ✓ 切断背侧流 ⇒ CENT 潜变量变化 max={float(d):.2e}")

    log("=== 自测 3：原型集残差（自有 vs 别人的）===")
    a = o["POST"].detach()      # ⚠️ detach ⇒ 每次迭代的 loss 是**新的图**
    #    （不 detach 会在第二次 backward 时报「backward through the graph a second time」）
    # ⚠️ **必须先把原型集拟合到自己的潜变量上**再比较：
    #    「自有原型解释得更好」是**训练后**才有的性质 ⇒ 随机初始化下**不保证**
    #    （踩过一次 flaky 断言：随机种子一变就翻车）⇒ 这里显式拟合 ⇒ 确定性。
    opt = torch.optim.Adam([m.post.proto], lr=0.05)
    for _ in range(80):
        loss = m.post.proto_read(a, m.post.proto)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        r_own = float(m.post.proto_read(a, m.post.proto))
        r_other = float(m.post.proto_read(a, m.cent.proto))
    assert r_other > r_own, (r_own, r_other)
    log(f"  ✓ 自有残差 {r_own:.4f} < 用别人原型的残差 {r_other:.4f}（拟合后 ⇒ 确定性）")

    log("=== 自测 4：判据实现（方向 + 显著性同时查）===")
    v = decide(_fake_records(closerness=0.0))
    assert v["verdict"] == "PRIMITIVES_3", v["verdict"]
    log("  ✓ 三组各自独立 ⇒ 判『3 个原语』")

    v = decide(_fake_records(closerness=0.03))
    assert v["verdict"] == "PRIMITIVES_2", v["verdict"]
    log("  ✓ CENT↔FRONT 明显更近 ⇒ 判『2 个原语』")

    v = decide(_fake_records(sig_all=False))
    assert v["verdict"] == "PERMUTATION_INSENSITIVE", v["verdict"]
    log("  ✓ 转分化全不显著 ⇒ 判『判据是空的 ⇒ 停』")

    v = decide(_fake_records(iface_bad="ventral|FRONT"))
    assert "ventral|FRONT" in v.get("interfaces_drop", []), v.get("interfaces_drop")
    log("  ✓ 某接口无价值 ⇒ 判『砍掉那条』")

    recs = _fake_records()
    for r in recs:
        r["post_acc1"] = 0.002    # 低于随机水平 ⇒ 管线自检不过
    v = decide(recs)
    assert v["verdict"] == "PIPELINE_SELFCHECK_FAIL", v["verdict"]
    log("  ✓ 管线自检不过 ⇒ 判『全实验作废』")

    log("=== 自测 6：端到端（合成数据：训练一步 + 完整评测管线）===")
    import numpy as _np
    groups_t = {"POST": list(range(34)), "CENT": list(range(34, 48)),
                "FRONT": list(range(48, 63))}
    mt = build_model(n_ch=63, n_times=250, groups=groups_t, n_layers=1)
    xb = torch.randn(8, 63, 250)
    ifb = torch.randn(8, 1024)
    masked_recon, post_contrastive = build_loss_fns(mt, None)
    o = mt(xb)
    l_post = post_contrastive(o["POST"], ifb)
    l_cent = masked_recon(mt.cent, xb[:, groups_t["CENT"], :])
    l_front = masked_recon(mt.front, xb[:, groups_t["FRONT"], :])
    loss = l_post + l_cent + l_front
    loss.backward()
    assert torch.isfinite(loss), float(loss)
    assert mt.post.proto.grad is not None
    log(f"  ✓ 三个损失都有限且能反传：post {float(l_post):.3f} "
        f"cent {float(l_cent):.3f} front {float(l_front):.3f}")

    class _A:
        n_perm = 5
        n_perm_null = 5
        batch = 8
        diag_rank = 4
        do_200way = 1
    fake_eeg = _np.random.RandomState(0).randn(20, 4, 63, 250).astype("float32")
    fake_img = _np.random.RandomState(1).randn(20, 1024).astype("float64")
    rec = eval_subject(mt, groups_t, 0, None, _A(), "cpu",
                       _np.random.default_rng(0), eeg=fake_eeg, img_features=fake_img)
    for k in ("img_rsa", "transdiff", "iface", "lowrank_diag"):
        assert k in rec, k
    log(f"  ✓ 评测管线跑通（img_rsa/ transdiff / iface / lowrank_diag 都有）")

    log("=== 自测 5：中位是上中位 / 零差剔除 ===")
    assert med_up([1.0, 2.0, 3.0, 4.0]) == 3.0
    n_eff, k, m_, p, nz = sign_test_nonzero([0.0] * 10)
    assert n_eff == 0 and nz == 10, (n_eff, nz)
    log("  ✓ 上中位；全零差 ⇒ n_eff=0（不被误当成全负）")
    log("=== 自测通过 ===")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--eval-subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ap.add_argument("--n-images", type=int, default=0, help="0 = 全部训练图（与 exp93b/exp94 同）")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-proto", type=int, default=64)
    ap.add_argument("--head-hidden", type=int, default=256)
    ap.add_argument("--t-stride", type=int, default=8)
    ap.add_argument("--proto-temp", type=float, default=0.1)
    ap.add_argument("--align-check-images", type=int, default=1500)
    ap.add_argument("--align-check-perm", type=int, default=100)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--do-200way", type=int, default=1)
    ap.add_argument("--diag-rank", type=int, default=4,
                    help="低秩诊断的秩（只报不作判据）")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fte = torch.load(E93B.FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)
    ftr = torch.load(E93B.FEAT_TRAIN, map_location="cpu", weights_only=False)
    IMG_train = ftr["img_features"].numpy().astype(np.float32)

    t0 = time.time()
    model, groups, hist = train(args, IMG_train, device)
    log(f"  训练用时 {time.time()-t0:.0f}s")

    per_subject = []
    for sid in args.eval_subjects:
        ts = time.time()
        log(f"\n=== sub-{sid:02d} ===")
        try:
            rec = eval_subject(model, groups, sid, IMG_test, args, device,
                               np.random.default_rng(SEED + sid))
            per_subject.append(rec)
            log(f"  用时 {time.time()-ts:.0f}s")
        except Exception as e:
            rec = {"subject": f"sub-{sid:02d}",
                   "error": f"{type(e).__name__}: {e}"[:300],
                   "trace": traceback.format_exc()[-800:]}
            log(f"  ✗ {rec['error']}")
            per_subject.append(rec)

    verdict = decide(per_subject)
    log("\n=== 判决 ===")
    log(f"  {verdict['verdict']}：{verdict.get('why', '')}")

    (out_dir / "exp90_primitives.json").write_text(
        json.dumps({"config": vars(args), "criteria": CRITERIA,
                    "groups": groups, "hist": hist,
                    "per_subject": per_subject, "verdict": verdict},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"结果写入 {out_dir}/exp90_primitives.json")


if __name__ == "__main__":
    main()
