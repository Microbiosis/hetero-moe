"""variants.py — 阶段 D：跨模型重组的变体（**零件资格赛**）。

## 为什么不能走「拆成零件 → 重建整条链」

`assembler.py` 的路子是「把 block 拆成零件 → 用统一接口重建整条链」。
exp76 的结构勘查证明这条路在 cbramod 上**不成立**（实测，不是推测）：

1. `find_block_modules` 在 **cbramod / biot / mirepnet** 上都返回 **0 个 block**
   —— 它们的 block 里没有「同一层下同时有像注意力的和像 FFN 的」这种配对结构。
2. **CBraMod 的 token 是 4 维二维网格** (B, rows, cols, D)，Criss-Cross 是
   **卷积**实现 ⇒ 与 `interfaces.py` 的 3 维 token 约定根本不兼容
   （`assembler.find_tokenizer` 在它身上必然返回 None，实测 d_model=None）。
3. 就算硬重建，cbramod 自己的权重**搬不动**（卷积注意力 vs 注意力实现，
   结构不同族）⇒ 产出的是**随机初始化的 transformer**。
   跑 RSA 只会测到「随机 Transformer 没有脑对齐」，**测不到重组有没有用**。

⇒ 阶段 D 走两条**真正可测**的路：

## 路线 A：权重级嫁接（G1 / G2）

在**真实宿主模型树**上把供体的权重/模块搬进宿主的槽位。
- 判定是**机器可判**的：逐键形状相同才搬，搬完逐键**比对数值**。
- **零训练 ⇒ 不插学习投影**：随机投影等于白搬（D1 的「两侧学习投影」是
  给**有训练**的场景设计的）⇒ 维度不同一律记为**负结果**，不硬凑。

## 路线 B：表征级拼接（G3）

宿主最终表征 ⊕ 供体某层表征。

- 供体层特征用 **exp75 的 `layerwise_features`** 抽（同一套抽取 + `mean_all`
  池化）⇒ 「exp75 说这层有语义」与「这层在拼接里有没有用」**口径一致**。
- ⚠️ **必须配 shuffle 对照**：拼接让维度变大 ⇒ RSA 的 RDM 混合与 ridge 的
  参数增多都会让分数**天然变好**，那不是供体的功劳。
  对照行 = 宿主 ⊕ **打乱图像对应关系**的供体特征：维度完全相同、边际统计不变，
  但供体携带的图像信息被摧毁。
  ⇒ **变体 − 对照** 才是「供体贡献」；单看变体高低没有意义。

## 供体清单（D9，依据 exp75 的逐层语义承载曲线）

| 供体 | 取哪一层 | exp75 证据 |
|---|---|---|
| signal-jepa | 第 0 层（唯一显著层） | `transformer.encoder.layers.0.dropout1` r=+0.0140 p=0.005；**第 6 层显著为负** |
| mirepnet | FFN 内部 | `transformer.1.1.fn` r=+0.0633 p=0.015 |
| braintokenizer | **量化码**（最终输出） | 内部连续特征 0 显著，量化码 +0.0155（8/10 显著） |
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# 变体清单
# ---------------------------------------------------------------------------
HOST = "cbramod"          # D9：语义铺满 12 个 block ⇒ 拆得动、换得动


@dataclass
class DonorSpec:
    """一个供体：模型 + 取哪一层的特征。"""
    vid: str                     # 变体名
    donor: str                   # 供体模型名
    layer: str                   # 模块名（与 exp75 一致）或 "__final__"
    note: str = ""
    kind: str = "layer"          # "layer"（中间层）| "final"（模型最终输出）

    @property
    def label(self):
        return f"{self.host_label} ⊕ {self.donor}/{self.layer}"

    @property
    def host_label(self):
        return HOST


# ⚠️ 模块名必须与 **exp75 逐层曲线里的名字逐字一致** ⇒ 同一套抽取、同一套池化。
DONORS = [
    DonorSpec(
        vid="v1_sjepa_l0", donor="signal-jepa",
        layer="transformer.encoder.layers.0.dropout1",
        note="signal-jepa 第 0 层：exp75 里唯一显著层（+0.0140 p=0.005）",
    ),
    DonorSpec(
        vid="v2_mirepnet_ffn", donor="mirepnet",
        layer="transformer.1.1.fn",
        note="mirepnet 的 FFN：exp75 最强层（+0.0633 p=0.015）",
    ),
    DonorSpec(
        vid="v3_braintok_code", donor="braintokenizer",
        layer="__final__", kind="final",
        note="braintokenizer 的量化码（最终输出，+0.0155，8/10 显著）",
    ),
]


def donors_for_trip(donor_names):
    """按本趟能装的供体过滤（stable / dev 分两趟 ⇒ 供体也不同）。"""
    want = {d.strip() for d in donor_names if d.strip()}
    return [d for d in DONORS if d.donor in want]


# ---------------------------------------------------------------------------
# 路线 B：表征级拼接
# ---------------------------------------------------------------------------
def shuffle_images(X, rng):
    """沿**图像轴**打乱试次特征（同一图的所有重复一起换位置）。

    X : (n_img, n_rep, D) → 打乱后 (n_img, n_rep, D)

    为什么这样打乱：我们要的是「供体携带的**图像对应关系**被摧毁」，
    而「每维的边际统计、维度、噪声结构」全部保留
    ⇒ 它是「维度膨胀」这条零信息的基线。
    ⚠️ 训练集与测试集**必须各自独立打乱**（用不同的 rng 状态）：
    若两边用同一置换，供体的图像信息只是被换了个标签 ⇒ 根本没被摧毁。
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"期望 (n_img, n_rep, D)，收到 {X.shape}")
    perm = rng.permutation(X.shape[0])
    return X[perm]


def concat_feats(a, b):
    """沿特征维拼接两个 (n_img, n_rep, D) 数组。"""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape[:2] != b.shape[:2]:
        raise ValueError(f"前两维不一致：{a.shape} vs {b.shape}")
    return np.concatenate([a, b], axis=-1)


# ---------------------------------------------------------------------------
# 路线 A：权重级嫁接
# ---------------------------------------------------------------------------
def graft_weights(host_module, donor_module, tol=1e-6):
    """把供体模块的权重搬进宿主模块（**逐键同形才搬**）。

    返回 (ok, report)：
    - ok：所有同形键都搬成功且**数值逐键一致**（机器可判的核实）；
    - report：逐键的 {key, shape_src, shape_dst, copied, match}。

    ⚠️ **不做投影**：维度不同就是搬不动（零训练下插学习投影 = 插随机权重）。
    """
    import torch
    s = donor_module.state_dict()
    d = host_module.state_dict()
    report = []
    ok = True
    loadable = {}
    for k, sv in s.items():
        if not torch.is_tensor(sv) or not sv.is_floating_point():
            continue
        dv = d.get(k)
        if dv is None:
            report.append({"key": k, "src": list(sv.shape), "dst": None,
                           "copied": False, "reason": "宿主没有这个键"})
            ok = False
            continue
        if tuple(dv.shape) != tuple(sv.shape):
            report.append({"key": k, "src": list(sv.shape), "dst": list(dv.shape),
                           "copied": False, "reason": "形状不同（需投影 ⇒ 零训练不做）"})
            ok = False
            continue
        loadable[k] = sv
        report.append({"key": k, "src": list(sv.shape), "dst": list(dv.shape),
                       "copied": True})
    if loadable:
        host_module.load_state_dict(loadable, strict=False)
    # 逐键**数值核实**（provenance 说是 reuse，值就必须真的是）
    d2 = host_module.state_dict()
    for e in report:
        if not e.get("copied"):
            continue
        k = e["key"]
        same = bool(torch.allclose(s[k].float(), d2[k].float(), atol=tol))
        e["match"] = same
        if not same:
            ok = False
    return ok, report


def get_module_by_path(model, path):
    """按点分路径取子模块（找不到返回 None）。"""
    try:
        return model.get_submodule(path)
    except Exception:
        return None


def set_module_by_path(model, path, new_module):
    """把 path 处的子模块**整体替换**为 new_module（原地改父模块的属性）。

    返回 True/False。用于「模块级替换」：供体模块的类与权重整体进宿主槽位。
    """
    if "." not in path:
        return False
    parent_path, _, attr = path.rpartition(".")
    parent = get_module_by_path(model, parent_path) if parent_path else model
    if parent is None or not hasattr(parent, attr):
        return False
    setattr(parent, attr, new_module)
    return True


# ---------------------------------------------------------------------------
# 嫁接可行性：纯数据判定（吃结构勘查的 JSON，不碰模型）
# ---------------------------------------------------------------------------
VERDICT_COMPATIBLE = "compatible"        # 逐键同形 ⇒ 可搬
VERDICT_NEEDS_PROJ = "needs_projection"  # 结构同族但维度不同 ⇒ 零训练不做
VERDICT_ARCH = "arch_mismatch"           # 结构不同族 ⇒ 不可搬

# 注意力权重键的判据（各实现命名不同，但都落在这些模式里）
_ATTN_KEY = re.compile(
    r"(in_proj|out_proj|to_q|to_k|to_v|to_out|qkv|queries|keys|values)", re.I)
# FFN 权重键的判据
_FFN_KEY = re.compile(r"(linear1|linear2|w1|w2|fc1|fc2|ffn)", re.I)


def _classify_group(shapes):
    """给一个权重组分类：'attn' / 'ffn' / None（键名判据，保留给逐子模块用）。"""
    ks = list(shapes)
    if any(_ATTN_KEY.search(k) for k in ks):
        return "attn"
    if any(_FFN_KEY.search(k) for k in ks):
        return "ffn"
    return None


def layer_parts(layer):
    """把一层的**全部**二维权重张量按角色归组。

    ⚠️ 必须**摊平到层**再做：同一个 FFN 在不同实现里摊法不同 ——
    cbramod 是 `linear1` + `linear2` **两个子模块**，
    mirepnet 是 `fn` **一个 Sequential**（键名 `1.0.weight` / `1.2.weight`），
    biot 是 `fn.fn.w1/w2`。
    逐子模块分类会两边都落空 ⇒ 判定永远是 arch_mismatch（实测踩过）。

    归组规则（**按形状**，不按键名）：
    - FFN：一对互相转置的矩阵 (h,d) 与 (d,h)，h≠d；
    - attn：剩下的同形方阵（≥2 个），如 (E,E)×3 〔q/k/v/o〕或 (3E,E)〔MHA in_proj〕。

    返回 {"attn": [[shape,...], ...], "ffn": [[shape,...], ...]}，
    每个内层列表是该角色下**全部张量形状的多重集**（排序后）。
    """
    mats = []
    for c in layer.get("children", []):
        for k, v in (c.get("shapes") or {}).items():
            if len(v) == 2:
                mats.append((k, tuple(v)))

    ffn, used = [], set()
    for i, (ka, a) in enumerate(mats):
        if i in used:
            continue
        for j, (kb, b) in enumerate(mats):
            if j <= i or j in used:
                continue
            if a[0] == b[1] and b[0] == a[1] and a[0] != a[1]:
                ffn.append(sorted([a, b]))
                used.add(i)
                used.add(j)
                break

    byshape = {}
    for i, (k, s) in enumerate(mats):
        if i in used:
            continue
        byshape.setdefault(s, []).append(s)
    attn = [sorted(v) for v in byshape.values()
            if len(v) >= 2 and v[0][0] == v[0][1]]

    # 键名判据补刀：MHA 的 in_proj 是 (3E,E) 不是方阵 ⇒ 上面抓不到
    for c in layer.get("children", []):
        sh = c.get("shapes", {}) or {}
        if _classify_group(sh) == "attn":
            grp = sorted(tuple(v) for v in sh.values() if len(v) == 2)
            if grp and grp not in attn:
                attn.append(grp)
    return {"attn": attn, "ffn": ffn}


def shape_groups(layer):
    """（保留旧接口）逐子模块按角色归类。"""
    attn, ffn = [], []
    for c in layer.get("children", []):
        sh = c.get("shapes", {}) or {}
        if not sh:
            continue
        role = _classify_group(sh)
        if role == "attn":
            attn.append(sh)
        elif role == "ffn":
            ffn.append(sh)
    return {"attn": attn, "ffn": ffn}


def _groups_compatible(host_shapes, donor_shapes):
    """两个权重组能否直接搬：键集相同 **且** 逐键形状相同。"""
    hk, dk = set(host_shapes), set(donor_shapes)
    if hk != dk:
        return False, f"键集不同（宿主 {sorted(hk)[:3]} vs 供体 {sorted(dk)[:3]}）"
    for k in hk:
        if tuple(host_shapes[k]) != tuple(donor_shapes[k]):
            return False, (f"{k} 形状不同：宿主 {host_shapes[k]} vs "
                           f"供体 {donor_shapes[k]}")
    return True, f"逐键同形（{len(hk)} 键）"


def compare_layers(host_layer, donor_layer):
    """比一对「层」的 mixer / ffn，给出三值判定。

    ⚠️ 判定基于**形状多重集**（不是键名）：键名不同可以按位置重命名后搬，
    形状不同才真的搬不动。
    """
    out = {"host": host_layer.get("name"), "donor": donor_layer.get("name")}
    hp = layer_parts(host_layer)
    dp = layer_parts(donor_layer)
    for part, key in (("attn", "mixer"), ("ffn", "ffn")):
        hs, ds = hp[part], dp[part]
        if not hs or not ds:
            out[key] = {"verdict": VERDICT_ARCH,
                        "why": f"一侧没有该部件（宿主 {len(hs)} 组 / "
                               f"供体 {len(ds)} 组）"}
            continue
        hit = None
        for h in hs:
            if h in ds:
                hit = h
                break
        if hit is not None:
            out[key] = {"verdict": VERDICT_COMPATIBLE,
                        "why": f"形状多重集完全相同：{hit}"}
            continue
        out[key] = {"verdict": VERDICT_NEEDS_PROJ,
                    "why": f"形状不同：宿主 {hs[0]} vs 供体 {ds[0]} ⇒ "
                           f"零训练下要插学习投影 ⇒ 不做"}
    return out


def feasibility(host_struct, donor_struct, max_pairs=400):
    """宿主 × 供体 的逐层嫁接可行性表（纯数据判定）。"""
    hl = host_struct.get("layer_detail", [])
    dl = donor_struct.get("layer_detail", [])
    rows = []
    for h in hl:
        for d in dl:
            cmp = compare_layers(h, d)
            if (cmp["mixer"]["verdict"] == VERDICT_ARCH
                    and cmp["ffn"]["verdict"] == VERDICT_ARCH):
                continue
            rows.append(cmp)
            if len(rows) >= max_pairs:
                return rows
    return rows


def resolve_layer(struct, module_name):
    """把 exp75 的**模块名**解析到结构勘查里**所属的那一层**。

    exp75 记的是「每个 3/4 维输出」的模块名（如
    `transformer.encoder.layers.0.dropout1` —— 那是 Dropout 子模块），
    而嫁接的槽位是**层**（`transformer.encoder.layers.0`）⇒ 必须解析。

    规则：取 layer_detail 里**最长的、且是 module_name 前缀**的那个。
    """
    ld = struct.get("layer_detail", [])
    if not module_name:
        return None
    cands = [d for d in ld
             if d.get("name") == module_name
             or module_name.startswith(d.get("name", "") + ".")]
    if not cands:
        return None
    # ⚠️ 只看「有权重子模块」的那些：exp75 的模块名常常是 Dropout/Norm 这类
    #    **没有权重**的叶子（如 `...layers.0.dropout1`）⇒ 拿它当嫁接槽位
    #    会得到空的权重组，判定永远是 arch_mismatch（实测踩过）。
    weighted = [d for d in cands if any(c.get("shapes") for c in d.get("children", []))]
    pool = weighted or cands
    return max(pool, key=lambda d: (len(d.get("children", [])), len(d.get("name", ""))))


def graft_verdict(host_struct, donor_struct, donor_layer_name=None):
    """给定供体的**目标层**，给出它能不能嫁进宿主的结论（机器可判）。

    供体层名给不出时，就在供体的层明细里找第一个非 arch 的候选。
    """
    hl = host_struct.get("layer_detail", [])
    dlayer = resolve_layer(donor_struct, donor_layer_name)
    cand = [dlayer] if dlayer else donor_struct.get("layer_detail", [])
    best = None
    for d in cand:
        for h in hl:
            cmp = compare_layers(h, d)
            for part in ("mixer", "ffn"):
                if cmp[part]["verdict"] == VERDICT_COMPATIBLE:
                    return {"verdict": VERDICT_COMPATIBLE, "part": part,
                            "host": h.get("name"), "donor": d.get("name"),
                            "why": cmp[part]["why"]}
            if best is None and (cmp["mixer"]["verdict"] != VERDICT_ARCH
                                  or cmp["ffn"]["verdict"] != VERDICT_ARCH):
                best = (cmp, h, d)
    if best is None:
        return {"verdict": VERDICT_ARCH, "why": "供体没有可比的层"}
    cmp, h, d = best
    key = "mixer" if cmp["mixer"]["verdict"] != VERDICT_ARCH else "ffn"
    out = {"verdict": cmp[key]["verdict"], "host": h.get("name"),
           "donor": d.get("name"), "why": cmp[key]["why"]}
    # 附上双方的实际形状（负结果也要能核对）
    hp = layer_parts(h)
    dp = layer_parts(d)
    role = "attn" if key == "mixer" else "ffn"
    out["host_shapes"] = hp[role][:2]
    out["donor_shapes"] = dp[role][:2]
    return out
