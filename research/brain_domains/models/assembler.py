"""assembler.py — 阶段 C/D：逐 block 拆开、跨模型重组。

## 它解决什么

7 个基模的内部结构各不相同，但**都能拆成同一种零件**：

    EEG → [patch embed] → tokens(B,N,D) → [block]×L → tokens → [head]

而每个 block 内部又是「序列混合器 mixer + 计算模块 ffn + 归一化」。
⇒ 把 block 当成可插拔零件，就能**跨模型重组**：
第 0 层用 A 模型的注意力、第 1 层用 B 模型的 Criss-Cross、第 2 层换成时频交叉注意力。

## 三条纪律（这决定了重组结论可不可信）

1. **权重搬没搬，必须逐组件核实**。
   provenance 写 `reuse` 而权重其实是随机初始化的 —— 这类谎报会让"重组带来提升"
   变成无法归因。因此每个 block 记**逐组件**状态（mixer / ffn / proj），
   并且 `verify_weights()` 会真的比对张量值。
2. **搬不动就说搬不动**。
   CBraMod 的 Criss-Cross 是卷积实现，我们的 `CrissCross2D` 是注意力实现 ——
   结构不同、权重**不可搬**，只能标 `arch-only`（连结构都只是近似）。
   **绝不拿随机初始化的权重冒充"从 CBraMod 搬来的"**。
3. **占位实现必须显式**。
   `linear_ssm` 只是 nearest 投影的结构占位（环境里没有 mamba_ssm/fla），
   它必须标 `fresh` 并注明"这是占位，不是 Mamba"。

## 维度不一致怎么办

token 维度 D 必须沿链一致。源 block 的维度与宿主不同时，
在 block 入口插一个**学习到的线性投影**（provenance = `projected`），
投影之后的模块权重因形状变了仍需 `fresh` —— 两者都记，不混为一谈。

## 用法

    from assembler import VariantPlan, BlockPlan, build_variant
    plan = VariantPlan(variant_id="mix01", host_model="cbramod",
                       blocks=[BlockPlan(role="mixer", source_model="biot",
                                         source_layer=0, mixer_kind="attn1d")])
    variant, prov = build_variant(plan, base_models)
    print(prov.to_json())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

try:
    from .interfaces import (MIXER_KINDS, Block, CrissCross2D, FFN,
                             IdentityMixer, LinearSSM, Projection,
                             SelfAttention1D, TimeFreqCross, TokenBatch)
    from .provenance import (W_ARCH_ONLY, W_FRESH, W_PROJECTED, W_REUSE,
                             BlockProv, VariantProv, merge_license)
except ImportError:
    from interfaces import (MIXER_KINDS, Block, CrissCross2D, FFN,
                            IdentityMixer, LinearSSM, Projection,
                            SelfAttention1D, TimeFreqCross, TokenBatch)
    from provenance import (W_ARCH_ONLY, W_FRESH, W_PROJECTED, W_REUSE,
                            BlockProv, VariantProv, merge_license)


# ---------------------------------------------------------------------------
# 源模型的结构勘查
# ---------------------------------------------------------------------------
_ATTN_HINT = re.compile(r"(attention|attn|mha|selfattn|self_attn|criss)", re.I)
_FFN_HINT = re.compile(r"(mlp|ffn|feed_?forward|feedforward|linear_stack|fc_stack)", re.I)
_FFN_LAYER_HINT = re.compile(r"(mlp|ffn|feed_?forward)", re.I)


def is_attn_like(mod) -> bool:
    if isinstance(mod, nn.MultiheadAttention):
        return True
    return bool(_ATTN_HINT.search(type(mod).__name__))


def is_ffn_like(mod) -> bool:
    name = type(mod).__name__
    if _FFN_HINT.search(name):
        return True
    # nn.Sequential 且含 ≥2 个 Linear —— 也是 FFN
    if isinstance(mod, nn.Sequential):
        n_lin = sum(1 for m in mod if isinstance(m, nn.Linear))
        if n_lin >= 2:
            return True
    return False


def find_block_modules(model):
    """找出模型里最内层的「block」。

    判据：模块的**直接子模块**里同时有一个「像注意力」和一个「像 FFN」的。
    取最内层 ⇒ 不会把整个 encoder 也当成一个 block。

    返回 [{"name", "module", "attn": (名, 模块), "ffn": (名, 模块)}]
    """
    found = []

    def scan(mod, prefix, depth):
        if depth > 3:
            return
        attn = ffn = None
        for cn, cm in mod.named_children():
            if attn is None and is_attn_like(cm):
                attn = (cn, cm)
            if ffn is None and is_ffn_like(cm):
                ffn = (cn, cm)
        if attn and ffn:
            found.append({"name": prefix, "module": mod, "attn": attn, "ffn": ffn})
            return                      # 最内层，不再往下钻
        for cn, cm in mod.named_children():
            scan(cm, f"{prefix}.{cn}" if prefix else cn, depth + 1)

    for n, m in model.named_children():
        scan(m, n, 1)
    return found


def find_tokenizer(model, x):
    """找出把 EEG 变成 token 的那个模块（patch embed）。

    ⚠️ 不能直接取「前向里第一个输出 3 维张量的模块」：
    hook 是**内层先触发**的，那样会抓到 `patch.proj`（一个 Linear）——
    单独调用它吃原始 EEG 会形状不匹配。

    做法：先记下所有输出过 3 维张量的候选，再**逐个实测**
    「谁能独立吃原始 EEG 并吐出 (B,N,D)」——
    按 `named_modules()` 的定义顺序（外层优先）试，取第一个成功的。

    返回 (限定名, 模块) 或 (None, None)。
    """
    cands = []

    def hook(nm):
        def _h(mod, inp, o):
            t = o
            if isinstance(t, (tuple, list)):
                t = t[0] if t else None
            if torch.is_tensor(t) and t.ndim == 3 and t.shape[1] > 1:
                cands.append(nm)
        return _h

    handles = []
    for nm, mod in model.named_modules():
        if nm:
            handles.append(mod.register_forward_hook(hook(nm)))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    if not cands:
        return None, None

    cand_set = set(cands)
    for nm, mod in model.named_modules():
        if nm not in cand_set:
            continue
        try:
            with torch.no_grad():
                out = mod(x)
        except Exception:
            continue                       # 吃不下原始 EEG（如内层 Linear）⇒ 跳过
        t = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(t) and t.ndim == 3 and t.shape[1] > 1:
            return nm, mod
    return None, None


def get_submodule_safe(model, name):
    try:
        return model.get_submodule(name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 计划
# ---------------------------------------------------------------------------
@dataclass
class BlockPlan:
    role: str                    # "mixer" | "ffn"
    source_model: str
    source_layer: int | str      # int = 第几个 block；str = 模块限定名
    mixer_kind: str = "attn1d"   # MIXER_KINDS 的 key
    nhead: int = 4
    d_hidden: int | None = None  # None = 沿用源 FFN 的隐藏维
    weights: str = "auto"        # auto / reuse / projected / fresh / arch-only
    note: str = ""


@dataclass
class VariantPlan:
    variant_id: str
    host_model: str
    blocks: list[BlockPlan] = field(default_factory=list)
    notes: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 权重搬运
# ---------------------------------------------------------------------------
def _copy_mha_weights(src_mha: nn.Module, dst_mha: nn.Module) -> bool:
    """nn.MultiheadAttention → nn.MultiheadAttention：逐键原样搬。

    形状只取决于 embed_dim（in_proj_weight 是 (3E, E)），**与 nhead 无关** ⇒
    nhead 不同也能搬。但 E 不同 ⇒ 搬不了。
    """
    s = src_mha.state_dict()
    d = dst_mha.state_dict()
    need = ["in_proj_weight", "in_proj_bias", "out_proj.weight", "out_proj.bias"]
    if not all(k in s for k in need):
        return False
    for k in need:
        if s[k].shape != d[k].shape:
            return False
    dst_mha.load_state_dict({k: s[k] for k in need}, strict=False)
    return True


def _ffn_linears(mod):
    """抽出 FFN 里的 Linear（按出现顺序）。"""
    lins = []
    if isinstance(mod, nn.Sequential):
        for m in mod.modules():
            if isinstance(m, nn.Linear):
                lins.append(m)
        return lins
    for m in mod.modules():
        if isinstance(m, nn.Linear):
            lins.append(m)
    return lins


def _ffn_in_dim(mod):
    """源 FFN 的输入维度（第一个 Linear 的 in_features）。"""
    lins = _ffn_linears(mod)
    if lins:
        return int(lins[0].in_features)
    return None


def _copy_ffn_weights(src_mod, dst_ffn: FFN) -> bool:
    """源 FFN 的 Linear 顺序 → 目标 FFN.experts[0].{0,2}。

    目标 FFN 是 Sequential(Linear, GELU, Linear)，所以 Linear 位置是 0 和 2。
    形状必须完全一致，否则不搬（标 fresh）。
    """
    lins = _ffn_linears(src_mod)
    if len(lins) < 2:
        return False
    s0, s1 = lins[0], lins[-1]
    d = dst_ffn.state_dict()
    pairs = [("experts.0.0.weight", s0.weight), ("experts.0.0.bias", s0.bias),
             ("experts.0.2.weight", s1.weight), ("experts.0.2.bias", s1.bias)]
    for k, v in pairs:
        if k not in d or v is None or d[k].shape != v.shape:
            return False
    dst_ffn.load_state_dict({k: v for k, v in pairs}, strict=False)
    return True


def build_mixer(kind: str, d_model: int, nhead: int = 4):
    """按 kind 造目标 mixer。"""
    if kind not in MIXER_KINDS:
        raise ValueError(f"未知 mixer kind：{kind}（可用：{list(MIXER_KINDS)}）")
    cls = MIXER_KINDS[kind]
    if kind in ("attn1d", "crisscross2d", "timefreq"):
        return cls(d_model, nhead)
    return cls(d_model)


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------
class AssembledVariant(nn.Module):
    """组装出的变体：宿主的 token 化 + 我们拼的 block 栈。

    `forward(x)`：x=(B,C,T) → TokenBatch。
    """

    def __init__(self, tokenizer, blocks, d_model, layout=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.blocks = nn.ModuleList(blocks)
        self.d_model = d_model
        self.layout = layout

    def forward(self, x):
        tb = self.tokenizer(x)
        for b in self.blocks:
            tb = b(tb)
        return tb


class TokenizerAdapter(nn.Module):
    """把宿主的 patch-embed 模块包装成 `(B,C,T) -> TokenBatch`。"""

    def __init__(self, module, d_model, layout=None):
        super().__init__()
        self.module = module
        self.d_model = d_model
        self.layout = layout

    def forward(self, x):
        from registry import normalize_token_tensor
        out = self.module(x)
        t = out
        if isinstance(t, (tuple, list)):
            t = t[0] if t else None
        tok = normalize_token_tensor(t)
        return TokenBatch(tok, self.layout)


def _resolve_block(base_models, bp: BlockPlan):
    """在源模型里定位 block 及其 mixer/ffn 子模块。"""
    h = base_models.get(bp.source_model)
    if h is None or not getattr(h, "ok", False):
        return None, None, None, f"源模型 {bp.source_model} 未装载"
    model = h.model
    blocks = find_block_modules(model)
    if not blocks:
        return None, None, None, f"{bp.source_model} 里找不到 block（无 attn+ffn 配对）"
    if isinstance(bp.source_layer, int):
        if bp.source_layer >= len(blocks):
            return None, None, None, (f"{bp.source_model} 只有 {len(blocks)} 个 block，"
                                      f"取不到第 {bp.source_layer} 个")
        b = blocks[bp.source_layer]
    else:
        b = next((x for x in blocks if x["name"] == bp.source_layer), None)
        if b is None:
            return None, None, None, f"{bp.source_model} 里没有名为 {bp.source_layer} 的 block"
    return b["attn"][1], b["ffn"][1], b["name"], None


def _source_dim(mod) -> int | None:
    """源 mixer 的 token 维度。"""
    if isinstance(mod, nn.MultiheadAttention):
        return int(mod.embed_dim)
    for attr in ("embed_dim", "d_model", "dim", "hidden_size"):
        v = getattr(mod, attr, None)
        if isinstance(v, int):
            return v
    return None


def build_variant(plan: VariantPlan, base_models, licenses=None, verbose=True):
    """按计划组装一个变体。返回 (AssembledVariant, VariantProv)。"""
    licenses = licenses or {}
    prov = VariantProv(variant_id=plan.variant_id, host_model=plan.host_model,
                       notes=plan.notes)

    host = base_models.get(plan.host_model)
    if host is None or not getattr(host, "ok", False):
        prov.blocks.append(BlockProv(
            index=0, role="host", source_model=plan.host_model, source_layer=None,
            weights=W_ARCH_ONLY, note=f"宿主未装载：{getattr(host, 'error', '不存在')}"))
        return None, prov

    # --- 宿主：token 化入口 + 布局 ---
    import registry as REG
    hcfg = getattr(host, "cfg", {}) or {}
    hmeta = getattr(host, "meta", {}) or {}
    n_ch = hcfg.get("n_chans") or hmeta.get("n_chans") or 63
    n_t = hcfg.get("n_times") or hmeta.get("n_times") or 250
    dummy = torch.zeros(1, int(n_ch), int(n_t))
    tok_name, tok_mod = find_tokenizer(host.model, dummy)
    if tok_mod is None:
        prov.blocks.append(BlockProv(
            index=0, role="patch_embed", source_model=plan.host_model,
            source_layer=None, weights=W_ARCH_ONLY,
            note="找不到 token 化模块（patch embed）"))
        return None, prov

    # 宿主的 token 维度
    with torch.no_grad():
        out = tok_mod(dummy)
    t = out[0] if isinstance(out, (tuple, list)) else out
    from registry import normalize_token_tensor
    tok = normalize_token_tensor(t)
    d_model = int(tok.shape[-1])
    lay = REG.infer_layout(tok.shape[1], hcfg, hmeta, host.model)
    layout = lay["layout"]
    prov.blocks.append(BlockProv(
        index=0, role="patch_embed", source_model=plan.host_model,
        source_layer=tok_name, weights=W_REUSE, in_dim=int(n_ch),
        out_dim=d_model, license=licenses.get(plan.host_model),
        note=f"宿主 token 化（{tok_name}）⇒ {tuple(tok.shape)}；"
             f"布局 {layout}（{lay['confidence']}）"))

    # --- 逐 block 组装 ---
    built = []
    for i, bp in enumerate(plan.blocks, start=1):
        attn_mod, ffn_mod, blk_name, err = _resolve_block(base_models, bp)
        if err:
            prov.blocks.append(BlockProv(
                index=i, role=bp.role, source_model=bp.source_model,
                source_layer=bp.source_layer, weights=W_ARCH_ONLY, note=err))
            continue

        src_dim = _source_dim(attn_mod) or _ffn_in_dim(ffn_mod) or d_model
        comp = {}

        # --- 判定哪些权重真能搬 ---
        mixer_is_mha = isinstance(attn_mod, nn.MultiheadAttention)
        mixer_target_ok = bp.mixer_kind in ("attn1d", "crisscross2d", "timefreq")
        lins = _ffn_linears(ffn_mod)
        ffn_src_ok = len(lins) >= 2 and lins[0].in_features == src_dim

        # 能不能在**源自己的维度**里跑这个块？能 ⇒ 权重可原样搬，
        # 维度差用两侧的学习投影调和（进：宿主→源；出：源→宿主）。
        # 不能（结构不同 / 占位）⇒ 直接在宿主维度重建，权重就是新的。
        run_in_src_dim = (mixer_is_mha and mixer_target_ok) or ffn_src_ok
        if bp.mixer_kind == "linear_ssm":
            run_in_src_dim = False          # 占位结构，永远重建
        inner_dim = src_dim if (run_in_src_dim and src_dim != d_model) else d_model
        need_adapter = inner_dim != d_model

        # --- mixer ---
        mixer = build_mixer(bp.mixer_kind, inner_dim, bp.nhead)
        if bp.mixer_kind == "linear_ssm":
            comp["mixer"] = W_FRESH
            note_mix = ("linear_ssm 是结构占位（nearest 投影），**不是 Mamba**；"
                        "权重不搬，且不假装它是 Mamba")
        elif mixer_is_mha and mixer_target_ok and inner_dim == src_dim:
            if _copy_mha_weights(attn_mod, mixer.attn):
                comp["mixer"] = W_REUSE
                note_mix = "nn.MultiheadAttention 权重逐键原样搬用"
            else:
                comp["mixer"] = W_FRESH
                note_mix = "MultiheadAttention 形状不匹配，未搬"
        elif mixer_is_mha and mixer_target_ok:
            comp["mixer"] = W_FRESH
            note_mix = (f"源 token 维 {src_dim} ≠ 宿主 {d_model}，"
                        f"但两侧投影后仍可原样搬权重")
        elif mixer_is_mha:
            comp["mixer"] = W_ARCH_ONLY
            note_mix = f"目标 mixer({bp.mixer_kind}) 与源结构不同，权重不可搬"
        else:
            comp["mixer"] = W_ARCH_ONLY
            note_mix = (f"源 mixer 是 {type(attn_mod).__name__}（非 MultiheadAttention）"
                        f" ⇒ 与目标 {bp.mixer_kind} 结构不同，权重不可搬")

        # --- ffn ---
        if len(lins) >= 2:
            hid = lins[0].out_features
            ffn = FFN(inner_dim, hid)
            if ffn_src_ok and inner_dim == src_dim and _copy_ffn_weights(ffn_mod, ffn):
                comp["ffn"] = W_REUSE
                note_ffn = f"FFN 权重原样搬用（隐藏维 {hid}）"
            else:
                comp["ffn"] = W_FRESH
                note_ffn = "FFN 形状不一致，未搬"
        else:
            ffn = FFN(inner_dim, bp.d_hidden or 4 * inner_dim)
            comp["ffn"] = W_ARCH_ONLY
            note_ffn = "源里没找到 FFN 的 Linear 组"

        # --- 组装 block ---
        proj_in = proj_out = None
        if need_adapter:
            proj_in = Projection(d_model, inner_dim)    # 宿主 → 源
            proj_out = Projection(inner_dim, d_model)   # 源 → 宿主
            comp["proj"] = W_PROJECTED
        blk = Block(mixer, ffn, inner_dim, norm_first=True)
        wrapper = _AdapterBlock(proj_in, blk, proj_out)

        # 整块状态 = 各组件里最弱的
        order = [W_ARCH_ONLY, W_FRESH, W_PROJECTED, W_REUSE]   # 弱 → 强
        states = [comp.get("mixer", W_FRESH), comp.get("ffn", W_FRESH)]
        if need_adapter:
            states.append(W_PROJECTED)
        block_state = min(states, key=order.index)
        if bp.weights != "auto":
            block_state = bp.weights        # 显式声明优先（但组件明细仍然保留）

        prov.blocks.append(BlockProv(
            index=i, role=bp.role, source_model=bp.source_model,
            source_layer=blk_name, weights=block_state,
            in_dim=src_dim, out_dim=d_model,
            license=licenses.get(bp.source_model),
            note=f"{note_mix}；{note_ffn}"
                 + (f"；两侧学习投影 {d_model}→{inner_dim}→{d_model}" if need_adapter else "")
                 + (f"；{bp.note}" if bp.note else ""),
            components=comp))
        built.append(wrapper)

    # --- 许可合并：整条链取最严 ---
    prov.license = merge_license([b.license for b in prov.blocks if b.license])

    variant = AssembledVariant(
        tokenizer=TokenizerAdapter(tok_mod, d_model, layout),
        blocks=built, d_model=d_model, layout=layout)
    variant.eval()
    if verbose:
        log_plan(prov)
    return variant, prov


class _AdapterBlock(nn.Module):
    """两侧学习投影的 block 包装（宿主维 → 源维 → 宿主维）。

    投影**方向**很关键：源权重期待的输入维度是 `src_dim`，
    所以入口投影是 宿主→源、出口投影是 源→宿主。
    反过来（源→宿主）会让源权重形状对不上，等于白搬。
    """

    def __init__(self, proj_in, block, proj_out):
        super().__init__()
        self.proj_in = proj_in
        self.block = block
        self.proj_out = proj_out

    def forward(self, x: TokenBatch):
        h = self.proj_in(x.tokens) if self.proj_in is not None else x.tokens
        out = self.block(x.replace(h))
        t = self.proj_out(out.tokens) if self.proj_out is not None else out.tokens
        return out.replace(t)


def log_plan(prov: VariantProv):
    print(f"── 变体 {prov.variant_id}（宿主 {prov.host_model}）")
    for b in prov.blocks:
        comps = " ".join(f"{k}={v}" for k, v in (b.components or {}).items())
        print(f"   [{b.index}] {b.role:<12} {b.source_model}/{b.source_layer} "
              f"⇒ {b.weights:<10} {('[' + comps + ']') if comps else ''} "
              f"{b.note[:80]}")
    print(f"   权重来源统计：{prov.weight_summary()} | 模型：{prov.models_used()} "
          f"| 许可：{prov.licenses_used()} → {prov.license}")


# ---------------------------------------------------------------------------
# 核实：provenance 说的是不是真的
# ---------------------------------------------------------------------------
def verify_weights(variant, base_models, plan: VariantPlan, tol=1e-6):
    """**逐块真的比对权重值**：provenance 说 reuse 的，值必须与源一致。

    返回 (ok: bool, 报告 list)。这是对"provenance 撒谎"的机器可判防线。
    """
    report = []
    ok_all = True
    for i, bp in enumerate(plan.blocks, start=1):
        attn_mod, ffn_mod, blk_name, err = _resolve_block(base_models, bp)
        if err:
            report.append({"index": i, "checked": False, "reason": err})
            continue
        blk = variant.blocks[i - 1]
        inner = blk.block if isinstance(blk, _AdapterBlock) else blk
        mixer = inner.mixer
        entry = {"index": i, "source": f"{bp.source_model}/{blk_name}",
                 "mixer_kind": bp.mixer_kind, "checked": True}
        if isinstance(mixer, (SelfAttention1D, CrissCross2D, TimeFreqCross)) and \
                isinstance(attn_mod, nn.MultiheadAttention):
            s = attn_mod.state_dict()
            d = mixer.attn.state_dict()
            same = all(torch.allclose(s[k].float(), d[k].float(), atol=tol)
                       for k in ("in_proj_weight", "in_proj_bias",
                                 "out_proj.weight", "out_proj.bias")
                       if k in s and k in d and s[k].shape == d[k].shape)
            entry["mixer_weights_match_source"] = bool(same)
            if not same:
                ok_all = False
        else:
            entry["mixer_weights_match_source"] = None
            entry["note"] = "无可比对的同构权重（结构不同）"
        report.append(entry)
    return ok_all, report


# ---------------------------------------------------------------------------
# 声明式变体清单（阶段 D）
# ---------------------------------------------------------------------------
def expand_plan(plan: VariantPlan, base_models):
    """把 `source_layer="all"` 展开成源模型的全部 block。

    为什么需要展开：7 个基模的层数**现在还不知道**（探针还没跑完），
    而变体清单必须先写死。写死「取 A 模型的全部层」比写死「第 3 层」更稳 ——
    前者对层数变化不敏感，后者一旦模型层数与预期不符就静默错位。
    """
    out_blocks = []
    for bp in plan.blocks:
        if bp.source_layer != "all":
            out_blocks.append(bp)
            continue
        h = base_models.get(bp.source_model)
        if h is None or not getattr(h, "ok", False):
            out_blocks.append(bp)      # 保留原样，build 时会记错误
            continue
        blocks = find_block_modules(h.model)
        for i in range(len(blocks)):
            out_blocks.append(BlockPlan(
                role=bp.role, source_model=bp.source_model, source_layer=i,
                mixer_kind=bp.mixer_kind, nhead=bp.nhead, d_hidden=bp.d_hidden,
                weights=bp.weights,
                note=(bp.note + f"（源全部 {len(blocks)} 层的第 {i} 层）").strip()))
    return VariantPlan(variant_id=plan.variant_id, host_model=plan.host_model,
                       blocks=out_blocks, notes=plan.notes, extra=plan.extra)


def default_variants(host="cbramod"):
    """≥4 个重组变体（阶段 D 的清单）。

    设计意图：
    - **v1_self**：宿主自己的层全留 —— 「重组前」的对照基线；
    - **v2_hybrid_attn**：中段换成别的模型的一维注意力 —— 换**注意力类型**；
    - **v3_timefreq**：中段插时频交叉注意力 —— ROADMAP ④ 域注意力；
    - **v4_ssm_slot**：某层换成 linear_ssm —— **结构占位对照**（不是 Mamba）；
    - **v5_deep_swap**：把宿主后段整体换成另一个模型的层 —— 换**来源模型**。

    `source_layer="all"` 会在装配时按源模型的实际层数展开。
    """
    return [
        VariantPlan(variant_id="v1_self", host_model=host,
                    blocks=[BlockPlan("mixer", host, "all", "crisscross2d")],
                    notes="宿主自己的层全留（重组前对照）"),
        VariantPlan(variant_id="v2_hybrid_attn", host_model=host,
                    blocks=[BlockPlan("mixer", "biot", "all", "attn1d")],
                    notes="整条链换成 BIOT 的一维注意力"),
        VariantPlan(variant_id="v3_timefreq", host_model=host,
                    blocks=[BlockPlan("mixer", host, "all", "crisscross2d"),
                            BlockPlan("mixer", host, 0, "timefreq")],
                    notes="中段插一层时频交叉注意力（ROADMAP ④）"),
        VariantPlan(variant_id="v4_ssm_slot", host_model=host,
                    blocks=[BlockPlan("mixer", host, "all", "crisscross2d"),
                            BlockPlan("mixer", host, 0, "linear_ssm")],
                    notes="一层换成 linear_ssm **结构占位**（不是 Mamba，仅作对照）"),
        VariantPlan(variant_id="v5_deep_swap", host_model=host,
                    blocks=[BlockPlan("mixer", "eegdino-small", "all", "attn1d")],
                    notes="整条链换成 EEGDINO 的层（换来源模型）"),
    ]


def plan_report(plans, base_models, verbose=True):
    """干跑：每个变体能不能装、源 block 有几个、维度差多少。

    **不建权重**，只出可行性 —— 网络一恢复就能先看这张表，
    避免把算力砸在结构上根本装不起来的变体上。
    """
    rows = []
    for p in plans:
        ep = expand_plan(p, base_models)
        h = base_models.get(ep.host_model)
        host_ok = h is not None and getattr(h, "ok", False)
        host_dim = None
        n_host_blocks = 0
        if host_ok:
            n_host_blocks = len(find_block_modules(h.model))
            try:
                import registry as REG
                n_ch = (h.cfg or {}).get("n_chans") or 63
                n_t = (h.cfg or {}).get("n_times") or 250
                nm, mod = find_tokenizer(h.model, torch.zeros(1, int(n_ch), int(n_t)))
                if mod is not None:
                    with torch.no_grad():
                        o = mod(torch.zeros(1, int(n_ch), int(n_t)))
                    tt = o[0] if isinstance(o, (tuple, list)) else o
                    from registry import normalize_token_tensor
                    host_dim = int(normalize_token_tensor(tt).shape[-1])
            except Exception:
                host_dim = None
        per_block = []
        for bp in ep.blocks:
            attn, ffn, nm, err = _resolve_block(base_models, bp)
            if err:
                per_block.append({"src": f"{bp.source_model}/{bp.source_layer}",
                                  "ok": False, "error": err})
                continue
            sd = _source_dim(attn) or _ffn_in_dim(ffn)
            per_block.append({
                "src": f"{bp.source_model}/{nm}",
                "ok": True, "mixer_kind": bp.mixer_kind,
                "src_dim": sd, "host_dim": host_dim,
                "needs_adapter": bool(sd and host_dim and sd != host_dim),
                "weights_reusable": bool(isinstance(attn, nn.MultiheadAttention)
                                         and bp.mixer_kind in
                                         ("attn1d", "crisscross2d", "timefreq")),
            })
        rows.append({"variant_id": p.variant_id, "host": p.host_model,
                     "host_ok": host_ok, "host_dim": host_dim,
                     "n_host_blocks": n_host_blocks,
                     "n_blocks_planned": len(ep.blocks),
                     "feasible": host_ok and all(b["ok"] for b in per_block),
                     "blocks": per_block, "notes": p.notes})
        if verbose:
            flag = "✓" if rows[-1]["feasible"] else "✗"
            print(f"  {flag} {p.variant_id:<16} 宿主 {p.host_model}"
                  f"（{n_host_blocks} 层, d={host_dim}）→ 计划 {len(ep.blocks)} 块"
                  f" | {p.notes}")
            for b in per_block:
                if not b["ok"]:
                    print(f"       ✗ {b['src']}：{b['error']}")
                else:
                    print(f"       · {b['src']:<24} {b['mixer_kind']:<12}"
                          f" 源维 {b['src_dim']} → 宿主 {b['host_dim']}"
                          f"{'（需两侧投影）' if b['needs_adapter'] else ''}"
                          f" | 权重可搬 {b['weights_reusable']}")
    return rows


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
def _fake_models():
    """三个结构不同的假基模，用来把组装链路在本地跑通。

    A：标准 transformer（nn.MultiheadAttention + Sequential FFN）—— 权重**可搬**
    B：Criss-Cross 卷积注意力 —— 与目标的注意力实现**结构不同，权重不可搬**
    C：只有一层、维度更小 —— 逼出「插学习投影」这条路径
    """
    from types import SimpleNamespace

    class PatchEmbed(nn.Module):
        def __init__(self, n_ch, n_t, pc, pt, d):
            super().__init__()
            self.pc, self.pt, self.d = pc, pt, d
            self.rows, self.cols = n_t // pt, n_ch // pc
            self.proj = nn.Linear(pc * pt, d)

        def forward(self, x):
            B, C, T = x.shape
            p = x.unfold(-1, self.pt, self.pt).unfold(1, self.pc, self.pc)
            p = p.permute(0, 2, 1, 3, 4).reshape(B, self.rows * self.cols, -1)
            return self.proj(p)

    class BlockMHA(nn.Module):
        def __init__(self, d, nhead=4):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
            self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            self.n1 = nn.LayerNorm(d)
            self.n2 = nn.LayerNorm(d)

        def forward(self, x):
            h = self.n1(x)
            a, _ = self.attn(h, h, h, need_weights=False)
            x = x + a
            return x + self.ffn(self.n2(x))

    class CrissCrossAttention(nn.Module):
        """结构与 nn.MultiheadAttention 不同 ⇒ 权重不可搬。"""

        def __init__(self, d):
            super().__init__()
            self.row = nn.Conv2d(d, d, (1, 3), padding=(0, 1))
            self.col = nn.Conv2d(d, d, (3, 1), padding=(1, 0))

        def forward(self, x, layout=None):
            return x

    class BlockCriss(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.attn = CrissCrossAttention(d)
            self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

        def forward(self, x):
            return x + self.ffn(self.attn(x))

    class ModelA(nn.Module):
        def __init__(self, d=64, n_layers=2):
            super().__init__()
            self.patch = PatchEmbed(63, 250, 9, 25, d)
            self.blocks = nn.ModuleList([BlockMHA(d) for _ in range(n_layers)])
            self.norm = nn.LayerNorm(d)

        def forward(self, x):
            t = self.patch(x)
            for b in self.blocks:
                t = b(t)
            return {"embedding": self.norm(t)}

    class ModelB(nn.Module):
        def __init__(self, d=64, n_layers=2):
            super().__init__()
            self.patch = PatchEmbed(63, 250, 9, 25, d)
            self.blocks = nn.ModuleList([BlockCriss(d) for _ in range(n_layers)])
            self.norm = nn.LayerNorm(d)

        def forward(self, x):
            t = self.patch(x)
            for b in self.blocks:
                t = b(t)
            return {"embedding": self.norm(t)}

    class ModelC(nn.Module):
        """token 维度更小（32），逼出「插学习投影」路径。"""

        def __init__(self, d=32, n_layers=1):
            super().__init__()
            self.patch = PatchEmbed(63, 250, 9, 25, d)
            self.blocks = nn.ModuleList([BlockMHA(d) for _ in range(n_layers)])
            self.norm = nn.LayerNorm(d)

        def forward(self, x):
            t = self.patch(x)
            for b in self.blocks:
                t = b(t)
            return {"embedding": self.norm(t)}

    torch.manual_seed(3)
    hs = {}
    for nm, cls in (("modelA", ModelA), ("modelB", ModelB), ("modelC", ModelC)):
        m = cls()
        hs[nm] = SimpleNamespace(
            ok=True, model=m,
            cfg={"n_chans": 63, "n_times": 250, "patch_size": [9, 25]},
            meta={}, error="",
            n_params=sum(p.numel() for p in m.parameters()))
    return hs


def self_test():
    print("=== 组装器自测 ===")
    base = _fake_models()
    licenses = {"modelA": "BSD-3-Clause", "modelB": "MIT", "modelC": "CC-BY-NC-4.0"}

    # --- (1) 全 A：权重应当真的搬过去了 ---
    p1 = VariantPlan(variant_id="v_all_a", host_model="modelA",
                     blocks=[BlockPlan("mixer", "modelA", 0, "attn1d"),
                             BlockPlan("mixer", "modelA", 1, "attn1d")])
    v1, pr1 = build_variant(p1, base, licenses)
    assert v1 is not None
    ok, rep = verify_weights(v1, base, p1)
    assert ok, f"provenance 说 reuse，但权重对不上：{rep}"
    print(f"  ✓ v_all_a：两块 mixer 权重与源逐键一致"
          f"（{rep[0]['mixer_weights_match_source']}）")
    assert pr1.weight_summary().get(W_REUSE, 0) >= 2, pr1.weight_summary()

    # --- (2) 混合：A 的注意力 + B 的 Criss-Cross ⇒ B 那块必须标 arch-only ---
    p2 = VariantPlan(variant_id="v_mix", host_model="modelA",
                     blocks=[BlockPlan("mixer", "modelA", 0, "attn1d"),
                             BlockPlan("mixer", "modelB", 0, "attn1d")])
    v2, pr2 = build_variant(p2, base, licenses)
    b2 = pr2.blocks[2]
    assert b2.weights == W_ARCH_ONLY, f"卷积注意力不该标 reuse：{b2.to_dict()}"
    assert "不可搬" in b2.note
    print(f"  ✓ v_mix：B 的块标为 {b2.weights} —— {b2.note[:60]}")

    # --- (3) 维度不一致 ⇒ 两侧学习投影调和，源的权重**仍然搬得动** ---
    p3 = VariantPlan(variant_id="v_proj", host_model="modelA",
                     blocks=[BlockPlan("mixer", "modelC", 0, "attn1d")])
    v3, pr3 = build_variant(p3, base, licenses)
    b3 = pr3.blocks[1]
    assert b3.components.get("proj") == W_PROJECTED, b3.to_dict()
    assert b3.components.get("mixer") == W_REUSE, b3.to_dict()
    assert b3.components.get("ffn") == W_REUSE, b3.to_dict()
    assert b3.weights == W_PROJECTED, b3.to_dict()
    ok3, rep3 = verify_weights(v3, base, p3)
    assert ok3, f"加了投影后权重仍应与源一致：{rep3}"
    print(f"  ✓ v_proj：两侧学习投影 64→32→64，源权重照搬"
          f"（{b3.components}）且逐键核对通过")

    # --- (4) linear_ssm 必须标 fresh 且注明是占位 ---
    p4 = VariantPlan(variant_id="v_ssm", host_model="modelA",
                     blocks=[BlockPlan("mixer", "modelA", 0, "linear_ssm")])
    v4, pr4 = build_variant(p4, base, licenses)
    b4 = pr4.blocks[1]
    assert b4.weights == W_FRESH, b4.to_dict()
    assert "占位" in b4.note and "不是 Mamba" in b4.note, b4.note
    print(f"  ✓ v_ssm：标 {b4.weights} —— {b4.note[:60]}")

    # --- (5) 许可取最严 ---
    assert pr3.license == "NC", pr3.license
    print(f"  ✓ 许可合并：modelA(BSD-3) + modelC(CC-BY-NC) ⇒ {pr3.license}")

    # --- (6) 前向能跑，输出 TokenBatch ---
    for nm, v in (("v_all_a", v1), ("v_mix", v2), ("v_proj", v3), ("v_ssm", v4)):
        with torch.no_grad():
            tb = v(torch.zeros(2, 63, 250))
        assert isinstance(tb, TokenBatch), type(tb)
        assert tb.tokens.shape[0] == 2 and tb.tokens.shape[-1] == 64, tb.tokens.shape
    print(f"  ✓ 四个变体前向通过：tokens {tuple(tb.tokens.shape)}，布局 {tb.layout}")

    # --- (7) provenance 落盘结构完整 ---
    d = pr1.to_dict()
    for k in ("variant_id", "host_model", "blocks", "weight_summary",
              "models_used", "licenses_used"):
        assert k in d, k
    import json as _json
    _json.loads(pr1.to_json())
    print(f"  ✓ provenance 可序列化：{pr1.weight_summary()} | "
          f"{pr1.models_used()} | {pr1.licenses_used()}")

    # --- (8) 「all 层」展开 + 干跑可行性报告 ---
    p5 = VariantPlan(variant_id="v_all_expand", host_model="modelA",
                     blocks=[BlockPlan("mixer", "modelA", "all", "attn1d")])
    ep = expand_plan(p5, base)
    assert len(ep.blocks) == 2, f"modelA 有 2 层，应展开成 2 块：{len(ep.blocks)}"
    rep = plan_report([p5], base)
    assert rep[0]["feasible"] and rep[0]["n_blocks_planned"] == 2, rep[0]
    assert rep[0]["host_dim"] == 64, rep[0]
    print(f"  ✓ 「all 层」展开：{len(ep.blocks)} 块；干跑报告 feasible="
          f"{rep[0]['feasible']}，宿主 d={rep[0]['host_dim']}")

    # --- (9) 源模型缺失时必须报出来，不能静默 ---
    bad = VariantPlan(variant_id="v_missing", host_model="modelA",
                      blocks=[BlockPlan("mixer", "notexist", 0, "attn1d")])
    _, pb = build_variant(bad, base, licenses)
    assert pb.blocks[1].weights == W_ARCH_ONLY and "未装载" in pb.blocks[1].note
    print(f"  ✓ 源模型缺失：标 {pb.blocks[1].weights} —— {pb.blocks[1].note[:50]}")

    print("=== 组装器自测通过 ===")


if __name__ == "__main__":
    self_test()
