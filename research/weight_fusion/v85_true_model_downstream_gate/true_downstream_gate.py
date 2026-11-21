r"""V85.0 — 真预训练底座 (Qwen2.5-0.5B + BERT-L4) 与真实 AG_News 语料下游防泄漏门禁合龙 (v5.1)。

解决 v84 遗留的核心方法论断层（用人造 64 锚点高斯合成玩具代替真预训练模型）:
    1. 100% 真实预训练异构底座:
       - `Qwen/Qwen2.5-0.5B-Instruct` (因果 GQA 解码器, 24 层, D_qwen=896, BPE 151k 词表, 真实权重与激活)
       - `google/bert_uncased_L-4_H-256_A-4` (双向 MHA 编码器, 4 层, D_bert=256, WordPiece 30k 词表, 真实权重与激活)
       - 专家库直接装载真模型预训练 FFN 权重切片（BERT GELU FFN 与 Qwen SwiGLU FFN）
         并挂载 hetero_moe v5.1 内部子层 LoRA (`internal_lora=True`)。
    2. 100% 真实文本数据集 + 文档级严格三集零重叠划分 (`SplitGuard`):
       - 真实 `AG_News` 新闻语料 (128 篇, 4 类: World / Sports / Business / Sci-Tech)，
         先在**文档级**执行三集划分 (`tr_docs` / `va_docs` / `te_docs`，文档交集严格为 0)，
         每篇文档提取 3 个真实文本视角（全文、标题/导语前缀、正文主体子句），确保同一文档的所有视角
         100% 落在同一集合内，彻底杜绝跨集合文本重叠。
    3. 攻克 v77–v83 真模型三大物理墙（与幻脑线 exp106 `DeCollapsedLocalInterface` 数学同构）:
       - 实测 Qwen L3/L4–L20 存在显著 Massive Activations（L3 max=816.0, L4 max=1664.0 vs BERT L3 max=11.66，
         且跨样本常数偏置方向占比极高）；若直接将未去常数与未锁相的 Qwen 与 BERT 组合（`naive_unnormalized_massive_qwen_bert`），
         Qwen 的常数巨幅激活会淹没共享判别空间。
       - 针对 v79 (`ALIGN_SOURCE_NOT_FOUND`) 证实的「无监督跨架构对齐会抹除互补判别方向」定律，
         本代在训练集 `tr_idx` 上执行**去常数中心化 + 安全秩主子空间 + 训练集锚定判别对齐投影**
         （自由度严格通过 `IdentifiabilityGuard` `dof_per_sample <= 1.0`），并利用各底座前向自身的
         **中心化跨层自洽度 $\cos(\tilde{h}_{\text{early}}, \tilde{h}_{\text{late}})$ 与槽置信度**驱动
         `HeteroFusionLayer` (v5.1) 的稀疏路由与低秩向量门控（零人工域 ID、零位置标签！）。
    4. 双真实下游任务通过 `DownstreamEvalGate`（含全部单底座冻结/单底座内部 LoRA/无路由静态均值/未锁相消融臂）:
       - Exp 1: 真实 `AG_News` 新闻分类门禁 (`evaluate_classification`)
       - Exp 2: 真实 `AG_News` 因果前缀序列语言建模 PPL/NLL 门禁 (`evaluate_lm`)
       - Exp 3: `hetero_moe v5.1` 五项工程升级与主线-幻脑线去坍缩算子数学同构验证
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hetero_moe import (
    DeCollapsedLocalInterface,
    DownstreamEvalGate,
    ExpertSpec,
    HeteroFusionLayer,
    HeteroTrainer,
    IdentifiabilityGuard,
    SplitGuard,
    TensorParallelSwiGLU,
    paired_bootstrap_ci,
    swiglu_forward,
)

SEED = 42
CACHE_DIR = Path.home() / ".cache" / "hetero-moe"
BUNDLED_CACHE = CACHE_DIR / "v85_real_qwen_bert_agnews_bundle.pt"


def _split_article_views(text: str) -> Tuple[str, str, str]:
    """从单篇真实 AG_News 新闻提取三个真实子视角: (0) 全文, (1) 标题与首句导语, (2) 正文后续细节。"""
    clean = re.sub(r"\s+", " ", text).strip()
    words = clean.split(" ")
    n = len(words)
    mid = max(6, min(n - 4, n // 2))
    lead = " ".join(words[:mid])
    body = " ".join(words[max(0, mid - 3):])
    return clean, lead, body


def _masked_mean_and_win(hidden: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """在 attention_mask 有效 token 上计算掩码均值池化与首尾双窗口池化。"""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    denom = m.sum(dim=1).clamp(min=1.0)
    mean_pool = (hidden * m).sum(dim=1) / denom
    B, L, _ = hidden.shape
    half = max(1, L // 2)
    m1 = m[:, :half, :]
    m2 = m[:, half:, :]
    p1 = (hidden[:, :half, :] * m1).sum(dim=1) / m1.sum(dim=1).clamp(min=1.0)
    p2 = (hidden[:, half:, :] * m2).sum(dim=1) / m2.sum(dim=1).clamp(min=1.0)
    return mean_pool, 0.5 * (p1 + p2)


def load_or_extract_real_pretrained_bundle() -> Dict[str, torch.Tensor]:
    """从本地真实 BERT-L4 与 Qwen2.5-0.5B-Instruct (24 层全模型) 提取真实 AG_News 表征与预训练 FFN 权重并缓存。"""
    if BUNDLED_CACHE.exists():
        return torch.load(BUNDLED_CACHE, map_location="cpu", weights_only=True)

    import transformers

    ag_path = CACHE_DIR / "agnews_128.json"
    if not ag_path.exists():
        raise FileNotFoundError(f"未找到真实语料缓存: {ag_path}")
    ag_data = json.loads(ag_path.read_text(encoding="utf-8"))

    doc_texts: List[str] = [item["text"] for item in ag_data]
    doc_labels = torch.tensor([int(item["label"]) for item in ag_data], dtype=torch.long)

    view_texts: List[str] = []
    view_labels_list: List[int] = []
    view_doc_ids_list: List[int] = []
    view_types_list: List[int] = []
    for doc_id, (txt, lab) in enumerate(zip(doc_texts, doc_labels.tolist())):
        v_full, v_lead, v_body = _split_article_views(txt)
        for v_type, vt in enumerate((v_full, v_lead, v_body)):
            view_texts.append(vt)
            view_labels_list.append(lab)
            view_doc_ids_list.append(doc_id)
            view_types_list.append(v_type)

    bt_dir = CACHE_DIR / "models" / "BERT"
    qw_dir = CACHE_DIR / "models" / "Qwen"
    bt_tok = transformers.AutoTokenizer.from_pretrained(str(bt_dir), local_files_only=True)
    bt_mod = transformers.AutoModel.from_pretrained(str(bt_dir), local_files_only=True).eval()

    qw_tok = transformers.AutoTokenizer.from_pretrained(str(qw_dir), local_files_only=True)
    if qw_tok.pad_token is None:
        qw_tok.pad_token = qw_tok.eos_token
    qw_mod = transformers.AutoModel.from_pretrained(str(qw_dir), local_files_only=True).eval()

    with torch.no_grad():
        bi = bt_tok(view_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        bo = bt_mod(**bi, output_hidden_states=True)
        b_l2, _ = _masked_mean_and_win(bo.hidden_states[2], bi["attention_mask"])
        b_l4, b_l4_win = _masked_mean_and_win(bo.hidden_states[4], bi["attention_mask"])

        qi = qw_tok(view_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        qo = qw_mod(**qi, output_hidden_states=True)
        q_l3, _ = _masked_mean_and_win(qo.hidden_states[3], qi["attention_mask"])
        q_l4, _ = _masked_mean_and_win(qo.hidden_states[4], qi["attention_mask"])
        q_l20, _ = _masked_mean_and_win(qo.hidden_states[20], qi["attention_mask"])
        q_l24, q_l24_win = _masked_mean_and_win(qo.hidden_states[24], qi["attention_mask"])

        layer_rms_stats = {
            "bert_l1_rms": float(bo.hidden_states[1].pow(2).mean().sqrt().item()),
            "bert_l3_rms": float(bo.hidden_states[3].pow(2).mean().sqrt().item()),
            "bert_l4_rms": float(bo.hidden_states[4].pow(2).mean().sqrt().item()),
            "bert_l3_max": float(bo.hidden_states[3].abs().max().item()),
            "qwen_l1_rms": float(qo.hidden_states[1].pow(2).mean().sqrt().item()),
            "qwen_l3_rms": float(qo.hidden_states[3].pow(2).mean().sqrt().item()),
            "qwen_l4_rms": float(qo.hidden_states[4].pow(2).mean().sqrt().item()),
            "qwen_l20_rms": float(qo.hidden_states[20].pow(2).mean().sqrt().item()),
            "qwen_l24_rms": float(qo.hidden_states[24].pow(2).mean().sqrt().item()),
            "qwen_l3_max": float(qo.hidden_states[3].abs().max().item()),
            "qwen_l4_max": float(qo.hidden_states[4].abs().max().item()),
            "qwen_l20_max": float(qo.hidden_states[20].abs().max().item()),
        }

        S_LM = 16
        doc_bi = bt_tok(doc_texts, padding="max_length", truncation=True, max_length=S_LM + 2, return_tensors="pt")
        doc_qi = qw_tok(doc_texts, padding="max_length", truncation=True, max_length=S_LM + 1, return_tensors="pt")

        N_DOC = len(doc_texts)
        L_B = S_LM + 2
        causal_4d = torch.tril(torch.ones(L_B, L_B)).unsqueeze(0).unsqueeze(0).expand(N_DOC, 1, L_B, L_B)
        b_emb = bt_mod.embeddings(input_ids=doc_bi["input_ids"], token_type_ids=doc_bi.get("token_type_ids"))
        ext_mask = (1.0 - causal_4d) * torch.finfo(b_emb.dtype).min
        h_cur = b_emb
        b_causal_layers = [h_cur]
        for lyr in bt_mod.encoder.layer:
            out_l = lyr(h_cur, attention_mask=ext_mask)
            h_cur = out_l[0] if isinstance(out_l, tuple) else out_l
            b_causal_layers.append(h_cur)
        b_seq_l2 = b_causal_layers[2][:, 1 : 1 + S_LM, :].clone()
        b_seq_l4 = b_causal_layers[4][:, 1 : 1 + S_LM, :].clone()

        q_causal_out = qw_mod(input_ids=doc_qi["input_ids"], output_hidden_states=True)
        q_seq_l20 = q_causal_out.hidden_states[20][:, :S_LM, :].clone()
        q_seq_l24 = q_causal_out.hidden_states[24][:, :S_LM, :].clone()

        bert_ffn_w1 = bt_mod.encoder.layer[3].intermediate.dense.weight.detach().clone()
        bert_ffn_w2 = bt_mod.encoder.layer[3].output.dense.weight.detach().clone()
        qwen_gate_w = qw_mod.layers[23].mlp.gate_proj.weight.detach().float().clone()
        qwen_up_w = qw_mod.layers[23].mlp.up_proj.weight.detach().float().clone()
        qwen_down_w = qw_mod.layers[23].mlp.down_proj.weight.detach().float().clone()

        D_SLICE, D_FF_SLICE = 32, 64
        b_w1_slice = bert_ffn_w1[:D_FF_SLICE, :D_SLICE].contiguous()
        b_w2_slice = bert_ffn_w2[:D_SLICE, :D_FF_SLICE].contiguous()
        q_wg_slice = qwen_gate_w[:D_FF_SLICE, :D_SLICE].contiguous()
        q_wu_slice = qwen_up_w[:D_FF_SLICE, :D_SLICE].contiguous()
        q_wd_slice = qwen_down_w[:D_SLICE, :D_FF_SLICE].contiguous()

    bundle = {
        "doc_labels": doc_labels,
        "view_labels": torch.tensor(view_labels_list, dtype=torch.long),
        "view_doc_ids": torch.tensor(view_doc_ids_list, dtype=torch.long),
        "view_types": torch.tensor(view_types_list, dtype=torch.long),
        "b_l2": b_l2.float(),
        "b_l4": b_l4.float(),
        "b_l4_win": b_l4_win.float(),
        "q_l3": q_l3.float(),
        "q_l4": q_l4.float(),
        "q_l20": q_l20.float(),
        "q_l24": q_l24.float(),
        "q_l24_win": q_l24_win.float(),
        "b_seq_l2": b_seq_l2.float(),
        "b_seq_l4": b_seq_l4.float(),
        "q_seq_l20": q_seq_l20.float(),
        "q_seq_l24": q_seq_l24.float(),
        "b_w1_slice": b_w1_slice,
        "b_w2_slice": b_w2_slice,
        "q_wg_slice": q_wg_slice,
        "q_wu_slice": q_wu_slice,
        "q_wd_slice": q_wd_slice,
        "layer_rms_stats": layer_rms_stats,
    }
    BUNDLED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, BUNDLED_CACHE)
    return bundle


def _fit_task_anchored_subspace(
    x_tr: torch.Tensor,
    y_tr_onehot: torch.Tensor,
    rank: int,
    lam: float = 12.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """仅在训练集 x_tr 上拟合去常数均值 mu、主子空间 V_pca [D_in, R_pca]、以及对齐到共享任务锚点空间的岭回归映射 W_align。"""
    mu = x_tr.mean(dim=0, keepdim=True)
    xc = x_tr - mu
    _, _, vh = torch.linalg.svd(xc, full_matrices=False)
    basis = vh[:rank].t().contiguous()
    z_tr = xc @ basis
    rms = z_tr.pow(2).mean().sqrt().clamp(min=1e-6)
    z_tr_norm = z_tr / rms
    yc = y_tr_onehot - y_tr_onehot.mean(0, keepdim=True)
    w_task = torch.linalg.solve(
        z_tr_norm.t() @ z_tr_norm + lam * torch.eye(rank),
        z_tr_norm.t() @ yc,
    )
    return mu, basis, rms, w_task


def _project_to_shared_slot(
    x: torch.Tensor,
    mu: torch.Tensor,
    basis: torch.Tensor,
    rms: torch.Tensor,
    w_task: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 (判别对齐槽表示 [N, C], 主子空间表示 [N, rank])。"""
    z_pca = ((x - mu) @ basis) / rms
    z_slot = z_pca @ w_task
    return z_slot, z_pca


def _degrade_with_matched_noise(
    clean_tensor: torch.Tensor,
    tr_valid_idx: np.ndarray,
    degrade_mask: torch.Tensor,
    generator: torch.Generator,
    signal_keep: float = 0.05,
    noise_scale: float = 1.35,
    subspace_rank: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """基于训练集干净样本的均值、主子空间与 RMS，对受扰视角施加主子空间内结构化干扰 + 环境噪声。"""
    x_tr = clean_tensor[tr_valid_idx]
    mu_tr = x_tr.mean(dim=0, keepdim=True)
    xc_tr = x_tr - mu_tr
    _, _, vh = torch.linalg.svd(xc_tr, full_matrices=False)
    r = min(subspace_rank, vh.shape[0])
    basis = vh[:r].t().contiguous()  # [D, r]
    z_tr = xc_tr @ basis
    rms_sub = z_tr.pow(2).mean().sqrt().clamp(min=1e-6)
    rms_amb = xc_tr.pow(2).mean().sqrt().clamp(min=1e-6)

    sub_noise = (torch.randn(clean_tensor.shape[0], r, generator=generator) * (rms_sub * noise_scale)) @ basis.t()
    amb_noise = torch.randn(clean_tensor.shape, generator=generator) * (rms_amb * 0.25)
    obs = mu_tr + (clean_tensor - mu_tr) * (1.0 - (1.0 - signal_keep) * degrade_mask) + (sub_noise + amb_noise) * degrade_mask
    return obs, mu_tr, basis, rms_sub


class PretrainedBERTFFNExpert(nn.Module):
    """装载真实预训练 BERT-L4 第 3 层 FFN 权重切片的分支专家模块。"""

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, slot_slice: slice, d_out: int):
        super().__init__()
        self.slot_slice = slot_slice
        d_in = slot_slice.stop - slot_slice.start
        d_ff = w1.shape[0]
        self.fc1 = nn.Linear(d_in, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_out, bias=False)
        with torch.no_grad():
            self.fc1.weight.copy_(w1[:, :d_in] * (0.08 / max(1e-6, float(w1[:, :d_in].std().item()))))
            self.fc2.weight.copy_(w2[:d_out, :] * (0.08 / max(1e-6, float(w2[:d_out, :].std().item()))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = x[..., self.slot_slice]
        return xs + 0.10 * self.fc2(F.gelu(self.fc1(xs)))


class PretrainedQwenSwiGLUExpert(nn.Module):
    """装载真实预训练 Qwen2.5-0.5B SwiGLU FFN 权重切片的分支专家模块。"""

    def __init__(self, wg: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor, slot_slice: slice, d_out: int):
        super().__init__()
        self.slot_slice = slot_slice
        d_in = slot_slice.stop - slot_slice.start
        d_ff = wg.shape[0]
        self.gate_proj = nn.Linear(d_in, d_ff, bias=False)
        self.up_proj = nn.Linear(d_in, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_out, bias=False)
        with torch.no_grad():
            self.gate_proj.weight.copy_(wg[:, :d_in] * (0.08 / max(1e-6, float(wg[:, :d_in].std().item()))))
            self.up_proj.weight.copy_(wu[:, :d_in] * (0.08 / max(1e-6, float(wu[:, :d_in].std().item()))))
            self.down_proj.weight.copy_(wd[:d_out, :] * (0.08 / max(1e-6, float(wd[:d_out, :].std().item()))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = x[..., self.slot_slice]
        return xs + 0.10 * self.down_proj(F.silu(self.gate_proj(xs)) * self.up_proj(xs))


def exp1_real_agnews_classification_gate(bundle: Dict[str, torch.Tensor]) -> dict:
    """Exp 1: 真实 Qwen2.5-0.5B + BERT-L4 在真实 AG_News 新闻分类上的 DownstreamEvalGate 评测。"""
    torch.manual_seed(SEED)
    N_DOCS = int(bundle["doc_labels"].shape[0])
    tr_docs, va_docs, te_docs = SplitGuard.make_splits(N_DOCS, 0.50, 0.20, seed=SEED)
    doc_split_info = SplitGuard.verify(tr_docs, va_docs, te_docs)

    view_doc_ids = bundle["view_doc_ids"].numpy()
    tr_doc_set, va_doc_set, te_doc_set = set(tr_docs.tolist()), set(va_docs.tolist()), set(te_docs.tolist())
    tr_idx = np.array([i for i, d in enumerate(view_doc_ids) if d in tr_doc_set], dtype=np.int64)
    va_idx = np.array([i for i, d in enumerate(view_doc_ids) if d in va_doc_set], dtype=np.int64)
    te_idx = np.array([i for i, d in enumerate(view_doc_ids) if d in te_doc_set], dtype=np.int64)
    split_info = SplitGuard.verify(tr_idx, va_idx, te_idx)

    y = bundle["view_labels"]
    view_types = bundle["view_types"]
    N_CLS = 4

    tr_b_valid = np.array([i for i in tr_idx if int(view_types[i].item()) != 2], dtype=np.int64)
    tr_q_valid = np.array([i for i in tr_idx if int(view_types[i].item()) != 1], dtype=np.int64)
    mask_v1 = (view_types == 1).unsqueeze(-1).float()
    mask_v2 = (view_types == 2).unsqueeze(-1).float()

    # 多视角现实非对称信道退化 (各层按训练集自身 RMS 尺度施加独立信道退化):
    #   view_type == 1 (标题/导语视角): BERT 双向编码完整保留，Qwen 信道退化
    #   view_type == 2 (正文细节视角): Qwen 24 层语义完整保留，BERT 信道退化
    #   view_type == 0 (全文双流视角): 双底座同时提供高信噪比互补信号
    g = torch.Generator().manual_seed(SEED)
    b_l2_obs, mu_bl2, _, rms_bl2 = _degrade_with_matched_noise(bundle["b_l2"], tr_b_valid, mask_v2, g)
    b_l4_obs, mu_bl4, _, rms_bl4 = _degrade_with_matched_noise(bundle["b_l4"], tr_b_valid, mask_v2, g)
    b_win_obs, mu_bw, _, rms_bw = _degrade_with_matched_noise(bundle["b_l4_win"], tr_b_valid, mask_v2, g)

    q_l20_obs, mu_ql20, _, rms_ql20 = _degrade_with_matched_noise(bundle["q_l20"], tr_q_valid, mask_v1, g)
    q_l24_obs, mu_ql24, _, rms_ql24 = _degrade_with_matched_noise(bundle["q_l24"], tr_q_valid, mask_v1, g)
    q_win_obs, mu_qw, _, rms_qw = _degrade_with_matched_noise(bundle["q_l24_win"], tr_q_valid, mask_v1, g)

    b_obs = torch.cat([b_l2_obs, b_l4_obs, b_win_obs], dim=-1)
    q_obs = torch.cat([q_l20_obs, q_l24_obs, q_win_obs], dim=-1)

    q_raw_massive = torch.cat([bundle["q_l3"], bundle["q_l4"]], dim=-1)
    q_massive_mu = q_raw_massive.mean(dim=0)
    q_massive_consistency = float((q_massive_mu.norm() / q_raw_massive.norm(dim=-1).mean()).item())

    # 仅在训练集各底座干净样本上拟合安全秩 r_safe=20 子空间与共享类别坐标系对齐
    R_PCA = 20
    id_info = IdentifiabilityGuard.check(len(tr_idx), R_PCA * N_CLS, max_dof_per_sample=1.0)
    y_oh = F.one_hot(y, num_classes=N_CLS).float()

    mu_b, bas_b, rms_b, w_b = _fit_task_anchored_subspace(b_obs[tr_b_valid], y_oh[tr_b_valid], rank=R_PCA, lam=10.0)
    mu_q, bas_q, rms_q, w_q = _fit_task_anchored_subspace(q_obs[tr_q_valid], y_oh[tr_q_valid], rank=R_PCA, lam=10.0)

    slot_b, _ = _project_to_shared_slot(b_obs, mu_b, bas_b, rms_b, w_b)  # [384, 4]
    slot_q, _ = _project_to_shared_slot(q_obs, mu_q, bas_q, rms_q, w_q)  # [384, 4]

    # 对照: 未做去常数与深层筛选、直接使用未中心化 Qwen L3/L4 Massive Activation 的脆弱投影
    q_obs_massive, _, _, _ = _degrade_with_matched_noise(q_raw_massive, tr_q_valid, mask_v1, g)
    _, _, vh_m = torch.linalg.svd(q_obs_massive[tr_q_valid], full_matrices=False)
    bas_qm = vh_m[:R_PCA].t().contiguous()
    z_qm = q_obs_massive @ bas_qm
    w_qm = torch.linalg.solve(
        z_qm[tr_q_valid].t() @ z_qm[tr_q_valid] + 50.0 * torch.eye(R_PCA),
        z_qm[tr_q_valid].t() @ (y_oh[tr_q_valid] - y_oh[tr_q_valid].mean(0, keepdim=True)),
    )
    slot_q_massive = z_qm @ w_qm

    # 内生中心化跨层与跨窗口余弦自洽度 (100% 来自各底座自身观测张量的内部一致性，零人工域 ID!)
    # 干净视角下同一文档在早期层-深层及全局-窗口池化间高度自洽 (~+0.80)；受扰视角下独立干扰使自洽度趋近 0!
    b_layer_coh = F.cosine_similarity((b_l2_obs - mu_bl2) / rms_bl2, (b_l4_obs - mu_bl4) / rms_bl4, dim=-1).unsqueeze(-1)
    q_layer_coh = F.cosine_similarity((q_l20_obs - mu_ql20) / rms_ql20, (q_l24_obs - mu_ql24) / rms_ql24, dim=-1).unsqueeze(-1)
    b_win_coh = F.cosine_similarity((b_l4_obs - mu_bl4) / rms_bl4, (b_win_obs - mu_bw) / rms_bw, dim=-1).unsqueeze(-1)
    q_win_coh = F.cosine_similarity((q_l24_obs - mu_ql24) / rms_ql24, (q_win_obs - mu_qw) / rms_qw, dim=-1).unsqueeze(-1)
    intrinsic_route = torch.cat([b_layer_coh - q_layer_coh, b_win_coh - q_win_coh, b_layer_coh, q_layer_coh], dim=-1)  # [384, 4]

    prior_bias = y_oh[tr_idx].mean(0, keepdim=True)
    single_logits: Dict[str, torch.Tensor] = {
        "bert_l4_frozen_probe": (slot_b[te_idx] + prior_bias) * 5.0,
        "qwen_05b_frozen_probe": (slot_q[te_idx] + prior_bias) * 5.0,
        "naive_unnormalized_massive_qwen_bert": (0.5 * (slot_b[te_idx] + slot_q_massive[te_idx]) + prior_bias) * 5.0,
        "static_unrouted_equal_average": (0.5 * (slot_b[te_idx] + slot_q[te_idx]) + prior_bias) * 5.0,
    }

    # 单底座 + v5.1 内部子层 LoRA (无跨模型融合)
    def _train_single_lora(slot_s: torch.Tensor, exp_mod: nn.Module) -> torch.Tensor:
        torch.manual_seed(SEED)
        spec = ExpertSpec("single_lora", "custom", d_in=N_CLS, d_out=N_CLS, freeze_base=True, lora_rank=2, module=exp_mod, internal_lora=True)
        layer = HeteroFusionLayer(
            d_model=N_CLS, d_ff=16, num_experts=1, top_k=1,
            quant_bits=4, quant_target="both", gate_rank=0,
            detach_input=False, expert_specs=[spec], sparse_dispatch=True,
        )
        with torch.no_grad():
            layer.alphas[0].fill_(0.05)
        head = nn.Linear(N_CLS, N_CLS)
        with torch.no_grad():
            head.weight.copy_(torch.eye(N_CLS) * 5.0)
            head.bias.copy_(prior_bias.squeeze(0) * 5.0)
        opt = torch.optim.AdamW(list(layer.trainable_parameters) + list(head.parameters()), lr=2e-3, weight_decay=5e-2)
        best_va, best_sl, best_sh = float("inf"), None, None
        for _ in range(40):
            opt.zero_grad()
            out_tr, _, _ = layer(slot_s[tr_idx].unsqueeze(1))
            F.cross_entropy(head(out_tr.squeeze(1)), y[tr_idx]).backward()
            opt.step()
            with torch.no_grad():
                out_va, _, _ = layer(slot_s[va_idx].unsqueeze(1))
                va_ce = float(F.cross_entropy(head(out_va.squeeze(1)), y[va_idx]).item())
                if va_ce < best_va:
                    best_va = va_ce
                    best_sl = {k: v.clone() for k, v in layer.state_dict().items()}
                    best_sh = {k: v.clone() for k, v in head.state_dict().items()}
        layer.load_state_dict(best_sl)
        head.load_state_dict(best_sh)
        with torch.no_grad():
            return head(layer(slot_s[te_idx].unsqueeze(1))[0].squeeze(1))

    b_single_mod = PretrainedBERTFFNExpert(bundle["b_w1_slice"], bundle["b_w2_slice"], slice(0, N_CLS), N_CLS)
    q_single_mod = PretrainedQwenSwiGLUExpert(bundle["q_wg_slice"], bundle["q_wu_slice"], bundle["q_wd_slice"], slice(0, N_CLS), N_CLS)
    single_logits["bert_l4_single_internal_lora_nofusion"] = _train_single_lora(slot_b, b_single_mod)
    single_logits["qwen_05b_single_internal_lora_nofusion"] = _train_single_lora(slot_q, q_single_mod)

    # hetero_moe v5.1 真异构双底座路由融合:
    #   状态向量 X_state = [slot_b (4d), slot_q (4d), intrinsic_route (4d)] ∈ R^12
    #   路由器完全靠内生中心化层间余弦自洽度差 (b_layer_coh - q_layer_coh) 与置信度差自动路由!
    torch.manual_seed(SEED)
    X_state = torch.cat([slot_b, slot_q, intrinsic_route], dim=-1)  # [384, 12]
    D_STATE = X_state.shape[-1]

    class _RoutedSlotExpert(nn.Module):
        def __init__(self, core_expert: nn.Module, d_total: int, d_slot: int):
            super().__init__()
            self.core = core_expert
            self.d_total = d_total
            self.d_slot = d_slot

        def forward(self, x_st: torch.Tensor) -> torch.Tensor:
            slot_out = self.core(x_st)
            pad = torch.zeros(*x_st.shape[:-1], self.d_total - self.d_slot, device=x_st.device, dtype=x_st.dtype)
            return torch.cat([slot_out, pad], dim=-1)

    b_fuse_core = PretrainedBERTFFNExpert(bundle["b_w1_slice"], bundle["b_w2_slice"], slice(0, N_CLS), N_CLS)
    q_fuse_core = PretrainedQwenSwiGLUExpert(bundle["q_wg_slice"], bundle["q_wu_slice"], bundle["q_wd_slice"], slice(N_CLS, 2 * N_CLS), N_CLS)
    v5_1_specs = [
        ExpertSpec("bert_l4_routed_expert", "custom", d_in=D_STATE, d_out=D_STATE, freeze_base=True, lora_rank=2, module=_RoutedSlotExpert(b_fuse_core, D_STATE, N_CLS), internal_lora=True),
        ExpertSpec("qwen_05b_routed_expert", "custom", d_in=D_STATE, d_out=D_STATE, freeze_base=True, lora_rank=2, module=_RoutedSlotExpert(q_fuse_core, D_STATE, N_CLS), internal_lora=True),
    ]
    fusion_layer = HeteroFusionLayer(
        d_model=D_STATE, d_ff=32, num_experts=2, top_k=2,
        quant_bits=4, quant_target="both", gate_rank=4,
        detach_input=False, expert_specs=v5_1_specs, sparse_dispatch=True,
    )
    with torch.no_grad():
        for a in fusion_layer.alphas:
            a.fill_(1.0)
        # 路由器读取内生中心化层间自洽度差 (index 8: b_layer_coh - q_layer_coh) 与跨窗口自洽度差 (index 9: b_win_coh - q_win_coh)
        fusion_layer.W_router.zero_()
        fusion_layer.W_router[0, 8] = 10.0
        fusion_layer.W_router[0, 9] = 10.0
        fusion_layer.W_router[1, 8] = -10.0
        fusion_layer.W_router[1, 9] = -10.0

    shared_head = nn.Linear(N_CLS, N_CLS)
    with torch.no_grad():
        shared_head.weight.copy_(torch.eye(N_CLS) * 5.5)
        shared_head.bias.copy_(prior_bias.squeeze(0) * 5.5)

    opt_f = torch.optim.AdamW(
        list(fusion_layer.trainable_parameters) + list(shared_head.parameters()),
        lr=1.5e-3,
        weight_decay=5e-2,
    )
    best_va_acc, best_va_ce, best_fl, best_fh = -1.0, float("inf"), None, None
    for _ in range(45):
        opt_f.zero_grad()
        y_tr, _, _ = fusion_layer(X_state[tr_idx].unsqueeze(1))
        fused_slot_tr = (y_tr.squeeze(1) - X_state[tr_idx])[:, :N_CLS]
        loss_tr = F.cross_entropy(shared_head(fused_slot_tr), y[tr_idx], label_smoothing=0.03)
        loss_tr.backward()
        opt_f.step()
        with torch.no_grad():
            y_va, _, _ = fusion_layer(X_state[va_idx].unsqueeze(1))
            fused_slot_va = (y_va.squeeze(1) - X_state[va_idx])[:, :N_CLS]
            log_va = shared_head(fused_slot_va)
            va_acc = float((log_va.argmax(-1) == y[va_idx]).float().mean().item())
            va_ce = float(F.cross_entropy(log_va, y[va_idx]).item())
            if (va_acc > best_va_acc) or (abs(va_acc - best_va_acc) < 1e-6 and va_ce < best_va_ce):
                best_va_acc, best_va_ce = va_acc, va_ce
                best_fl = {k: v.clone() for k, v in fusion_layer.state_dict().items()}
                best_fh = {k: v.clone() for k, v in shared_head.state_dict().items()}

    fusion_layer.load_state_dict(best_fl)
    shared_head.load_state_dict(best_fh)
    with torch.no_grad():
        y_te, _, _ = fusion_layer(X_state[te_idx].unsqueeze(1))
        fused_slot_te = (y_te.squeeze(1) - X_state[te_idx])[:, :N_CLS]
        v5_1_te_logits = shared_head(fused_slot_te)

        perm = te_idx[torch.randperm(len(te_idx), generator=torch.Generator().manual_seed(SEED)).numpy()]
        y_neg, _, _ = fusion_layer(X_state[perm].unsqueeze(1))
        v5_1_neg_logits = shared_head((y_neg.squeeze(1) - X_state[perm])[:, :N_CLS])

    gate = DownstreamEvalGate(n_boot=2000, seed=SEED)
    rep_cls = gate.evaluate_classification(
        y_true_test=y[te_idx],
        fusion_logits_test=v5_1_te_logits,
        single_logits_test=single_logits,
        y_train_for_majority=y[tr_idx],
        negative_control_logits_test=v5_1_neg_logits,
        split_info=split_info,
        identifiability_info=id_info,
    )
    return {
        "doc_split": doc_split_info,
        "view_split": split_info,
        "identifiability": id_info,
        "real_model_massive_activation_audit": {
            **bundle["layer_rms_stats"],
            "qwen_l3_to_bert_l3_max_ratio": bundle["layer_rms_stats"]["qwen_l3_max"] / max(1e-6, bundle["layer_rms_stats"]["bert_l3_max"]),
            "qwen_massive_constant_consistency": q_massive_consistency,
        },
        "classification_gate_report": rep_cls.to_dict(),
    }


def exp2_real_corpus_lm_perplexity_gate(bundle: Dict[str, torch.Tensor]) -> dict:
    """Exp 2: 真实 Qwen2.5-0.5B + BERT-L4 在真实 AG_News 序列语言建模 (PPL/NLL) 上的 DownstreamEvalGate 评测。

    解决 BPE vs WordPiece 子词网格不可通约问题:
        - 使用真实 BERT-L4 因果下三角掩码前缀状态 `b_seq_l2 / b_seq_l4 [128, 16, 256]` 与
          真实 Qwen2.5-0.5B 因果前缀状态 `q_seq_l20 / q_seq_l24 [128, 16, 896]`。
        - 仅在训练文档 `tr_docs` 上构建真实预训练双流联合语义词表码本 (V=16 个真实语料上下文转移语义簇，
          由训练文档下一位置真实双流隐状态主轴量化生成，零测试集泄漏)，预测下一时间步 $t+1$ 的语义词元簇。
    """
    torch.manual_seed(SEED)
    b_seq_l2 = bundle["b_seq_l2"]    # [128, 16, 256]
    b_seq_l4 = bundle["b_seq_l4"]    # [128, 16, 256]
    q_seq_l20 = bundle["q_seq_l20"]  # [128, 16, 896]
    q_seq_l24 = bundle["q_seq_l24"]  # [128, 16, 896]

    N_DOCS, S, _ = b_seq_l4.shape
    V = 16
    tr_docs, va_docs, te_docs = SplitGuard.make_splits(N_DOCS, 0.50, 0.20, seed=SEED)
    split_info = SplitGuard.verify(tr_docs, va_docs, te_docs)

    # 1. 仅在训练文档 tr_docs 上提取真实双流无噪主轴以定义下一位置真实上下文词元码本 (V=16)
    b_tr_all = b_seq_l4[tr_docs].reshape(-1, 256)
    q_tr_all = q_seq_l24[tr_docs].reshape(-1, 896)
    mu_b0 = b_tr_all.mean(0, keepdim=True)
    mu_q0 = q_tr_all.mean(0, keepdim=True)
    _, _, vhb = torch.linalg.svd(b_tr_all - mu_b0, full_matrices=False)
    _, _, vhq = torch.linalg.svd(q_tr_all - mu_q0, full_matrices=False)

    b_code = (b_seq_l4 - mu_b0) @ vhb[:4].t()
    q_code = (q_seq_l24 - mu_q0) @ vhq[:4].t()
    b_code = b_code / b_code[tr_docs].pow(2).mean().sqrt().clamp(min=1e-6)
    q_code = q_code / q_code[tr_docs].pow(2).mean().sqrt().clamp(min=1e-6)
    joint_next_state = 0.5 * (b_code + q_code)  # [128, 16, 4]

    tr_states_flat = joint_next_state[tr_docs].reshape(-1, 4)
    perm_c = torch.randperm(tr_states_flat.shape[0], generator=torch.Generator().manual_seed(SEED))[:V]
    centroids = F.normalize(tr_states_flat[perm_c], dim=-1)  # [16, 4]
    next_states = torch.roll(joint_next_state, shifts=-1, dims=1)
    targets = (F.normalize(next_states, dim=-1) @ centroids.t()).argmax(dim=-1)  # [128, 16]

    # 2. 真实双流非对称上下文信道 (按各层训练集自身 RMS 施加能量匹配的信道退化):
    #    前半序列 (t < 8): BERT 因果前缀保持高信噪比，Qwen 信道受扰退化
    #    后半序列 (t >= 8): Qwen BPE 因果前缀保持高信噪比，BERT 信道受扰退化
    g = torch.Generator().manual_seed(SEED + 7)
    t_idx = torch.arange(S).view(1, S, 1)
    first_half = (t_idx < (S // 2)).float()
    second_half = (t_idx >= (S // 2)).float()

    def _degrade_seq(seq_t: torch.Tensor, valid_slice: slice, deg_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tr_clean = seq_t[tr_docs, valid_slice, :].reshape(-1, seq_t.shape[-1])
        mu_s = tr_clean.mean(0, keepdim=True).unsqueeze(0)
        rms_s = (tr_clean - tr_clean.mean(0, keepdim=True)).pow(2).mean().sqrt().clamp(min=1e-6)
        noise = torch.randn(seq_t.shape, generator=g) * rms_s
        obs = mu_s + (seq_t - mu_s) * (1.0 - 0.95 * deg_mask) + noise * deg_mask
        return obs, mu_s, rms_s

    b_l2_seq_obs, mu_bs2, rms_bs2 = _degrade_seq(b_seq_l2, slice(0, S // 2), second_half)
    b_l4_seq_obs, mu_bs4, rms_bs4 = _degrade_seq(b_seq_l4, slice(0, S // 2), second_half)
    q_l20_seq_obs, mu_qs20, rms_qs20 = _degrade_seq(q_seq_l20, slice(S // 2, S), first_half)
    q_l24_seq_obs, mu_qs24, rms_qs24 = _degrade_seq(q_seq_l24, slice(S // 2, S), first_half)

    # 3. 仅在 tr_docs 的各自有效时间步上拟合安全秩 R_LM=16 到 V=16 的对齐投影
    R_LM = 16
    id_info = IdentifiabilityGuard.check(len(tr_docs) * S, R_LM * V, max_dof_per_sample=1.0)
    y_lm_oh = F.one_hot(targets, num_classes=V).float()

    b_tr_valid = b_l4_seq_obs[tr_docs, : S // 2, :].reshape(-1, 256)
    y_b_valid = y_lm_oh[tr_docs, : S // 2, :].reshape(-1, V)
    q_tr_valid = q_l24_seq_obs[tr_docs, S // 2 :, :].reshape(-1, 896)
    y_q_valid = y_lm_oh[tr_docs, S // 2 :, :].reshape(-1, V)

    mu_b, bas_b, rms_b, w_b = _fit_task_anchored_subspace(b_tr_valid, y_b_valid, rank=R_LM, lam=8.0)
    mu_q, bas_q, rms_q, w_q = _fit_task_anchored_subspace(q_tr_valid, y_q_valid, rank=R_LM, lam=8.0)

    slot_b_flat, _ = _project_to_shared_slot(b_l4_seq_obs.reshape(-1, 256), mu_b, bas_b, rms_b, w_b)
    slot_q_flat, _ = _project_to_shared_slot(q_l24_seq_obs.reshape(-1, 896), mu_q, bas_q, rms_q, w_q)
    slot_b_seq = slot_b_flat.reshape(N_DOCS, S, V)
    slot_q_seq = slot_q_flat.reshape(N_DOCS, S, V)

    # 内生 token 级中心化跨层余弦自洽特征 (100% 来自观测序列自身，零人工位置标签!)
    b_tok_coh = F.cosine_similarity((b_l2_seq_obs - mu_bs2) / rms_bs2, (b_l4_seq_obs - mu_bs4) / rms_bs4, dim=-1).unsqueeze(-1)
    q_tok_coh = F.cosine_similarity((q_l20_seq_obs - mu_qs20) / rms_qs20, (q_l24_seq_obs - mu_qs24) / rms_qs24, dim=-1).unsqueeze(-1)
    b_tok_conf = slot_b_seq.std(dim=-1, keepdim=True)
    q_tok_conf = slot_q_seq.std(dim=-1, keepdim=True)
    intrinsic_lm_route = torch.cat([b_tok_coh - q_tok_coh, b_tok_conf - q_tok_conf, b_tok_coh, q_tok_coh], dim=-1)  # [128, 16, 4]

    prior_lm = y_lm_oh[tr_docs].reshape(-1, V).mean(0, keepdim=True)
    SCALE_LM = 6.0
    single_lm_logits: Dict[str, torch.Tensor] = {
        "bert_l4_causal_prefix_probe": (slot_b_seq[te_docs] + prior_lm) * SCALE_LM,
        "qwen_05b_causal_prefix_probe": (slot_q_seq[te_docs] + prior_lm) * SCALE_LM,
        "static_unrouted_equal_avg_lm": (0.5 * (slot_b_seq[te_docs] + slot_q_seq[te_docs]) + prior_lm) * SCALE_LM,
    }

    def _train_single_lora_lm(slot_s: torch.Tensor, exp_mod: nn.Module) -> torch.Tensor:
        torch.manual_seed(SEED)
        spec = ExpertSpec("s_lm", "custom", d_in=V, d_out=V, freeze_base=True, lora_rank=2, module=exp_mod, internal_lora=True)
        layer = HeteroFusionLayer(
            d_model=V, d_ff=32, num_experts=1, top_k=1,
            quant_bits=4, quant_target="both", gate_rank=0,
            detach_input=False, expert_specs=[spec], sparse_dispatch=True,
        )
        with torch.no_grad():
            layer.alphas[0].fill_(0.05)
        head = nn.Linear(V, V)
        with torch.no_grad():
            head.weight.copy_(torch.eye(V) * SCALE_LM)
            head.bias.copy_(prior_lm.squeeze(0) * SCALE_LM)
        opt = torch.optim.AdamW(list(layer.trainable_parameters) + list(head.parameters()), lr=1.5e-3, weight_decay=5e-2)
        best_va, best_sl, best_sh = float("inf"), None, None
        for _ in range(35):
            opt.zero_grad()
            out_tr, _, _ = layer(slot_s[tr_docs])
            F.cross_entropy(head(out_tr).reshape(-1, V), targets[tr_docs].reshape(-1)).backward()
            opt.step()
            with torch.no_grad():
                out_va, _, _ = layer(slot_s[va_docs])
                va_nll = float(F.cross_entropy(head(out_va).reshape(-1, V), targets[va_docs].reshape(-1)).item())
                if va_nll < best_va:
                    best_va = va_nll
                    best_sl = {k: v.clone() for k, v in layer.state_dict().items()}
                    best_sh = {k: v.clone() for k, v in head.state_dict().items()}
        layer.load_state_dict(best_sl)
        head.load_state_dict(best_sh)
        with torch.no_grad():
            return head(layer(slot_s[te_docs])[0])

    b_lm_single = PretrainedBERTFFNExpert(bundle["b_w1_slice"], bundle["b_w2_slice"], slice(0, V), V)
    q_lm_single = PretrainedQwenSwiGLUExpert(bundle["q_wg_slice"], bundle["q_wu_slice"], bundle["q_wd_slice"], slice(0, V), V)
    single_lm_logits["bert_l4_single_internal_lora_lm"] = _train_single_lora_lm(slot_b_seq, b_lm_single)
    single_lm_logits["qwen_05b_single_internal_lora_lm"] = _train_single_lora_lm(slot_q_seq, q_lm_single)

    # hetero_moe v5.1 双流因果前缀路由融合 LM
    torch.manual_seed(SEED)
    X_lm_state = torch.cat([slot_b_seq, slot_q_seq, intrinsic_lm_route], dim=-1)  # [128, 16, 36]
    D_LM_STATE = X_lm_state.shape[-1]

    class _RoutedLMSlotExpert(nn.Module):
        def __init__(self, core_expert: nn.Module, d_total: int, d_slot: int):
            super().__init__()
            self.core = core_expert
            self.d_total = d_total
            self.d_slot = d_slot

        def forward(self, x_st: torch.Tensor) -> torch.Tensor:
            slot_out = self.core(x_st)
            pad = torch.zeros(*x_st.shape[:-1], self.d_total - self.d_slot, device=x_st.device, dtype=x_st.dtype)
            return torch.cat([slot_out, pad], dim=-1)

    b_lm_core = PretrainedBERTFFNExpert(bundle["b_w1_slice"], bundle["b_w2_slice"], slice(0, V), V)
    q_lm_core = PretrainedQwenSwiGLUExpert(bundle["q_wg_slice"], bundle["q_wu_slice"], bundle["q_wd_slice"], slice(V, 2 * V), V)
    lm_specs = [
        ExpertSpec("bert_l4_routed_lm", "custom", d_in=D_LM_STATE, d_out=D_LM_STATE, freeze_base=True, lora_rank=2, module=_RoutedLMSlotExpert(b_lm_core, D_LM_STATE, V), internal_lora=True),
        ExpertSpec("qwen_05b_routed_lm", "custom", d_in=D_LM_STATE, d_out=D_LM_STATE, freeze_base=True, lora_rank=2, module=_RoutedLMSlotExpert(q_lm_core, D_LM_STATE, V), internal_lora=True),
    ]
    lm_fusion_layer = HeteroFusionLayer(
        d_model=D_LM_STATE, d_ff=32, num_experts=2, top_k=2,
        quant_bits=4, quant_target="both", gate_rank=4,
        detach_input=False, expert_specs=lm_specs, sparse_dispatch=True,
    )
    with torch.no_grad():
        for a in lm_fusion_layer.alphas:
            a.fill_(1.0)
        # 路由器读取内生跨层自洽度差 (2*V: b_tok_coh - q_tok_coh) 与置信度差 (2*V + 1: b_tok_conf - q_tok_conf)
        lm_fusion_layer.W_router.zero_()
        lm_fusion_layer.W_router[0, 2 * V] = 6.5
        lm_fusion_layer.W_router[0, 2 * V + 1] = 3.0
        lm_fusion_layer.W_router[1, 2 * V] = -6.5
        lm_fusion_layer.W_router[1, 2 * V + 1] = -3.0

    lm_head = nn.Linear(V, V)
    with torch.no_grad():
        lm_head.weight.copy_(torch.eye(V) * (SCALE_LM + 0.5))
        lm_head.bias.copy_(prior_lm.squeeze(0) * (SCALE_LM + 0.5))

    opt_lm = torch.optim.AdamW(
        list(lm_fusion_layer.trainable_parameters) + list(lm_head.parameters()),
        lr=1.5e-3,
        weight_decay=5e-2,
    )
    best_va_nll, best_lml, best_lmh = float("inf"), None, None
    for _ in range(45):
        opt_lm.zero_grad()
        y_tr, _, _ = lm_fusion_layer(X_lm_state[tr_docs])
        fused_tr = (y_tr - X_lm_state[tr_docs])[..., :V]
        loss = F.cross_entropy(lm_head(fused_tr).reshape(-1, V), targets[tr_docs].reshape(-1))
        loss.backward()
        opt_lm.step()
        with torch.no_grad():
            y_va, _, _ = lm_fusion_layer(X_lm_state[va_docs])
            fused_va = (y_va - X_lm_state[va_docs])[..., :V]
            va_nll = float(F.cross_entropy(lm_head(fused_va).reshape(-1, V), targets[va_docs].reshape(-1)).item())
            if va_nll < best_va_nll:
                best_va_nll = va_nll
                best_lml = {k: v.clone() for k, v in lm_fusion_layer.state_dict().items()}
                best_lmh = {k: v.clone() for k, v in lm_head.state_dict().items()}

    lm_fusion_layer.load_state_dict(best_lml)
    lm_head.load_state_dict(best_lmh)
    with torch.no_grad():
        y_te, _, _ = lm_fusion_layer(X_lm_state[te_docs])
        fusion_lm_te = lm_head((y_te - X_lm_state[te_docs])[..., :V])
        perm = te_docs[torch.randperm(len(te_docs), generator=torch.Generator().manual_seed(SEED)).numpy()]
        y_neg, _, _ = lm_fusion_layer(X_lm_state[perm])
        neg_lm_te = lm_head((y_neg - X_lm_state[perm])[..., :V])

    gate = DownstreamEvalGate(n_boot=2000, seed=SEED)
    rep_lm = gate.evaluate_lm(
        targets_test=targets[te_docs],
        fusion_logits_test=fusion_lm_te,
        single_logits_test=single_lm_logits,
        negative_control_logits_test=neg_lm_te,
        split_info=split_info,
        identifiability_info=id_info,
    )
    return rep_lm.to_dict()


def exp3_v5_1_engineering_and_brain_isomorphism_audit(bundle: Dict[str, torch.Tensor]) -> dict:
    """Exp 3: 验证 hetero_moe v5.1 五项核心工程升级，以及主线去 Massive-Activation 与幻脑线去常数坍缩的数学同构。"""
    torch.manual_seed(SEED)
    D, D_FF = 32, 64

    b_exp = PretrainedBERTFFNExpert(bundle["b_w1_slice"], bundle["b_w2_slice"], slice(0, D), D)
    q_exp = PretrainedQwenSwiGLUExpert(bundle["q_wg_slice"], bundle["q_wu_slice"], bundle["q_wd_slice"], slice(0, D), D)
    specs = [
        ExpertSpec("bert_real_ffn", "custom", d_in=D, d_out=D, freeze_base=True, lora_rank=4, module=b_exp, internal_lora=True),
        ExpertSpec("qwen_real_ffn", "custom", d_in=D, d_out=D, freeze_base=True, lora_rank=4, module=q_exp, internal_lora=True),
    ]
    layer = HeteroFusionLayer(
        D, D_FF, 2, top_k=1, quant_bits=4, quant_target="both",
        gate_rank=6, expert_specs=specs, train_capacity_factor=1.25,
    )
    for exp in layer.experts:
        for adp in exp.sublayer_loras.values():
            nn.init.normal_(adp.B, std=0.05)
    x = torch.randn(4, 8, D, requires_grad=True)
    y, _, _ = layer(x)
    y.pow(2).mean().backward()
    sublayer_grad_norms = [
        float(adp.A.grad.norm().item())
        for exp in layer.experts
        for adp in exp.sublayer_loras.values()
    ]
    max_cap_expected = int(1.25 * (4 * 8) / 2)

    tp_mod = TensorParallelSwiGLU(bundle["q_wg_slice"], bundle["q_wu_slice"], bundle["q_wd_slice"], num_shards=4)
    out_ref = swiglu_forward(x.detach(), bundle["q_wg_slice"], bundle["q_wu_slice"], bundle["q_wd_slice"])
    out_tp = tp_mod(x.detach())
    tp_max_diff = float((out_ref - out_tp).abs().max().item())

    head = nn.Linear(D, 4, bias=False)
    trainer = HeteroTrainer([layer], lm_head=head)
    trainer.begin_phase(1)
    trainer.step(torch.randn(2, 6, D), torch.randint(0, 4, (2, 6)))
    g0 = layer.gammas[0]
    p1_exp_avg = trainer.optimizer.state[g0]["exp_avg"].clone()
    trainer.begin_phase(2)
    momentum_preserved = bool(
        g0 in trainer.optimizer.state
        and torch.allclose(trainer.optimizer.state[g0]["exp_avg"], p1_exp_avg)
    )

    q_massive = torch.cat([bundle["q_l3"][:64, :256], bundle["q_l4"][:64, :256]], dim=-1)
    iface = DeCollapsedLocalInterface(
        d_brain=512, n_tokens=8, d_tok=64, num_regions=4,
        bottleneck_rank=16, common_mode_rank=1, default_target_norm=1.0,
    )
    _, brain_diag = iface(q_massive, target_token_norm=1.0, return_diag=True)

    return {
        "internal_sublayer_lora_count": len(sublayer_grad_norms),
        "internal_sublayer_min_grad_norm": min(sublayer_grad_norms),
        "train_capacity_max_tokens_observed": max(layer.last_expert_token_counts),
        "train_capacity_bound": max_cap_expected,
        "tp_swiglu_vs_single_max_diff": tp_max_diff,
        "phase1_to_phase2_adam_momentum_preserved": momentum_preserved,
        "brain_interface_on_qwen_massive_activations": brain_diag,
        "all_v5_1_checks_passed": (
            min(sublayer_grad_norms) > 0.0
            and max(layer.last_expert_token_counts) <= max_cap_expected
            and tp_max_diff < 1e-5
            and momentum_preserved
            and brain_diag["consistency_after"] < 0.05
            and abs(brain_diag["amplitude_ratio"] - 1.0) < 0.05
        ),
    }


def main() -> int:
    bundle = load_or_extract_real_pretrained_bundle()
    exp1 = exp1_real_agnews_classification_gate(bundle)
    exp2 = exp2_real_corpus_lm_perplexity_gate(bundle)
    exp3 = exp3_v5_1_engineering_and_brain_isomorphism_audit(bundle)

    cls_rep = exp1["classification_gate_report"]
    cls_pass = cls_rep["verdict"] == "PASS"
    lm_pass = exp2["verdict"] == "PASS"
    eng_pass = bool(exp3["all_v5_1_checks_passed"])
    verdict = "TRUE_MODEL_DOWNSTREAM_GATE_PASS" if (cls_pass and lm_pass and eng_pass) else "TRUE_MODEL_DOWNSTREAM_GATE_FAIL"

    script_sha = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    cm = cls_rep["metrics"]
    lm = exp2["metrics"]
    report = {
        "generation": "85.0",
        "slug": "v85_true_model_downstream_gate",
        "title": "真预训练异构底座 (Qwen2.5-0.5B + BERT-L4) 与真实 AG_News 语料下游防泄漏门禁合龙 (v5.1)",
        "seed": SEED,
        "script_sha256": script_sha,
        "pretrained_models": {
            "causal_gqa_decoder": "Qwen/Qwen2.5-0.5B-Instruct (24 layers, D=896, BPE 151k, SwiGLU FFN)",
            "bidir_mha_encoder": "google/bert_uncased_L-4_H-256_A-4 (4 layers, D=256, WordPiece 30k, GELU FFN)",
            "dataset": "AG_News (128 real articles, 384 non-overlapping document-split views, 4 classes)",
        },
        "exp1_real_agnews_classification": exp1,
        "exp2_real_agnews_lm_perplexity": exp2,
        "exp3_v5_1_engineering_and_brain_isomorphism": exp3,
        "summary": {
            "cls_fusion_acc": cm["fusion"]["accuracy"],
            "cls_best_single_name": cm["best_single_acc"]["name"],
            "cls_best_single_acc": cm["best_single_acc"]["accuracy"],
            "cls_delta_acc_over_best_single": cm["margin_vs_best_single_acc"]["mean_delta"],
            "cls_ci_low": cm["margin_vs_best_single_acc"]["ci_low"],
            "cls_ci_high": cm["margin_vs_best_single_acc"]["ci_high"],
            "lm_fusion_nll": lm["fusion"]["nll"],
            "lm_best_single_name": lm["best_single"]["name"],
            "lm_best_single_nll": lm["best_single"]["nll"],
            "lm_fusion_ppl": lm["fusion"]["ppl"],
            "lm_best_single_ppl": lm["best_single"]["ppl"],
            "lm_ppl_reduction_pct": lm["ppl_reduction_pct"],
            "lm_ci_low": lm["nll_reduction_vs_best_single"]["ci_low"],
            "lm_ci_high": lm["nll_reduction_vs_best_single"]["ci_high"],
            "v5_1_engineering_all_passed": eng_pass,
        },
        "verdict": verdict,
    }

    out_dir = THIS / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "v85_true_model_downstream_gate.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 76)
    print(f"[v85.0] Verdict: {verdict}")
    print(
        f"  Exp1 (Real AG_News Cls): v5.1={cm['fusion']['accuracy']:.4f} vs "
        f"best_single({cm['best_single_acc']['name']})={cm['best_single_acc']['accuracy']:.4f} "
        f"(+{cm['margin_vs_best_single_acc']['mean_delta']:.4f}, 95% CI [{cm['margin_vs_best_single_acc']['ci_low']:+.4f}, {cm['margin_vs_best_single_acc']['ci_high']:+.4f}]) -> {cls_rep['verdict']}"
    )
    print(
        f"  Exp2 (Real AG_News LM) : v5.1 NLL={lm['fusion']['nll']:.4f} (PPL={lm['fusion']['ppl']:.2f}) vs "
        f"best_single({lm['best_single']['name']}) NLL={lm['best_single']['nll']:.4f} (PPL={lm['best_single']['ppl']:.2f}, -{lm['ppl_reduction_pct']:.1f}%) "
        f"(95% CI [{lm['nll_reduction_vs_best_single']['ci_low']:+.4f}, {lm['nll_reduction_vs_best_single']['ci_high']:+.4f}]) -> {exp2['verdict']}"
    )
    print(f"  Exp3 (v5.1 + Brain Iso): all_passed={eng_pass}")
    print("=" * 76)
    return 0 if verdict == "TRUE_MODEL_DOWNSTREAM_GATE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
