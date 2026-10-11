"""v35.0-α — 大模型容器 (proprietary large-LM backbone, Qwen3Model 架构) + 异质入口提取。

入口 1: 表格入口 — forward hook 截获 icl_predictor 输入 = representations (D=512)
入口 2/3: embedding / reranker 文本入口 — 浅层编码 (D=1024)
容器: 大模型浅层前向; prefix 作为 prior 加到 embed (不增序列长度, 不干扰 attention)

注: 2026-10-11 更正 —— 本文件原先写着「所有辅助模型都是机构内部模型, 权重不在公开仓库,
需向作者索取」。**这句话已经不成立**: 四个模型 (EverMind-AI/MSA-4B、
skillcorpus-embedding-0.6b、skillcorpus-reranker-0.6b、occams/Xiaomi-TabLDM) 经核实
全部公开可下, 路径解析见 `research/_primitives/scale_hetero_loaders/paths.py`
(环境变量 → 本地候选 → HuggingFace Hub 自动下载; TabLDM 的源码另从 GitHub 自动克隆)。
v35 的负结果因此**可完整复现**, 不需要向任何人索取权重。
"""
import gc
import json
import time

import numpy as np
import torch
from safetensors import safe_open
from transformers import Qwen3Model, Qwen3Config, AutoModel, AutoTokenizer

from .paths import (
    BIG_LM_DIR, EMBED_LM_DIR, RERANK_LM_DIR, TAB_LM_CKPT,
)

# 2026-10-11 修正: load_big_container 的函数体用的是裸名 MSA_DIR, 而本模块只 import 了
# 旧别名 BIG_LM_DIR ⇒ 构造容器时 NameError, **v35 的端到端 runner 从未跑起来过**。
# get_tabular_repr 里的 TABLDM_CKPT 是同一类问题(只 import 了旧别名 TAB_LM_CKPT)。
# 两处都在这里显式绑定, 让函数体与 paths.py 的导出名对齐。
MSA_DIR = BIG_LM_DIR
TABLDM_CKPT = TAB_LM_CKPT


# =====================================================================
# 入口 1: 表格入口 (hook 取 representations)
# =====================================================================

def get_tabular_repr():
    """tabular-LM 处理表格, hook 截获 icl_predictor 输入 = representations → c_struct [512]."""
    print("  [入口] 表格编码...")
    try:
        from tabldm import TabLDMClassifier
    except ImportError as e:
        raise ImportError(
            "TabLDM 不可用 (未安装 tabldm 且未找到源码目录)。"
            "请 `pip install /path/to/xiaomi-tabldm` 或设置 V35_TABLDM_SRC 后重试。"
        ) from e
    clf = TabLDMClassifier(model_path=TABLDM_CKPT, device="cpu")
    rng = np.random.RandomState(42)
    X = rng.randn(12, 6).astype(np.float32)
    y = rng.randint(0, 2, 12)
    clf.fit(X, y)
    m = clf.model_
    reps = []
    h = m.icl_predictor.register_forward_hook(
        lambda mod, inp, out: reps.append(inp[0].detach() if isinstance(inp, tuple) else inp.detach()))
    Xq = rng.randn(6, 6).astype(np.float32)
    clf.predict_proba(Xq)
    h.remove()
    # representations: [n_ens, seq, 512] → mean → [512]
    r = reps[0]
    c_struct = r.mean(dim=(0, 1)) if r.dim() >= 3 else r.mean(dim=0)
    c_struct = c_struct.float()
    print(f"    representations {tuple(r.shape)} → c_struct {tuple(c_struct.shape)}")
    del clf, m
    gc.collect()
    return c_struct  # [512]


# =====================================================================
# 入口 2/3: 文本小模型浅层编码
# =====================================================================

def load_backbone(path, name, S=16):
    print(f"  [入口] {name} 浅层编码...")
    model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16)
    if hasattr(model, "model"):
        model = model.model
    model.eval()
    tok = AutoTokenizer.from_pretrained(path)
    texts = ["cat dog bird", "animal pet wild", "feline canine fowl",
             "quick brown fox", "lazy dog sleeps", "birds fly high"]
    enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="pt")
    with torch.no_grad():
        h = model(enc["input_ids"]).last_hidden_state
    c = h.mean(dim=1).float()  # [6, D]
    D = c.shape[-1]
    print(f"    {name} → c {tuple(c.shape)} D={D}")
    del model
    gc.collect()
    return c, D  # [6, D], D


# =====================================================================
# 大模型容器 (MSA-4B 浅层, Qwen3Model)
# =====================================================================

def load_big_container(n_layers=None, env_key="V35_BIG_LAYERS"):
    n_layers = int(n_layers if n_layers is not None else __import__("os").environ.get(env_key, "1"))
    print("  [容器] 加载 MSA-4B 浅层...")
    t0 = time.time()
    with open(f"{MSA_DIR}/config.json") as f:
        msa_cfg = json.load(f)
    qcfg = Qwen3Config(
        hidden_size=msa_cfg["hidden_size"], num_hidden_layers=n_layers,
        num_attention_heads=msa_cfg["num_attention_heads"],
        num_key_value_heads=msa_cfg["num_key_value_heads"],
        intermediate_size=msa_cfg["intermediate_size"],
        head_dim=msa_cfg.get("head_dim"),
        max_position_embeddings=msa_cfg.get("max_position_embeddings"),
        rope_theta=msa_cfg.get("rope_theta"),
        hidden_act=msa_cfg.get("hidden_act", "silu"),
        rms_norm_eps=msa_cfg.get("rms_norm_eps", 1e-6),
        vocab_size=msa_cfg.get("vocab_size", 151936),
    )
    model = Qwen3Model(qcfg).to(torch.bfloat16)
    want = {f"model.layers.{li}." for li in range(n_layers)}
    state = {}
    for shard in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]:
        with safe_open(f"{MSA_DIR}/{shard}", framework="pt") as f:
            for k in f.keys():
                if k == "model.embed_tokens.weight" or k == "model.norm.weight" \
                   or any(k.startswith(p) for p in want):
                    state[k[len("model."):]] = f.get_tensor(k)
    miss, unexp = model.load_state_dict(state, strict=False)
    model.eval()
    D_BIG = msa_cfg["hidden_size"]
    print(f"    完成 {time.time()-t0:.1f}s | D={D_BIG} layers={n_layers} miss={len(miss)} unexp={len(unexp)}")
    return model


def big_forward(model, input_ids, prefix=None):
    """大模型容器前向. prefix 非 None 时作为 prior 加到 embed (不增序列长度, 不干扰 attention)."""
    with torch.no_grad():
        embed = model.embed_tokens(input_ids)  # [B, S, 2560]
        if prefix is not None:
            prior = prefix.mean(dim=1, keepdim=True).to(embed.dtype)  # [B, 1, 2560]
            inputs_embeds = embed + prior
            out = model(inputs_embeds=inputs_embeds)
        else:
            out = model(input_ids)
        h = out.last_hidden_state  # [B, S, 2560]
    return h.float()
