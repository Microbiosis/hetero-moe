"""v36.0 — TabLDM 表格入口 (逐样本) + 大模型浅层容器。

相对 v35 的修复:
  v35 bug: TabLDM 表征 `r.mean(dim=(0,1))` 折叠成 [512] 常向量 (无 per-sample 信息),
           再 `prefix.mean(dim=1, keepdim=True)` 塌缩成 [B,1,2560] 常数注入
           → 信息量为零, ProjHead 只能把常数视作偏置 (由此得出 v35 假阴性)
  v36:    TabLDM 表征保留 [B, 512] 逐样本 + 用**无参数 k-NN** 评估
           (干净避免线性探针过拟合, 是真实表征质量的镜像)
  任务:   跨模态 — 文本无关, 表格分类 10 类
"""
import gc
import glob
import json
import time

import numpy as np
import torch
from safetensors import safe_open
from transformers import Qwen3Model, Qwen3Config, AutoTokenizer

from . import paths as _paths

D_BIG = 2560  # MSA-4B hidden_size
B_DEFAULT = 12  # query batch
S_DEFAULT = 16  # token seq length


def get_tabldm_repr(n=B_DEFAULT, n_classes=10, seed=42):
    """TabLDM 处理表格, hook 取查询样本 representations → c_struct [B, 512] 逐样本 + y_table [B].

    关键修复 (相对 v35): 仅对查询样本取 [B, 512], 不 `mean(dim=(0,1))` 折叠成常数。
    """
    print(f"  [入口] TabLDM 表格编码 (逐样本, n={n}, n_classes={n_classes})...")
    _ = _paths.TABLDM_SRC
    try:
        from tabldm import TabLDMClassifier
    except ImportError as e:
        raise ImportError(
            "TabLDM 不可用 (未安装 tabldm 且未找到源码目录)。"
            "请 `pip install /path/to/xiaomi-tabldm` 或设置 V36_TABLDM_SRC 后重试。"
        ) from e
    clf = TabLDMClassifier(model_path=_paths.TABLDM_CKPT, device="cpu")
    rng = np.random.RandomState(seed)
    X_train = rng.randn(12, 6).astype(np.float32)
    y_train = rng.randint(0, n_classes, 12)
    clf.fit(X_train, y_train)
    m = clf.model_
    reps = []
    h = m.icl_predictor.register_forward_hook(
        lambda mod, inp, out: reps.append(inp[0].detach() if isinstance(inp, tuple) else inp.detach()))
    Xq = rng.randn(n, 6).astype(np.float32)
    yq = rng.randint(0, n_classes, n)
    clf.predict_proba(Xq)
    h.remove()
    r = reps[0]  # [n_ens, seq, 512]
    rep_q = r[:, -n:, :]  # [n_ens, B, 512] 取查询样本部分
    c_struct = rep_q.mean(dim=0).float()  # [B, 512] 逐样本 (修复 v35 折叠 bug)
    y_table = torch.tensor(yq, dtype=torch.long)
    print(f"    representations {tuple(r.shape)} → c_struct {tuple(c_struct.shape)} y_table {tuple(y_table.shape)}")
    del clf, m
    gc.collect()
    return c_struct, y_table


def load_big_container(n_layers=None, env_key="V36_BIG_LAYERS", msa_dir=None):
    n_layers = int(n_layers if n_layers is not None else __import__("os").environ.get(env_key, "1"))
    msa_dir = msa_dir or _paths.MSA_DIR
    print("  [容器] 加载 MSA-4B 浅层...")
    t0 = time.time()
    with open(f"{msa_dir}/config.json", encoding="utf-8") as f:
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
    shards = sorted(glob.glob(f"{msa_dir}/*.safetensors"))
    if not shards:
        raise RuntimeError(
            f"{msa_dir} 下没有 *.safetensors —— MSA-4B 权重未就位。"
            "请先下载 EverMind-AI/MSA-4B (或设 V36_MSA_DIR 指向本地目录)")
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if k == "model.embed_tokens.weight" or k == "model.norm.weight" \
                   or any(k.startswith(p) for p in want):
                    state[k[len("model."):]] = f.get_tensor(k)
    miss, unexp = model.load_state_dict(state, strict=False)
    model.eval()
    print(f"    完成 {time.time()-t0:.1f}s | D={D_BIG} layers={n_layers} miss={len(miss)} unexp={len(unexp)}")
    return model


def big_forward(model, input_ids, attention_mask=None):
    """无注入的大模型前向 (纯净基线，支持传入 attention_mask 屏蔽 pad)."""
    with torch.no_grad():
        if attention_mask is not None:
            return model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
        return model(input_ids).last_hidden_state.float()


def encode_baseline_text(model, tokenizer, texts, s=S_DEFAULT):
    """文本 → MSA-4B 浅层 → 掩码平均池化 → [B, D_BIG]。与表格完全无关 (baseline)。"""
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=s, return_tensors="pt")
    ids = enc["input_ids"]
    mask = enc.get("attention_mask")
    with torch.no_grad():
        h = big_forward(model, ids, attention_mask=mask)
        if mask is not None:
            m = mask.to(dtype=h.dtype, device=h.device).unsqueeze(-1)
            return (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return h.mean(dim=1)  # [B, 2560]