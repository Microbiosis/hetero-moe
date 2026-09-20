"""v33.0 — 真实底座浅层加载与编码。

规模异构融合的输入编码阶段:
  大: MSA-4B        (D=2560) — Qwen3 风格权重, 浅层加载 (省内存)
  小: embedding-0.6b / reranker-0.6b (D=1024) — Qwen3 backbone

MSA-4B 的 model_type='msa' 不被 transformers 原生识别, 但其权重 key 结构
(q_norm/k_norm, SwiGLU, GQA, RMSNorm, RoPE) 与 Qwen3Model 完全兼容,
故构造匹配 Qwen3Config → Qwen3Model → 去 'model.' 前缀加载 (missing=0, unexpected=0)。
"""
import json
import os
import time

import torch
from safetensors import safe_open
from transformers import Qwen3Model, Qwen3Config, AutoModel, AutoTokenizer

from .paths import MSA_DIR, EMB_DIR, RER_DIR


def load_msa_big(msa_dir=None, n_layers=None, env_key="V33_BIG_LAYERS"):
    """加载 MSA-4B backbone 浅层 (Qwen3Model, 无 lm_head)。

    资源优化: 不加载完整 36 层 8GB 权重, 只取 embed_tokens + 前 N 层 + norm
    (~1GB)。融合实验只需 D=2560 的隐藏态表征, 单层即保留大维度特征。
    """
    msa_dir = msa_dir or MSA_DIR
    if n_layers is None:
        n_layers = int(os.environ.get(env_key, "1"))
    print(f"  [大] 加载 MSA-4B 浅层 ({n_layers}层, 非完整36层)...")
    t0 = time.time()
    with open(f"{msa_dir}/config.json") as f:
        msa_cfg = json.load(f)
    qcfg = Qwen3Config(
        hidden_size=msa_cfg["hidden_size"],
        num_hidden_layers=n_layers,
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
    # 选择性加载: 只取 embed_tokens + 前 N 层 + norm (省内存)
    want_prefixes = {f"model.layers.{li}." for li in range(n_layers)}
    state = {}
    for shard in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]:
        with safe_open(f"{msa_dir}/{shard}", framework="pt") as f:
            for k in f.keys():
                if k == "model.embed_tokens.weight" or k == "model.norm.weight" \
                   or any(k.startswith(p) for p in want_prefixes):
                    state[k[len("model."):]] = f.get_tensor(k)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    print(f"    完成 {time.time()-t0:.1f}s | D={msa_cfg['hidden_size']} "
          f"layers={n_layers}/{msa_cfg['num_hidden_layers']} | "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    return model, msa_cfg["hidden_size"], msa_cfg.get("vocab_size")


def load_small(name, path):
    """加载小模型 backbone (embedding 是 Qwen3Model; reranker 是 Qwen3ForCausalLM -> 取 .model)。"""
    print(f"  [小] 加载 {name} ...")
    t0 = time.time()
    model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16)
    if hasattr(model, "model"):  # ForCausalLM -> 底层 backbone
        model = model.model
    model.eval()
    cfg = model.config
    print(f"    完成 {time.time()-t0:.1f}s | D={cfg.hidden_size} "
          f"layers={cfg.num_hidden_layers} vocab={cfg.vocab_size}")
    return model, cfg.hidden_size, cfg.vocab_size


def load_tokenizer(path):
    return AutoTokenizer.from_pretrained(path)


def encode(texts, tok, model, S):
    """用模型原生 tokenizer 编码文本, 返回 last_hidden_state [B, S, D] (detached)。"""
    enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="pt")
    with torch.no_grad():
        h = model(enc["input_ids"]).last_hidden_state
    return h.detach()  # [B, S, D]
