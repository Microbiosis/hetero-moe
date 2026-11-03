"""路线5：投影空间的 LLM 语义解码。

v62 四槽功能靠 CKA/余弦数值推断。本脚本把 SlotSubspace 权重序列化为
结构化 JSON（奇异值谱、列范数、top-k 索引），生成给 LLM 的 prompt，
让 LLM 用自然语言解释每槽在对齐/路由中的角色。

依赖：任一带 OpenAI 兼容 API 的 LLM（或 Anthropic）。
不依赖 GPU，纯文本处理。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def serialize_slot(W: np.ndarray, topk: int = 8) -> dict:
    """把投影矩阵 [d_out, d_in] 压缩为 LLM 可读的结构化摘要。"""
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    col_norms = np.linalg.norm(W, axis=0)
    row_norms = np.linalg.norm(W, axis=1)
    return {
        'shape': list(W.shape),
        'rank': int((S > 1e-6).sum()),
        'top_singular_values': S[:topk].round(4).tolist(),
        'sv_ratio_top1_topk': float(S[0] / (S[:topk].sum() + 1e-9)),
        'col_norm_top_idx': np.argsort(-col_norms)[:topk].tolist(),
        'col_norm_top_val': np.sort(-col_norms)[:topk].round(4).tolist(),
        'row_norm_top_idx': np.argsort(-row_norms)[:topk].tolist(),
        'row_norm_top_val': np.sort(-row_norms)[:topk].round(4).tolist(),
        'weight_std': float(W.std()),
        'weight_mean': float(W.mean()),
        'sparsity': float((np.abs(W) < 1e-3).mean()),
    }


def build_prompt(slot_data: dict, slot_name: str) -> str:
    return f"""你是一个神经网络可解释性专家。下面是一个 MoE 模型中
「{slot_name} 槽」的投影矩阵权重统计（JSON 格式）。

背景：该模型有三个异构骨干（Qwen/BERT/ViT），每个功能槽（Q/K/V/O）
有一个 256×256 的投影矩阵，负责把不同模态的 hidden state 映射到
共享子空间。已知：
- K/V 槽只有 BERT/ViT 参与（两两 CKA 0.987）
- Q/O 槽三模型都参与（两两 0.987，三模型同时 0.68/0.72）
- 训练后 Q 槽 router 权重约 0.13，K 槽约 0.27

请根据下面的权重统计，用中文回答：
1. 这个槽的投影矩阵呈现什么结构？（低秩/正交/稀疏/集中？）
2. 列范数 top 索引和行范数 top 索引告诉你什么？
3. 结合奇异值谱，这个槽更像「对齐用」还是「路由用」？为什么？
4. 如果让你给这个槽起一个语义名字，会叫什么？

权重统计：
{json.dumps(slot_data, indent=2, ensure_ascii=False)}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='v62.4 .pt（含 proj_space state_dict）')
    ap.add_argument('--out', default='slot_prompts.jsonl')
    ap.add_argument('--llm', default=None,
                    help='openai:gpt-4o-mini 或 anthropic:claude-...，留空则只生成 prompt')
    args = ap.parse_args()

    import torch
    sd = torch.load(args.ckpt, map_location='cpu')
    model_sd = sd.get('model', sd)
    slots = ['Q', 'K', 'V', 'O']
    lines = []
    for s in slots:
        # 取该槽第一个模型的投影矩阵
        key = f'proj_space.proj.{s}|Qwen.weight'
        if key not in model_sd:
            key = f'proj.proj.{s}|Qwen.weight'
        if key not in model_sd:
            print(f'[warn] 找不到 {s} 槽权重，跳过')
            continue
        W = model_sd[key].numpy()
        data = serialize_slot(W)
        prompt = build_prompt(data, s)
        rec = {'slot': s, 'stats': data, 'prompt': prompt}
        if args.llm:
            rec['llm_response'] = call_llm(args.llm, prompt)
        lines.append(rec)
        print(f'[slot {s}] rank={data["rank"]} '
              f'sv_ratio={data["sv_ratio_top1_topk"]:.3f} '
              f'sparsity={data["sparsity"]:.3f}')

    with open(args.out, 'w') as f:
        for r in lines:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'saved {len(lines)} slots → {args.out}')


def call_llm(spec: str, prompt: str) -> str:
    """极简 LLM 调用：支持 openai: 和 anthropic: 前缀。"""
    provider, model = spec.split(':', 1)
    if provider == 'openai':
        from openai import OpenAI
        client = OpenAI()
        r = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.3)
        return r.choices[0].message.content
    elif provider == 'anthropic':
        import anthropic
        client = anthropic.Anthropic()
        r = client.messages.create(
            model=model, max_tokens=1024,
            messages=[{'role': 'user', 'content': prompt}])
        return r.content[0].text
    raise ValueError(f'不支持的 provider: {provider}')


if __name__ == '__main__':
    main()
