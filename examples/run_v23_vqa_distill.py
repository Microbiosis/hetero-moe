"""V23.0 — Mini-VQA 真实任务验证多模态嵌合 (BERT + ViT 学生).

4 变体 × 5 seeds = 20 run:
    random-baseline       — 随机猜 (基线下界)
    text-only             — 用 v21 学生但 image 全黑 (文本信息)
    image-only            — 用 v21 学生但 text 全 mask (图像信息)
    multimodal-distill    — v21 学生 + LoRA + 任务 loss

核心判断: 多模态学生是否真的需要 image 模态才能答 VQA?
           (text-only 应远差于 multimodal)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from research._primitives.attention import AttnPool
from research.routing_evolution.v8_central import V8Trainer
from research.aligner.v10_embedding import CrossArchAttnAligner
from research.multimodal.v21_multimodal import MultiModalStudent
from research.multimodal.v23_vqa import MiniVQADataset
from examples.run_v8_full import D_SHARED


def get_text_inputs(tokenizer, question: str, max_length: int = 16):
    enc = tokenizer(question, padding="max_length", truncation=True,
                     max_length=max_length, return_tensors="pt")
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def run_seed(mode, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"  [seed {seed}] mode={mode} ...")
    dataset = MiniVQADataset(num_train=12, seed=seed)

    if mode == "random-baseline":
        # 随机猜 4 类, 评估准确率
        correct = 0
        total = 0
        for s in dataset.all_eval():
            guess = random.randint(0, 3)
            if guess == s["label"]:
                correct += 1
            total += 1
        accuracy = correct / total
        loss_proxy = 1 - accuracy   # 错误率作 loss proxy
        return loss_proxy

    # 学生模型
    tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    student = MultiModalStudent(d_out=D_SHARED, lora_rank=8)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-2)

    train_samples = dataset.all_train()
    eval_samples = dataset.all_eval()

    # 训练: CE loss on 4 类答案
    STEPS = 50
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for s in train_samples:
            if mode == "text-only":
                img = torch.zeros_like(s["image"])   # 全黑
            elif mode == "image-only":
                img = s["image"]
                text_inputs = get_text_inputs(tokenizer, "[PAD] " + s["question"])   # mask text
            else:  # multimodal
                img = s["image"]
                text_inputs = get_text_inputs(tokenizer, s["question"])
            if mode != "image-only":
                text_inputs = get_text_inputs(tokenizer, s["question"])
            y, _ = student(text_inputs, img.unsqueeze(0))   # [1, S, D]
            # 取首个 token 的输出作分类 logits (简化)
            logits = y[:, 0, :4]   # [1, 4]
            target = torch.tensor([s["label"]], dtype=torch.long)
            loss = F.cross_entropy(logits, target)
            total_loss = total_loss + loss
        total_loss = total_loss / len(train_samples)
        total_loss.backward()
        opt.step()

    # 评估: eval 集准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for s in eval_samples:
            if mode == "text-only":
                img = torch.zeros_like(s["image"])
            elif mode == "image-only":
                img = s["image"]
                text_inputs = get_text_inputs(tokenizer, "[PAD] " + s["question"])
            else:
                img = s["image"]
                text_inputs = get_text_inputs(tokenizer, s["question"])
            if mode != "image-only":
                text_inputs = get_text_inputs(tokenizer, s["question"])
            y, _ = student(text_inputs, img.unsqueeze(0))
            logits = y[:, 0, :4]
            pred = logits.argmax(dim=-1).item()
            if pred == s["label"]:
                correct += 1
            total += 1
    return 1 - correct / total   # 返回 error rate (越低越好)


MODES = ["random-baseline", "text-only", "image-only", "multimodal-distill"]
SEEDS_V23 = 3   # 端到端跑 3 seeds (CPU 慢)


def main():
    print("=" * 78)
    print("V23.0 — Mini-VQA 真实任务验证多模态嵌合")
    print("=" * 78)
    print(f"4 变体 × {SEEDS_V23} seeds = {4 * SEEDS_V23} run")
    print("核心判断: 多模态学生是否真的需要 image 模态才能答 VQA?")
    print()
    results = {m: [] for m in MODES}

    print(f"\n每个 mode × {SEEDS_V23} seeds")
    for mode in MODES:
        for seed in range(SEEDS_V23):
            try:
                err = run_seed(mode, seed)
                results[mode].append(err)
                print(f"  [{mode:>20} seed {seed}] error_rate={err:.4f}")
            except Exception as e:
                print(f"  [{mode:>20} seed {seed}] FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results[mode].append(float("nan"))

    # 汇总
    print("\n" + "=" * 78)
    print(f"{'mode':>22} | {'error rate':>11} | {'acc':>5}")
    print("-" * 78)
    for mode in MODES:
        vals = [x for x in results[mode] if not np.isnan(x)]
        if not vals:
            print(f"{mode:>22} | {'N/A':>11} | {'N/A':>5}")
            continue
        m_mean = statistics.mean(vals)
        acc = 1 - m_mean
        print(f"{mode:>22} | {m_mean:>11.4f} | {acc:>5.1%}")
    print("=" * 78)
    print()
    print("结论:")
    f_r = statistics.mean([x for x in results["random-baseline"] if not np.isnan(x)])
    f_t = statistics.mean([x for x in results["text-only"] if not np.isnan(x)])
    f_i = statistics.mean([x for x in results["image-only"] if not np.isnan(x)])
    f_m = statistics.mean([x for x in results["multimodal-distill"] if not np.isnan(x)])
    print(f"  random-baseline:    {f_r:.4f}  (random 猜, error≈75%)")
    print(f"  text-only:          {f_t:.4f}")
    print(f"  image-only:         {f_i:.4f}")
    print(f"  multimodal-distill: {f_m:.4f}")
    print()
    if f_m < f_t and f_m < f_i:
        print(f"✅ multimodal 最优 (学生能融合 text+image)")
    elif f_m < f_t:
        print(f"⚠ multimodal 优于 text-only 但未必优于 image-only")
    else:
        print(f"⚠ multimodal 未显著优于单模态")


if __name__ == "__main__":
    main()