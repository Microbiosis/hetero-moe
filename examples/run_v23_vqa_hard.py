"""V23b — Hard VQA 真实任务端到端 (8 类, 24 训练 + 8 eval).

4 变体 × 3 seeds = 12 run:
    random-baseline       — 随机猜 8 类 (基线下界)
    text-only             — 用 v21 学生但 image 全黑 (只用文本)
    image-only            — 用 v21 学生但 text mask (只用图像)
    multimodal-distill    — v21 学生 + LoRA + CE loss

核心判断: 8 类陷阱任务下, multimodal 是否优于单模态?
           (text-only 至多 12.5% (1/8), image-only 至多 62.5% (5/8), multimodal 应 > 62.5%)
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

from research.multimodal.v21_multimodal import MultiModalStudent
from research.multimodal.v23_vqa import HardVQADataset
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
    dataset = HardVQADataset(train_per_class=3, seed=seed)

    if mode == "random-baseline":
        correct = 0
        total = 0
        for s in dataset.all_eval():
            guess = random.randint(0, 7)
            if guess == s["label"]:
                correct += 1
            total += 1
        accuracy = correct / total
        return 1 - accuracy

    tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    student = MultiModalStudent(d_out=D_SHARED, lora_rank=8)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-2)

    train_samples = dataset.all_train()
    eval_samples = dataset.all_eval()
    NUM_LABELS = 8

    STEPS = 50
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for s in train_samples:
            img = s["image"]
            if mode == "text-only":
                img_input = torch.zeros_like(img)
            elif mode == "image-only":
                img_input = img
                text_inputs = get_text_inputs(tokenizer, "[PAD] " + s["question"])
            else:  # multimodal
                img_input = img
                text_inputs = get_text_inputs(tokenizer, s["question"])
            if mode != "image-only":
                text_inputs = get_text_inputs(tokenizer, s["question"])
            y, _ = student(text_inputs, img_input.unsqueeze(0))
            logits = y[:, 0, :NUM_LABELS]  # [1, 8]
            target = torch.tensor([s["label"]], dtype=torch.long)
            loss = F.cross_entropy(logits, target)
            total_loss = total_loss + loss
        total_loss = total_loss / len(train_samples)
        total_loss.backward()
        opt.step()

    correct = 0
    total = 0
    with torch.no_grad():
        for s in eval_samples:
            img = s["image"]
            if mode == "text-only":
                img_input = torch.zeros_like(img)
            elif mode == "image-only":
                img_input = img
                text_inputs = get_text_inputs(tokenizer, "[PAD] " + s["question"])
            else:
                img_input = img
                text_inputs = get_text_inputs(tokenizer, s["question"])
            if mode != "image-only":
                text_inputs = get_text_inputs(tokenizer, s["question"])
            y, _ = student(text_inputs, img_input.unsqueeze(0))
            logits = y[:, 0, :NUM_LABELS]
            pred = logits.argmax(dim=-1).item()
            if pred == s["label"]:
                correct += 1
            total += 1
    return 1 - correct / total   # error rate


MODES = ["random-baseline", "text-only", "image-only", "multimodal-distill"]
SEEDS_V23B = 3


def main():
    print("=" * 78)
    print("V23b — Hard VQA 真实任务 (8 类陷阱设计)")
    print("=" * 78)
    print(f"4 变体 × {SEEDS_V23B} seeds = {4 * SEEDS_V23B} run")
    print("核心判断: 8 类陷阱任务下 multimodal 是否优于单模态?")
    print()
    results = {m: [] for m in MODES}

    print(f"\n每个 mode × {SEEDS_V23B} seeds")
    for mode in MODES:
        for seed in range(SEEDS_V23B):
            try:
                err = run_seed(mode, seed)
                results[mode].append(err)
                print(f"  [{mode:>20} seed {seed}] error_rate={err:.4f}")
            except Exception as e:
                print(f"  [{mode:>20} seed {seed}] FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results[mode].append(float("nan"))

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
    print(f"  random-baseline:    {f_r:.4f}  (random 猜, error≈87.5%)")
    print(f"  text-only:          {f_t:.4f}")
    print(f"  image-only:         {f_i:.4f}")
    print(f"  multimodal-distill: {f_m:.4f}")
    print()
    if f_m < f_i and f_m < f_t:
        improvement_i = (f_i - f_m) / max(f_i, 1e-9) * 100
        improvement_t = (f_t - f_m) / max(f_t, 1e-9) * 100
        print(f"✅ multimodal 最优 (比 image-only 改善 {improvement_i:.1f}%, 比 text-only 改善 {improvement_t:.1f}%)")
    elif f_m < f_t:
        print(f"⚠ multimodal 优于 text-only 但未必优于 image-only")
    else:
        print(f"⚠ multimodal 未显著优于单模态")


if __name__ == "__main__":
    main()