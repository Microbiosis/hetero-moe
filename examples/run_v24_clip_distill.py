"""V24.0 — CLIP 预训练教师 + 真实多模态数据蒸馏 (3 mode × 3 seeds = 9 run).

3 变体 × 3 seeds:
    random-baseline       — 随机投影 image-text → 512 (无学习)
    student-no-distill    — MultiModalStudent 不蒸馏 (从零对齐)
    student-distill       — MultiModalStudent + LoRA 蒸馏 CLIP 融合 embedding

核心判断: CLIP 已有 image-text 对齐空间, 学生能否用极少样本(12 train)
          蒸馏后匹配 CLIP 融合空间 (验证 v23b 失败的根因是 toy 数据 + 从零学)
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
from research.multimodal.v24_clip import CLIPTeacher, MiniImageTextCorpus

D_CLIP = 512


def get_text_inputs(tokenizer, text: str, max_length: int = 16):
    enc = tokenizer(text, padding="max_length", truncation=True,
                    max_length=max_length, return_tensors="pt")
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def run_seed(mode: str, seed: int, epochs: int = 20) -> float:
    """单 seed 单 mode 训练, 返回最终 distillation MSE."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print(f"  [seed {seed}] mode={mode} ...")
    corpus = MiniImageTextCorpus(train_size=12, eval_size=4, seed=seed)
    teacher = CLIPTeacher()

    if mode == "random-baseline":
        # 不训练任何学生, 只用一个固定随机投影把学生 hidden → D_CLIP
        # 评估"完全不学习时的基线 MSE"
        random_proj = torch.randn(256, D_CLIP) * 0.01  # 小初始化
        mse_list = []
        for s in corpus.all_eval():
            pixel_values = s["image"].unsqueeze(0)
            text_inputs = teacher.processor(
                text=[s["text"]], return_tensors="pt",
                padding=True, truncation=True,
            )
            target = teacher.fused_target(
                pixel_values, text_inputs["input_ids"],
                text_inputs["attention_mask"], alpha=0.5,
            )
            # 模拟学生预测 (固定随机向量, 不依赖样本)
            student_pred = torch.randn(1, D_CLIP) * 0.1
            mse_list.append(F.mse_loss(student_pred, target).item())
        return statistics.mean(mse_list)

    tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    student = MultiModalStudent(d_out=256, lora_rank=8)

    if mode == "student-no-distill":
        # 不蒸馏, 让学生 hidden 跟随机目标比较 (作为无蒸馏基线)
        proj = nn.Linear(256, D_CLIP).requires_grad_(False)
        optimizer = None
    else:  # student-distill
        proj = nn.Linear(256, D_CLIP)
        optimizer = torch.optim.Adam(
            list(student.parameters()) + list(proj.parameters()),
            lr=1e-3,
        )

    for epoch in range(epochs):
        total_loss = 0.0
        for s in corpus.all_train():
            pixel_values = s["image"].unsqueeze(0)
            text_inputs = get_text_inputs(tokenizer, s["text"], max_length=16)

            with torch.no_grad():
                target = teacher.fused_target(
                    pixel_values, text_inputs["input_ids"],
                    text_inputs["attention_mask"], alpha=0.5,
                )

            y, weights = student(text_inputs, pixel_values)
            # y: [B, S_text, 256] → mean pool → [B, 256]
            h = y.mean(dim=1)
            pred = proj(h)
            loss = F.mse_loss(pred, target)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/{epochs}: loss={total_loss/len(corpus.all_train()):.4f}")

    # 评估
    eval_mse = []
    for s in corpus.all_eval():
        pixel_values = s["image"].unsqueeze(0)
        text_inputs = get_text_inputs(tokenizer, s["text"], max_length=16)
        with torch.no_grad():
            target = teacher.fused_target(
                pixel_values, text_inputs["input_ids"],
                text_inputs["attention_mask"], alpha=0.5,
            )
            y, _ = student(text_inputs, pixel_values)
            h = y.mean(dim=1)
            pred = proj(h)
            eval_mse.append(F.mse_loss(pred, target).item())
    return statistics.mean(eval_mse)


def main():
    modes = ["random-baseline", "student-no-distill", "student-distill"]
    seeds = [0, 1, 2]
    print("=== V24.0 — CLIP 预训练教师 + 多模态蒸馏 (3 mode × 3 seeds) ===")
    results: dict = {m: [] for m in modes}
    for mode in modes:
        for seed in seeds:
            try:
                mse = run_seed(mode, seed, epochs=20)
                results[mode].append(mse)
                print(f"    -> eval MSE = {mse:.4f}")
            except Exception as e:
                print(f"    !! seed {seed} failed: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results[mode].append(float("nan"))

    print("\n--- 汇总 (mean ± std over seeds) ---")
    for mode in modes:
        vals = [v for v in results[mode] if not (isinstance(v, float) and np.isnan(v))]
        if vals:
            m = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"  {mode:25s}: {m:.4f} ± {sd:.4f}")
        else:
            print(f"  {mode:25s}: ALL FAIL")
    return results


if __name__ == "__main__":
    main()
