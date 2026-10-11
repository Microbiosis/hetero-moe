"""v40.0 — 分派式多教师 (runner).

检验 ``research/distillation/v17_multi_teacher/MANIFEST.json`` 的
``root_cause_identified_later``: v17 多教师蒸馏的失败是否源于
"同时多目标回归的实现形式", 而非多教师本身不可行.

协议
----
**教师、数据、训练循环、评估口径全部从 v17 复用**
(``from examples.run_v17_multi_teacher import make_teacher, prepare_data,
train_teacher``), 以保证 arm 之间以及与 v17 之间可比.

**两处刻意偏离 v17, 均为提高测量精度**:

1. **评估输入固定**: v17 的评估 ``torch.randint`` 消耗全局 RNG, 不同 arm
   抽到的评估样本不同. 本 runner 用**独立且固定种子**的 generator 生成评估
   输入, 使所有 arm 面对**同一批**评估样本 (配对比较).
2. **成对评估**: v17 逐样本评估; 本 runner 一次前向 6 个样本, 各 arm 同批.

对照臂
------
======================================  ================================
arm                                     角色
======================================  ================================
``weighted-3-v17style``                 复现锚点 — v17 逐样本循环原样
``weighted-3``                         加权和 (批量版)
``single-mixture`` / ``single-mom``    单教师基线 (= v17 的两个 single)
``dispatch-cyclic``                    **零信息** 分派对照
``dispatch-consensus``                 无泄漏分派
``dispatch-oracle``                    ⚠️ **泄漏上界**, 禁止引用为结论
======================================  ================================

无训练的诊断 (``seed 0``): 教师两两分歧 + 各教师/共识对目标的 MSE.

用法::

    python3 examples/run_v40_dispatch.py            # 5 seeds
    V40_SEEDS=1 python3 examples/run_v40_dispatch.py  # 冒烟
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from research.distillation.v16_bert_student import BERTStudent
from research.distillation.v40_dispatch import (
    consensus_assign,
    cyclic_assign,
    dispatch_loss,
    oracle_assign,
    pairwise_disagreement,
    teacher_mean,
    teacher_target_mse,
    weighted_loss,
)

# ── 协议常量: 全部复用 v17 / v8_full ──────────────────────────────────────
from examples.run_v17_multi_teacher import make_teacher, prepare_data, train_teacher
from examples.run_v8_full import D_SHARED, LR, N_CLS, S, SEEDS, STEPS, load_encoders, make_pools

EVAL_SEED = 20261010          # 评估输入的固定种子 (配对比较)
N_EVAL = 6                    # BERT 模态样本数

MODES = [
    "weighted-3-v17style",
    "weighted-3",
    "consensus-target",
    "single-mixture",
    "single-mom",
    "dispatch-cyclic",
    "dispatch-consensus",
    "dispatch-oracle",
]

LEAKING_MODES = {"dispatch-oracle"}

THREE = ["attn", "mixture", "mom"]


def build_seed_context(seed, encoders, attn_pools):
    """训练 v17 的三个对齐器教师并返回其融合输出 + 评估目标.

    与 v17 ``run_seed`` 的 97-136 行逐行等价, 两处差别:
    - 教师只训一次, **在所有 arm 之间共享** (v17 是每个 mode 重训一遍)
    - 直接返回 ``[6, S, D_SHARED]`` 的堆叠张量而非逐样本列表

    注意 ``AlignedFusionLayer`` 需要**全部三个模态同时输入**
    (``h_list = [h_text, h_code, h_img]``, 每个 ``[6, S, D_m]``),
    输出是融合后的 ``[6, S, D_SHARED]`` —— 不是只喂 BERT 模态.
    """
    torch.manual_seed(seed)
    modal_seqs, modal_indices, targets = prepare_data(seed, encoders)

    teachers = {}
    for kind in THREE:
        t = make_teacher(kind, attn_pools, encoders)
        train_teacher(t, modal_seqs, modal_indices, targets)
        t.eval()
        for p in t.parameters():
            p.requires_grad = False
        teachers[kind] = t

    h_by_modal = [[] for _ in range(N_CLS)]
    for i, m in enumerate(modal_indices):
        h_by_modal[m].append(modal_seqs[i])
    h_list = [torch.stack(h_by_modal[c], dim=0) for c in range(N_CLS)]

    stack = {}
    with torch.no_grad():
        for kind, t in teachers.items():
            stack[kind] = t(h_list).detach()           # [6, S, D_SHARED]

    # v17 的评估目标: targets[0..5] (prepare_data 中 class 0 的 6 份副本, 同一张量重复 6 次)
    eval_target = torch.stack(targets[:N_EVAL], dim=0)  # [6, S, D_SHARED]
    return stack, eval_target


def run_arm(mode, stack, eval_target, bert_tok):
    """在给定教师输出上训练学生并做配对评估."""
    assign, _ = _assign_for(mode, stack, eval_target)

    student = BERTStudent(model_name="huawei-noah/TinyBERT_General_4L_312D",
                          d_out=D_SHARED, lora_rank=8)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=LR)
    vocab_size = bert_tok.vocab_size

    if mode == "weighted-3-v17style":
        # v17 原始循环: 逐样本前向, 每个教师只对齐**同一个样本**
        # (v17 的 teacher_outputs[kind][s].unsqueeze(0) 是 [S, D] -> [1, S, D],
        #  与 student_outs[kind] = y 的 [1, S, D] 对齐)
        for _ in range(STEPS):
            opt.zero_grad(set_to_none=True)
            losses = []
            for s in range(N_EVAL):
                input_ids = torch.randint(0, vocab_size, (S,), dtype=torch.long)
                y = student(input_ids.unsqueeze(0))                       # [1, S, D]
                sub = {k: v[s:s + 1] for k, v in stack.items()}           # 每个教师只取第 s 个样本
                losses.append(weighted_loss(y, sub, weights={k: 1.0 / 3 for k in THREE})["total"])
            (sum(losses) / len(losses)).backward()
            opt.step()
    else:
        for _ in range(STEPS):
            opt.zero_grad(set_to_none=True)
            input_ids = torch.randint(0, vocab_size, (N_EVAL, S), dtype=torch.long)
            y = student(input_ids)                                        # [6, S, D]
            _training_loss(mode, y, stack, assign, eval_target).backward()
            opt.step()

    # 配对评估: 固定 generator, 所有 arm 面对同一批输入
    gen = torch.Generator().manual_seed(EVAL_SEED)
    fuse_loss = 0.0
    with torch.no_grad():
        for s in range(N_EVAL):
            input_ids = torch.randint(0, vocab_size, (S,), dtype=torch.long, generator=gen)
            y = student(input_ids.unsqueeze(0))
            fuse_loss += F.mse_loss(y, eval_target[s].unsqueeze(0)).item()
    return fuse_loss / N_EVAL, assign


def _assign_for(mode, stack, eval_target):
    """返回 (assign, 泄漏标记). assign 为长度 N_EVAL 的教师名序列.

    加权臂不用分派, 返回空列表占位.
    """
    if mode.startswith("single-"):
        return [mode.split("-", 1)[1]] * N_EVAL, False
    if mode == "dispatch-cyclic":
        return cyclic_assign(stack), False
    if mode == "dispatch-consensus":
        return consensus_assign(stack), False
    if mode == "dispatch-oracle":
        return oracle_assign(stack, eval_target), True   # ⚠️ 读目标
    if mode.startswith("weighted-") or mode == "consensus-target":
        return [], False
    raise ValueError(f"未知 arm: {mode}")


def _training_loss(mode, y, stack, assign, target):
    """给定批量学生输出, 返回该 arm 的标量损失."""
    if mode == "weighted-3-v17style":
        return weighted_loss(y, stack, weights={k: 1.0 / 3 for k in THREE})["total"]
    if mode == "weighted-3":
        return weighted_loss(y, stack)["total"]
    if mode == "consensus-target":
        # 朝**教师均值**蒸馏 (不是分派). 无泄漏, 且不需要选谁.
        return torch.nn.functional.mse_loss(y, teacher_mean(stack).detach())
    return dispatch_loss(y, stack, assign)


def main():
    seeds = int(os.environ.get("V40_SEEDS", SEEDS))
    print("=" * 78)
    print("V40.0 — 分派式多教师")
    print("=" * 78)
    print(f"检验对象: v17 MANIFEST.root_cause_identified_later")
    print(f"对照: MOPD (arXiv:2606.30406) 的 per-prompt 分派")
    print(f"{len(MODES)} arm × {seeds} seeds = {len(MODES) * seeds} run")
    print(f"⚠️ dispatch-oracle 读目标, 仅为上界, 不得引用为结论")
    print()

    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    bert_tok = encoders[0][0]

    results = {m: [] for m in MODES}
    assignments = {m: [] for m in MODES}
    diagnostic = {}

    # seed 在外层: 教师只训一次, 在所有 arm 间共享 (v17 是每个 mode 重训)
    for seed in range(seeds):
        try:
            stack, eval_target = build_seed_context(seed, encoders, attn_pools)
        except Exception as e:
            print(f"  [seed {seed}] 教师构建 FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            for m in MODES:
                results[m].append(float("nan"))
            continue

        if seed == 0:
            diagnostic["pairwise_disagreement"] = pairwise_disagreement(stack)
            diagnostic["teacher_target_mse"] = teacher_target_mse(stack, eval_target)
            diagnostic["teacher_names"] = list(stack)
            # 共享目标的事实: 6 个样本的目标是同一张量
            diagnostic["targets_all_identical"] = bool(
                all(torch.equal(eval_target[0], eval_target[i]) for i in range(N_EVAL))
            )
            print(f"\n  [seed {seed}] 教师诊断: "
                  f"两两分歧={diagnostic['pairwise_disagreement']:.6f}  "
                  f"目标全同={diagnostic['targets_all_identical']}  "
                  f"教师目标MSE={ {k: round(v, 6) for k, v in diagnostic['teacher_target_mse'].items()} }")

        for mode in MODES:
            try:
                fuse, assign = run_arm(mode, stack, eval_target, bert_tok)
                results[mode].append(fuse)
                assignments[mode].append(list(assign))
                print(f"    [{mode:>20} seed {seed}] fuse={fuse:.4f}  assign={assign}")
            except Exception as e:
                print(f"    [{mode:>20} seed {seed}] FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results[mode].append(float("nan"))

    # ── 无训练诊断 ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("教师侧诊断 (seed 0, 无训练)")
    print("-" * 78)
    print(f"  两两平均平方距离      : {diagnostic.get('pairwise_disagreement', float('nan')):.6f}")
    print(f"  6 个样本目标是否全同 : {diagnostic.get('targets_all_identical')}")
    print("  各教师 / 共识 对目标 MSE:")
    for k, v in (diagnostic.get("teacher_target_mse") or {}).items():
        print(f"      {k:<20} {v:.6f}")

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'arm':>22} | {'fuse MSE':>9} | {'vs single-mixture':>17} | {'泄漏'}")
    print("-" * 78)
    base_vals = [x for x in results["single-mixture"] if not np.isnan(x)]
    base_mean = statistics.mean(base_vals) if base_vals else float("nan")
    summary = {}
    for m in MODES:
        vals = [x for x in results[m] if not np.isnan(x)]
        if not vals:
            print(f"{m:>22} | {'N/A':>9} | {'N/A':>17} |")
            continue
        mean = statistics.mean(vals)
        delta = (base_mean - mean) / max(base_mean, 1e-9) * 100
        summary[m] = {
            "mean": mean,
            "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
            "delta_pct_vs_single_mixture": delta,
            "leaks": m in LEAKING_MODES,
        }
        print(f"{m:>22} | {mean:>9.4f} | {delta:>+16.1f}% | {'⚠ 是' if m in LEAKING_MODES else '否'}")

    # 分派是否退化 (只用一个教师?)
    print("\n分派退化检查 (每 arm 用到的不同教师数 / arm):")
    for m in MODES:
        if m.startswith("weighted-") or not assignments[m]:
            continue
        uniq = sorted({t for a in assignments[m] for t in a})
        print(f"    {m:<22} {len(uniq)} / 3  -> {uniq}")
    print("=" * 78)

    payload = {
        "experiment": "v40_dispatch",
        "seeds": seeds,
        "constants": {"D_SHARED": D_SHARED, "S": S, "STEPS": STEPS, "LR": LR,
                      "N_EVAL": N_EVAL, "EVAL_SEED": EVAL_SEED},
        "diagnostic": diagnostic,
        "arms": summary,
        "raw": results,
        "assignments": assignments,
        "leaking_modes": sorted(LEAKING_MODES),
    }
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "research", "distillation", "v40_dispatch", "results", "raw.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n写入: {out}")


if __name__ == "__main__":
    main()
