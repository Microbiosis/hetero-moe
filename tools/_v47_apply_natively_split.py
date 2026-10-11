#!/usr/bin/env python3
"""按 v47 审计结论回写各包 MANIFEST 的 leakage_audit 段。

只改 leakage_audit 一个键, 其余字段一律不碰 —— 审计结论不应该顺手改别的数字。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# pkg -> (status, audits, note)
UPDATES = {
    "research/scale_heterogeneous/v36_alpha2_knn": (
        "clean", ["36.0"],
        "**更正此前 MANIFEST 里『本包无 held-out 评估』的错误说法**。v36 原生就是样本外: "
        "评估用无参数 leave-one-out 1-NN (`research/_primitives/scale_hetero_loaders/eval.py:13-22`, "
        "第 19 行 `dists[i] = float(\"inf\")` 把查询样本自己排除), 被审计的分类器**没有任何可训练参数**, "
        "因此不存在目标泄漏面。\n"
        "⚠️ 但要说清另一件事: 该协议的**统计功效**很低 —— `get_tabldm_repr(seed=42)` 的 12 个 y_table "
        "是 `rng.randint` 随机标签, 无可学结构, 所以绝对准确率接近随机; 结论靠的是"
        "『同结构同维度的随机对照』(X_ctl) 与入口臂的配对比较, 不是靠绝对准确率。",
    ),
    "research/scale_heterogeneous/v37_b_selftrained_entry": (
        "clean", ["37.0"],
        "**更正此前 MANIFEST 里『本包无 held-out 评估』的错误说法**。v37 原生 96/32 留出: "
        "`examples/run_v37_b_selftrained_entry.py:77-79` 用 `RandomState(1234)` 的置换切成互不相交的 "
        "`tr_idx`/`te_idx`, 训练循环只从 `tr_idx` 取 batch, 准确率只在 `te_idx` 上量 "
        "(`research/_primitives/scale_hetero_loaders/arm.py:41-47`)。四臂共用同一个 split, 对照公平。\n"
        "⚠️ 旁证 `knn_acc(h_all, y)` 走的是全部 128 条的 leave-one-out, 含训练样本 —— 但它是**无参数**的"
        "拓扑读数, 报告里已与 test_acc 分列, 不构成泄漏。",
    ),
    "research/scale_heterogeneous/v38_g_adult_boundary": (
        "clean", ["38.0"],
        "**更正此前 MANIFEST 里『本包无 held-out 评估』的错误说法**。v38 原生 384/128 留出: "
        "`examples/run_v38_g_adult_boundary.py:130-133` 置换切分互不相交, 第 99 行的训练 batch 只取 "
        "`tr_idx`, 第 106 行的 `acc` 只在 `te_idx` 上量。五臂共用同一 split。\n"
        "注: TabLDM 被动表征是对**全部**样本(含测试行)跑的无监督 encoder, 不接触标签, 不构成标签泄漏。",
    ),
    "research/scale_heterogeneous/v39_g2_interaction": (
        "clean", ["39.0"],
        "**更正此前 MANIFEST 里『本包无 held-out 评估』的错误说法**。v39 原生 384/128 留出: "
        "`examples/run_v39_g2_interaction.py:143-146` 置换切分, 第 111 行的准确率只在 `te_idx` 上量。"
        "五臂 × 双数据集共用同一 split。",
    ),
    "research/multimodal/v23_vqa": (
        "clean", ["23.0"],
        "**更正此前 MANIFEST 里『本包无 held-out 评估』的错误说法**。v23 原生 train/eval 分离: "
        "`research/multimodal/v23_vqa/dataset.py:78-79` 用 `seed` 生成训练集、`seed + 1000` 生成评估集"
        "(不同随机图像); `examples/run_v23_vqa_distill.py:63-64` 分别取 `all_train()` / `all_eval()`, "
        "第 92-96 行的准确率只在 eval 集上量。",
    ),
    "research/multimodal/v24_clip": (
        "clean", ["24.0"],
        "**更正此前 MANIFEST 里『本包无 held-out 评估』的错误说法**。v24 原生 train/eval 分离: "
        "`research/multimodal/v24_clip/corpus.py:105-111` 两个 split 用不同的 seed 偏移构造"
        "(源码注释即写明『不同 split 用不同的 seed 偏移避免完全重复』); "
        "`examples/run_v24_clip_distill.py:81` 只在 `all_train()` 上训练, 第 106-118 行的 eval MSE "
        "只在 `all_eval()` 上量。",
    ),
    "archive/negative_findings/v23b_hard_vqa": (
        "clean", ["23b.0"],
        "**更正此前 MANIFEST 里『本包无 held-out 评估』的错误说法**。v23b 原生 24/8 留出: "
        "`archive/negative_findings/v23b_hard_vqa/hard_dataset.py:104-106` 训练取 idx 0-23, "
        "评估取 idx 24-31 且换 `seed + 10000` 且 `is_train=False`; "
        "`examples/run_v23_vqa_hard.py:59-60, 91` 分别取 `all_train()` / `all_eval()`。\n"
        "四臂全部恰为 50%(=随机) 这一**否定**结论不依赖 held-out —— 泄漏只会虚高 in-sample 优势, "
        "不会凭空造出一个负结果; 但切分确实是原生的, 上面这条是据实核实而非推定。",
    ),
}


def main():
    touched = 0
    for rel, (status, audits, note) in UPDATES.items():
        path = ROOT / rel / "MANIFEST.json"
        man = json.loads(path.read_text(encoding="utf-8"))
        before = man.get("leakage_audit", {})
        man["leakage_audit"] = {
            "status": status,
            "audits": audits,
            "note": note,
            "audited_by": "47.0",
        }
        path.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        touched += 1
        print(f"{man['package']:<28} {before.get('status')} -> {status}")
    print(f"\n回写 {touched} 个包")


if __name__ == "__main__":
    main()
