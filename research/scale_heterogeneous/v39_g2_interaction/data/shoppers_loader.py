"""Online Shoppers 数据集加载器 — 第二个真实数据集验证 v9 跨模态结论.

UCI Online Shoppers Purchasing Intention: 12330 样本 × 18 列.
- 数值特征 10 个 (表格侧): Administrative, *_Duration, ProductRelated, BounceRates,
  ExitRates, PageValues, SpecialDay
- 类别特征 7 个 (文本侧): Month, OperatingSystems, Browser, Region, TrafficType,
  VisitorType, Weekend
- 标签: Revenue

跨模态 XOR 标签:
  y = 1 iff (VisitorType == New_Visitor [只在文本侧]) XOR (ProductRelated ≥ 18 [只在表格侧])
  两条件近似独立且各自约半 → 单侧信息为零, 跨模态必需.

decouple=True: 非 ProductRelated 数值列样本间随机置换, 切断与 VisitorType 的关联
(ProductRelated_Duration 等与访客类型相关, 否则表格侧间接泄漏).
"""
import os
import numpy as np

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_CSV = os.path.join(_DATA_DIR, "online_shoppers_intention.csv")

_NUM_COLS = ["Administrative", "Administrative_Duration", "Informational",
             "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
             "BounceRates", "ExitRates", "PageValues", "SpecialDay"]
_CAT_COLS = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType",
             "VisitorType", "Weekend"]
_PR_THR = 18  # ProductRelated 阈值 (≈中位数, 满足比例 50.6%)
_PR_IDX = _NUM_COLS.index("ProductRelated")  # 4 — XOR 表格条件, 解耦时保持不动


def _rows():
    import csv
    out = []
    with open(_CSV, encoding="ascii", errors="ignore") as f:
        for r in csv.DictReader(f):
            try:
                nums = [float(r[c]) for c in _NUM_COLS]
            except (ValueError, KeyError):
                continue
            cats = [r[c] for c in _CAT_COLS]
            is_new = cats[5] == "New_Visitor"
            pr_hi = nums[_PR_IDX] >= _PR_THR
            out.append((cats, nums, 1 if is_new != pr_hi else 0))
    return out


def load_shoppers_crossmodal(n=512, seed=42, decouple=True):
    """跨模态构造: 文本(类别特征) + 表格(数值特征), XOR 标签需两侧联合.

    四格均衡采样: (New/Returning) × (PR</≥阈值) 各取 n/4 → 两条件在采样集中
    各自 50% 且独立, 严格切断 XOR 的边际泄漏 (否则稀缺的 New 客会使 PR 与 y
    产生边际关联, 纯表格单侧虚高).
    """
    import torch
    rows = _rows()
    cells = {(False, False): [], (False, True): [], (True, False): [], (True, True): []}
    for cats, nums, y in rows:
        is_new = cats[5] == "New_Visitor"
        pr_hi = nums[_PR_IDX] >= _PR_THR
        cells[(is_new, pr_hi)].append((cats, nums, y))
    rng = np.random.RandomState(seed)
    q = n // 4
    sel = []
    for key, lst in cells.items():
        idx = rng.choice(len(lst), q, replace=len(lst) < q)
        sel += [lst[i] for i in idx]
    order = rng.permutation(len(sel)).tolist()
    sel = [sel[i] for i in order]

    texts, Xs, ys = [], [], []
    for cats, nums, y in sel:
        wk = "on a weekend" if cats[6] == "TRUE" else "on a weekday"
        t = (f"A {cats[5].lower().replace('_', '-')} in {cats[0].lower()}, using browser "
             f"{cats[2]} on os {cats[1]}, from region {cats[3].lower()} via traffic "
             f"type {cats[4]}, {wk}.")
        texts.append(t); Xs.append(list(nums)); ys.append(y)
    Xa = np.asarray(Xs, dtype=np.float32)
    if decouple:
        for c in range(len(_NUM_COLS)):
            if c != _PR_IDX:
                Xa[:, c] = Xa[rng.permutation(len(Xa)), c]
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    X = (Xa - mu) / sd
    return texts, torch.tensor(X), torch.tensor(ys, dtype=torch.long)


if __name__ == "__main__":
    texts, X, y = load_shoppers_crossmodal(64)
    print(f"Shoppers: X{tuple(X.shape)} 正样本率 {y.float().mean():.2%}")
    print(f"  text[0]={texts[0]!r}")
