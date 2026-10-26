"""scan_movable_components.py — 「把能用的拿来」的清单。

## 做什么

以**自研 tokenizer 骨架的目标 token 维 D=64** 为基准，
把 6 个基模的每个组件（注意力 / FFN）逐键比对，判定三档：

| 判定 | 含义 | 能不能直接搬 |
|---|---|---|
| `movable` | **逐键同形**（键名 + 形状） | ✅ 直接 `load_state_dict`，零改动 |
| `projectable` | 同角色但维度不同 | ⚠️ 需要学习投影 ⇒ **必须训练** |
| `incompatible` | 结构不同族 / 键集不同 | ❌ 搬不了（除非改结构） |

并把 exp75 的**语义强度**挂在每个组件上 ⇒
「哪些组件既搬得进去、又带语义」= 组装新架构的候选清单。

## 为什么以 D=64 为基准

自研 tokenizer（exp80）的 token 维：ch_mix Conv1d(63→64) → t_down Conv1d(64→64)
⇒ **D=64**；VQ 码本 (K, 64)。这是我们自己可控的那个宽度，
⇒ 用它当目标宽度，组装的「学习投影」负担最小。
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
STRUCT = HERE / "structure"
TARGET_D = 64          # 自研 tokenizer 的 token 维
TARGET_H = 1024        # 目标 FFN 隐藏维（我们那一档的量级）

ATTN_RE = re.compile(
    r"(in_proj_weight|out_proj\.weight|qkv\.weight|to_q\.weight|to_k\.weight|"
    r"to_v\.weight|to_out\.weight|proj\.weight|queries\.weight|keys\.weight)", re.I)
FFN_RE = re.compile(r"^(weight|w1\.weight|w2\.weight|fc1\.weight|fc2\.weight|"
                    r"1\.0\.weight|1\.2\.weight|0\.weight|2\.weight)$", re.I)


def load_models():
    out = {}
    for tag in ("stable", "dev"):
        p = STRUCT / f"{tag}.json"
        if not p.exists():
            continue
        for rec in json.loads(p.read_text(encoding="utf-8"))["models"]:
            out[rec["model"]] = rec
    return out


def classify(child):
    """一个子模块 ⇒ (角色, {键: 形状})。"""
    sh = child.get("shapes", {}) or {}
    if not sh:
        return None, {}
    if any(ATTN_RE.search(k) for k in sh):
        return "attn", sh
    if any(FFN_RE.search(k) for k in sh):
        return "ffn", sh
    return None, sh


def movable_to_target(role, shapes):
    """逐键同形判定。"""
    if role == "attn":
        # 目标：一个 embed_dim=64 的 nn.MultiheadAttention
        want = {"in_proj_weight": [192, 64], "in_proj_bias": [192],
                "out_proj.weight": [64, 64], "out_proj.bias": [64]}
        keys = set(shapes) & set(want)
        if len(keys) >= 2 and all(list(shapes[k]) == want[k] for k in keys) \
                and len(keys) >= 2:
            return "movable", f"逐键同形（{len(keys)} 键）"
        # 维度对不对
        ip = shapes.get("in_proj_weight")
        if ip is not None and len(ip) == 2 and ip[1] == TARGET_D:
            return "projectable", f"结构同族但 embed_dim≠{TARGET_D}（{ip[1]}）"
        return "incompatible", "键集/结构不同族"
    if role == "ffn":
        pairs = [(k, v) for k, v in shapes.items() if len(v) == 2]
        # 找一对互相转置的 (h,d) 与 (d,h)
        for i, (ka, a) in enumerate(pairs):
            for kb, b in pairs[i + 1:]:
                if a[0] == b[1] and b[0] == a[1] and a[0] != a[1]:
                    if a[1] == TARGET_D and b[0] == TARGET_D:
                        return "movable", f"逐键同形（{ka}{a} / {kb}{b}）"
                    return "projectable", (f"同角色但维度不同：{ka}{a} / {kb}{b}"
                                          f"（目标 D={TARGET_D}）")
        return "incompatible", "不是标准两层 FFN"
    return "incompatible", ""


def main():
    models = load_models()
    print(f"目标宽度 D={TARGET_D}（自研 tokenizer 的 token 维）\n")
    rows = []
    for name, rec in models.items():
        seen = set()
        for L in rec.get("layer_detail", []):
            for c in L.get("children", []):
                role, shapes = classify(c)
                if role is None:
                    continue
                key = (role, c["name"], json.dumps(shapes, sort_keys=True))
                if key in seen:
                    continue
                seen.add(key)
                verdict, why = movable_to_target(role, shapes)
                rows.append({"model": name, "layer": L["name"], "child": c["name"],
                             "role": role, "verdict": verdict, "why": why,
                             "shapes": {k: v for k, v in list(shapes.items())[:4]}})
    order = {"movable": 0, "projectable": 1, "incompatible": 2}
    rows.sort(key=lambda r: (order[r["verdict"]], r["model"], r["layer"]))

    print(f"{'判定':<13}{'模型':<16}{'组件':<16}{'层':<34}说明")
    print("-" * 132)
    for r in rows:
        if r["verdict"] == "incompatible":
            continue
        print(f"{r['verdict']:<13}{r['model']:<16}{r['child']:<16}"
              f"{r['layer'][:33]:<34}{r['why']}")

    n_mov = sum(1 for r in rows if r["verdict"] == "movable")
    n_prj = sum(1 for r in rows if r["verdict"] == "projectable")
    print(f"\n合计：movable {n_mov} 个组件 / projectable {n_prj} 个 / "
          f"incompatible {len(rows)-n_mov-n_prj} 个")
    out = HERE / "movable_components.json"
    out.write_text(json.dumps({"target_d": TARGET_D, "rows": rows},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"明细写入 {out}")


if __name__ == "__main__":
    main()
