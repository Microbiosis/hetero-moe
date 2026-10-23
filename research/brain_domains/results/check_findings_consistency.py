"""check_findings_consistency.py — 把 FINDINGS 里的表与**解析出的数据**自动比对。

## 为什么需要

今天这轮在 FINDINGS 里同时存在过两组数字（第一轮 vs 第二轮），
还存过两条**已被证伪**的解读 ⇒ 手工维护必然漂移。
⇒ 这条脚本把「文档里的关键数字」与「结果 JSON」绑在一起：
**改数据不改文档、或改文档不改数据，都会被立刻指出来。"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FINDINGS = ROOT / "FINDINGS_TIER0_RSA.md"
VARIANTS = HERE / "variant-results" / "variant_parsed_from_logs.json"
S1 = HERE / "s1"

# FINDINGS 里的关键断言：(小节标题关键字, 变体行名, 期望的 Δr vs 宿主 中位, 期望正向数)
CLAIMS = [
    ("二十二·三、第二轮结果", "v1_sjepa_l0", 0.0178, 10),
    ("二十二·三、第二轮结果", "v4_s1a_recon", 0.0177, 10),
    ("二十二·三、第二轮结果", "v3_braintok_code", 0.0057, 7),
    ("二十二·三、第二轮结果", "v2_mirepnet_ffn", -0.0074, 2),
]

# S1a 的关键断言：{文件: {读出: 期望中位 A r}}
S1_CLAIMS = {
    "k1024": {"recon": 0.0372, "emb_mean": 0.0045, "onehot_mean": 0.0014,
              "continuous": 0.0032, "random_codebook": 0.0021},
    "k256": {"recon": 0.0276, "emb_mean": 0.0053, "onehot_mean": 0.0028,
             "continuous": 0.0019, "random_codebook": 0.0060},
}


def find_section(text, key):
    """截出含 key 的小节文本。"""
    i = text.find(key)
    if i < 0:
        return ""
    j = text.find("\n### ", i + 1)
    if j < 0:
        j = text.find("\n## ", i + 1)
    return text[i: j if j > 0 else len(text)]


def parse_delta_table(section):
    """从「供体贡献」表里解析 {行名: (Δr vs 宿主 中位, 正向数)}。

    ⚠️ **按表头映射列**，不写死列号 —— 表头顺序会变（今天就加过列）。
    """
    out = {}
    header = None
    for line in section.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            header = None
            continue
        cells = [c.strip().strip("*") for c in s.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):      # 分隔行
            continue
        if header is None:
            header = cells
            continue
        # 定位列：Δr vs 宿主 与它后面的「正向」列
        try:
            i_delta = next(i for i, c in enumerate(header) if c.startswith("Δr vs 宿主"))
            i_npos = next(i for i, c in enumerate(header)
                         if c.startswith("正向") and i > i_delta)
        except StopIteration:
            header = None
            continue
        name = cells[0]
        # 行名可能带括号注解（如「**v4_s1a_recon**（自训重建层）」）
        # ⇒ 先切括号、再去星号（顺序反了会留下 "v4_s1a_recon**"）
        name = re.split(r"[（(]", name)[0].strip().strip("*").strip()
        if not name.startswith("v"):
            continue
        if len(cells) <= max(i_delta, i_npos):
            continue
        m = re.search(r"([+−-]?\d+\.\d+)", cells[i_delta])
        npos = re.search(r"(\d+)/(\d+)", cells[i_npos])
        if m and npos:
            out[name] = (float(m.group(1).replace("−", "-")), int(npos.group(1)))
    return out


def main():
    problems = []
    if not FINDINGS.exists():
        print(f"✗ 找不到 {FINDINGS}")
        return 1
    text = FINDINGS.read_text(encoding="utf-8")

    # --- 1. 变体表（阶段 D 最终表）---
    if VARIANTS.exists():
        data = json.loads(VARIANTS.read_text(encoding="utf-8"))
        deltas = data.get("deltas", [])
        got = {d["variant"]: d for d in deltas}
        sec = find_section(text, "二十二·三、第二轮结果")
        doc = parse_delta_table(sec)
        for sec_key, vid, exp_delta, exp_npos in CLAIMS:
            if not sec:
                problems.append(f"FINDINGS 里找不到小节「{sec_key}」")
                continue
            if vid not in doc:
                problems.append(f"[表] 文档里找不到 {vid} 的 Δr vs 宿主 行")
                continue
            d_doc, n_doc = doc[vid]
            if abs(d_doc - exp_delta) > 5e-4:
                problems.append(f"[表] {vid}：文档 Δr={d_doc}，FINDINGS 断言 {exp_delta}")
            if n_doc != exp_npos:
                problems.append(f"[表] {vid}：文档正向 {n_doc}，FINDINGS 断言 {exp_npos}")
            d = got.get(vid)
            if d is None:
                problems.append(f"[数据] {vid} 不在解析结果里")
                continue
            if abs(d["d_r_vs_host_median"] - d_doc) > 5e-4:
                problems.append(f"[数据] {vid}：解析得 Δr={d['d_r_vs_host_median']}，"
                                f"文档写 {d_doc}")
            if d["d_r_vs_host_n_pos"] != n_doc:
                problems.append(f"[数据] {vid}：解析得正向 {d['d_r_vs_host_n_pos']}，"
                                f"文档写 {n_doc}")
    else:
        print(f"（缺 {VARIANTS}，跳过变体表比对）")

    # --- 2. S1a 的读出中位数 ---
    for k, claims in S1_CLAIMS.items():
        fp = S1 / f"{k}.json"
        if not fp.exists():
            continue
        d = json.loads(fp.read_text(encoding="utf-8"))
        for name, s in d.get("summary", {}).items():
            if name not in claims:
                continue
            exp = claims[name]
            if abs(s["r_median"] - exp) > 5e-4:
                problems.append(f"[S1-{k}] {name}：解析得 {s['r_median']}，"
                                f"FINDINGS 断言 {exp}")

    if problems:
        print("✗ 文档与数据不一致：")
        for p in problems:
            print("   -", p)
        return 1
    print("✓ 文档与数据一致（变体表 Δr vs 宿主 / S1a 读出中位数）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
