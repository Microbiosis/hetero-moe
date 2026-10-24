"""check_findings_consistency.py — 把 FINDINGS 里的表与**解析出的数据**自动比对。

## 为什么需要

今天这轮在 FINDINGS 里同时存在过两组数字（第一轮 vs 第二轮），
还存过两条**已被证伪**的解读 ⇒ 手工维护必然漂移。
⇒ 这条脚本把「文档里的关键数字」与「结果 JSON」绑在一起：
**改数据不改文档、或改文档不改数据，都会被立刻指出来。**

## 覆盖的表

1. 阶段 D 最终表的「Δr vs 宿主」行（§二十二·三）
2. S1a 的读出中位数（`results/s1/`）
3. **组件级组装的变体表**（§二十三·三）
4. **「学出来的码本 vs 随机码本」决定性对照表**（§二十三·四）
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FINDINGS = ROOT / "FINDINGS_TIER0_RSA.md"
VARIANTS = HERE / "variant-results" / "variant_parsed_from_logs.json"
S1 = HERE / "s1"
ASSEMBLED = HERE / "assembled-results"

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

# --- 组装变体表（§二十三·三）---
# 表头 → 读出名（不同表里同一个读出的列名不一样，所以两张表各有一组）
ASM_COLS = {"重建层": "recon", "码本向量": "emb_mean", "离散 one-hot": "onehot_mean",
            "码号": "code_ids", "连续(不量化)": "continuous",
            "随机码本": "random_codebook"}
# 决定性对照表（§二十三·四）的表头
DEC_COLS = {"学出来的码本": "emb_mean", "随机码本": "random_codebook"}
# 文档里的臂名 → 结果 JSON 文件名（S1a 单独取）
ASM_FILES = {
    "S1a": "s1/k1024.json",
    "v1-none": "assembled-results/v1-ffn-none.json",
    "v1-mirepnet": "assembled-results/v1-ffn-mirepnet.json",
    "v1-biot": "assembled-results/v1-ffn-biot.json",
    "v1-eegdino-small": "assembled-results/v1-ffn-eegdino-small.json",
    "v2-none": "assembled-results/v2-ffn-none.json",
    "v2-mirepnet": "assembled-results/v2-ffn-mirepnet.json",
    "v2-biot": "assembled-results/v2-ffn-biot.json",
    "v2-eegdino-small": "assembled-results/v2-ffn-eegdino-small.json",
    "v2-none-nosem": "assembled-results/v2-ffn-none-nosem.json",
    "v2-plain": "assembled-results/v2-ffn-plain.json",
    "v2-plain-nosem": "assembled-results/v2-ffn-plain-nosem.json",
}

# --- 决定性对照表（§二十三·四）：臂 → (学出来的码本, 随机码本, Δr, 正向) ---
DECISIVE = {
    "v2-plain": (0.0221, 0.0149, 0.0068, 8),
    "v2-none": (0.0294, 0.0495, -0.0180, 1),
    "v2-mirepnet": (0.0270, 0.0267, -0.0034, 4),
    "v2-biot": (0.0348, 0.0369, -0.0004, 5),
    "v2-eegdino-small": (0.0233, 0.0418, -0.0171, 0),
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


def parse_num_cell(cell):
    """`+0.0372 (10)` / `**+0.0372 (10)**` → (0.0372, 10)；解析不出来返回 None。"""
    s = cell.replace("*", "").strip()
    m = re.search(r"([+−-]?\d+\.\d+)", s)
    n = re.search(r"(\d+)\s*/\s*(\d+)", s)
    if not m:
        return None
    val = float(m.group(1).replace("−", "-"))
    npos = int(n.group(1)) if n else None
    return val, npos


def parse_readout_table(section, header_key, cols_map=None, extra_cols=()):
    """找**第一张**表头含 header_key 的表 → {臂名: {列名: (值, 显著数)}}。

    ⚠️ **只取第一张**：同一个小节里可能有多张表都含这个表头
    （例如「组装变体表」与「两个变量的分解」表都含「重建层」，
    且**都有 `v2-none` 这一行**）⇒ 若把多张表混着解析，后面的行会覆盖前面的
    ⇒ 守卫就会拿「分解表里的 v2-none」去比「组装表里的 v2-none」，
    **改错了也查不出来**（实测踩到）。
    ⇒ 遇到表尾（非 `|` 行）就**立刻收工**。

    `cols_map`：表头 → 读出/列名（默认用 ASM_COLS）
    `extra_cols`：额外要抽的原始列（列名 → 键），例如 Δr / 正向
    """
    cols_map = cols_map or ASM_COLS
    out = {}
    header = None
    for line in section.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            if header is not None:        # 表尾 ⇒ 这张表结束了
                break
            header = None
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        if header is None:
            if header_key not in "".join(cells):
                continue
            header = cells
            continue
        # 表头 → 列号
        cols = {}
        for i, c in enumerate(header):
            key = c.replace("*", "").strip()
            for hk, rk in cols_map.items():
                if key.startswith(hk) and rk not in cols:
                    cols[rk] = i
                    break
        for hk, rk in extra_cols:
            for i, c in enumerate(header):
                if c.replace("*", "").strip().startswith(hk) and rk not in cols:
                    cols[rk] = i
                    break
        if not cols:
            break
        name = re.split(r"[（(]", cells[0])[0].replace("*", "").strip()
        if not name:
            continue
        row = {}
        for rk, i in cols.items():
            if i >= len(cells):
                continue
            v = parse_num_cell(cells[i])
            if v:
                row[rk] = v
        if row:
            out[name] = row
    return out


def paired_delta(d, readout_a, readout_b):
    """逐被试配对差（a − b）→ (中位, 正向数, n, 单侧二项 p)。"""
    from math import comb

    def per_subj(readout):
        out = {}
        for rec in d.get("per_subject", []):
            rows = rec.get("rows") or {}
            if readout in rows and "r_img" in rows[readout]:
                out[rec.get("subject", "?")] = float(rows[readout]["r_img"])
        return out

    a, b = per_subj(readout_a), per_subj(readout_b)
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    diffs = [a[k] - b[k] for k in keys]
    npos = sum(1 for x in diffs if x > 0)
    srt = sorted(diffs)
    n = len(srt)
    med = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2
    p = sum(comb(n, k) for k in range(npos, n + 1)) / (2 ** n)
    return med, npos, n, p


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

    # --- 3. 组件级组装的变体表（§二十三·三）---
    sec = find_section(text, "二十三·三、主结果")
    if not sec:
        problems.append("FINDINGS 里找不到小节「二十三·三、主结果」")
    else:
        doc = parse_readout_table(sec, "重建层")
        for arm, fn in ASM_FILES.items():
            fp = ROOT / "results" / fn
            if not fp.exists():
                problems.append(f"[组装] 缺结果文件 {fn}")
                continue
            d = json.loads(fp.read_text(encoding="utf-8"))
            summ = d.get("summary", {})
            for readout, (dval, dnpos) in doc.get(arm, {}).items():
                s = summ.get(readout)
                if s is None:
                    problems.append(f"[组装] {arm} 的 {readout} 不在结果里")
                    continue
                if abs(s["r_median"] - dval) > 5e-4:
                    problems.append(f"[组装] {arm}/{readout}：文档 {dval}，"
                                    f"结果 {s['r_median']}")
                if dnpos is not None and s["n_sig"] != dnpos:
                    problems.append(f"[组装] {arm}/{readout}：文档显著 {dnpos}，"
                                    f"结果 {s['n_sig']}")
        if len(doc) < len(ASM_FILES):
            problems.append(f"[组装] 变体表只解析出 {len(doc)} 行，"
                            f"期望 {len(ASM_FILES)} 行")

    # --- 4. 决定性对照表（§二十三·四 与 §二十四·五 各有一份 ⇒ 两份都要查）---
    for sec_key in ("二十三·四、决定性对照", "二十四·五、2×2 补齐"):
        sec4 = find_section(text, sec_key)
        if not sec4:
            problems.append(f"FINDINGS 里找不到小节「{sec_key}」")
            continue
        doc4 = parse_readout_table(sec4, "学出来的码本",
                                    cols_map=DEC_COLS,
                                    extra_cols=(("Δr", "delta"), ("正向", "npos")))
        for arm, (emb, rnd, d_dr, npos) in DECISIVE.items():
            row = doc4.get(arm)
            if row is None:
                problems.append(f"[决定性/{sec_key[:6]}] 文档里找不到 {arm} 行")
                continue
            for key, exp in (("emb_mean", emb), ("random_codebook", rnd)):
                if key in row and abs(row[key][0] - exp) > 5e-4:
                    problems.append(f"[决定性/{sec_key[:6]}] {arm}/{key}："
                                    f"文档 {row[key][0]}，期望 {exp}")
            if "delta" in row and abs(row["delta"][0] - d_dr) > 5e-4:
                problems.append(f"[决定性/{sec_key[:6]}] {arm}：文档 Δr="
                                f"{row['delta'][0]}，期望 {d_dr}")
            if "npos" in row and row["npos"][0] != npos:
                problems.append(f"[决定性/{sec_key[:6]}] {arm}：文档正向 "
                                f"{row['npos'][0]}，期望 {npos}")
            # 数据侧：自己算一遍配对差
            fp = ASSEMBLED / f"{arm}.json"
            if fp.exists():
                d = json.loads(fp.read_text(encoding="utf-8"))
                st = paired_delta(d, "emb_mean", "random_codebook")
                if st:
                    med, np_, n, p = st
                    if abs(med - d_dr) > 5e-4:
                        problems.append(f"[决定性/{sec_key[:6]}] {arm}：数据算得 "
                                        f"Δr={med:.4f}，文档写 {d_dr}")
                    if np_ != npos:
                        problems.append(f"[决定性/{sec_key[:6]}] {arm}：数据算得正向 "
                                        f"{np_}/{n}，文档写 {npos}")

    if problems:
        print("✗ 文档与数据不一致：")
        for p in problems:
            print("   -", p)
        return 1
    print("✓ 文档与数据一致（变体表 Δr vs 宿主 / S1a 读出中位数 / "
          "组装变体表 / 决定性对照表）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
