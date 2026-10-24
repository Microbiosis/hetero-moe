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
5. **整合台账**（`INTEGRATION.md`）的 A 类数字不得与数据漂移
6. **exp89 位置轴**（§二十五·二 的通道组消融表 + 台账 A14/A15）
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
POSAX = HERE / "posaxis-results" / "exp89_posaxis.json"

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


def find_section_h2(text, key):
    """截出含 key 的 **H2 级**章节（到下一个 `## ` 为止）。

    ⚠️ `find_section` 在 H2 标题下会**立刻**被下一个 `### ` 截断
    ⇒ 用来查 H2 章节会取到空串（实测：左右轴那一节的所有数字都查不到）。
    """
    i = text.find(key)
    if i < 0:
        return ""
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


def paired_two_docs(doc_a, doc_b, readout="onehot_mean"):
    """两个 checkpoint 之间的逐被试配对差（a − b）→ (中位, 正向数, n, p)。"""
    from math import comb

    def per_subj(d):
        out = {}
        for rec in d.get("per_subject", []):
            rows = rec.get("rows") or {}
            if readout in rows and "r_img" in rows[readout]:
                out[rec.get("subject", "?")] = float(rows[readout]["r_img"])
        return out

    a, b = per_subj(doc_a), per_subj(doc_b)
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


def parse_posaxis_table(section):
    """exp89 的通道组消融表 → {组: (r_median, n_sig)}。

    表行长这样：`| **POST 后部** | 34 | **+0.0628** | **10/10** | +0.0402 ... |`
    ⇒ 组名取第一格的**首个词**（`POST` / `ANT` / `OCC` …）。
    """
    out = {}
    for line in section.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 4:
            continue
        key = re.split(r"[\s（(]", cells[0].replace("*", "").strip())[0]
        if key not in ("ALL", "POST", "ANT", "OCC", "PARI", "TEMP", "CENT",
                       "FRONT"):
            continue
        r = parse_num_cell(cells[2])
        n = parse_num_cell(cells[3])
        if r:
            out[key] = (r[0], n[1] if n else None)
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

    # --- 5. 整合台账（INTEGRATION.md）不得与数据漂移 ---
    integ = ROOT / "INTEGRATION.md"
    if integ.exists():
        # 文档里用的是排版减号「−」⇒ 比较前归一化
        itext = integ.read_text(encoding="utf-8").replace("−", "-")
        must = []

        def _sum(path, readout="onehot_mean"):
            fp = ASSEMBLED / f"{path}.json"
            if not fp.exists():
                return None
            e = json.loads(fp.read_text(encoding="utf-8"))["summary"].get(readout)
            return e

        # A3：v2-none 的离散层中位数
        e = _sum("v2-none")
        if e:
            must.append((f"{e['r_median']:+.4f}", "A3 v2-none 离散 one-hot 中位"))
        # A4：语义监督的配对差（两种配方）
        for a, b, label in (("v2-plain", "v2-plain-nosem", "A4 语义监督(不嫁接注意力)"),
                            ("v2-none", "v2-none-nosem", "A4 语义监督(嫁接注意力)")):
            fa, fb = ASSEMBLED / f"{a}.json", ASSEMBLED / f"{b}.json"
            if fa.exists() and fb.exists():
                st = paired_two_docs(json.loads(fa.read_text(encoding="utf-8")),
                                     json.loads(fb.read_text(encoding="utf-8")))
                if st:
                    must.append((f"{st[0]:+.4f}", label))
        # A5：组装（无语义监督）vs S1a 自研骨架
        fa = ASSEMBLED / "v2-plain-nosem.json"
        s1 = S1 / "k1024.json"
        if fa.exists() and s1.exists():
            st = paired_two_docs(json.loads(fa.read_text(encoding="utf-8")),
                                 json.loads(s1.read_text(encoding="utf-8")))
            if st:
                must.append((f"{st[0]:+.4f}", "A5 组装 vs S1a"))
        # A6：嫁接注意力的净效应（v2-none − v2-plain）
        fa, fb = ASSEMBLED / "v2-none.json", ASSEMBLED / "v2-plain.json"
        if fa.exists() and fb.exists():
            st = paired_two_docs(json.loads(fa.read_text(encoding="utf-8")),
                                 json.loads(fb.read_text(encoding="utf-8")))
            if st:
                must.append((f"{st[0]:+.4f}", "A6 嫁接注意力 净效应"))
        # A7：决定性对照的 Δr（学出来的码本 − 随机码本）
        for arm, (emb, rnd, d_dr, npos) in DECISIVE.items():
            if arm in ("v2-none", "v2-plain"):
                must.append((f"{d_dr:+.4f}", f"A7 {arm} Δr(学出来的码本-随机码本)"))
        # A1：S1a 的重建层
        if s1.exists():
            e1 = json.loads(s1.read_text(encoding="utf-8"))["summary"].get("recon")
            if e1:
                must.append((f"{e1['r_median']:+.4f}", "A1 S1a 重建层"))
        for s, what in must:
            if s not in itext:
                problems.append(f"[台账] INTEGRATION.md 缺 {what} 的数字 {s}")
    else:
        problems.append("找不到 INTEGRATION.md（整合台账）")

    # --- 6. exp89 位置轴（§二十五·二 的表 + 台账 A14/A15）---
    if POSAX.exists():
        pd = json.loads(POSAX.read_text(encoding="utf-8"))
        psumm = pd.get("summary", {})
        sec = find_section(text, "二十五·二、(a) 通道组消融")
        if not sec:
            problems.append("FINDINGS 里找不到小节「二十五·二、(a) 通道组消融」")
        else:
            pdoc = parse_posaxis_table(sec)
            for g, (r_med, n_sig) in pdoc.items():
                e = psumm.get(g)
                if not e:
                    problems.append(f"[位置轴] 结果里没有组 {g}")
                    continue
                if abs(e["r_median"] - r_med) > 5e-4:
                    problems.append(f"[位置轴] {g}：文档 {r_med}，"
                                    f"结果 {e['r_median']}")
                if n_sig is not None and e["n_sig"] != n_sig:
                    problems.append(f"[位置轴] {g}：文档显著 {n_sig}，"
                                    f"结果 {e['n_sig']}")
            # 配对差（后部 − 前部）那一行：数字与 p 值都要在
            pma = psumm.get("POST_minus_ANT")
            if pma:
                ln = ""
                for line in sec.splitlines():
                    if "后部 - 前部（配对）" in line.replace("−", "-"):
                        ln = line.replace("−", "-")
                        break
                if not ln:
                    problems.append("[位置轴] 文档里找不到「后部 − 前部（配对）」那一行")
                else:
                    for s, what in ((f"{pma['median']:+.4f}", "后−前差中位"),
                                    (f"{pma['p_sign']:.4f}", "后−前差 p")):
                        if s not in ln:
                            problems.append(f"[位置轴] 「后部 − 前部（配对）」行缺 {what} 的数字 {s}")
            for g in ("ALL", "POST", "ANT", "OCC"):
                if g not in pdoc:
                    problems.append(f"[位置轴] 文档表里没有 {g} 行")
            # 配对差（后部 − 前部）
            pma = psumm.get("POST_minus_ANT")
            if pma:
                s = f"{pma['median']:+.4f}"
                if s not in sec.replace("−", "-"):
                    problems.append(f"[位置轴] 文档里找不到后−前差 {s}")
            # 台账 A14/A15
            integ_text = (ROOT / "INTEGRATION.md").read_text(
                encoding="utf-8").replace("−", "-") if (
                ROOT / "INTEGRATION.md").exists() else ""
            pos_s = psumm.get("position_summary", {})
            must89 = []
            if "POST" in psumm:
                must89.append((f"{psumm['POST']['r_median']:+.4f}", "A14 后部"))
            if "ANT" in psumm:
                must89.append((f"{psumm['ANT']['r_median']:+.4f}", "A14 前部"))
            if pma:
                must89.append((f"{pma['median']:+.4f}", "A14 后−前配对差"))
            for key, label in (("amp_r", "A15 跨 split 幅度图 r"),
                               ("amp_vs_anat_spearman", "A15 幅度~解剖位 sp")):
                if key in pos_s:
                    must89.append((f"{pos_s[key]['median']:+.4f}", label))
            for s, what in must89:
                if s not in integ_text:
                    problems.append(f"[台账] INTEGRATION.md 缺 {what} 的数字 {s}")
    else:
        print(f"（缺 {POSAX}，跳过 exp89 位置轴比对）")

    # --- 7. 左右轴（exp90 系列）：文档数字不得与结果漂移 ---
    latf = HERE / "lateral-results" / "lateral_sitepairs.json"
    ampf = HERE / "lateral-results" / "lateral_ampcheck.json"
    if latf.exists() and ampf.exists():
        ld = json.loads(latf.read_text(encoding="utf-8"))
        ad = json.loads(ampf.read_text(encoding="utf-8"))
        aggs = ld.get("per_subject_aggregate", {})
        agg_all = aggs.get("all_pairs", {})
        agg_post = aggs.get("posterior_O_PO_P_T", {})

        def _doc(name):
            p = ROOT / name
            if not p.exists():
                return None
            return p.read_text(encoding="utf-8").replace("−", "-")

        ftext = _doc("FINDINGS_TIER0_RSA.md")
        itext2 = _doc("INTEGRATION.md")
        # ⚠️ **必须限定到具体小节 / 具体行**：
        #    同一个数字在别处也会出现（`0.0020` 既是左右轴的 p，也是 A14 后−前差的 p）
        #    ⇒ 全文件搜索会让检查**永远通过**（反向测试抓到过这个坑）
        fsec = find_section_h2(ftext, "## 二十六、左右轴") if ftext else ""
        a16 = ""
        if itext2:
            for line in itext2.splitlines():
                if "**A16**" in line:
                    a16 = line
                    break
        if fsec:
            lines = fsec.splitlines()

            def _line(marker):
                # ⚠️ 文档里的排版减号「−」已被归一成 ASCII「-」⇒ **标记也要一起归一**
                #    （否则标记永远匹配不上，守卫会误报「找不到行」）
                m = marker.replace("−", "-")
                for ln in lines:
                    if m in ln.replace("−", "-"):
                        return ln.replace("−", "-")
                return ""

            # ⚠️ **按行定位**（不能整节搜）：同一个数可能在别处重复出现
            #    （`0.0020` 既是表里的 p，也是 §二十六·四 正文里的 p）
            #    ⇒ 整节搜 ⇒ 篡改抓不到（实测）
            for marker, pairs in (
                    ("逐被试聚合",
                     ((f"{agg_all.get('p_sign', 0):.4f}", "逐被试聚合 p"),
                      (f"{agg_all.get('median_of_subject_means', 0):+.4f}",
                       "逐被试均值中位"))),
                    ("后部位点族",
                     ((f"{agg_post.get('median_of_subject_means', 0):+.4f}",
                       "后部族中位"),)),
                    ("**后部**（O/PO/P/T",
                     ((f"{ad['posterior']['median_diff']:+.3f}", "后部幅度差中位"),)),
                    ("**前部**（F/FC/AF/FT/C/CP",
                     ((f"{ad['anterior']['median_diff']:+.3f}", "前部幅度差中位"),))):
                ln = _line(marker)
                if not ln:
                    problems.append(f"[左右轴] FINDINGS §二十六 找不到行「{marker}」")
                    continue
                for s, what in pairs:
                    if s not in ln:
                        problems.append(
                            f"[左右轴] FINDINGS §二十六「{marker}」行缺 {what} 的数字 {s}")
            # --- exp90-lateral 的组级/迁移数（§二十六·三）---
            lf = HERE / "lateral-results" / "exp90_lateral.json"
            if lf.exists():
                lj = json.loads(lf.read_text(encoding="utf-8"))
                vs = (lj.get("v2") or {}).get("summary", {})
                lr = vs.get("LEFT_minus_RIGHT", {})
                for marker, pairs in (
                        ("语义对齐 RSA（中位，10/10 显著）",
                         ((f"{vs.get('LEFT_RSA', {}).get('median', 0):+.4f}", "LEFT RSA"),
                          (f"{vs.get('RIGHT_RSA', {}).get('median', 0):+.4f}", "RIGHT RSA"),
                          (f"{vs.get('ALL_RSA', {}).get('median', 0):+.4f}", "ALL RSA"))),
                        ("逐被试配对差（左 − 右）",
                         ((f"{lr.get('median', 0):+.4f}", "左右配对差中位"),
                          (f"{lr.get('p_sign', 0):.4f}", "左右配对差 p"))),
                        ("左右两组的「图像相似度结构」相关",
                         ((f"{vs.get('RDMRED_L_R', {}).get('median', 0):+.4f}",
                           "左右结构冗余"),)),
                        ("同侧分半可靠度上限",
                         ((f"{vs.get('RDMRED_CEILING', {}).get('median', 0):+.4f}",
                           "同侧分半上限"),)),
                        ("跨侧 左→右（位点镜像对齐）",
                         ((f"{vs.get('MB_XL2R', {}).get('median', 0):.4f}",
                           "跨侧 transfer acc1"),
                          (f"{vs.get('MB_XM2R', {}).get('median', 0):.4f}",
                           "错位对照 acc1"))),
                        ("对齐/错位 = ",
                         ((f"{vs.get('TRANSFER_XL2R_over_XM2R', {}).get('median', 0):.3f}",
                           "对齐/错位比"),))):
                    ln = _line(marker)
                    if not ln:
                        problems.append(f"[左右轴] FINDINGS §二十六 找不到行「{marker}」")
                        continue
                    for s, what in pairs:
                        if s not in ln:
                            problems.append(
                                f"[左右轴] FINDINGS §二十六「{marker}」行缺 {what} 的数字 {s}")
            else:
                problems.append("缺 lateral-results/exp90_lateral.json")
        elif ftext:
            problems.append("FINDINGS 里找不到「二十六、左右轴」小节")
        if a16:
            for s, what in ((f"{agg_all.get('p_sign', 0):.4f}", "左右轴 p"),
                            (f"{ad['posterior']['n_pos']}/{ad['posterior']['n_pos']+ad['posterior']['n_neg']}",
                             "后部幅度 右>左 被试数"),
                            (f"{ad['anterior']['n_pos']}/{ad['anterior']['n_pos']+ad['anterior']['n_neg']}",
                             "前部幅度 右>左 被试数")):
                if s not in a16:
                    problems.append(f"[左右轴] INTEGRATION A16 行缺 {what} 的数字 {s}")
        elif itext2:
            problems.append("INTEGRATION.md 里找不到 A16 行")
    else:
        print(f"（缺 {latf} 或 {ampf}，跳过左右轴比对）")

    if problems:
        print("✗ 文档与数据不一致：")
        for p in problems:
            print("   -", p)
        return 1
    print("✓ 文档与数据一致（变体表 Δr vs 宿主 / S1a 读出中位数 / "
          "组装变体表 / 决定性对照表 / exp89 位置轴 / 左右轴）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
