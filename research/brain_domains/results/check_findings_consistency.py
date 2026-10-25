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


def sign_test_p(diffs):
    """双侧符号检验（精确二项）。返回 (n_pos, n_neg, p)。"""
    from math import comb
    d = [float(x) for x in diffs if x == x]
    n = len(d)
    if n == 0:
        return 0, 0, float("nan")
    k = sum(1 for x in d if x > 0)
    m = n - k
    p = 2.0 * sum(comb(n, i) for i in range(min(k, m) + 1)) / (2 ** n)
    return k, m, float(min(1.0, p))


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
            # ⚠️ A16 的归因数字现在来自**支线 B（10 个被试）**，不是 5 个被试的 ampcheck
            #    （那 5 个被试的读法已被支线 B 推翻 ⇒ 见 FINDINGS §二十七·二）
            bfile = HERE / "cross-validation" / "lateral_baseline.json"
            bsum = None
            if bfile.exists():
                bj = json.loads(bfile.read_text(encoding="utf-8"))
                # 支线 B 的 JSON：per_subject[].vals
                vs = [(r.get("vals") or {}) for r in bj.get("per_subject", [])]
                if vs:
                    def _stat(key):
                        xs = [v.get(key) for v in vs if v.get(key) is not None]
                        if not xs:
                            return None
                        kk = sum(1 for x in xs if x > 0)
                        srt = sorted(xs)
                        return {"n_pos": kk, "n": len(xs),
                                "median": srt[len(srt) // 2],
                                "p_sign": sign_test_p(xs)[2]}
                    bsum = {k: _stat(k) for k in
                            ("post_RL", "ant_RL", "post_resid_ant_base")}
            if bsum:
                for key, label in (("post_RL", "后部幅度 右>左 被试数"),
                                   ("ant_RL", "前部幅度 右>左 被试数")):
                    e = bsum.get(key)
                    if e:
                        s = f"{e['n_pos']}/{e['n']}"
                        if s not in a16:
                            problems.append(
                                f"[左右轴] INTEGRATION A16 行缺 {label} 的数字 {s}")
                r = bsum.get("post_resid_ant_base")
                if r:
                    for s, what in ((f"{r['median']:+.4f}", "后部残余中位"),
                                    (f"{r['p_sign']:.4f}", "后部残余 p")):
                        if s not in a16:
                            problems.append(
                                f"[左右轴] INTEGRATION A16 行缺 {what} 的数字 {s}")
            else:
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

    # --- 8. exp91a 读出强度检验（§二十七·五 的三行表）---
    rdf = HERE / "readout-results" / "exp91a_readout.json"
    if rdf.exists():
        rd = json.loads(rdf.read_text(encoding="utf-8"))
        rs = rd.get("summary", {})
        sec27 = find_section_h2(text, "## 二十七、外部核对")
        lines27 = sec27.splitlines()
        for ro in ("amp630", "bandpow", "covpat"):
            e = rs.get(ro) or {}
            if not e:
                problems.append(f"[读出检验] 结果里没有 {ro}")
                continue
            ln = next((l for l in lines27
                       if l.startswith(f"| `{ro}` |")), "")
            if not ln:
                problems.append(f"[读出检验] §二十七·五 找不到 {ro} 那一行")
                continue
            ln_n = ln.replace("−", "-")
            for g in ("ALL", "POST", "ANT"):
                if g not in e:
                    continue
                s = f"{e[g]['r_median']:+.4f}"
                if s not in ln_n:
                    problems.append(
                        f"[读出检验] §二十七·五 {ro}/{g} 行缺数字 {s}")
            d = e.get("_POST_minus_ANT")
            if d:
                for s in (f"{d['median']:+.4f}", f"{d['p_sign']:.4f}"):
                    if s not in ln_n:
                        problems.append(
                            f"[读出检验] §二十七·五 {ro} 的后−前缺数字 {s}")
    else:
        print(f"（缺 {rdf}，跳过读出强度检验比对）")

    # --- 9. exp92 三支线（§二十八·二）---
    cvf = HERE / "cross-validation" / "cross_validation.json"
    if cvf.exists():
        cv = json.loads(cvf.read_text(encoding="utf-8"))
        br = cv.get("branches", {})
        A, B, C = br.get("A", {}), br.get("B", {}), br.get("C", {})
        sec28 = find_section_h2(text, "## 二十八、三条支线并行")
        lines28 = sec28.splitlines() if sec28 else []

        def row28(marker):
            for ln in lines28:
                if ln.startswith(marker):
                    return ln.replace("−", "-")
            return ""

        # A 支线：两个模型 × 后部/前部 + 后−前
        for tag, marker in (("s1a", "| S1a（纯自监督）"), ("s1b", "| S1b（带语义监督）")):
            e = A.get(tag) or {}
            ln = row28(marker)
            if not e or not ln:
                problems.append(f"[exp92-A] 找不到 {tag} 的行或结果")
                continue
            for g in ("POST", "ANT"):
                if g in e and f"{e[g]['median']:+.4f}" not in ln:
                    problems.append(f"[exp92-A] §二十八 {tag}/{g} 行缺数字 "
                                    f"{e[g]['median']:+.4f}")
            d = e.get("_POST_minus_ANT")
            if d:
                for s, what in ((f"{d['median']:+.4f}", "后−前中位"),
                                (f"{d['p_sign']:.4f}", "后−前 p"),
                                (f"{d['n_pos']}/{d['n']}", "后−前正向数")):
                    if s not in ln:
                        problems.append(f"[exp92-A] §二十八 {tag} 行缺 {what} 的数字 {s}")
        # B 支线
        for key, marker in (("post_RL", "| 后部 R−L（原始）"),
                            ("ant_RL", "| 前部 R−L（全局增益差的代理）"),
                            ("resid", "| 后部残余（后部 R−L − 前部 R−L）")):
            e = B.get(key) or {}
            ln = row28(marker)
            if not e or not ln:
                problems.append(f"[exp92-B] 找不到 {key} 的行或结果")
                continue
            for s, what in ((f"{e['median']:+.4f}", "中位"),
                            (f"{e.get('n_pos','-')}/{e.get('n','-')}", "正向数"),
                            (f"{e.get('p_sign', 0):.4f}", "p")):
                if s not in ln:
                    problems.append(f"[exp92-B] §二十八 {key} 行缺 {what} 的数字 {s}")
        # C 支线
        for tag, marker in (("raw630", "| **630-d 手工特征**"),
                            ("s1a_recon", "| S1a 重建层（我们自训）"),
                            ("s1b_onehot_mean", "| S1b 离散层（带语义监督）"),
                            ("s1a_onehot_mean", "| S1a 离散层 |")):
            e = C.get(tag) or {}
            ln = row28(marker)
            if not e or not ln:
                problems.append(f"[exp92-C] 找不到 {tag} 的行或结果")
                continue
            s = f"{e['median']:.4f}"
            if s not in ln:
                problems.append(f"[exp92-C] §二十八 {tag} 行缺数字 {s}")
    else:
        print(f"（缺 {cvf}，跳过 exp92 三支线比对）")

    # --- 10. exp93：档位归因 / 对比学习 encoder / 模型层地板 ---
    R93 = HERE / "exp93-results"
    CV = HERE / "cross-validation"
    sec29 = find_section_h2(text, "## 二十九、exp93")
    if not sec29:
        problems.append("FINDINGS 里找不到「## 二十九、exp93」小节")
    else:
        lines29 = sec29.splitlines()

        def row29(prefix):
            for ln in lines29:
                if ln.startswith(prefix):
                    return ln.replace("−", "-")
            return ""

        def med_up(xs):
            xs = [float(x) for x in xs if x == x]
            if not xs:
                return None
            s = sorted(xs)
            return s[len(s) // 2]

        # (a) exp93b 的三条读出 + RSA（绑定**汇总** JSON ⇒ 不受原始文件改名影响）
        e93b = R93 / "exp93b_contrastive.summary.json"
        if e93b.exists():
            bd = json.loads(e93b.read_text(encoding="utf-8"))
            for key, marker, label in (
                    ("zeroshot_acc1", "| zeroshot top-1（**单试次**） |", "exp93b 单试次 top-1"),
                    ("zeroshot_acc1_featavg", "| zeroshot top-1（**试次平均**） |",
                     "exp93b 试次平均 top-1"),
                    ("zeroshot_acc5", "| zeroshot top-5 |", "exp93b top-5"),
                    ("ridge_feat_acc1", "| ridge-on-features top-1（A13 口径） |",
                     "exp93b ridge 读出"),
                    ("rsa", "| **RSA** |", "exp93b RSA")):
                e = bd.get(key) or {}
                if e.get("median_up") is None:
                    continue
                ln = row29(marker)
                if not ln:
                    problems.append(f"[exp93b] §二十九·四 找不到行「{marker}」")
                    continue
                s = (f"{e['median_up']:+.4f}" if key == "rsa"
                     else f"{e['median_up']:.4f}")
                if s not in ln:
                    problems.append(f"[exp93b] §二十九·四「{label}」行缺数字 {s}")
            # 跨被试：完全没进过训练的第 10 个被试
            ps = bd.get("per_subject") or []
            hold = [r for r in ps if r.get("subject") == "sub-10"]
            if hold:
                v = ((hold[0].get("zeroshot") or {}).get("acc1_cos"))
                if v is not None:
                    ln = next((l for l in lines29 if "sub-10" in l), "")
                    if not ln:
                        problems.append("[exp93b] §二十九·四 找不到 sub-10 那一句")
                    else:
                        s = f"{float(v):.4f}"
                        if s not in ln:
                            problems.append(
                                f"[exp93b] §二十九·四 的 sub-10 缺数字 {s}")
        else:
            print(f"（缺 {e93b}，跳过 exp93b 比对）")

        # (b) exp93a 的因子表（**合并后的 10 被试版**：s1-5 来自 Job 日志、s6-10 来自桶内 JSON）
        m93 = R93 / "exp93a_merged.json"
        if m93.exists():
            merged = json.loads(m93.read_text(encoding="utf-8"))
            for marker, cellname, label in (
                    ("| 拟合集 1000 张、单试次 |", "P1/ridge/1000", "exp93a n_train=1000"),
                    ("| 拟合集 4000 张、单试次 |", "P1/ridge/4000", "exp93a n_train=4000"),
                    ("| **拟合集 16540 张、单试次** |", "P1/ridge/16540",
                     "exp93a 全量单试次"),
                    ("| **拟合集 16540 张、试次平均** |", "P1/ridge/16540/试次平均",
                     "exp93a 全量试次平均")):
                ln = row29(marker)
                e = merged.get(cellname)
                if not ln or not e:
                    continue
                if f"{e['median_up']:.4f}" not in ln:
                    problems.append(
                        f"[exp93a] §二十九·二「{label}」行缺数字 {e['median_up']:.4f}")
        else:
            print(f"（缺 {m93}，跳过 exp93a 比对）")

        # (c) 模型层地板：POST/ANT 相对全部随机组的被试数
        fl = [CV / "posaxis_model_floor_s1-5.json",
              CV / "posaxis_model_floor.json"]
        per = {}
        for fp in fl:
            if not fp.exists():
                continue
            d = json.loads(fp.read_text(encoding="utf-8"))
            for r in d.get("per_subject", []):
                if "error" not in r:
                    per[r["subject"]] = r
        if per:
            cnt = {}
            for tag in ("s1a", "s1b"):
                up = dn = 0
                for r in per.values():
                    g = (r.get("groups") or {}).get(tag, {})
                    n = (r.get("random_null") or {}).get(tag, {})
                    gp = (g.get("POST") or {}).get("r")
                    ga = (g.get("ANT") or {}).get("r")
                    rp = (n.get("POST") or {}).get("r_max")
                    ra = (n.get("ANT") or {}).get("r_max")
                    if gp is not None and rp is not None and gp > rp:
                        up += 1
                    if ga is not None and ra is not None and ga < ra:
                        dn += 1
                cnt[tag] = (up, dn, len(per))
            for tag, marker in (("s1a", "| S1a（纯自监督） |"),
                                ("s1b", "| S1b（带语义监督） |")):
                up, dn, n = cnt.get(tag, (0, 0, 0))
                if not n:
                    continue
                ln = row29(marker)
                if not ln:
                    problems.append(f"[地板] §二十九·五 找不到行「{marker}」")
                    continue
                for s, what in ((f"{up}/{n}", "POST 高于全部随机组"),
                                (f"{dn}/{n}", "ANT 低于全部随机组")):
                    if s not in ln:
                        problems.append(
                            f"[地板] §二十九·五 {tag} 行缺{what}的数字 {s}")
        else:
            print(f"（缺地板结果，跳过 exp92-A 地板比对）")

        # (d) exp93c 剂量-反应表（§二十九·五）
        dosf = R93 / "exp93c_dose.summary.json"
        if dosf.exists():
            ds = json.loads(dosf.read_text(encoding="utf-8"))
            for tag, marker, label in (
                    ("feature", "| **特征层** 630-d |", "exp93c 特征层"),
                    ("s1a", "| **模型层 S1a**（纯自监督） |", "exp93c 模型层 S1a"),
                    ("s1b", "| **模型层 S1b**（带语义监督） |", "exp93c 模型层 S1b")):
                row = ds.get(tag) or {}
                if not row:
                    continue
                ln = row29(marker)
                if not ln:
                    problems.append(f"[exp93c] §二十九·五 找不到行「{marker}」")
                    continue
                for k in sorted(row, key=lambda x: int(x)):
                    s = f"{row[k]['median_up']:+.4f}"
                    if s not in ln:
                        problems.append(
                            f"[exp93c] §二十九·五 {label} 行缺 k={k} 的数字 {s}")
        else:
            print(f"（缺 {dosf}，跳过 exp93c 比对）")

        # (e) exp93e 窗口表（§二十九·六）—— 优先用**合并后的 n=10 汇总**，
        #     没有就退回日志版（n=6–7），两条路的键名不同 ⇒ 两种都认。
        wn = None
        sumf = R93 / "exp93e_window.summary.json"
        logf = R93 / "exp93e_window.from_log.json"
        if sumf.exists():
            wn = json.loads(sumf.read_text(encoding="utf-8"))
            wn = wn.get("windows", wn)
        elif logf.exists():
            wn = json.loads(logf.read_text(encoding="utf-8")).get("windows", {})
        if wn:
            for w, v in wn.items():
                if not isinstance(v, dict) or "POST" not in v:
                    continue          # 汇总文件的元数据键（n_ok/n_err 等）不是窗口
                # 文档里窗口名带 `w` 前缀（日志版不带）⇒ 两种都试
                ln = row29(f"| `w{w}`") or row29(f"| `{w}`")
                if not ln:
                    problems.append(f"[exp93e] §二十九·六 找不到窗口行「{w}」")
                    continue
                for s, what in ((f"{v['POST']['median_up']:+.4f}", "POST"),
                                (f"{v['ANT']['median_up']:+.4f}", "ANT"),
                                (f"{v['level']['median_up']:.4f}", "档位"),
                                (f"{v['null_median']:+.4f}", "随机组中位"),
                                (f"{v['POST_above_all']}/{v['n']}", "POST>全部随机组"),
                                (f"{v['ANT_below_all']}/{v['n']}", "ANT<全部随机组")):
                    if s not in ln:
                        problems.append(
                            f"[exp93e] §二十九·六 窗口 {w} 行缺{what}的数字 {s}")
        else:
            print(f"（缺 {sumf} 与 {logf}，跳过 exp93e 比对）")

        # (f) exp93d 模型层读出 @ n_train=16540（§二十九·四 的 A18 检验表）
        d93 = R93 / "exp93d_level.summary.json"
        if d93.exists():
            dd = json.loads(d93.read_text(encoding="utf-8"))
            for r in dd.get("rows", []):
                tag = r["tag"]
                ln = row29(f"| {tag} |")
                if not ln:
                    problems.append(f"[exp93d] §二十九·四 找不到 {tag} 那一行")
                    continue
                if f"{r['median_up']:.4f}" not in ln:
                    problems.append(
                        f"[exp93d] §二十九·四 {tag} 行缺数字 {r['median_up']:.4f}")
        else:
            print(f"（缺 {d93}，跳过 exp93d 比对）")

        # (g) exp93f 读出混淆预检表（§二十九·九）
        f93 = R93 / "exp93f_summary.json"
        if f93.exists():
            fs = json.loads(f93.read_text(encoding="utf-8"))
            # ⚠️ **只搜 §二十九·九 子节**：整节里还有 exp93c 的剂量表（同样含 S1a/recon 字样）
            i99 = text.find("### 二十九·九")
            sub99 = text[i99:] if i99 >= 0 else ""
            j99 = sub99.find("\n### ")
            if j99 > 0:
                sub99 = sub99[:j99]
            lines99 = sub99.splitlines()
            for r in fs.get("rows", []):
                if "slope" not in r:
                    continue
                tag, ro = r["tag"], r["readout"]
                cand = [l for l in lines99
                        if (f"**{tag}**".lower() in l.lower()
                            or f"| {tag} |".lower() in l.lower()) and ro in l]
                if not cand:
                    problems.append(f"[exp93f] §二十九·九 找不到 {tag}/{ro} 那一行")
                    continue
                slope_s = f"{r['slope']:+.4f}"
                mono_s = str(r.get("monotonic", ""))
                if not any(slope_s in l and (not mono_s or mono_s in l) for l in cand):
                    problems.append(
                        f"[exp93f] §二十九·九 {tag}/{ro} 行缺斜率 {slope_s}"
                        + (f" / 单调 {mono_s}" if mono_s else ""))
        else:
            print(f"（缺 {f93}，跳过 exp93f 比对）")

        # --- 11. exp94：骨干对照 × 交叉验证 ---
        e94 = HERE / "exp94-results" / "exp94_summary.json"
        if e94.exists():
            d94 = json.loads(e94.read_text(encoding="utf-8"))
            dec94 = d94.get("decision") or {}
            s94 = d94.get("summaries") or {}
            sec30 = find_section(text, "三十·四、结果")
            if not sec30:
                problems.append("FINDINGS 里找不到「三十·四、结果」小节")
            else:
                rows30 = sec30.splitlines()

                def row94(prefix):
                    for ln in rows30:
                        if ln.startswith(prefix):
                            return ln.replace("−", "-")
                    return ""

                # (a) C7 自一致性表：4 个量的「已发表 / 本实验 V0」
                #     ⚠️ 标记用**文档里的可读行标签**（不是 JSON 键名）⇒ 改文档会报错
                for k, marker in (("zeroshot_acc1", "| zeroshot acc1 |"),
                                  ("zeroshot_acc1_featavg", "| zeroshot acc1（试次平均）|"),
                                  ("rsa", "| RSA |"),
                                  ("zeroshot_acc5", "| zeroshot acc5 |")):
                    got = (s94.get("v0") or {}).get(k)
                    if got is None:
                        continue
                    sc = (dec94.get("self_consistency") or {}).get(k) or {}
                    pub = sc.get("published")
                    row = row94(marker)
                    if not row:
                        problems.append(f"[exp94] §三十·四 找不到自一致性行「{marker}」")
                        continue
                    for val, what in ((pub, "已发表值"), (got, "本实验 V0")):
                        if val is None:
                            continue
                        if f"{float(val):.4f}" not in row:
                            problems.append(
                                f"[exp94] §三十·四 自一致性行 {k} 缺{what} {float(val):.4f}")

                # (b) 主表：C1/C2/C3 的 V0 / V1 / 配对差 / n_pos / p
                pr = dec94.get("paired") or {}
                for key, marker in (("zeroshot_acc1", "| **C1 单试次 top-1** |"),
                                    ("zeroshot_acc1_featavg", "| **C2 试次平均 top-1** |"),
                                    ("rsa", "| **C3 RSA** |")):
                    row = row94(marker)
                    if not row:
                        problems.append(f"[exp94] §三十·四 找不到主表行「{marker}」")
                        continue
                    v0 = (s94.get("v0") or {}).get(key)
                    v1 = (s94.get("v1") or {}).get(key)
                    pd = pr.get(key) or {}
                    for val, what in ((v0, "V0"), (v1, "V1"),
                                      (pd.get("median_diff"), "配对差中位")):
                        if val is None:
                            continue
                        if f"{float(val):.4f}" not in row:
                            problems.append(
                                f"[exp94] §三十·四 主表 {key} 缺{what} {float(val):.4f}")
                    if pd.get("n") and pd.get("n_pos") is not None:
                        tag = f"{pd['n_pos']}/{pd['n']}"
                        if tag not in row:
                            problems.append(f"[exp94] §三十·四 主表 {key} 缺配对计数 {tag}")
                    if pd.get("p_sign") is not None:
                        ps = f"{float(pd['p_sign']):.4f}"
                        if ps not in row:
                            problems.append(
                                f"[exp94] §三十·四 主表 {key} 缺符号检验 p {ps}")

                # (c) 位置轴表：两个骨干的计数与 POST-ANT
                pa = dec94.get("posaxis") or {}
                for v, lab in (("v0", "| V0（混通道）|"), ("v1", "| V1（保留通道身份）|")):
                    row = row94(lab)
                    d = pa.get(v)
                    if not d:
                        continue
                    if not row:
                        problems.append(f"[exp94] §三十·四 找不到位置轴行「{lab}」")
                        continue
                    for cnt, what in ((f"{d['post_above_all_rand34']}/{d['n']}", "POST>随机34"),
                                      (f"{d['ant_below_all_rand29']}/{d['n']}", "ANT<随机29")):
                        if cnt not in row:
                            problems.append(
                                f"[exp94] §三十·四 位置轴 {v} 行缺{what} {cnt}")
                    for val, what in ((d.get("post_minus_ant_median"), "POST-ANT"),
                                      (d.get("post_acc1_median"), "POST acc1"),
                                      (d.get("ant_acc1_median"), "ANT acc1"),
                                      (d.get("post_rsa_median"), "POST RSA")):
                        if val is None:
                            continue
                        if f"{float(val):.4f}" not in row:
                            problems.append(
                                f"[exp94] §三十·四 位置轴 {v} 行缺{what} {float(val):.4f}")

                # (d) INTEGRATION 的 A20 台账行必须带关键数与 R4 限定
                it = (ROOT / "INTEGRATION.md").read_text(encoding="utf-8")
                a20 = next((l for l in it.splitlines() if l.startswith("| **A20**")), "")
                if not a20:
                    problems.append("[台账] INTEGRATION.md 里找不到 A20 那一行")
                else:
                    for val, what in (((s94.get("v1") or {}).get("zeroshot_acc1"), "V1 单试次"),
                                      ((s94.get("v0") or {}).get("zeroshot_acc1"), "V0 单试次"),
                                      ((pr.get("zeroshot_acc1") or {}).get("median_diff"), "配对差")):
                        if val is None:
                            continue
                        if f"{float(val):.4f}" not in a20:
                            problems.append(
                                f"[台账] A20 缺{what} {float(val):.4f}")
                    for need in ("p=0.0020", "R4", "9–10/10",
                                 "CENT 运动 14", "FRONT 执行 15"):
                        if need not in a20:
                            problems.append(f"[台账] A20 缺「{need}」")
        else:
            print(f"（缺 {e94}，跳过 exp94 比对）")

        # --- 12. exp95：合成版 V2（**未结案**：V2 被额度中断）→ 只绑已跑出的部分 ---
        e95 = HERE / "exp95-results" / "exp95_summary.json"
        if e95.exists():
            d95 = json.loads(e95.read_text(encoding="utf-8"))
            s95 = (d95.get("summaries") or {}).get("v1") or {}
            sec31a = find_section(text, "三十一·二、已跑出的部分")
            sec31b = find_section(text, "三十一·三、")
            if not sec31a or not sec31b:
                problems.append("FINDINGS 里找不到「三十一·二 / 三十一·三」小节")
            else:
                rows_a = sec31a.splitlines()

                def row95a(prefix):
                    for ln in rows_a:
                        if ln.startswith(prefix):
                            return ln.replace("−", "-")
                    return ""

                # (a) C7' 合并表：exp93b 已发表 / exp94 的 V1 / exp95 的 V1 / 漂移
                for k, marker in (("zeroshot_acc1", "| zeroshot acc1 |"),
                                  ("zeroshot_acc1_featavg", "| zeroshot acc1（试次平均）|"),
                                  ("rsa", "| RSA |"),
                                  ("zeroshot_acc5", "| zeroshot acc5 |")):
                    got = s95.get(k)
                    if got is None:
                        continue
                    row = row95a(marker)
                    if not row:
                        problems.append(f"[exp95] §三十一·二 找不到自一致性行「{marker}」")
                        continue
                    sc95 = (d95.get("self_consistency_exp93b") or {}).get(k) or {}
                    dr95 = (d95.get("job_drift_vs_exp94") or {}).get(k) or {}
                    for val, what in ((sc95.get("published"), "exp93b 已发表值"),
                                      (dr95.get("exp94"), "exp94 的 V1"),
                                      (got, "exp95 的 V1 复现")):
                        if val is None:
                            continue
                        if f"{float(val):.4f}" not in row:
                            problems.append(
                                f"[exp95] §三十一·二 行 {k} 缺{what} {float(val):.4f}")

                # (b) 位置轴表：exp95 的 V1 复现那一行
                pa95 = d95.get("posaxis_v1") or {}
                row = next((ln for ln in sec31b.splitlines()
                            if ln.startswith("| **exp95 的 V1 复现**")), "")
                if not row:
                    problems.append("[exp95] §三十一·三 找不到「exp95 的 V1 复现」那一行")
                else:
                    n95 = pa95.get("n")
                    for cnt, what in ((f"{pa95.get('post_above_all_rand34')}/{n95}",
                                       "POST>全部随机34"),
                                      (f"{pa95.get('ant_below_all_rand29')}/{n95}",
                                       "ANT<全部随机29")):
                        if cnt not in row:
                            problems.append(f"[exp95] §三十一·三 位置轴行缺{what} {cnt}")
                    for val, what in ((pa95.get("post_minus_ant_median"), "POST−ANT"),
                                      (pa95.get("post_acc1_median"), "POST acc1"),
                                      (pa95.get("ant_acc1_median"), "ANT acc1")):
                        if val is None:
                            continue
                        if f"{float(val):.4f}" not in row:
                            problems.append(
                                f"[exp95] §三十一·三 位置轴行缺{what} {float(val):.4f}")

                # (c) INTEGRATION 的 A14 必须带「9–10/10」这个**降级后的**区间表述
                it95 = (ROOT / "INTEGRATION.md").read_text(encoding="utf-8")
                a14 = next((l for l in it95.splitlines()
                            if l.startswith("| **A14**")), "")
                if not a14:
                    problems.append("[台账] INTEGRATION.md 里找不到 A14 那一行")
                else:
                    for need in ("9–10/10", "骨干交叉验证"):
                        if need not in a14:
                            problems.append(
                                f"[台账] A14 缺「{need}」"
                                f"（exp95 复现把 POST 侧的 10/10 降级成 9–10/10）")

                # (d) §三十一·四：V1 vs V2 的主表 + 位置轴（**判决已出**）
                dec95 = d95.get("decision") or {}
                sec31c = find_section(text, "三十一·四、")
                if not sec31c:
                    problems.append("FINDINGS 里找不到「三十一·四」小节")
                else:
                    rows_c = sec31c.splitlines()

                    def row95c(prefix):
                        for ln in rows_c:
                            if ln.startswith(prefix):
                                return ln.replace("−", "-")
                        return ""

                    pr95 = dec95.get("paired_v2_minus_v1") or {}
                    s95v1 = (d95.get("summaries") or {}).get("v1") or {}
                    s95v2 = (d95.get("summaries") or {}).get("v2") or {}
                    for key, marker in (("zeroshot_acc1", "| **C1 单试次 top-1** |"),
                                        ("zeroshot_acc1_featavg", "| **C2 试次平均 top-1** |"),
                                        ("rsa", "| C3 RSA |")):
                        row = row95c(marker)
                        if not row:
                            problems.append(f"[exp95] §三十一·四 找不到主表行「{marker}」")
                            continue
                        for val, what in ((s95v1.get(key), "V1"), (s95v2.get(key), "V2"),
                                          ((pr95.get(key) or {}).get("median_diff"), "配对差中位")):
                            if val is None:
                                continue
                            if f"{float(val):.4f}" not in row:
                                problems.append(
                                    f"[exp95] §三十一·四 主表 {key} 缺{what} {float(val):.4f}")
                        pd = pr95.get(key) or {}
                        if pd.get("n") and pd.get("n_pos") is not None:
                            tag = f"{pd['n_pos']}/{pd['n']}"
                            if tag not in row:
                                problems.append(
                                    f"[exp95] §三十一·四 主表 {key} 缺配对计数 {tag}")
                        if pd.get("p_sign") is not None:
                            ps = f"{float(pd['p_sign']):.4f}"
                            if ps not in row:
                                problems.append(
                                    f"[exp95] §三十一·四 主表 {key} 缺符号检验 p {ps}")
                    pa95c = dec95.get("posaxis") or {}
                    for v, marker in (("v1", "| V1 |"), ("v2", "| V2 |")):
                        row = row95c(marker)
                        d = pa95c.get(v)
                        if not d:
                            continue
                        if not row:
                            problems.append(f"[exp95] §三十一·四 找不到位置轴行「{marker}」")
                            continue
                        for cnt, what in ((f"{d['post_above_all_rand34']}/{d['n']}", "POST>随机34"),
                                          (f"{d['ant_below_all_rand29']}/{d['n']}", "ANT<随机29")):
                            if cnt not in row:
                                problems.append(
                                    f"[exp95] §三十一·四 位置轴 {v} 行缺{what} {cnt}")
                        for val, what in ((d.get("post_minus_ant_median"), "POST−ANT"),
                                          (d.get("post_acc1_median"), "POST acc1"),
                                          (d.get("ant_acc1_median"), "ANT acc1")):
                            if val is None:
                                continue
                            if f"{float(val):.4f}" not in row:
                                problems.append(
                                    f"[exp95] §三十一·四 位置轴 {v} 行缺{what} {float(val):.4f}")
                    # 判决必须落在「V2 被砍、用 V1」这一支（HYBRID_WORSE / REDUNDANT）
                    v95 = dec95.get("verdict")
                    if v95 not in (None, "HYBRID_WORSE", "HYBRID_REDUNDANT"):
                        problems.append(
                            f"[exp95] §三十一·四 的判决是 {v95}"
                            f" ⇒ 与「剃刀砍掉 V2、用 V1」的表述不一致")
                    # 判决**双向绑定**：JSON 里的判决字符串必须出现在文档里
                    if v95 and v95 not in sec31c:
                        problems.append(
                            f"[exp95] §三十一·四 里找不到判决字符串「{v95}」")
        else:
            print(f"（缺 {e95}，跳过 exp95 比对）")

        # --- 13. 三组切分（按脑科学解剖）：§4.1 的分组剖面表 ← exp89 的组级数字 ---
        pos89 = HERE / "posaxis-results" / "exp89_posaxis.json"
        specf = ROOT / "BRAIN_STRUCTURE_SPEC.md"
        if pos89.exists() and specf.exists():
            p89 = json.loads(pos89.read_text(encoding="utf-8")).get("summary", {})
            stext = specf.read_text(encoding="utf-8")
            i = stext.find("### 4.1 前提已实测")
            j = stext.find("### 4.2 ", i + 1) if i >= 0 else -1
            sec41 = stext[i: j if j > 0 else len(stext)]
            for g, marker in (("POST", "| **POST** 感觉/语义 |"),
                              ("CENT", "| **CENT** 运动 |"),
                              ("FRONT", "| **FRONT** 执行 |"),
                              ("ANT", "| （原 **ANT**")):
                e = p89.get(g) or {}
                v = e.get("r_median")
                ns = e.get("n_sig")
                if v is None:
                    continue
                row = next((ln for ln in sec41.splitlines()
                            if ln.startswith(marker)), "")
                if not row:
                    problems.append(
                        f"[三组] BRAIN_STRUCTURE_SPEC §4.1 找不到分组行「{marker}」")
                    continue
                if f"{float(v):+.4f}" not in row:
                    problems.append(
                        f"[三组] §4.1 分组行 {g} 的 RSA 不符：文档里没有 {float(v):+.4f}")
                if ns is not None and f"{ns}/10" not in row:
                    problems.append(
                        f"[三组] §4.1 分组行 {g} 的显著被试数不符：文档里没有 {ns}/10")

            # 通道数：**同一事实两处呈现**（exp89 原表 §二十五·二 与 §4.1 的三组表）⇒ 机器绑定
            sec89 = find_section(text, "二十五·二、(a) 通道组消融")
            if not sec89:
                problems.append("FINDINGS 里找不到「二十五·二」小节（通道数的正本）")
            else:
                ch89 = {}
                for ln in sec89.splitlines():
                    if not ln.startswith("|"):
                        continue
                    cells = [c.strip() for c in ln.strip().strip("|").split("|")]
                    if len(cells) < 2 or not cells[1].isdigit():
                        continue
                    name = cells[0].replace("**", "").split(" ")[0]
                    ch89[name] = cells[1]
                for g, marker in (("POST", "| **POST** 感觉/语义 |"),
                                  ("CENT", "| **CENT** 运动 |"),
                                  ("FRONT", "| **FRONT** 执行 |"),
                                  ("ANT", "| （原 **ANT**")):
                    if g not in ch89:
                        continue
                    row = next((ln for ln in sec41.splitlines()
                                if ln.startswith(marker)), "")
                    if not row:
                        continue
                    cells = [c.strip() for c in row.strip().strip("|").split("|")]
                    if len(cells) < 2 or cells[1] != ch89[g]:
                        problems.append(
                            f"[三组] §4.1 分组行 {g} 的通道数 {cells[1] if len(cells) > 1 else '?'}"
                            f" 与 exp89 原表的 {ch89[g]} 不一致")
        else:
            print(f"（缺 {pos89} 或 {specf}，跳过三组切分比对）")

        # --- 14. exp96：三组模型层剖面 —— 数字绑定 + **两份判决都必须留痕** ---
        e96 = HERE / "exp96-results" / "exp96_summary.json"
        if e96.exists():
            d96 = json.loads(e96.read_text(encoding="utf-8"))
            sec32 = find_section(text, "三十二·三、结果")
            sec32d = find_section(text, "三十二·四、判决")
            if not sec32 or not sec32d:
                problems.append("FINDINGS 里找不到「三十二·三」或「三十二·四」小节")
            else:
                bb = d96.get("per_backbone") or {}
                # (a) 逐骨干的 B / C：数字 + 计数 + p 必须与文档一致
                for tag, info in bb.items():
                    b = info.get("B") or {}
                    c = info.get("C") or {}
                    row = next((ln for ln in sec32.splitlines()
                                if ln.startswith(f"| **{tag}**")), "")
                    if not row:
                        problems.append(f"[exp96] §三十二·三 找不到骨干行「{tag}」")
                        continue
                    # ⚠️ 文档里负号是 Unicode「−」⇒ 查找前归一化（与 exp95 那块同一处理）
                    row = row.replace("−", "-")
                    for val, what in ((b.get("median"), "B 中位差"),
                                      (c.get("median"), "C 中位差")):
                        if val is None:
                            continue
                        if f"{float(val):+.4f}" not in row:
                            problems.append(
                                f"[exp96] §三十二·三 {tag} 行缺{what} {float(val):+.4f}")
                    for d, what in ((b, "B"), (c, "C")):
                        n_eff = d.get("n_eff")
                        n_pos = d.get("n_pos")
                        n_neg = d.get("n_neg")
                        if n_eff is None or n_pos is None or n_neg is None:
                            continue
                        # ⚠️ 规范写法：「n_pos/n_eff 正、n_neg/n_eff 负」
                        #    （两个计数都查 ⇒ 「0/10」不能被「10/10」的子串糊弄过去）
                        for cnt in (f"{n_pos}/{n_eff} 正", f"{n_neg}/{n_eff} 负"):
                            if cnt not in row:
                                problems.append(
                                    f"[exp96] §三十二·三 {tag} 行缺{what} 计数「{cnt}」")
                        p = d.get("p_sign")
                        if p is not None and f"{float(p):.4f}" not in row:
                            problems.append(
                                f"[exp96] §三十二·三 {tag} 行缺{what} p {float(p):.4f}")
                # (b) **两份判决都必须出现在文档里**（预登记的 ANCHOR_FAIL 不能被删掉）
                # ⚠️ 必须查**唯一的判决句**（①/② 编号 + 完整短语）⇒
                #    只查判决字符串会漏（同一个字符串出现两次时，改第一处抓不到）。
                v_job = (d96.get("job_verdict") or {}).get("verdict")
                v_s1a = (d96.get("s1a_only_verdict_post_hoc") or {}).get("verdict")
                if v_job:
                    sent = f"① 预登记判决 = `{v_job}`"
                    if sent not in sec32d:
                        problems.append(
                            f"[exp96] §三十二·四 里找不到预登记判决句「{sent}」"
                            "（预登记的那份判决不能被改成事后的那份）")
                if v_s1a:
                    sent = f"② 事后只看 s1a（**非预登记**）= `{v_s1a}`"
                    if sent not in sec32d:
                        problems.append(
                            f"[exp96] §三十二·四 里找不到事后判决句「{sent}」")
                v_rec = (d96.get("recomputed_verdict") or {}).get("verdict")
                if v_rec and v_rec not in sec32d:
                    problems.append(
                        f"[exp96] §三十二·四 里找不到离线重算判决字符串「{v_rec}」")
                # (c) 判据 A 的漂移量
                mdiff = d96.get("criterion_A_max_abs_diff")
                if mdiff is not None and f"{float(mdiff):.2e}" not in text:
                    problems.append(
                        f"[exp96] FINDINGS 里找不到判据 A 的最大漂移 {float(mdiff):.2e}")
                # (d) 台账 A21 必须带**骨干限定**与**两份判决**
                itext96 = (ROOT / "INTEGRATION.md").read_text(encoding="utf-8")
                a21 = next((l for l in itext96.splitlines()
                            if l.startswith("| **A21**")), "")
                if not a21:
                    problems.append("[台账] INTEGRATION.md 里找不到 A21 那一行")
                else:
                    a21 = a21.replace("−", "-")
                    for need in ("ANCHOR_FAIL", "THREE_GROUPS_HOLD", "s1a",
                                 "10/10", "0.0020", "两份判决", "骨干限定"):
                        if need not in a21:
                            problems.append(
                                f"[exp96] 台账 A21 缺「{need}」"
                                "（两份判决 / 骨干限定 / 关键数字都不能丢）")
        else:
            print(f"（缺 {e96}，跳过 exp96 比对）")

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
