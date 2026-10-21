#!/usr/bin/env python3
"""patch_framework_json2 — 从实际目录重建 framework.json 的 tables / docs / entry_points。

设计约束
--------
1. **一一覆盖**：`framework/` 下除 `framework.json` 自身外的每个文件，必须出现在
   `REGISTRY`（作为 canonical）或某个条目的 `twins` 里。漏一个就退出码 1，不写文件。
2. **行数不手填**：`rows` / `cols` / `bytes` 全从实文件现算，避免索引与产物脱节
   （上一版就是这样把 `papers_manifest` 记成 5 行，实际已 8 行）。
3. **JSON 不算 DataFrame**：`rows` 取顶层键数（dict）或长度（list），并用
   `rows_unit="top_level_keys"` 明示，不与表的行数混为一谈。
4. **作废不删除**：被推翻的产物保留文件，在索引里标 `status="superseded"`。
"""
import json
import sys
from pathlib import Path

import pandas as pd

FWROOT = Path("/domains")
FW = FWROOT / "framework"

# file -> (name, purpose, status)
REGISTRY = {
    "framework/allen_region_map.parquet":
        ("allen_region_map", "Allen 缩写 / 全名 → UBERON 对照（73 个区）", "active"),
    "framework/decay_stats.json":
        ("decay_stats", "ρ_half 衰减统计汇总（按 view 分组的两段列表）", "active"),
    "framework/duplicate_map.parquet":
        ("duplicate_map", "L1 去重边表（185 条，含每条的判定依据）", "active"),
    "framework/framework_index.parquet":
        ("framework_index", "桶内全部对象的入口索引（文件 / 形状 / 阶数 / rank）", "active"),
    "framework/isi_quality.json":
        ("isi_quality", "ISI<2 ms 违反率的首批探查（DMFC 2 成员 / 286 单元）；10 session 版见 isi_quality2", "active"),
    "framework/isi_quality2.json":
        ("isi_quality2", "10 session 猕猴 ISI 违反率 + 速率校正；correction 字段记录被推翻的判词", "active"),
    "framework/joins.json":
        ("joins", "8 个跨表连接键的合法性与空洞率（L1 只放结构键）", "active"),
    "framework/l2_decay.parquet":
        ("l2_decay", "L2 衰减统计逐切片（l2_invariants 的子集，388 行）", "active"),
    "framework/l2_invariants.parquet":
        ("l2_invariants", "L2 不变量逐切片（605 行，M 层窗口）", "active"),
    "framework/l2_region_invariants.parquet":
        ("l2_region_invariants", "L2 区级不变量（454 行，区身份核验后）", "active"),
    "framework/macaque_sensitivity.json":
        ("macaque_sensitivity", "猕猴留一 session / 无漂移子集 / 配对区比较的稳健性", "active"),
    "framework/macaque_unit_region.parquet":
        ("macaque_unit_region", "猕猴 13,126 个单元的逐单元区标签（PMd / M1 / S1 / DMFC）", "active"),
    "framework/matched_stats.json":
        ("matched_stats", "小鼠 vs 猕猴等样本量匹配后的 ρ_half 比值与 CI", "active"),
    "framework/multi_stats.json":
        ("multi_stats", "omnibrain_multi 视图的统计汇总（三段列表）", "active"),
    "framework/paper_qc_notes.json":
        ("paper_qc_notes", "论文全文与框架口径的 5 条核对，逐条标明是否影响结论", "active"),
    "framework/papers_manifest.parquet":
        ("papers_manifest", "已落地论文文件清单（含 sha256 与字节数）", "active"),
    "framework/papers_oa_status.parquet":
        ("papers_oa_status", "6 篇论文的开放获取落地状态与未落地原因", "active"),
    "framework/qc_impact.json":
        ("qc_impact", "论文口径核对对已有数字的实际影响（细胞数 / 区秩相关）", "active"),
    "framework/quality_axis_macaque2.parquet":
        ("quality_axis_macaque2", "猕猴修正版质量轴逐 cell 打分（同 session 内参照）", "active"),
    "framework/quality_axis_macaque2.json":
        ("quality_axis_macaque2_metrics", "猕猴修正版质量轴参数、LOSO 与可靠度/上限", "active"),
    "framework/quality_bias.json":
        ("quality_bias", "质量分与 rho1 的耦合诊断（秩相关 / 阈值过滤漂移 / 分层）", "active"),
    "framework/quality_label_reliability.json":
        ("quality_label_reliability", "标签分半可靠度与完美特征上限（含逐区 log-SD）", "active"),
    "framework/quality_macaque_confound.json":
        ("quality_macaque_confound", "猕猴片段序号 / session 混杂的四项检验", "active"),
    "framework/quality_model.json":
        ("quality_model", "质量模型训练与置换检验的完整指标（CV AUC / null / 单特征）", "active"),
    "framework/region_table_full.parquet":
        ("region_table_full", "36 个区的完整表（n_units / rho_half / 置信度 / Allen id）", "active"),
    "framework/retrieval_index.json":
        ("retrieval_index", "L3 检索层的字段、词表与全局限制（gaps）", "active"),
    "framework/rho_matched_multi.parquet":
        ("rho_matched_multi", "等样本量匹配后的 multi 视图样本（212 行）", "active"),
    "framework/rho_matched_units.parquet":
        ("rho_matched_units", "等样本量匹配后的 unit 级样本（106 行）", "active"),
    "framework/slice_catalog.parquet":
        ("slice_catalog", "L3 检索目录（1,059 行 = 605 slice + 454 区，逐行 caveats）", "active"),
    "framework/sources.parquet":
        ("sources", "L0 来源表（12 条，含 pin_kind / pin_value / pin_status）", "active"),
    "framework/structure.parquet":
        ("structure", "区 / 域结构表（244 行，五级置信度 + 五级归属粒度 + Allen id）", "active"),
    "framework/unit_isi.parquet":
        ("unit_isi", "首批 ISI 指标逐单元（286 单元，DMFC）", "active"),
    "framework/unit_isi2.parquet":
        ("unit_isi2", "10 session 猕猴 ISI 指标逐单元（1,920，PMd / M1 各 960）", "active"),
    "framework/unit_region_registry.parquet":
        ("unit_region_registry", "全部 37,155 个单元的区归属登记", "active"),
}

# canonical -> 同内容的另一种编码（不单独占索引条目）
TWINS = {}   # CSV 已全部转为 parquet，不再有同内容的第二种编码


DOC_PURPOSE = {
    "METHODS.md": "口径、结果与限制（唯一的权威说明）",
    "RETRIEVAL.md": "L3 检索层用法：字段、示例查询、全局限制",
}


def measure(p: Path):
    suf = p.suffix.lower()
    if suf == ".parquet":
        df = pd.read_parquet(p)
        return len(df), len(df.columns), "rows"
    if suf == ".csv":
        df = pd.read_csv(p)
        return len(df), len(df.columns), "rows"
    if suf == ".json":
        o = json.loads(p.read_text(encoding="utf-8"))
        return (len(o) if isinstance(o, (dict, list)) else 1), None, "top_level_keys"
    return None, None, "unknown"


def main():
    actual = sorted(p for p in FW.iterdir()
                    if p.is_file() and p.name != "framework.json")
    covered, missing_on_disk, unregistered = set(), [], []

    tables = []
    for rel, (name, purpose, status) in REGISTRY.items():
        p = FWROOT / rel
        if not p.exists():
            missing_on_disk.append(rel)
            continue
        rows, cols, unit = measure(p)
        tables.append(dict(
            name=name, file=rel, rows=rows, cols=cols,
            purpose=purpose, status=status,
            rows_unit=unit, bytes=p.stat().st_size,
            twins=TWINS.get(rel, []),
        ))
        covered.add(p.name)
        for t in TWINS.get(rel, []):
            covered.add(Path(t).name)

    twin_declared_missing = []
    for canon, tws in TWINS.items():
        for t in tws:
            if not (FWROOT / t).exists():
                twin_declared_missing.append(t)

    for p in actual:
        if p.name not in covered:
            unregistered.append(p.name)

    print(f"entries={len(tables)}  files_on_disk(excl. index)={len(actual)}  "
          f"covered={len(covered)}")
    if unregistered:
        print(f"UNREGISTERED ({len(unregistered)}): {unregistered}")
    if missing_on_disk:
        print(f"MISSING_ON_DISK ({len(missing_on_disk)}): {missing_on_disk}")
    if twin_declared_missing:
        print(f"TWIN_DECLARED_MISSING: {twin_declared_missing}")

    if unregistered or missing_on_disk or twin_declared_missing:
        print("ABORT: 索引与磁盘不一致，未写入。")
        return 1

    names = [t["name"] for t in tables]
    assert len(names) == len(set(names)), "重复的 name"

    docs = []
    for p in sorted((FW / "docs").glob("*.md")):
        docs.append(dict(name=p.name, file=f"framework/docs/{p.name}",
                         bytes=p.stat().st_size,
                         purpose=DOC_PURPOSE.get(p.name, "")))

    old = json.loads((FW / "framework.json").read_text(encoding="utf-8"))
    sup = sorted(t["name"] for t in tables if t["status"] == "superseded")

    new = dict(old)
    new["schema_version"] = "1.3"
    new["tables"] = sorted(tables, key=lambda t: t["name"])
    new["docs"] = docs
    new["entry_points"] = {
        "read_first": "framework/docs/DATA_LADDER.md（L0–L4 现状与缺口）",
        "methods": "framework/docs/FRAMEWORK_METHODS.md（口径、统计、限制）",
        "retrieval": ("framework/docs/RETRIEVAL.md + framework/slice_catalog.parquet"
                      "（+ retrieval_index.json 字段/词表/全局限制）"),
        "sources": "framework/sources.parquet（含 pin_kind / pin_value / pin_status）",
        "check": "scripts/verify_framework.py（自动对账；条数以运行输出为准）",
    }
    new["coverage"] = {
        "n_entries": len(tables),
        "n_files_excl_index": len(actual),
        "n_twins": sum(len(t["twins"]) for t in tables),
        "unregistered": [],
        "superseded": sup,
    }

    (FW / "framework.json").write_text(
        json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"schema_version={new['schema_version']}  tables={len(tables)}  docs={len(docs)}")
    print(f"superseded={sup}")
    print(f"framework.json bytes={(FW / 'framework.json').stat().st_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
