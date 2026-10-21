#!/usr/bin/env python3
"""verify_framework — 校验 framework/ 下产物的自洽性。不重算，只对账。

检查分五组：索引、结构、L2、decay、joins。任何一条不过就返回非零退出码。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

OUT = Path("/domains/framework")
fails, checks = [], 0


def ok(name, cond, detail=""):
    global checks
    checks += 1
    if not cond:
        fails.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")
    else:
        print(f"  ok    {name}")


print("=" * 66)
print("A. framework_index")
idx = pd.read_parquet(OUT / "framework_index.parquet")
ok("行数 > 0", len(idx) > 0, len(idx))
ok("成员数 == 14", idx.member.nunique() == 14, idx.member.nunique())
ok("rank ⊆ 0..4", set(idx["rank"].unique()) <= {0, 1, 2, 3, 4}, set(idx["rank"].unique()))
bad = idx[(idx["rank"] >= 2) & idx["shape"].isna()]
ok("rank>=2 都有 shape", len(bad) == 0, bad[["member", "obj"]].to_dict("records"))
ragged = idx[(idx["rank"] == 1) & idx["shape"].isna()]
ok("rank==1 的空 shape 只能是 ragged 对象",
   set(ragged.obj) <= {"units/obs_intervals+spike_times"}, [o for o in ragged.obj])
per = idx.groupby("member").n_files.sum()
ok("每个成员至少 1 个文件", (per > 0).all(), per[per <= 0].to_dict())

print("=" * 66)
print("B. structure")
S = pd.read_parquet(OUT / "structure.parquet")
ok("行数 > 0", len(S) > 0, len(S))
ok("层级 == {unit,segment,window,node}",
   set(S.level.unique()) == {"unit", "segment", "window", "node"}, set(S.level.unique()))
EXPECT = {"steinmetz": 24029, "counts_v4": 175, "omnibrain_multi": 1409850,
          "flywire": 107265}
tot = S.groupby("member").n.sum().to_dict()
for m, v in EXPECT.items():
    ok(f"{m} 总数 == {v}", int(tot.get(m, -1)) == v, tot.get(m))
ok("region_assignment 取值合法",
   set(S.region_assignment.unique()) <= {"per_unit", "per_segment", "per_node",
                                         "window_area", "session_majority"},
   set(S.region_assignment.unique()))
ok("region_resolution 取值合法",
   set(S.region_resolution.unique()) <= {"as_labeled", "parent_rollup"},
   set(S.region_resolution.unique()))
ok("region_confidence 取值合法",
   set(S.region_confidence.unique()) <= {"anatomical", "anatomical_unmapped", "inferred",
                                         "compound", "coarse", "none"},
   set(S.region_confidence.unique()))

print("=" * 66)
print("C. l2_invariants")
L = pd.read_parquet(OUT / "l2_invariants.parquet")
ok("行数 == 605", len(L) == 605, len(L))
ok("成员数 == 13", L.member.nunique() == 13, L.member.nunique())
ok("counts_v4_ragged 不在 L2", "counts_v4_ragged" not in set(L.member.unique()))
ok("(member,slice) 唯一", not L.duplicated(["member", "slice"]).any(),
   int(L.duplicated(["member", "slice"]).sum()))
ok("tau_available 与 tau_status 一致",
   bool((L.tau_available == (L.tau_status == "ok")).all()))
ok("tau_status=ok 必有 tau_ms",
   bool(L.loc[L.tau_status == "ok", "tau_ms"].notna().all()))
ok("pooled_acf_half_status=ok 必有值",
   bool(L.loc[L.pooled_acf_half_status == "ok", "pooled_acf_half_ms"].notna().all()))
ok("tau_status 分布符合预期",
   L.tau_status.value_counts().to_dict() ==
   {"not_estimable_8bins": 439, "shape_not_plateau": 58, "ok": 50,
    "inapplicable_whitened": 31, "not_estimated_4d": 20, "too_few_points": 3,
    "static": 2, "not_computed": 2},
   L.tau_status.value_counts().to_dict())

print("=" * 66)
print("D. l2_decay")
D = pd.read_parquet(OUT / "l2_decay.parquet")
ok("成员 ⊆ L2 成员", set(D.member.unique()) <= set(L.member.unique()),
   set(D.member.unique()) - set(L.member.unique()))
key_d = set(zip(D.member, D.slice))
key_l = set(zip(L.member, L.slice))
ok("(member,slice) 都存在于 L2", key_d <= key_l, len(key_d - key_l))
ok("rho_half_status=ok 必有值",
   bool(D.loc[D.rho_half_status == "ok", "rho_half_ms"].notna().all()))
dm = D[D.member == "primate_dmfc_rsg"]
ok("dmfc 两切片 rho_half 非空", dm.rho_half_ms.notna().all(), dm.rho_half_ms.tolist())
ok("dmfc 两切片 rho_zero 非空", dm.rho_zero_ms.notna().all(), dm.rho_zero_ms.tolist())
sp = D[D.member.isin(["steinmetz", "primate_macaque_nlb"])]
ok("尖峰切片数 == 111", len(sp) == 111, len(sp))

print("=" * 66)
print("D2. unit_region_registry")
R = pd.read_parquet(OUT / "unit_region_registry.parquet")
ok("行数 == 37155", len(R) == 37155, len(R))
ok("成员 == {steinmetz, primate_macaque_nlb}",
   set(R.member.unique()) == {"steinmetz", "primate_macaque_nlb"}, set(R.member.unique()))
ok("(member,slice,unit_idx) 唯一", not R.duplicated(["member", "slice", "unit_idx"]).any(),
   int(R.duplicated(["member", "slice", "unit_idx"]).sum()))
ok("region_label 无空值", R.region_label.notna().all())
ok("小鼠 72/73 区有 Allen MBA id",
   R[R.member == "steinmetz"].allen_mba_id.notna().sum() == 24021,
   int(R[R.member == "steinmetz"].allen_mba_id.notna().sum()))
ok("猕猴 PMd+M1 各 6432",
   R[(R.member == "primate_macaque_nlb") & (R.region_label == "PMd")].shape[0] == 6432
   and R[(R.member == "primate_macaque_nlb") & (R.region_label == "M1")].shape[0] == 6432,
   R[R.member == "primate_macaque_nlb"].region_label.value_counts().to_dict())
ok("全部 per_unit", (R.region_assignment == "per_unit").all())

print("=" * 66)
print("E. joins")
J = json.loads((OUT / "joins.json").read_text(encoding="utf-8"))
ok("schema >= 1.1", J.get("schema_version", "0") >= "1.1", J.get("schema_version"))
ok("键数 == 8", len(J["keys"]) == 8, len(J["keys"]))
rk = next((k for k in J["keys"] if k["key"] == "region"), None)
ok("region 键存在且为 partial", rk is not None and rk["joinable_across_members"] == "partial")
ok("region 带 region_assignment 统计", "region_assignment" in (rk or {}))
ok("recommended_filter 提到 region_assignment",
   "region_assignment" in (rk or {}).get("recommended_filter", ""))
ok("空洞率在 (0,1)", 0 < J["holes"]["empty_fraction"] < 1, J["holes"]["empty_fraction"])

print("=" * 66)
print("F. L2 质量模型")
Q = pd.read_parquet(OUT / "superseded" / "slice_quality.parquet")
QM = json.loads((OUT / "quality_model.json").read_text(encoding="utf-8"))
QB = json.loads((OUT / "quality_bias.json").read_text(encoding="utf-8"))
ok("slice_quality 行数 == 315", len(Q) == 315, len(Q))
ok("slice_quality 全为 steinmetz", set(Q.member.unique()) == {"steinmetz"},
   set(Q.member.unique()))
ok("quality_prob 均非空且 ∈(0,1)",
   bool(Q.quality_prob.notna().all() and Q.quality_prob.between(0, 1, "neither").all()),
   [float(Q.quality_prob.min()), float(Q.quality_prob.max())])
ok("主标签合法", QM["primary_label"] in ("lab_tau", "lab_form"), QM["primary_label"])
ok("cv_auc_mean 与折均值一致",
   abs(QM["cv_auc_mean"] - sum(QM["cv_auc"]) / len(QM["cv_auc"])) < 1e-6,
   [QM["cv_auc_mean"], QM["cv_auc"]])
# 结论护栏：下面两条若失败，说明本次重训推翻了文档里写的结论，必须回去改文字
ok("[护栏] 主标签 CV AUC 显著高于置换零分布 q95",
   QM["cv_auc_mean"] > QM["perm_null_q95"], [QM["cv_auc_mean"], QM["perm_null_q95"]])
ok("[护栏] quality_prob 与 rho1 强共线(>0.8)，只能分层不能过滤",
   QB["overall"]["quality_vs_rho1"]["spearman"] > 0.8,
   QB["overall"]["quality_vs_rho1"]["spearman"])
ok("[护栏] 阈值 0.50 过滤会把 rho_half 中位抬高 >1.3x",
   QB["filter_shift"]["0.50"]["shift_vs_unfiltered"] > 1.3,
   QB["filter_shift"]["0.50"]["shift_vs_unfiltered"])

print("=" * 66)
print("G. L2 独立质量轴（跨 session 一致性）")
A = pd.read_parquet(OUT / "superseded" / "quality_axis.parquet")
AX = json.loads((OUT / "superseded" / "quality_axis.json").read_text(encoding="utf-8"))
LR = json.loads((OUT / "quality_label_reliability.json").read_text(encoding="utf-8"))
ok("quality_axis 行数 == 231", len(A) == 231, len(A))
ok("特征不含 rho1/acf",
   not any(f in AX["features"] for f in ("rho1", "pooled_acf_lag1", "erank", "acf1")),
   AX["features"])
ok("特征与 rho1 均不共线 (|rho| < 0.6)",
   max(abs(v) for v in AX["feature_vs_rho1"].values()) < 0.6,
   AX["feature_vs_rho1"])
ok("标签与 rho1 不共线 (|rho| < 0.3)", abs(AX["label_vs_rho1"]) < 0.3, AX["label_vs_rho1"])
# 结论护栏：这三条共同支撑"该轴不可用"的判词
ok("[护栏] 按区分组 AUC 不高于随机 (<=0.50)", AX["cv"]["mean"] <= 0.50, AX["cv"]["mean"])
ok("[护栏] 置换 p > 0.05（确实无信号）", AX["cv"]["p"] > 0.05, AX["cv"]["p"])
ok("[护栏] 实测 AUC 显著低于完美特征上限",
   AX["cv"]["mean"] < LR["ceiling_auc_median"] - 0.2,
   [AX["cv"]["mean"], LR["ceiling_auc_median"]])
ok("[护栏] 区内跨 session 离散度中位 > 0.5（log 尺度）",
   LR["within_region_log_sd_median"] > 0.5, LR["within_region_log_sd_median"])

print("=" * 66)
print("H. 猕猴：同 session 内参照 + 留一 session")
M2 = json.loads((OUT / "quality_axis_macaque2.json").read_text(encoding="utf-8"))
MS = json.loads((OUT / "macaque_sensitivity.json").read_text(encoding="utf-8"))
sup = json.loads((OUT / "superseded" / "quality_axis_macaque.json").read_text(encoding="utf-8"))
ok("修正版 cell 数 == 128", M2["n_cells"] == 128, M2["n_cells"])
ok("修正版 session 数 == 5", M2["n_sessions"] == 5, M2["n_sessions"])
ok("旧版已标作废并指向修正版",
   sup.get("superseded") is True and sup.get("superseded_by") == "quality_axis_macaque2.json",
   sup.get("superseded"))
# 结论护栏：这三条支撑"猕猴同样没有独立质量轴，但已有结论稳健"
ok("[护栏] 留一 session AUC ≤ 0.56（无信号）", M2["loso"]["mean"] <= 0.56, M2["loso"]["mean"])
ok("[护栏] 留一 session 置换 p > 0.05", M2["loso"]["p"] > 0.05, M2["loso"]["p"])
ok("[护栏] session 解释 log rho_half 方差 > 0.5",
   M2["session_r2_on_log_rho_half"] > 0.5, M2["session_r2_on_log_rho_half"])
ok("[护栏] 留一 session 后中位仍在全量 ±15% 内",
   all(0.85 <= v["ratio_vs_full"] <= 1.15
       for v in MS["leave_one_session_out"].values()),
   {k: round(v["ratio_vs_full"], 3) for k, v in MS["leave_one_session_out"].items()})
ok("[护栏] PMd/M1 在每个 session 都 100% 同向",
   all(v["frac_pmd_gt_m1"] == 1.0 for v in MS["paired_pmd_m1_by_session"].values()),
   {k: v["frac_pmd_gt_m1"] for k, v in MS["paired_pmd_m1_by_session"].items()})
ok("[护栏] 配对比中位 > 1.0", MS["paired_pmd_m1_all"] > 1.0, MS["paired_pmd_m1_all"])

print("=" * 66)
print("I. 文档落地（L4 可追溯的前提：口径说明必须在桶里）")
DOCS = ["METHODS.md", "RETRIEVAL.md"]
for name in DOCS:
    p = OUT / "docs" / name
    sz = p.stat().st_size if p.exists() else 0
    ok(f"docs/{name} 存在且 > 1 KB", sz > 1024, sz)
FJ = json.loads((OUT / "framework.json").read_text(encoding="utf-8"))
ok("framework.json 可解析且非空", isinstance(FJ, dict) and len(FJ) > 0, list(FJ)[:8])
ok("schema_version == 1.3（精简后：被推翻的产物不再入索引）",
   FJ.get("schema_version") == "1.3", FJ.get("schema_version"))
ok("tables 索引 == 34", len(FJ.get("tables", [])) == 34, len(FJ.get("tables", [])))
ok("docs 索引 == 2", len(FJ.get("docs", [])) == 2,
   [d["name"] for d in FJ.get("docs", [])])
bad_doc = [(d["name"], d["bytes"]) for d in FJ["docs"]
           if not (OUT.parent / d["file"]).exists()
           or (OUT.parent / d["file"]).stat().st_size != d["bytes"]]
ok("docs 的 bytes 与实文件一致", not bad_doc, bad_doc)
ok("framework.json 有 entry_points", len(FJ.get("entry_points", {})) >= 4,
   list(FJ.get("entry_points", {})))
ok("索引里没有 superseded 条目（被推翻的已移出索引）",
   not any(t["status"] == "superseded" for t in FJ["tables"]),
   [t["name"] for t in FJ["tables"] if t["status"] == "superseded"])
ok("superseded/ 里仍有 7 件被推翻的产物（留档，不删）",
   len(list((OUT / "superseded").glob("*"))) == 7,
   sorted(p.name for p in (OUT / "superseded").glob("*")))

bad_rows = []
for t in FJ["tables"]:
    f = OUT.parent / t["file"]
    if not f.exists():
        bad_rows.append((t["name"], "missing"))
        continue
    if t.get("rows_unit") == "top_level_keys":
        o = json.loads(f.read_text(encoding="utf-8"))
        n = len(o) if isinstance(o, (dict, list)) else 1
    elif str(f).endswith(".parquet"):
        n = len(pd.read_parquet(f))
    else:
        n = len(pd.read_csv(f))
    if n != t["rows"]:
        bad_rows.append((t["name"], "rows", t["rows"], n))
    if f.stat().st_size != t["bytes"]:
        bad_rows.append((t["name"], "bytes", t["bytes"], f.stat().st_size))
ok("tables 登记的行数 / 字节数与实文件一致", not bad_rows, bad_rows)

DISK = {p.name for p in OUT.iterdir() if p.is_file() and p.name != "framework.json"}
COVERED = set()
for t in FJ["tables"]:
    COVERED.add(Path(t["file"]).name)
    for tw in t.get("twins", []):
        COVERED.add(Path(tw).name)
ok("framework/ 下每个文件都登记在索引里（无漏登）", DISK == COVERED,
   {"未登记": sorted(DISK - COVERED), "索引里但盘上无": sorted(COVERED - DISK)})
COV = FJ.get("coverage", {})
ok("coverage 自述与实际一致",
   COV.get("n_entries") == len(FJ["tables"])
   and COV.get("n_files_excl_index") == len(DISK)
   and not COV.get("unregistered"), COV)

print("=" * 66)
print("J. 来源 pin（L0 可追溯的下限：每条来源都要有固定标识）")
SC = pd.read_parquet(OUT / "sources.parquet")
NEED = {"source_id", "name", "kind", "url", "version", "license", "used_by",
        "verified_this_session", "note", "pin_kind", "pin_value",
        "pin_checked_at", "pin_status"}
ok("sources.parquet 含全部 pin 列", NEED <= set(SC.columns), sorted(NEED - set(SC.columns)))
ok("来源数 == 12", len(SC) == 12, len(SC))
ok("每行都有 pin_value", SC.pin_value.notna().all() and (SC.pin_value.str.len() > 8).all(),
   SC[SC.pin_value.isna()].source_id.tolist())
ok("pin_status 取值合法",
   set(SC.pin_status) <= {"verified", "partial", "degraded", "broken"}, set(SC.pin_status))
ok("至少 8 条 verified", int((SC.pin_status == "verified").sum()) >= 8,
   SC.pin_status.value_counts().to_dict())
ok("没有仍为 broken 的来源", not (SC.pin_status == "broken").any(),
   SC.loc[SC.pin_status == "broken", "source_id"].tolist())
dandi = SC[SC.source_id.str.startswith("dandi")]
fixed = dandi[dandi.source_id != "dandi000954"]
ok("已发布的 DANDI 版本号都已固定（不再是 draft）",
   fixed.pin_value.str.match(r"^\d\.\d{6}\.\d{4}\b").all(), fixed[["source_id", "pin_value"]]
   .to_dict("records"))
ok("dandi000954 标注为 draft-only / degraded",
   bool((SC.loc[SC.source_id == "dandi000954", "pin_status"] == "degraded").all()))
# 2026-10-20：falcon-code 从 broken 升为 verified。原来那条断言（必须 broken 且
# verified=0）是"没核对就不许声明核对过"的护栏；现在 h2_preproc.py 已在固定 commit 下
# 通读，护栏改为**更严**的形式：要求 pin_value 里同时出现 1 个 commit + 2 个 blob 哈希，
# 少一个就说明只是口头声称，不能算固定。
FC = SC[SC.source_id == "falcon-code"]
fc_pin = str(FC.pin_value.iloc[0]) if len(FC) else ""
ok("falcon-code 用 commit + 2 个 blob 三重固定",
   len(FC) == 1 and FC.pin_status.iloc[0] == "verified"
   and int(FC.verified_this_session.iloc[0]) == 1
   and len(re.findall(r"\b[0-9a-f]{40}\b", fc_pin)) >= 3,
   fc_pin)
ok("旧版来源表已归档", (OUT / "superseded" / "sources_v1.parquet").exists())

print("=" * 66)
print("K. L3 检索层")
CAT = pd.read_parquet(OUT / "slice_catalog.parquet")
RI = json.loads((OUT / "retrieval_index.json").read_text(encoding="utf-8"))
ok("catalog 行数 == 1059", len(CAT) == 1059, len(CAT))
ok("unit_id 唯一", CAT.unit_id.is_unique, int(CAT.unit_id.duplicated().sum()))
ok("level 只有 slice/region", set(CAT.level) == {"slice", "region"}, set(CAT.level))
ok("成员元数据无 unknown", (CAT.species != "unknown").all(),
   sorted(set(CAT[CAT.species == "unknown"].member)))
ok("region 级只出现在有逐单元区标签的成员上",
   set(CAT.loc[CAT.level == "region", "member"]) <=
   set(CAT.loc[CAT.has_unit_region == True, "member"]),
   sorted(set(CAT.loc[CAT.level == "region", "member"]) -
          set(CAT.loc[CAT.has_unit_region == True, "member"])))
bad_cav = CAT[(CAT.rho_half_status.notna()) & (CAT.rho_half_status != "ok")
              & (CAT.n_caveats == 0)]
ok("rho_half 不可用的行必须带 caveat", len(bad_cav) == 0, len(bad_cav))
ok("retrieval_index 有 gaps（全局限制）", len(RI.get("gaps", {})) >= 4,
   list(RI.get("gaps", {})))
# 文档里的命中数不能悄悄过期：与 catalog 实算对账
eq = RI["example_queries"]
real_ca1 = int(((CAT.member == "steinmetz") & (CAT.level == "region")
                & (CAT.region == "CA1")).sum())
real_ok = int(((CAT.member == "primate_macaque_nlb") & (CAT.level == "region")
               & (CAT.rho_half_status == "ok")).sum())
real_noc = int((CAT.n_caveats == 0).sum())
ok("示例命中数与 catalog 一致（CA1 / 猴rho_ok / 无caveat）",
   (eq["mouse_CA1_region_level"], eq["macaque_rho_half_ok"], eq["no_caveat"])
   == (real_ca1, real_ok, real_noc),
   [(eq["mouse_CA1_region_level"], real_ca1), (eq["macaque_rho_half_ok"], real_ok),
    (eq["no_caveat"], real_noc)])
ok("RETRIEVAL.md 已在桶里", (OUT / "docs" / "RETRIEVAL.md").exists())

print("=" * 66)
print("L. 论文落地（L0 的可读性：声称拿到的，清单里必须真有那么一个文件）")
PM = pd.read_parquet(OUT / "papers_manifest.parquet")
PO = pd.read_parquet(OUT / "papers_oa_status.parquet")
ok("papers_manifest 行数 == 13", len(PM) == 13, len(PM))
ok("papers_oa_status 行数 == 6", len(PO) == 6, len(PO))
ok("manifest 每个文件都记了 64 位 sha256",
   PM.sha256.astype(str).str.fullmatch(r"[0-9a-f]{64}").all(),
   PM.loc[~PM.sha256.astype(str).str.fullmatch(r"[0-9a-f]{64}"), "file"].tolist())
ok("manifest 内 (source_id,file) 唯一", not PM.duplicated(["source_id", "file"]).any(),
   int(PM.duplicated(["source_id", "file"]).sum()))
ok("oa_status 的 status 取值合法",
   set(PO.status) <= {"landed", "landed_html", "xml_only", "blocked_pow"}, set(PO.status))
ok("preprint_status 取值合法",
   set(PO.preprint_status) <= {"landed_preprint_pdf", "available_not_fetched", "none_found"},
   set(PO.preprint_status))
PRE = set(PO.loc[PO.preprint_status == "landed_preprint_pdf", "source_id"])
ok("四篇预印本已落地（steinmetz / dandi000130 / flywire / things-eeg2）",
   PRE == {"steinmetz2019", "dandi000130", "flywire", "things-eeg2"}, sorted(PRE))
ok("[护栏] flywire 的预印本必须被记录（OpenAlex 会漏收，只能靠 PPR 索引兜住）",
   bool((PO.loc[PO.source_id == "flywire", "preprint_doi"] == "10.1101/2023.06.27.546656").all()),
   PO.loc[PO.source_id == "flywire", "preprint_doi"].tolist())
ok("六篇全部标 landed（6/6 都有 PDF）", int((PO.status == "landed").sum()) == 6,
   PO.status.value_counts().to_dict())
PDF_SRC = set(PM.loc[PM.kind.isin(["pdf", "pdf_preprint"]), "source_id"])
ok("标 landed 的每篇在 manifest 里都有 PDF 行",
   set(PO.loc[PO.status == "landed", "source_id"]) <= PDF_SRC,
   sorted(set(PO.loc[PO.status == "landed", "source_id"]) - PDF_SRC))
ok("[护栏] 没有 PDF 的不许标 landed",
   set(PO.loc[PO.status != "landed", "source_id"]).isdisjoint(PDF_SRC),
   sorted(set(PO.loc[PO.status != "landed", "source_id"]) & PDF_SRC))
ok("[护栏] 预印本行的 license 非空且写明 CC-BY 变体",
   PM.loc[PM.kind == "pdf_preprint", "license"].astype(str).str.contains("CC-BY").all(),
   PM.loc[PM.kind == "pdf_preprint", ["file", "license"]].to_dict("records"))
ok("dandi000070 标为无预印本（2012，不可声称走过预印本路线）",
   bool((PO.loc[PO.source_id == "dandi000070", "preprint_status"] == "none_found").all()),
   PO.loc[PO.source_id == "dandi000070", "preprint_status"].tolist())

print("=" * 66)
print(f"checks={checks}  fails={len(fails)}")
if fails:
    print("\n失败项：")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("全部通过")
