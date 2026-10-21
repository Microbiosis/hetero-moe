#!/usr/bin/env python3
"""build_sources — L0 来源登记表（唯一生成器）。

每行一个上游来源：数据集 / 论文 / 代码 / 本体。

列语义
------
verified_this_session : 1 = 本次会话里实际访问过该来源（文件或接口）；0 = 只是引用。
pin_kind             : 固定手段的类别（dandi_version / git_commit / bucket_manifest / ...）。
pin_value            : 固定到什么（版本串、commit、内容哈希）。必须可直接核验。
pin_checked_at       : 该 pin 最后一次实际核对日期。
pin_status           :
    verified — 版本与内容都已核对，且可重放
    partial  — 只能用不可变方式之一（例如端点无版本号、OSF 非 registration）
    degraded — 上游自身状态不完整（如 dandiset 只有 draft 且 status=Invalid）
    broken   — 原登记地址失效且未找到等价替代

输出 framework/sources.csv / .parquet
"""
import sys
from pathlib import Path

import pandas as pd

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/domains/framework")

TODAY = "2026-10-20"

# 2026-10-20 本会话新增证据：
#  1) steinmetz 桶镜像已按 path+size+xet_hash 规范化取 sha256（43 文件 / 1,761,487,147 B）；
#     DOI 经 Crossref 核对，标题 "Distributed coding of choice, action and engagement
#     across the mouse brain", Nature 2019。
#  2) flywire 论文全文（PMC11446842，已落地 papers/flywire__PMC11446842.xml）正文写明
#     "the analysis presented here is from version 783" 与 "version 783, required 3,013,513 edits"；
#     DOI 经 Crossref 核对，标题 "Neuronal wiring diagram of an adult brain", Nature 2024。
#  3) falcon-challenge 的 preproc/h2_preproc.py 已在固定 commit 下定位并通读：
#     全文没有 add_electrode / add_unit，只 add_acquisition(TimeSeries "binned_spikes")；
#     公用模块 preproc/nwb_create_utils.py 也只有 TimeSeries 助手。
#     ⇒ FALCON 导出的 NWB 结构上不可能带电极表/单元表，这条从"待复核"变为"代码级证实"。

ROWS = [
    dict(source_id="steinmetz2019",
         name="Steinmetz 2019 mouse Neuropixels",
         kind="dataset",
         url="https://doi.org/10.1038/s41586-019-1787-x",
         version="2019 Nature 576:266-273",
         license="CC-BY-4.0",
         used_by="steinmetz; omnibrain_spike_view",
         verified_this_session=1,
         note=("经 HF 桶镜像取得；DOI 本会话经 Crossref 核对（Nature 2019，"
               "Distributed coding of choice, action and engagement across the mouse brain）；"
               "桶目录 43 文件 / 1,761,487,147 B，按 path+size+xet_hash 规范化后取 sha256。"
               "原始 NWB 不在本桶，故只能固定镜像内容而非上游 artifact"),
         pin_kind="doi+bucket_manifest",
         pin_value=("10.1038/s41586-019-1787-x | hf://buckets/illusion-brain/training-data/"
                    "spikes/steinmetz-spikes manifest_sha256="
                    "af1c641e7f3c127b41d3620f937bde96630d29b13eb4293a3090aaa66f1a4381 "
                    "(files=43 bytes=1761487147; canon='path\\tsize\\txet_hash\\n' sorted by path)"),
         pin_checked_at=TODAY,
         pin_status="verified"),

    dict(source_id="dandi000070",
         name="Churchland 2012 reaching (NLB)",
         kind="dataset",
         url="https://dandiarchive.org/dandiset/000070",
         version="0.260528.0919",
         license="CC-BY-4.0",
         used_by="primate_macaque_nlb(churchland×67); macaque_unit_region",
         verified_this_session=1,
         note=("6 个 session、每个 192 units；electrodes/location 为 96 PMd + 96 M1；"
               "DANDI 侧 10 个 asset / 53,340,156,657 B。桶内 67 段是从这 6 个 session 切出的时间块"),
         pin_kind="dandi_version",
         pin_value="0.260528.0919 (assets=10 bytes=53340156657)",
         pin_checked_at=TODAY,
         pin_status="verified"),

    dict(source_id="dandi000127",
         name="Area2_Bump (macaque S1)",
         kind="dataset",
         url="https://dandiarchive.org/dandiset/000127",
         version="0.220113.0359",
         license="CC-BY-4.0",
         used_by="primate_macaque_nlb(area2-bump×2)",
         verified_this_session=1,
         note="96 电极全为 S1 Area 2；桶内 area 列的 S1_M1 是误标；2 asset / 1,823,368,810 B",
         pin_kind="dandi_version",
         pin_value="0.220113.0359 (assets=2 bytes=1823368810)",
         pin_checked_at=TODAY,
         pin_status="verified"),

    dict(source_id="dandi000130",
         name="DMFC_RSG (macaque)",
         kind="dataset",
         url="https://dandiarchive.org/dandiset/000130",
         version="0.241017.1448",
         license="CC-BY-4.0",
         used_by="primate_macaque_nlb(dmfc-rsg×3); primate_dmfc_rsg",
         verified_this_session=1,
         note="72 电极全 DMFC；NWB 为区间 + 稀疏 spike_times；2 asset / 15,673,496 B",
         pin_kind="dandi_version",
         pin_value="0.241017.1448 (assets=2 bytes=15673496)",
         pin_checked_at=TODAY,
         pin_status="verified"),

    dict(source_id="dandi000950",
         name="FALCON H2 human handwriting iBCI",
         kind="dataset",
         url="https://dandiarchive.org/dandiset/000950",
         version="0.241029.1403",
         license="CC-BY-4.0",
         used_by="human_intracortical(omnibrain_multi)",
         verified_this_session=1,
         note=("NWB 既无 electrodes 表也无 units 表：生成它的 preproc/h2_preproc.py"
               "（commit 770c9b9, blob bf42524）只 add_acquisition(TimeSeries 'binned_spikes'，"
               "20 ms 阈值跨越，data 形状 t×ch)，全文无 add_electrode / add_unit；"
               "公用模块 nwb_create_utils.py 亦只有 TimeSeries 助手。"
               "⇒ omnibrain_multi 里这批「单元」实为分箱矩阵的列号，不是 sorted unit。"
               "描述为左手中央前回 hand knob；47 asset / 1,224,956,736 B"),
         pin_kind="dandi_version",
         pin_value="0.241029.1403 (assets=47 bytes=1224956736)",
         pin_checked_at=TODAY,
         pin_status="verified"),

    dict(source_id="dandi000954",
         name="FALCON H1 human 7DoF reach-grasp",
         kind="dataset",
         url="https://dandiarchive.org/dandiset/000954",
         version="draft",
         license="CC-BY-4.0",
         used_by="human_intracortical(omnibrain_multi)",
         verified_this_session=1,
         note=("同样无 electrodes 表；半球未记录。注意：该 dandiset 只有 draft 且 status=Invalid，"
               "无已发布版本，因此版本维度不可固定。另外 falcon-challenge 仓库里没有 h1_preproc.py，"
               "生成 000954 的导出脚本本次未定位，故不能像 000950 那样给代码级证据；"
               "40 asset / 102,258,934 B"),
         pin_kind="dandi_draft_only",
         pin_value="draft status=Invalid (assets=40 bytes=102258934) — 无 published version",
         pin_checked_at=TODAY,
         pin_status="degraded"),

    dict(source_id="things-eeg2",
         name="THINGS-EEG2",
         kind="dataset",
         url="https://osf.io/3jk45/",
         version="v2",
         license="CC-BY-4.0",
         used_by="human_eeg; omnibrain_eeg_view; omnibrain4d",
         verified_this_session=0,
         note="250 Hz 白化预处理；跨通道相关已被抹掉。OSF 节点非 registration，无不可变版本",
         pin_kind="osf_node",
         pin_value=("3jk45 @ 2025-07-25T09:14:43 (registration=false) | "
                    "paper doi:10.1016/j.neuroimage.2022.119754"),
         pin_checked_at=TODAY,
         pin_status="partial"),

    dict(source_id="flywire",
         name="FlyWire FAFB v783",
         kind="dataset",
         url="https://doi.org/10.1038/s41586-024-07558-y",
         version="v783",
         license="CC-BY-4.0",
         used_by="flywire; omnibrain_graph_view",
         verified_this_session=1,
         note=("DOI 经 Crossref 核对（Neuronal wiring diagram of an adult brain, Nature 2024）；"
               "v783 由本会话已落地的论文全文直接确认：PMC11446842 正文写 "
               "'the analysis presented here is from version 783'、"
               "'version 783, required 3,013,513 edits'"),
         pin_kind="release+doi",
         pin_value="v783 | doi:10.1038/s41586-024-07558-y (Crossref ok; PMC11446842 正文确认 v783)",
         pin_checked_at=TODAY,
         pin_status="verified"),

    dict(source_id="allen-structure-graph",
         name="Allen Mouse Brain StructureGraph",
         kind="ontology",
         url="http://api.brain-map.org/api/v2/structure_graph_download/1.json",
         version="1.json",
         license="Allen Terms of Use",
         used_by="allen_region_map",
         verified_this_session=1,
         note="1,327 个结构，含 acronym/name/parent；端点无版本号，只能按抓取时点固定",
         pin_kind="api_snapshot",
         pin_value="structure_graph_download/1.json @ 2026-10-20（端点无版本）",
         pin_checked_at=TODAY,
         pin_status="partial"),

    dict(source_id="mbao",
         name="Mouse Brain Atlas Ontology (MBAO)",
         kind="ontology",
         url="https://github.com/brain-bican/mouse_brain_atlas_ontology/releases/tag/v2025-07-04",
         version="v2025-07-04",
         license="CC-BY-4.0",
         used_by="allen_region_map",
         verified_this_session=1,
         note=("3.2 MB；Allen ID → UBERON 但粒度粗（19 个区共用一个 UBERON）。"
               "该 release 把多条脑神经改映射到 root 结构"
               "（MBA_798 facial nerve → UBERON:0004674 facial nerve root）"),
         pin_kind="git_commit",
         pin_value="01d70bed4a444fdf9886f1aaf2f3b7071d4082b5 (tag v2025-07-04; PGP verified)",
         pin_checked_at=TODAY,
         pin_status="verified"),

    dict(source_id="uberon",
         name="UBERON nervous-minimal",
         kind="ontology",
         url="https://github.com/obophenotype/uberon/releases/tag/v2026-10-01",
         version="v2026-10-01",
         license="CC-BY-3.0",
         used_by="allen_region_map (name fallback)",
         verified_this_session=1,
         note="3.0 MB；缩写直接匹配会假阳性，只能做全名精确匹配",
         pin_kind="git_commit",
         pin_value="d225e8458efebec3d8df38fe936bf95fb3521d39 (tag v2026-10-01)",
         pin_checked_at=TODAY,
         pin_status="verified"),

    dict(source_id="falcon-code",
         name="FALCON preprocessing code",
         kind="code",
         url="https://github.com/snel-repo/falcon-challenge",
         version="main@770c9b9",
         license="见仓库",
         used_by="（用于确认 electrodes / units 表在导出时被丢弃）",
         verified_this_session=1,
         note=("原登记的 https://github.com/snel-repo/falcon 已 404，改指 falcon-challenge 并固定 commit。"
               "本会话在该 commit 下定位并通读 preproc/h2_preproc.py："
               "全文无 add_electrode / add_unit，只写 binned_spikes 一个 TimeSeries；"
               "preproc/nwb_create_utils.py 也只有 TimeSeries 助手。"
               "仓库内不存在 h1_preproc.py，000954 的导出脚本仍未定位"),
         pin_kind="git_commit+blob",
         pin_value=("770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f | "
                    "preproc/h2_preproc.py@bf425246593d4a076c359458f28a1872dc804fab | "
                    "preproc/nwb_create_utils.py@240c8dba4be8a0a6d8346933aac6462cfb9eb783"),
         pin_checked_at=TODAY,
         pin_status="verified"),
]

COLS = ["source_id", "name", "kind", "url", "version", "license", "used_by",
        "verified_this_session", "note", "pin_kind", "pin_value",
        "pin_checked_at", "pin_status"]

d = pd.DataFrame(ROWS)[COLS]

OUT.mkdir(parents=True, exist_ok=True)
d.to_csv(OUT / "sources.csv", index=False)
d.to_parquet(OUT / "sources.parquet", index=False)

print(f"rows={len(d)} cols={len(d.columns)}")
print(d.groupby("kind").size().to_string())
print("\npin_status:")
print(d.groupby("pin_status").size().to_string())
print(f"\n本次实际访问过的: {int(d.verified_this_session.sum())}/{len(d)}")
print(d[["source_id", "version", "pin_status", "verified_this_session"]].to_string(index=False))
