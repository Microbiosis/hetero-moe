#!/usr/bin/env python3
"""clean_release — 让 dataset 仓库只保留数据本身。

保留：
  README.md（数据集卡）
  数据目录：steinmetz / primate / omnibrain / omnibrain-multi / omnibrain4d /
            counts-v4 / flywire / hierarchy
  framework/：全部 parquet 表 + framework.json + joins.json + retrieval_index.json
              + docs/METHODS.md + docs/RETRIEVAL.md

排除（留在工作桶里，不进发布物）：
  exp0…exp38 / runs / MANIFEST.json
  framework/superseded / framework/docs/archive
  framework 下的分析结果 json（quality_* / macaque_* / isi_* / *_stats / qc_impact / paper_qc_notes）
  scripts/ 全部

用法： python clean_release.py [apply]
不带 apply 时只打印计划。
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "illusion-brain/brain-domains"
SRC = Path("/domains")
APPLY = len(sys.argv) > 1 and sys.argv[1] == "apply"

DATA_DIRS = {"steinmetz", "primate", "omnibrain", "omnibrain-multi",
             "omnibrain4d", "counts-v4", "flywire", "hierarchy"}
FW_ROOT_FILES = {"framework.json", "joins.json", "retrieval_index.json"}
FW_KEEP_DIRS = {"docs/METHODS.md", "docs/RETRIEVAL.md"}

# 仓库根 README.md 来自数据集卡，不是桶根的 README
SPECIAL_SRC = {"README.md": "scripts/dataset_card_README.md"}


def allowed(rel: str) -> bool:
    if rel == "README.md":
        return True
    parts = rel.split("/")
    if len(parts) == 1:
        return False                      # MANIFEST.json 等根级文件不要
    top = parts[0]
    if top in DATA_DIRS:
        return True
    if top != "framework":
        return False
    rest = "/".join(parts[1:])
    if rest.startswith("superseded/") or rest.startswith("docs/archive/"):
        return False
    if rest.endswith(".parquet"):
        return True
    if rest in FW_ROOT_FILES:
        return True
    if rest in FW_KEEP_DIRS:
        return True
    return False


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN 不存在")
        return 2
    api = HfApi(token=token)

    local = []
    for p in SRC.rglob("*"):
        if p.is_file():
            rel = p.relative_to(SRC).as_posix()
            local.append(rel)

    keep = sorted(r for r in local if allowed(r))
    drop_local = sorted(set(local) - set(keep))

    repo = set(api.list_repo_files(REPO, repo_type="dataset"))
    repo.discard(".gitattributes")

    to_delete = sorted(repo - set(keep))
    to_upload = sorted(set(keep) - repo)

    print(f"local={len(local)}  keep={len(keep)}  repo={len(repo)}")
    print(f"DELETE from repo: {len(to_delete)}")
    print(f"UPLOAD to repo  : {len(to_upload)}")
    if to_upload[:10]:
        print("   e.g. " + ", ".join(to_upload[:10]))
    print(f"本地保留但不上传（工作区）: {len(drop_local)}")

    if not APPLY:
        print("\n(只打印计划；加 apply 执行)")
        return 0

    # 1) 删除
    step = 100
    for i in range(0, len(to_delete), step):
        chunk = to_delete[i:i + step]
        api.create_commit(
            repo_id=REPO, repo_type="dataset",
            operations=[__import__("huggingface_hub").CommitOperationDelete(path_in_repo=x) for x in chunk],
            commit_message=f"clean release: remove {len(chunk)} non-data files",
        )
        print(f"  deleted {i + len(chunk)}/{len(to_delete)}", flush=True)

    # 2) 补传
    for rel in to_upload:
        src_path = SRC / SPECIAL_SRC.get(rel, rel)
        api.upload_file(path_or_fileobj=str(src_path), path_in_repo=rel,
                        repo_id=REPO, repo_type="dataset",
                        commit_message=f"clean release: add {rel}")
        print(f"  uploaded {rel}", flush=True)

    repo2 = set(api.list_repo_files(REPO, repo_type="dataset"))
    repo2.discard(".gitattributes")
    print(f"\nafter: repo={len(repo2)}  expected={len(keep)}  match={repo2 == set(keep)}")
    return 0 if repo2 == set(keep) else 1


if __name__ == "__main__":
    sys.exit(main())
