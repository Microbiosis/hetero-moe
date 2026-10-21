#!/usr/bin/env python3
"""verify_publish — 核对 dataset 仓库与 /domains 是否一致（逐文件）。"""
import os
from pathlib import Path

from huggingface_hub import HfApi

REPO = "illusion-brain/brain-domains"
SRC = Path("/domains")
JUNK = (".pyc", ".log", ".pid", ".DS_Store")

api = HfApi(token=os.environ["HF_TOKEN"])

repo_files = set(api.list_repo_files(REPO, repo_type="dataset"))
repo_files.discard(".gitattributes")   # HF 建仓时自动生成，不是数据

local = {}
for p in SRC.rglob("*"):
    if not p.is_file():
        continue
    rel = p.relative_to(SRC).as_posix()
    if rel.endswith(JUNK) or "__pycache__" in rel:
        continue
    local[rel] = p.stat().st_size

missing = sorted(set(local) - repo_files)
extra = sorted(repo_files - set(local))
print(f"local_files={len(local)}  repo_files={len(repo_files)}")
print(f"local_bytes={sum(local.values())} ({sum(local.values())/2**30:.2f} GiB)")
print(f"\nMISSING_IN_REPO ({len(missing)}):")
for x in missing[:30]:
    print("   " + x)
print(f"\nEXTRA_IN_REPO ({len(extra)}):")
for x in extra[:30]:
    print("   " + x)

ok = not missing and not extra
print("\nRESULT:", "CONSISTENT" if ok else "MISMATCH")
raise SystemExit(0 if ok else 1)
