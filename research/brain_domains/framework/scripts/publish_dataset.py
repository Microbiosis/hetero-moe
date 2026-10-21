#!/usr/bin/env python3
"""publish_dataset — 把 /domains 推送到 HF dataset 仓库（分阶段，可重复执行）。

用法： python publish_dataset.py <phase>
  init   建私有仓库 + 上传数据集卡
  core   framework / scripts / hierarchy / runs / MANIFEST.json / exp*
  data1  steinmetz / flywire / omnibrain / counts-v4
  data2  primate / omnibrain-multi
  big    omnibrain4d（25 GiB，单独一轮）

重复执行是安全的：Hub 侧按内容去重，已上传的文件不会重复传输数据。
"""
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = "illusion-brain/brain-domains"
SRC = Path("/domains")
CARD = SRC / "scripts" / "dataset_card_README.md"

IGNORE = ["**/*.pyc", "**/*.log", "**/*.pid", "**/__pycache__/**", "**/.DS_Store"]

PHASES = {
    "core":  ["framework", "scripts", "hierarchy", "runs"],
    "data1": ["steinmetz", "flywire", "omnibrain", "counts-v4"],
    "data2": ["primate", "omnibrain-multi"],
    "big":   ["omnibrain4d"],
}


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


def dir_size(p: Path):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else ""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN 不存在")
        return 2
    api = HfApi(token=token)

    if phase == "init":
        api.create_repo(repo_id=REPO, repo_type="dataset", private=True, exist_ok=True)
        print(f"repo ready: {REPO} (private)")
        if CARD.exists():
            api.upload_file(path_or_fileobj=str(CARD), path_in_repo="README.md",
                            repo_id=REPO, repo_type="dataset",
                            commit_message="dataset card")
            print(f"uploaded README.md ({CARD.stat().st_size} B)")
        else:
            print("WARN: 数据集卡不存在，跳过")
        return 0

    if phase == "files":
        name = sys.argv[2]
        root = SRC / name
        if not root.exists():
            print(f"ERROR: {root} 不存在")
            return 2
        have = set(api.list_repo_files(REPO, repo_type="dataset"))
        targets = [f for f in sorted(root.rglob("*")) if f.is_file()
                   and not any(str(f).endswith(x) or "__pycache__" in str(f)
                               for x in (".pyc", ".log", ".pid", ".DS_Store"))]
        total = sum(f.stat().st_size for f in targets)
        print(f"targets={len(targets)} bytes={human(total)} already_in_repo={len(have)}", flush=True)
        done = skipped = failed = 0
        failed_list = []
        for f in targets:
            rel = f.relative_to(SRC).as_posix()
            if rel in have:
                skipped += 1
                continue
            ok = False
            for attempt in range(3):
                try:
                    api.upload_file(path_or_fileobj=str(f), path_in_repo=rel,
                                    repo_id=REPO, repo_type="dataset",
                                    commit_message=f"add {rel}")
                    ok = True
                    break
                except Exception as e:
                    print(f"  RETRY {rel} attempt={attempt+1} {type(e).__name__}: {str(e)[:120]}", flush=True)
                    time.sleep(5 * (attempt + 1))
            if ok:
                done += 1
                print(f"  ok  {rel}  {human(f.stat().st_size)}  [{done} new]", flush=True)
            else:
                failed += 1
                failed_list.append(rel)
                print(f"  FAIL {rel}", flush=True)
        print(f"SUMMARY new={done} skipped={skipped} failed={failed}", flush=True)
        if failed_list:
            print("FAILED FILES: " + ", ".join(failed_list), flush=True)
            return 1
        return 0

    dirs = PHASES.get(phase)
    if dirs is None:
        print("usage: publisher <init|core|data1|data2|big>")
        return 2

    for name in dirs:
        p = SRC / name
        if not p.exists():
            print(f"SKIP  {name} (不存在)")
            continue
        sz = dir_size(p)
        t0 = time.time()
        print(f"START {name}  {human(sz)}", flush=True)
        api.upload_folder(
            folder_path=str(p),
            path_in_repo=name,
            repo_id=REPO,
            repo_type="dataset",
            ignore_patterns=IGNORE,
            commit_message=f"add {name}",
        )
        print(f"DONE  {name}  {human(sz)}  {time.time()-t0:.0f}s", flush=True)

    if phase == "core":
        m = SRC / "MANIFEST.json"
        if m.exists():
            api.upload_file(path_or_fileobj=str(m), path_in_repo="MANIFEST.json",
                            repo_id=REPO, repo_type="dataset",
                            commit_message="add v1 manifest")
            print("DONE  MANIFEST.json")
        t0 = time.time()
        print("START exp* (合计约 1 MB)", flush=True)
        api.upload_folder(
            folder_path=str(SRC),
            path_in_repo="",
            repo_id=REPO,
            repo_type="dataset",
            allow_patterns=["exp*/*"],
            ignore_patterns=IGNORE,
            commit_message="add historical experiments",
        )
        print(f"DONE  exp*  {time.time()-t0:.0f}s", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
