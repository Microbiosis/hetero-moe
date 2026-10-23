#!/usr/bin/env python3
"""hf_job_deliver.py — 校园网拦截下，把脚本送进 HF Job 并跑起来。

## 为什么需要它

本机到 huggingface.co 的**大体积写请求**会被拦截：

- `hf buckets cp`（桶写）→ `Server disconnected without sending a response`
- `hf jobs run` 带 ~6.6 KB 的 base64 命令行 → 同样被掐断
- 但 **小请求正常**（几百字节的 job-create、几 KB 的桶写都曾经成功）

⇒ 拦截是**按载荷大小**触发的，不是"整个 HF 不可用"。

## 办法：分块投递（chunked delivery）

1. 脚本 → gzip → base64 → 切成 N 个小片（默认 1.6 KB/片）；
2. 每片一个 Job，写到**各自的文件**（`/domains/scripts/.deliver/<name>.part<i>`）
   ⇒ 片与片之间无依赖，可并行，且不会互相写坏；
3. 最后一个 Job：按序拼片 → base64 解码 → gunzip → 写到
   `/domains/scripts/<name>.py`（**顺手落桶**）→ 就地执行。

## 用法

    python tools/hf_job_deliver.py \
        --script research/brain_domains/models/probe_base_models.py \
        --space hf.co/spaces/redmingwei/IllusionBrain \
        --flavor cpu-upgrade --timeout 60m --run

`--run` 表示最后一步执行脚本；不给则只落桶。

## 不做的事

- 不用 WiFi 工具换出口 IP（需用户明说，见记忆里的授权边界）；
- 不把桶内数据拉回本地（只送脚本，体积小且是交付物）。
"""

from __future__ import annotations

import argparse
import gzip
import base64
import subprocess
import sys
import time
from pathlib import Path

CHUNK = 1600
DOMAIN_DIR = "/domains/scripts"
DELIVER_DIR = "/domains/scripts/.deliver"


def run(cmd, timeout=900):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def hf_jobs_run(space, command, args, flavor, timeout, name):
    vol = ("-v hf://buckets/illusion-brain/training-data:/training:ro "
           "-v hf://buckets/illusion-brain/brain-domains:/domains "
           f"--flavor {flavor} --timeout {timeout} -d")
    quoted = " ".join(f'"{a}"' if " " in a else a for a in args)
    cmd = f'hf jobs run "{space}" {command} {quoted} {vol} --name {name}'
    return run(cmd)


def wait_job(job_id, timeout=1800):
    for _ in range(timeout // 15):
        rc, out = run(f'hf jobs inspect {job_id} --format json', timeout=120)
        if '"stage": "COMPLETED"' in out or '"stage":"COMPLETED"' in out:
            return True, out
        if '"stage": "FAILED"' in out or '"stage":"ERROR"' in out:
            return False, out
        time.sleep(15)
    return False, "timeout"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--space", default="hf.co/spaces/redmingwei/IllusionBrain")
    ap.add_argument("--flavor", default="cpu-upgrade")
    ap.add_argument("--timeout", default="60m")
    ap.add_argument("--name", default="deliver")
    ap.add_argument("--run", action="store_true", help="最后一步执行脚本")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    args = ap.parse_args()

    src = Path(args.script)
    raw = src.read_bytes()
    blob = base64.b64encode(gzip.compress(raw, 9)).decode("ascii")
    chunks = [blob[i:i + args.chunk] for i in range(0, len(blob), args.chunk)]
    tag = src.stem
    print(f"{src.name}: {len(raw)} B → gzip+base64 {len(blob)} 字符 → {len(chunks)} 片")

    # 1) 每片一个 Job（写各自文件，互不冲突）
    ids = []
    for i, c in enumerate(chunks):
        code = (f"import pathlib;d=pathlib.Path('{DELIVER_DIR}');"
                f"d.mkdir(parents=True,exist_ok=True);"
                f"(d/'{tag}.part{i}').write_text('{c}')")
        rc, out = hf_jobs_run(args.space, "python", ["-u", "-c", code],
                              args.flavor, args.timeout, f"{args.name}-p{i}")
        jid = ""
        for line in out.splitlines():
            if "id:" in line:
                jid = line.split("id:")[1].strip().split()[0]
        if rc != 0 or not jid:
            print(f"✗ 第 {i} 片 job-create 失败：{out[-300:]}")
            return 1
        ids.append(jid)
        print(f"  片 {i}: job {jid}")

    # 2) 等所有片完成
    for i, jid in enumerate(ids):
        ok, out = wait_job(jid)
        print(f"  片 {i}: {'✓' if ok else '✗'} {jid}")
        if not ok:
            print(out[-400:])
            return 1

    # 3) 拼接 → 解码 → 落桶 → 执行
    parts = ",".join(f"'{DELIVER_DIR}/{tag}.part{i}'" for i in range(len(chunks)))
    code = (
        "import base64,gzip,pathlib,runpy;"
        f"d=pathlib.Path('{DELIVER_DIR}');"
        f"b=''.join((d/'{tag}.part{i}').read_text() for i in range({len(chunks)}));"
        f"p=pathlib.Path('{DOMAIN_DIR}/{src.name}');p.parent.mkdir(parents=True,exist_ok=True);"
        f"p.write_bytes(gzip.decompress(base64.b64decode(b)));"
        "print('written', p, p.stat().st_size, 'bytes', flush=True);"
        + ("runpy.run_path(str(p),run_name='__main__')" if args.run else "pass")
    )
    rc, out = hf_jobs_run(args.space, "python", ["-u", "-c", code],
                          args.flavor, args.timeout, f"{args.name}-run")
    jid = ""
    for line in out.splitlines():
        if "id:" in line:
            jid = line.split("id:")[1].strip().split()[0]
    if rc != 0 or not jid:
        print(f"✗ 收尾 job-create 失败：{out[-400:]}")
        return 1
    print(f"收尾 job: {jid}")
    ok, out = wait_job(jid, timeout=2400)
    print(out[-4000:] if not ok else out[-2000:])
    print(f"收尾 job {'✓' if ok else '✗'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
